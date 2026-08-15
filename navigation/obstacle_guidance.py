"""
navigation/obstacle_guidance.py

Decides WHEN a detected obstacle deserves a guidance announcement.

This module receives DetectionResult objects (produced by spatial_analyzer.py)
and applies state-tracking logic to determine whether the obstacle guidance
system should announce anything to the user.

Rules:
  - An object must be seen in at least 3 consecutive frames before its
    first announcement (avoids false triggers).
  - Once announced, the same object is not announced again unless its
    zone, proximity, or priority changes in a meaningful way.
  - If an object disappears for 10 consecutive frames it is forgotten
    and can be announced again when it reappears.
  - IGNORE-priority objects never generate guidance.

No camera, no YOLO model, no OpenCV, no voice output — only decision logic.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from navigation.spatial_analyzer import DetectionResult, Priority


# ---------------------------------------------------------------------------
# Configuration constants (mirror yolo_voice_manager.py conventions)
# ---------------------------------------------------------------------------

# Frames an object must be visible before its first announcement
REQUIRED_FRAMES: int = 3

# Frames an object can be missing before it is forgotten entirely
MAX_MISSING_FRAMES: int = 10

# Priority order — higher index = more urgent
# Used to detect when priority increases (becomes more concerning)
_PRIORITY_RANK: Dict[str, int] = {
    Priority.IGNORE: 0,
    Priority.LOW:    1,
    Priority.MEDIUM: 2,
    Priority.HIGH:   3,
}


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """
    Internal state maintained for a single tracked object.

    Attributes
    ----------
    frames_seen : int
        Number of consecutive frames this object has been detected.
    frames_missing : int
        Number of consecutive frames this object has NOT been detected.
    announced : bool
        Whether an announcement has already been made for the current
        zone + proximity + priority combination.
    last_zone : str
        Zone reported the last time the object was announced.
    last_proximity : str
        Proximity tier reported the last time the object was announced.
    last_priority : str
        Priority reported the last time the object was announced.
    was_center_blocker : bool
        True if the last announcement was for a CENTER-zone obstacle.
        Used to generate a "path clear" notice when it disappears.
    """
    frames_seen       : int  = 0
    frames_missing    : int  = 0
    announced         : bool = False
    last_zone         : str  = ""
    last_proximity    : str  = ""
    last_priority     : str  = ""
    was_center_blocker: bool = False


# ---------------------------------------------------------------------------
# Guidance decision output
# ---------------------------------------------------------------------------

@dataclass
class GuidanceDecision:
    """
    The result produced by ObstacleGuidance for a single frame.

    Attributes
    ----------
    should_announce : bool
        True when the guidance system has decided an announcement is needed.
    detection : DetectionResult
        The full spatial analysis result for the obstacle being announced.
    is_clear_notice : bool
        True when this decision is a "path ahead clear" notice, meaning
        a previously blocking CENTER obstacle has now disappeared.
        When True, `detection` contains the last known state of that object.
    """
    should_announce : bool
    detection       : DetectionResult
    is_clear_notice : bool = False


# ---------------------------------------------------------------------------
# Main guidance class
# ---------------------------------------------------------------------------

class ObstacleGuidance:
    """
    Stateful guidance decision engine for Phase 3 obstacle guidance.

    Receives DetectionResult objects one frame at a time and decides
    whether the user should be informed about each obstacle.

    Usage
    -----
    guidance = ObstacleGuidance()

    # Each frame, pass all current detections:
    decisions = guidance.process_frame(detections, current_track_ids)

    # Then pass each decision to direction_assistant.py for message building.
    """

    def __init__(self) -> None:
        # Maps track_id → TrackState for every currently tracked object
        self._tracks: Dict[int, TrackState] = {}

        # Stores the last DetectionResult for each track_id so we can
        # still reference it even after the object disappears
        self._last_detection: Dict[int, DetectionResult] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_frame(
        self,
        detections      : list[DetectionResult],
        current_track_ids: set[int]
    ) -> list[GuidanceDecision]:
        """
        Process all detections from a single camera frame.

        Steps:
          1. Update state for every currently visible detection.
          2. Decide whether each visible obstacle warrants an announcement.
          3. Update missing-frame counts for objects not seen this frame.
          4. Forget objects that have been missing too long.
          5. Generate "path clear" notices for CENTER blockers that disappeared.

        Parameters
        ----------
        detections : list[DetectionResult]
            All spatial analysis results from the current frame.
            Should only include objects with is_obstacle=True (non-IGNORE).
        current_track_ids : set[int]
            Set of track IDs present in this frame (used to detect
            disappearances for objects that may not be in `detections`
            because they were filtered out upstream).

        Returns
        -------
        list[GuidanceDecision]
            Zero or more guidance decisions for this frame.
            Sorted by priority (HIGH first) so the caller can pick the
            most important one if only one announcement is possible.
        """
        decisions: list[GuidanceDecision] = []

        # Step 1 & 2 — process visible detections
        for det in detections:
            decision = self._update_and_decide(det)
            if decision is not None:
                decisions.append(decision)

        # Step 3, 4 & 5 — handle disappearances
        clear_notices = self._handle_missing(current_track_ids)
        decisions.extend(clear_notices)

        # Sort so HIGH priority announcements come first
        decisions.sort(
            key=lambda d: _PRIORITY_RANK.get(d.detection.priority, 0),
            reverse=True
        )

        return decisions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_and_decide(
        self,
        det: DetectionResult
    ) -> Optional[GuidanceDecision]:
        """
        Update the track state for one visible detection and decide
        whether it should be announced.

        Parameters
        ----------
        det : DetectionResult

        Returns
        -------
        GuidanceDecision or None
        """
        # IGNORE objects never generate guidance
        if det.priority == Priority.IGNORE:
            return None

        track_id = det.track_id

        # Retrieve or create state for this track
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState()

        state = self._tracks[track_id]

        # Always store the latest detection for later reference
        self._last_detection[track_id] = det

        # Object is visible this frame — reset missing count
        state.frames_missing = 0

        # Increment stable-frame count (cap at a large number to avoid overflow)
        state.frames_seen = min(state.frames_seen + 1, 9999)

        # Not yet stable — do not announce
        if state.frames_seen < REQUIRED_FRAMES:
            return None

        # Decide whether a (new) announcement is needed
        if self._needs_announcement(state, det):
            # Record the state at the time of announcement
            state.announced         = True
            state.last_zone         = det.zone
            state.last_proximity    = det.proximity
            state.last_priority     = det.priority
            state.was_center_blocker = (det.zone == "CENTER")

            return GuidanceDecision(
                should_announce = True,
                detection       = det,
                is_clear_notice = False,
            )

        return None

    def _needs_announcement(
        self,
        state: TrackState,
        det  : DetectionResult
    ) -> bool:
        """
        Determine whether this detection warrants a new announcement.

        An announcement is needed when:
          - The object has never been announced yet (first stable detection).
          - The zone changed    (e.g. LEFT → CENTER or CENTER → RIGHT).
          - The proximity tier changed to something closer
            (FAR → NEAR, or FAR/NEAR → CLOSE).
          - The priority increased (became more urgent).

        Parameters
        ----------
        state : TrackState
            Current state for this track.
        det : DetectionResult
            The new detection.

        Returns
        -------
        bool
        """
        # First-ever announcement for this track
        if not state.announced:
            return True

        # Zone changed
        if det.zone != state.last_zone:
            return True

        # Proximity moved CLOSER (not re-announce if it moved farther away)
        if self._proximity_increased(state.last_proximity, det.proximity):
            return True

        # Priority became more urgent
        if (
            _PRIORITY_RANK.get(det.priority, 0)
            > _PRIORITY_RANK.get(state.last_priority, 0)
        ):
            return True

        return False

    @staticmethod
    def _proximity_increased(old: str, new: str) -> bool:
        """
        Return True if the new proximity tier is closer than the old one.

        FAR → NEAR   : True
        FAR → CLOSE  : True
        NEAR → CLOSE : True
        CLOSE → NEAR : False  (object moved away — no re-announcement)
        NEAR → FAR   : False

        Parameters
        ----------
        old : str
            Previous proximity tier.
        new : str
            Current proximity tier.
        """
        proximity_rank = {"FAR": 0, "NEAR": 1, "CLOSE": 2}
        old_rank = proximity_rank.get(old, 0)
        new_rank = proximity_rank.get(new, 0)
        return new_rank > old_rank

    def _handle_missing(
        self,
        current_track_ids: set[int]
    ) -> list[GuidanceDecision]:
        """
        Increment missing-frame counts for absent tracks, forget tracks
        that have been missing too long, and emit "path clear" notices
        for CENTER blockers that just disappeared.

        Parameters
        ----------
        current_track_ids : set[int]
            Track IDs that are visible in the current frame.

        Returns
        -------
        list[GuidanceDecision]
            Zero or one "path clear" GuidanceDecision per forgotten track.
        """
        clear_notices: list[GuidanceDecision] = []

        for track_id in list(self._tracks.keys()):

            if track_id in current_track_ids:
                continue  # Visible this frame — already handled above

            state = self._tracks[track_id]
            state.frames_missing += 1

            if state.frames_missing >= MAX_MISSING_FRAMES:

                # If the object was a CENTER blocker, notify that path is clear
                if (
                    state.announced
                    and state.was_center_blocker
                    and track_id in self._last_detection
                ):
                    clear_notices.append(GuidanceDecision(
                        should_announce = True,
                        detection       = self._last_detection[track_id],
                        is_clear_notice = True,
                    ))

                # Forget this track entirely
                self._tracks.pop(track_id, None)
                self._last_detection.pop(track_id, None)

        return clear_notices
