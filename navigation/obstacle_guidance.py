"""
navigation/obstacle_guidance.py

Phase 3 safety guidance.

This module decides when a detected object should generate a
PATH-BLOCKING safety announcement.

Important design rule:

    Object detection != obstacle warning.

Normal objects such as:
    bottle
    laptop
    phone
    TV
    book

should not be announced by this module.

This module only receives detections that have already passed
the safety/path filter in navigation_main.py.

Rules:
    - Object must be seen for REQUIRED_FRAMES consecutive frames
      before first announcement.
    - Same object is not repeatedly announced unless its zone,
      proximity, or priority changes meaningfully.
    - Missing objects are eventually forgotten.
    - Disappearing objects DO NOT generate "Path ahead is clear".
      The system remains silent.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from navigation.spatial_analyzer import (
    DetectionResult,
    Priority,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of consecutive frames required before announcing.
REQUIRED_FRAMES: int = 3

# Number of missing frames before forgetting a track.
MAX_MISSING_FRAMES: int = 10


# Priority ranking.
_PRIORITY_RANK: Dict[str, int] = {
    Priority.IGNORE: 0,
    Priority.LOW: 1,
    Priority.MEDIUM: 2,
    Priority.HIGH: 3,
}


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackState:
    """
    State maintained for one tracked obstacle.
    """

    frames_seen: int = 0

    frames_missing: int = 0

    announced: bool = False

    last_zone: str = ""

    last_proximity: str = ""

    last_priority: str = ""


# ============================================================
# GUIDANCE DECISION
# ============================================================

@dataclass
class GuidanceDecision:
    """
    Result returned by ObstacleGuidance.

    should_announce:
        True when a safety announcement should be made.

    detection:
        Detection associated with the announcement.

    is_clear_notice:
        Always False in this version.

        We intentionally do NOT announce
        "Path ahead is clear" when an obstacle disappears.
    """

    should_announce: bool

    detection: DetectionResult

    is_clear_notice: bool = False


# ============================================================
# OBSTACLE GUIDANCE
# ============================================================

class ObstacleGuidance:
    """
    Stateful safety guidance engine.

    This class does NOT perform:

        YOLO
        camera capture
        distance estimation
        voice output

    It only decides whether a safety announcement is needed.
    """

    def __init__(self) -> None:

        # track_id -> TrackState
        self._tracks: Dict[int, TrackState] = {}

        # track_id -> latest DetectionResult
        self._last_detection: Dict[
            int,
            DetectionResult,
        ] = {}


    # ========================================================
    # PUBLIC
    # ========================================================

    def process_frame(
        self,
        detections: list[DetectionResult],
        current_track_ids: set[int],
    ) -> list[GuidanceDecision]:
        """
        Process detections from one frame.

        Only detections already considered safety obstacles
        should be passed here.

        Returns zero or more safety decisions.
        """

        decisions: list[GuidanceDecision] = []


        # ----------------------------------------------------
        # Process visible detections
        # ----------------------------------------------------

        for detection in detections:

            decision = self._update_and_decide(
                detection
            )

            if decision is not None:

                decisions.append(
                    decision
                )


        # ----------------------------------------------------
        # Update missing tracks
        #
        # IMPORTANT:
        # Missing objects no longer generate a "path clear"
        # announcement.
        # ----------------------------------------------------

        self._handle_missing(
            current_track_ids
        )


        # ----------------------------------------------------
        # Highest priority first
        # ----------------------------------------------------

        decisions.sort(
            key=lambda decision: _PRIORITY_RANK.get(
                decision.detection.priority,
                0,
            ),
            reverse=True,
        )


        return decisions


    # ========================================================
    # UPDATE ONE DETECTION
    # ========================================================

    def _update_and_decide(
        self,
        detection: DetectionResult,
    ) -> Optional[GuidanceDecision]:
        """
        Update state for one visible detection.
        """

        # IGNORE objects never generate safety guidance.
        if detection.priority == Priority.IGNORE:

            return None


        track_id = detection.track_id


        # ----------------------------------------------------
        # Create state if this is a new track.
        # ----------------------------------------------------

        if track_id not in self._tracks:

            self._tracks[track_id] = TrackState()


        state = self._tracks[
            track_id
        ]


        # ----------------------------------------------------
        # Store latest detection.
        # ----------------------------------------------------

        self._last_detection[
            track_id
        ] = detection


        # ----------------------------------------------------
        # Object is currently visible.
        # ----------------------------------------------------

        state.frames_missing = 0

        state.frames_seen = min(
            state.frames_seen + 1,
            9999,
        )


        # ----------------------------------------------------
        # Require stable detection.
        # ----------------------------------------------------

        if (
            state.frames_seen
            < REQUIRED_FRAMES
        ):

            return None


        # ----------------------------------------------------
        # Determine whether announcement is needed.
        # ----------------------------------------------------

        if not self._needs_announcement(
            state,
            detection,
        ):

            return None


        # ----------------------------------------------------
        # Save announced state.
        # ----------------------------------------------------

        state.announced = True

        state.last_zone = (
            detection.zone
        )

        state.last_proximity = (
            detection.proximity
        )

        state.last_priority = (
            detection.priority
        )


        return GuidanceDecision(
            should_announce=True,
            detection=detection,
            is_clear_notice=False,
        )


    # ========================================================
    # ANNOUNCEMENT DECISION
    # ========================================================

    def _needs_announcement(
        self,
        state: TrackState,
        detection: DetectionResult,
    ) -> bool:
        """
        Decide whether a new safety announcement is needed.

        Announces when:

            1. Object is seen for the first time.

            2. Zone changes.

            3. Object moves closer.

            4. Priority becomes more urgent.
        """

        # ----------------------------------------------------
        # First announcement.
        # ----------------------------------------------------

        if not state.announced:

            return True


        # ----------------------------------------------------
        # Direction changed.
        # ----------------------------------------------------

        if (
            detection.zone
            != state.last_zone
        ):

            return True


        # ----------------------------------------------------
        # Object became closer.
        # ----------------------------------------------------

        if self._proximity_increased(
            state.last_proximity,
            detection.proximity,
        ):

            return True


        # ----------------------------------------------------
        # Priority increased.
        # ----------------------------------------------------

        current_priority = (
            _PRIORITY_RANK.get(
                detection.priority,
                0,
            )
        )

        previous_priority = (
            _PRIORITY_RANK.get(
                state.last_priority,
                0,
            )
        )


        if current_priority > previous_priority:

            return True


        return False


    # ========================================================
    # PROXIMITY
    # ========================================================

    @staticmethod
    def _proximity_increased(
        old: str,
        new: str,
    ) -> bool:
        """
        Return True when an object moves closer.

        FAR -> NEAR      True
        FAR -> CLOSE     True
        NEAR -> CLOSE    True

        CLOSE -> NEAR    False
        NEAR -> FAR      False
        """

        proximity_rank = {
            "FAR": 0,
            "NEAR": 1,
            "CLOSE": 2,
        }


        old_rank = proximity_rank.get(
            old,
            0,
        )

        new_rank = proximity_rank.get(
            new,
            0,
        )


        return new_rank > old_rank


    # ========================================================
    # MISSING TRACKS
    # ========================================================

    def _handle_missing(
        self,
        current_track_ids: set[int],
    ) -> None:
        """
        Update missing-frame counters.

        IMPORTANT:

        When an obstacle disappears we simply remove its state
        after MAX_MISSING_FRAMES.

        We do NOT create a "Path ahead is clear" announcement.

        This keeps the voice assistant focused on:
            - requested objects
            - actual path hazards
        """

        for track_id in list(
            self._tracks.keys()
        ):

            # Still visible.
            if track_id in current_track_ids:

                continue


            state = self._tracks[
                track_id
            ]

            state.frames_missing += 1


            # ------------------------------------------------
            # Forget after too many missing frames.
            # ------------------------------------------------

            if (
                state.frames_missing
                >= MAX_MISSING_FRAMES
            ):

                self._tracks.pop(
                    track_id,
                    None,
                )

                self._last_detection.pop(
                    track_id,
                    None,
                )