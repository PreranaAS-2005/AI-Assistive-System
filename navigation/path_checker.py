"""
navigation/path_checker.py

Path-aware obstacle detection for the AI-Based Assistive Spatial Memory
and Guidance System.

This module determines whether any detected object is actually blocking
the route from the user toward their selected target.

Design principle:
    The user is at the bottom of the frame.
    The target defines the intended travel corridor (LEFT / CENTER / RIGHT).
    An obstacle blocks the path when it occupies the same corridor AND
    appears in front of (closer than) the target.

No camera, no YOLO, no voice output — pure geometry and classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import DetectionResult, Zone


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PathStatus:
    """
    Result of one path-check evaluation.

    Attributes
    ----------
    is_clear : bool
        True when the route toward the target is unobstructed.
    blocker : DetectionResult or None
        The highest-priority obstacle in the route corridor, or None.
    avoidance_direction : str
        Suggested avoidance direction when blocked: "left", "right", or "".
    avoidance_message : str
        Ready-to-speak avoidance instruction (e.g. "Move left.").
    """
    is_clear           : bool
    blocker            : Optional[DetectionResult]
    avoidance_direction: str
    avoidance_message  : str


# ---------------------------------------------------------------------------
# Priority rank used for blocker selection
# ---------------------------------------------------------------------------

_PRIORITY_RANK: dict[str, int] = {
    "IGNORE": 0,
    "LOW"   : 1,
    "MEDIUM": 2,
    "HIGH"  : 3,
}


# ---------------------------------------------------------------------------
# PathChecker
# ---------------------------------------------------------------------------

class PathChecker:
    """
    Evaluates whether the route toward a target is clear.

    Usage
    -----
    checker = PathChecker()

    status = checker.check(
        target_detection   = target_det,
        all_detections     = all_dets,
        target_name        = "bottle",
    )

    if not status.is_clear:
        voice.speak(status.avoidance_message)
    else:
        # Path is clear — proceed with target guidance
        ...

    Algorithm
    ---------
    1.  Determine the target's zone (LEFT / CENTER / RIGHT).
        This defines the route corridor.

    2.  Identify obstacle candidates:
        - Same zone as the target (in the corridor), OR
        - CENTER zone always qualifies as a potential blocker
          regardless of target zone (it is always on the way).
        - The candidate must not be the target itself.
        - The candidate must have a non-IGNORE priority.

    3.  Among candidates, keep only those that appear CLOSER than
        the target (larger bounding-box area approximates closer).
        This prevents labelling a farther object as a blocker.

    4.  If any blockers remain, select the one with the highest
        priority (and largest box if tied).

    5.  Suggest avoidance:
        - Target LEFT   → blocked → try going around on the RIGHT
        - Target CENTER → blocked → look at LEFT vs RIGHT openness
        - Target RIGHT  → blocked → try going around on the LEFT
    """

    def check(
        self,
        target_detection: DetectionResult,
        all_detections  : list[DetectionResult],
        target_name     : str,
    ) -> PathStatus:
        """
        Check whether the route to the target is clear.

        Parameters
        ----------
        target_detection : DetectionResult
            The best current detection of the target object.
        all_detections : list[DetectionResult]
            Every detection in the current frame (including non-obstacles).
        target_name : str
            Normalised YOLO class name of the target (e.g. "bottle").

        Returns
        -------
        PathStatus
        """
        target_zone = target_detection.zone
        target_area = target_detection.box_area
        target_id   = target_detection.track_id

        # ------------------------------------------------------------------
        # Build candidate blockers
        # ------------------------------------------------------------------
        candidates: list[DetectionResult] = []

        for det in all_detections:

            # Never treat the target itself as a blocker
            if det.track_id == target_id:
                continue
            if det.object_name.lower() == target_name.lower():
                continue

            # Must have a navigational priority (not IGNORE)
            if det.priority == "IGNORE":
                continue

            # Must be in the same corridor as the target, OR in CENTER
            # (center is always in every route)
            in_corridor = (
                det.zone == target_zone
                or det.zone == Zone.CENTER
            )
            if not in_corridor:
                continue

            # Must appear CLOSER than the target
            # (larger box area ≈ closer to camera)
            if det.box_area <= target_area:
                continue

            candidates.append(det)

        # ------------------------------------------------------------------
        # No blocker found
        # ------------------------------------------------------------------
        if not candidates:
            return PathStatus(
                is_clear            = True,
                blocker             = None,
                avoidance_direction = "",
                avoidance_message   = "",
            )

        # ------------------------------------------------------------------
        # Pick the most significant blocker
        # ------------------------------------------------------------------
        blocker = max(
            candidates,
            key=lambda d: (
                _PRIORITY_RANK.get(d.priority, 0),
                d.box_area,
            ),
        )

        # ------------------------------------------------------------------
        # Suggest avoidance direction
        # ------------------------------------------------------------------
        avoidance_dir, avoidance_msg = self._suggest_avoidance(
            target_zone   = target_zone,
            all_detections = all_detections,
            blocker       = blocker,
        )

        return PathStatus(
            is_clear            = False,
            blocker             = blocker,
            avoidance_direction = avoidance_dir,
            avoidance_message   = avoidance_msg,
        )

    # ------------------------------------------------------------------
    # Avoidance direction logic
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_avoidance(
        target_zone   : str,
        all_detections: list[DetectionResult],
        blocker       : DetectionResult,
    ) -> tuple[str, str]:
        """
        Suggest which way to go around the blocker.

        Strategy
        --------
        - Count obstacle density in LEFT vs RIGHT zones.
        - Prefer the side with fewer / less severe obstacles.
        - If target is LEFT/RIGHT, prefer staying on the same
          general side when possible; otherwise swap.

        Returns
        -------
        (direction_str, spoken_message) : tuple[str, str]
            direction_str is "left" or "right".
        """
        # Count total obstacle weight in each flank zone
        left_weight  = sum(
            _PRIORITY_RANK.get(d.priority, 0)
            for d in all_detections
            if d.zone == Zone.LEFT and d.priority != "IGNORE"
        )
        right_weight = sum(
            _PRIORITY_RANK.get(d.priority, 0)
            for d in all_detections
            if d.zone == Zone.RIGHT and d.priority != "IGNORE"
        )

        if target_zone == Zone.LEFT:
            # Target is left — try going even more left if possible,
            # otherwise route right around the blocker
            direction = "left" if left_weight <= right_weight else "right"

        elif target_zone == Zone.RIGHT:
            # Target is right — try going right if clearer, else left
            direction = "right" if right_weight <= left_weight else "left"

        else:
            # Target is CENTER — pick the clearer flank
            direction = "left" if left_weight <= right_weight else "right"

        blocker_name = blocker.object_name.lower()
        message = (
            f"Path is not clear. "
            f"{blocker_name.capitalize()} is blocking your path. "
            f"Move {direction}."
        )

        return direction, message
