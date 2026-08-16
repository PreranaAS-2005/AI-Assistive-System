"""
navigation/target_guidance.py

Continuous guidance for a requested target object.

Example:

    find chair

Then guide the user:

    Your chair is on your left.
    Your chair is ahead.
    Your chair is on your right.

Distance is handled separately by navigation_main.py.
"""

from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import (
    DetectionResult,
    Zone,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of consecutive frames required before accepting
# a direction change.
REQUIRED_ZONE_FRAMES: int = 2

# Number of consecutive missing frames before deciding
# that the target has really disappeared.
TARGET_LOST_FRAMES: int = 20


# ============================================================
# RESULT
# ============================================================

@dataclass
class TargetGuidanceResult:

    message: Optional[str]

    zone: str

    found: bool

    confidence: float


# ============================================================
# TARGET GUIDANCE
# ============================================================

class TargetGuidance:
    """
    Handles continuous guidance for the object requested
    by the user.

    It does NOT:

        - capture camera frames
        - run YOLO
        - calculate distance
        - speak directly

    It only decides when a target-direction message should
    be generated.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:

        self._active = False

        self._target_name: Optional[str] = None

        # Last direction that was actually announced.
        self._last_announced_zone: Optional[str] = None

        # Direction waiting for confirmation.
        self._pending_zone: Optional[str] = None

        # Number of consecutive frames supporting pending zone.
        self._pending_count = 0

        # Number of consecutive frames where target was missing.
        self._missing_frames = 0


    # ========================================================
    # PROPERTIES
    # ========================================================

    @property
    def is_active(self) -> bool:

        return (
            self._active
            and self._target_name is not None
        )


    @property
    def target_name(self) -> Optional[str]:

        return self._target_name


    # ========================================================
    # START
    # ========================================================

    def start(
        self,
        target_name: str,
        initial_zone: str = "",
    ) -> Optional[str]:
        """
        Start guidance for the requested object.

        If initial_zone is supplied, that zone is considered
        already announced.

        If initial_zone is empty, the first stable detection
        generates the first direction message.
        """

        target_name = (
            target_name
            .strip()
            .lower()
        )

        if not target_name:

            return None


        self._active = True

        self._target_name = target_name


        # ----------------------------------------------------
        # Initial zone
        #
        # This is important for the unit test:
        #
        # If CENTER was already announced, another CENTER
        # announcement should not be generated.
        # ----------------------------------------------------

        if initial_zone:

            self._last_announced_zone = (
                initial_zone
            )

        else:

            self._last_announced_zone = None


        # ----------------------------------------------------
        # Reset pending state
        # ----------------------------------------------------

        self._pending_zone = None

        self._pending_count = 0

        self._missing_frames = 0

        return None


    # ========================================================
    # STOP
    # ========================================================

    def stop(self) -> None:
        """
        Stop target guidance and clear all state.
        """

        self._active = False

        self._target_name = None

        self._last_announced_zone = None

        self._pending_zone = None

        self._pending_count = 0

        self._missing_frames = 0


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        detections: list[DetectionResult],
    ) -> TargetGuidanceResult:
        """
        Process the current frame.

        Only the requested object is considered.
        """

        # ----------------------------------------------------
        # Guidance inactive
        # ----------------------------------------------------

        if (
            not self.is_active
            or self._target_name is None
        ):

            return TargetGuidanceResult(
                message=None,
                zone="",
                found=False,
                confidence=0.0,
            )


        # ----------------------------------------------------
        # Find requested object
        # ----------------------------------------------------

        match = self._find_best_match(
            detections,
            self._target_name,
        )


        # ====================================================
        # TARGET NOT FOUND
        # ====================================================

        if match is None:

            self._missing_frames += 1

            # Temporary YOLO miss should not immediately
            # produce "no longer visible".
            self._pending_zone = None

            self._pending_count = 0


            # ------------------------------------------------
            # Target really disappeared
            # ------------------------------------------------

            if (
                self._missing_frames
                >= TARGET_LOST_FRAMES
            ):

                target_name = (
                    self._target_name
                )

                message = (
                    f"Your {target_name} "
                    "is no longer visible. "
                    "Please search again."
                )

                self.stop()

                return TargetGuidanceResult(
                    message=message,
                    zone="",
                    found=False,
                    confidence=0.0,
                )


            return TargetGuidanceResult(
                message=None,
                zone="",
                found=False,
                confidence=0.0,
            )


        # ====================================================
        # TARGET FOUND
        # ====================================================

        self._missing_frames = 0

        current_zone = match.zone

        confidence = match.confidence


        # ====================================================
        # SAME ZONE
        # ====================================================

        if (
            current_zone
            == self._last_announced_zone
        ):

            self._pending_zone = None

            self._pending_count = 0

            return TargetGuidanceResult(
                message=None,
                zone=current_zone,
                found=True,
                confidence=confidence,
            )


        # ====================================================
        # NEW ZONE
        # ====================================================

        if (
            current_zone
            == self._pending_zone
        ):

            self._pending_count += 1

        else:

            self._pending_zone = current_zone

            self._pending_count = 1


        # ====================================================
        # CONFIRM DIRECTION
        # ====================================================

        if (
            self._pending_count
            >= REQUIRED_ZONE_FRAMES
        ):

            self._last_announced_zone = (
                current_zone
            )

            self._pending_zone = None

            self._pending_count = 0


            message = self._build_message(
                current_zone
            )


            return TargetGuidanceResult(
                message=message,
                zone=current_zone,
                found=True,
                confidence=confidence,
            )


        # ----------------------------------------------------
        # Direction not stable enough yet.
        # ----------------------------------------------------

        return TargetGuidanceResult(
            message=None,
            zone=current_zone,
            found=True,
            confidence=confidence,
        )


    # ========================================================
    # BUILD DIRECTION MESSAGE
    # ========================================================

    def _build_message(
        self,
        zone: str,
    ) -> str:
        """
        Convert spatial zone into a natural voice message.
        """

        if zone == Zone.LEFT:

            direction = "on your left"

        elif zone == Zone.RIGHT:

            direction = "on your right"

        else:

            direction = "ahead"


        return (
            f"Your {self._target_name} "
            f"is {direction}."
        )


    # ========================================================
    # FIND BEST TARGET
    # ========================================================

    @staticmethod
    def _find_best_match(
        detections: list[DetectionResult],
        target: str,
    ) -> Optional[DetectionResult]:
        """
        Find the strongest detection matching the requested
        object name.
        """

        target_lower = (
            target
            .strip()
            .lower()
        )


        candidates = [
            detection
            for detection in detections
            if (
                detection.object_name
                .strip()
                .lower()
                == target_lower
            )
        ]


        if not candidates:

            return None


        # Prefer:
        #   1. highest confidence
        #   2. largest bounding box

        return max(
            candidates,
            key=lambda detection: (
                detection.confidence,
                detection.box_area,
            ),
        )