"""
navigation/target_guidance.py

Safe continuous guidance for a requested target.

Responsibilities:
    - Track requested target
    - Stabilize LEFT / CENTER / RIGHT direction
    - Generate clear direction guidance
    - Avoid unnecessary repeated direction announcements
    - Provide safety-priority messages for obstacles/steps

Distance and actual speech are handled by navigation_main.py.
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

# Frames required to confirm a direction.
REQUIRED_ZONE_FRAMES = 3

# Frames before target is considered lost.
TARGET_LOST_FRAMES = 20

# ============================================================
# SAFETY OBJECTS
# ============================================================

# These names can be supplied by a future/custom detector.
# Standard YOLOv8 COCO does NOT contain "stairs" or "steps".
STAIR_CLASSES = {
    "stairs",
    "stair",
    "steps",
    "step",
}

OBSTACLE_CLASSES = {
    "obstacle",
}


# ============================================================
# RESULT
# ============================================================

@dataclass
class TargetGuidanceResult:

    message: Optional[str]

    zone: str

    found: bool

    confidence: float

    # Safety information
    danger_ahead: bool = False

    danger_type: Optional[str] = None


# ============================================================
# TARGET GUIDANCE
# ============================================================

class TargetGuidance:

    def __init__(self) -> None:

        self._active = False

        self._target_name: Optional[str] = None

        # Last confirmed direction.
        self._last_announced_zone: Optional[str] = None

        # Direction waiting for confirmation.
        self._pending_zone: Optional[str] = None

        self._pending_count = 0

        # Missing target frames.
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

        target_name = (
            target_name
            .strip()
            .lower()
        )

        if not target_name:
            return None

        self._active = True

        self._target_name = target_name

        if initial_zone:
            self._last_announced_zone = initial_zone
        else:
            self._last_announced_zone = None

        self._pending_zone = None
        self._pending_count = 0
        self._missing_frames = 0

        return None


    # ========================================================
    # STOP
    # ========================================================

    def stop(self) -> None:

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

        # ----------------------------------------------------
        # Inactive
        # ----------------------------------------------------

        if not self.is_active:

            return TargetGuidanceResult(
                message=None,
                zone="",
                found=False,
                confidence=0.0,
            )


        # ----------------------------------------------------
        # SAFETY CHECK FIRST
        # ----------------------------------------------------

        danger_type = self._check_danger_ahead(
            detections
        )

        if danger_type is not None:

            return TargetGuidanceResult(
                message=self._build_safety_message(
                    danger_type
                ),
                zone=Zone.CENTER,
                found=True,
                confidence=0.0,
                danger_ahead=True,
                danger_type=danger_type,
            )


        # ----------------------------------------------------
        # Find requested target
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

            self._pending_zone = None
            self._pending_count = 0

            if (
                self._missing_frames
                >= TARGET_LOST_FRAMES
            ):

                target_name = self._target_name

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
        # FIRST CONFIRMED TARGET DIRECTION
        # ====================================================

        if self._last_announced_zone is None:

            if (
                current_zone
                == self._pending_zone
            ):

                self._pending_count += 1

            else:

                self._pending_zone = current_zone
                self._pending_count = 1


            if (
                self._pending_count
                >= REQUIRED_ZONE_FRAMES
            ):

                self._last_announced_zone = current_zone

                self._pending_zone = None
                self._pending_count = 0

                return TargetGuidanceResult(
                    message=self._build_message(
                        current_zone
                    ),
                    zone=current_zone,
                    found=True,
                    confidence=confidence,
                )


            return TargetGuidanceResult(
                message=None,
                zone=current_zone,
                found=True,
                confidence=confidence,
            )


        # ====================================================
        # SAME DIRECTION
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
        # NEW DIRECTION
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
        # CONFIRM NEW DIRECTION
        # ====================================================

        if (
            self._pending_count
            >= REQUIRED_ZONE_FRAMES
        ):

            self._last_announced_zone = current_zone

            self._pending_zone = None
            self._pending_count = 0

            return TargetGuidanceResult(
                message=self._build_message(
                    current_zone
                ),
                zone=current_zone,
                found=True,
                confidence=confidence,
            )


        return TargetGuidanceResult(
            message=None,
            zone=current_zone,
            found=True,
            confidence=confidence,
        )


    # ========================================================
    # DIRECTION MESSAGE
    # ========================================================

    def _build_message(
        self,
        zone: str,
    ) -> str:

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
    # SAFETY MESSAGE
    # ========================================================

    @staticmethod
    def _build_safety_message(
        danger_type: str,
    ) -> str:

        if danger_type == "steps":

            return (
                "Steps detected ahead. "
                "Stop. Do not walk forward."
            )

        return (
            "Obstacle detected ahead. "
            "Stop."
        )


    # ========================================================
    # DANGER DETECTION
    # ========================================================

    @staticmethod
    def _check_danger_ahead(
        detections: list[DetectionResult],
    ) -> Optional[str]:

        for detection in detections:

            name = (
                detection.object_name
                .strip()
                .lower()
            )

            # ----------------------------------------------
            # Steps / stairs
            # ----------------------------------------------

            if name in STAIR_CLASSES:

                if detection.zone == Zone.CENTER:

                    return "steps"


            # ----------------------------------------------
            # Generic obstacle
            # ----------------------------------------------

            if name in OBSTACLE_CLASSES:

                if detection.zone == Zone.CENTER:

                    return "obstacle"


        return None


    # ========================================================
    # FIND BEST TARGET
    # ========================================================

    @staticmethod
    def _find_best_match(
        detections: list[DetectionResult],
        target: str,
    ) -> Optional[DetectionResult]:

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

        return max(
            candidates,
            key=lambda detection: (
                detection.confidence,
                detection.box_area,
            ),
        )

