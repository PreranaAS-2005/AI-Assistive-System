"""
Continuous target guidance for the AI-Based Assistive Spatial Memory
and Guidance System.

After TargetSearch confirms an object, this module keeps tracking that
specific object and announces meaningful LEFT/CENTER/RIGHT changes.

No camera, YOLO, OpenCV, or voice output lives here.
"""

from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import DetectionResult, Zone


# Require the same direction for several frames before announcing it.
REQUIRED_ZONE_FRAMES: int = 3

# If the target is absent for this many frames, report that it is no longer
# visible and stop guidance.
TARGET_LOST_FRAMES: int = 10


@dataclass
class TargetGuidanceResult:
    message: Optional[str]
    zone: str
    found: bool
    confidence: float


class TargetGuidance:
    """Continuously guide toward one confirmed target object."""

    def __init__(self) -> None:
        self._active = False
        self._target_name: Optional[str] = None
        self._last_announced_zone: Optional[str] = None
        self._pending_zone: Optional[str] = None
        self._pending_count = 0
        self._missing_frames = 0

    @property
    def is_active(self) -> bool:
        return self._active and self._target_name is not None

    @property
    def target_name(self) -> Optional[str]:
        return self._target_name

    def start(self, target_name: str, initial_zone: str = "") -> Optional[str]:
        """Start continuous guidance after a target has been confirmed."""
        target_name = target_name.strip().lower()
        if not target_name:
            return None

        self._active = True
        self._target_name = target_name
        self._last_announced_zone = initial_zone or None
        self._pending_zone = None
        self._pending_count = 0
        self._missing_frames = 0

        # The confirmation message has already been spoken by TargetSearch.
        # Suppress an immediate duplicate for the same zone.
        return None

    def stop(self) -> None:
        self._active = False
        self._target_name = None
        self._last_announced_zone = None
        self._pending_zone = None
        self._pending_count = 0
        self._missing_frames = 0

    def process_frame(
        self,
        detections: list[DetectionResult],
    ) -> TargetGuidanceResult:
        """Track the target and announce only stable direction changes."""
        if not self.is_active or self._target_name is None:
            return TargetGuidanceResult(None, "", False, 0.0)

        match = self._find_best_match(detections, self._target_name)

        if match is None:
            self._missing_frames += 1
            self._pending_zone = None
            self._pending_count = 0

            if self._missing_frames >= TARGET_LOST_FRAMES:
                message = (
                    f"Your {self._target_name} is no longer visible. "
                    "Please search again."
                )
                self.stop()
                return TargetGuidanceResult(message, "", False, 0.0)

            return TargetGuidanceResult(None, "", False, 0.0)

        self._missing_frames = 0
        zone = match.zone
        confidence = match.confidence

        if zone == self._last_announced_zone:
            self._pending_zone = None
            self._pending_count = 0
            return TargetGuidanceResult(None, zone, True, confidence)

        if zone == self._pending_zone:
            self._pending_count += 1
        else:
            self._pending_zone = zone
            self._pending_count = 1

        if self._pending_count >= REQUIRED_ZONE_FRAMES:
            self._last_announced_zone = zone
            self._pending_zone = None
            self._pending_count = 0
            return TargetGuidanceResult(
                self._build_message(zone),
                zone,
                True,
                confidence,
            )

        return TargetGuidanceResult(None, zone, True, confidence)

    def _build_message(self, zone: str) -> str:
        if zone == Zone.LEFT:
            direction = "on your left"
        elif zone == Zone.RIGHT:
            direction = "on your right"
        else:
            direction = "ahead"
        return f"Your {self._target_name} is {direction}."

    @staticmethod
    def _find_best_match(
        detections: list[DetectionResult],
        target: str,
    ) -> Optional[DetectionResult]:
        candidates = [
            d for d in detections
            if d.object_name.lower() == target.lower()
        ]
        if not candidates:
            return None

        return max(candidates, key=lambda d: (d.confidence, d.box_area))