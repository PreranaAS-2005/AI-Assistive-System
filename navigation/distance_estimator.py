"""
Approximate monocular distance estimation for the assistive navigation system.

This module does NOT claim true depth. It estimates distance from a detected
object's bounding-box width using:

    distance = (known_width * focal_length) / pixel_width

The focal length must be calibrated for the actual camera.
"""

from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import DetectionResult


@dataclass(frozen=True)
class DistanceEstimate:
    object_name: str
    distance_m: float
    confidence: float
    method: str = "bounding_box_width"


# Initial approximate real-world widths in metres.
# These values are deliberately conservative and can be calibrated later.
REFERENCE_WIDTHS_M = {
    "person": 0.45,
    "bottle": 0.07,
    "cell phone": 0.075,
    "laptop": 0.32,
    "tv": 0.90,
    "chair": 0.45,
    "cup": 0.08,
    "book": 0.15,
    "couch": 1.80,
    "bed": 1.50,
    "dining table": 1.20,
    "backpack": 0.32,
    "keyboard": 0.44,
}


class DistanceEstimator:
    """Estimate approximate distance from a DetectionResult."""

    def __init__(
        self,
        focal_length_px: float = 700.0,
        reference_widths_m: Optional[dict[str, float]] = None,
        min_pixel_width: float = 5.0,
        max_distance_m: float = 20.0,
    ) -> None:
        if focal_length_px <= 0:
            raise ValueError("focal_length_px must be > 0")
        if min_pixel_width <= 0:
            raise ValueError("min_pixel_width must be > 0")
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be > 0")

        self.focal_length_px = float(focal_length_px)
        self.reference_widths_m = dict(
            reference_widths_m or REFERENCE_WIDTHS_M
        )
        self.min_pixel_width = float(min_pixel_width)
        self.max_distance_m = float(max_distance_m)

    def estimate(
        self,
        detection: DetectionResult,
    ) -> Optional[DistanceEstimate]:
        """Return an approximate distance, or None if unsupported/invalid."""

        object_name = detection.object_name.lower()
        real_width = self.reference_widths_m.get(object_name)

        if real_width is None:
            return None

        pixel_width = float(detection.box_width)

        if pixel_width < self.min_pixel_width:
            return None

        distance_m = (
            real_width * self.focal_length_px
        ) / pixel_width

        distance_m = min(distance_m, self.max_distance_m)

        return DistanceEstimate(
            object_name=object_name,
            distance_m=distance_m,
            confidence=float(detection.confidence),
        )

    @staticmethod
    def format_distance(
        estimate: DistanceEstimate,
    ) -> str:
        """Create a simple spoken-friendly distance phrase."""

        if estimate.distance_m < 1.0:
            return f"{estimate.object_name} is less than 1 meter away."

        return (
            f"{estimate.object_name} is approximately "
            f"{estimate.distance_m:.1f} meters away."
        )
