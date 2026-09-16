"""
navigation/spatial_analyzer.py

Pure spatial-analysis logic for Phase 3 of the
AI-Based Assistive Spatial Memory and Guidance System.

This module takes raw YOLO detection data and produces structured
spatial information:

    - LEFT / CENTER / RIGHT
    - LOW / MID / HIGH
    - FLOOR_LEVEL / TABLE_LEVEL / UNKNOWN
    - FAR / NEAR / CLOSE
    - obstacle priority
    - obstacle classification

No camera, no YOLO model, no voice output.
Only geometry and classification.
"""

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class Priority:
    """String constants for obstacle priority levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    IGNORE = "IGNORE"


# ---------------------------------------------------------------------------
# Spatial zones
# ---------------------------------------------------------------------------

class Zone:
    """String constants for horizontal spatial zones."""

    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


# ---------------------------------------------------------------------------
# Vertical positions
# ---------------------------------------------------------------------------

class VerticalPosition:
    """
    Approximate vertical position of an object in the camera frame.

    LOW:
        Object is near the bottom of the image.
        Often corresponds to an object on the floor or a low surface.

    MID:
        Object is around the middle of the image.

    HIGH:
        Object is near the top of the image.
    """

    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

class Surface:
    """
    Approximate surface/location classification.

    IMPORTANT:
    This is an image-position based estimate only.
    It is NOT a true depth or surface detector.
    """

    FLOOR_LEVEL = "FLOOR_LEVEL"
    TABLE_LEVEL = "TABLE_LEVEL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Proximity tiers
# ---------------------------------------------------------------------------

class Proximity:
    """
    Approximate proximity tiers.

    These are estimated from bounding-box area relative to
    the frame area.

    Larger bounding boxes generally suggest that an object
    is closer to the camera.
    """

    FAR = "FAR"
    NEAR = "NEAR"
    CLOSE = "CLOSE"


# ---------------------------------------------------------------------------
# Proximity thresholds
# ---------------------------------------------------------------------------

PROXIMITY_FAR_MAX = 0.05
PROXIMITY_CLOSE_MIN = 0.15


# ---------------------------------------------------------------------------
# Vertical position thresholds
# ---------------------------------------------------------------------------

"""
The vertical position is calculated using the CENTER of the
bounding box relative to the frame height.

Example for a 640-pixel-high frame:

    0% - 35%   -> HIGH
    35% - 70%  -> MID
    70% - 100% -> LOW

These values are intentionally conservative and can be tuned later.
"""

VERTICAL_HIGH_MAX = 0.35
VERTICAL_MID_MAX = 0.70


# ---------------------------------------------------------------------------
# Floor-level threshold
# ---------------------------------------------------------------------------

"""
Objects whose center is very low in the image are considered
potentially floor-level.

This does NOT prove that the object is on the floor.

It is only a visual estimate.
"""

FLOOR_LEVEL_MIN = 0.78


# ---------------------------------------------------------------------------
# Table-level threshold
# ---------------------------------------------------------------------------

"""
A rough image-space range where tabletop objects commonly appear.

This is only used as a preliminary classification.

Actual "on table" detection will be handled by checking the
relationship between the target object and a detected table.
"""

TABLE_LEVEL_MIN = 0.35
TABLE_LEVEL_MAX = 0.78


# ---------------------------------------------------------------------------
# Obstacle priority map
# ---------------------------------------------------------------------------

OBSTACLE_PRIORITY: dict[str, str] = {

    # HIGH — moving or large collision risks
    "person": Priority.HIGH,
    "bicycle": Priority.HIGH,
    "motorcycle": Priority.HIGH,
    "car": Priority.HIGH,
    "truck": Priority.HIGH,
    "bus": Priority.HIGH,
    "dog": Priority.HIGH,
    "cat": Priority.HIGH,

    # MEDIUM — larger stationary objects
    "chair": Priority.MEDIUM,
    "bench": Priority.MEDIUM,
    "couch": Priority.MEDIUM,
    "dining table": Priority.MEDIUM,
    "toilet": Priority.MEDIUM,
    "bed": Priority.MEDIUM,
    "potted plant": Priority.MEDIUM,
    "suitcase": Priority.MEDIUM,
    "backpack": Priority.MEDIUM,
    "handbag": Priority.MEDIUM,

    # LOW — small objects
    "bottle": Priority.LOW,
    "cup": Priority.LOW,
    "book": Priority.LOW,
    "laptop": Priority.LOW,
    "keyboard": Priority.LOW,
    "remote": Priority.LOW,
    "cell phone": Priority.LOW,

    # IGNORE — generally not useful for navigation
    "tv": Priority.IGNORE,
    "clock": Priority.IGNORE,
    "painting": Priority.IGNORE,
    "framed artwork": Priority.IGNORE,
}


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Structured spatial analysis result for a single YOLO detection.
    """

    track_id: int

    object_name: str

    confidence: float

    # Bounding-box center
    center_x: float
    center_y: float

    # Bounding-box dimensions
    box_width: float
    box_height: float
    box_area: float

    # Frame information
    frame_area: float

    # Proximity
    proximity_ratio: float
    proximity: str

    # Horizontal position
    zone: str

    # Vertical position
    vertical_position: str

    # Surface estimate
    surface: str

    # Obstacle information
    priority: str
    is_obstacle: bool


# ---------------------------------------------------------------------------
# Core geometry functions
# ---------------------------------------------------------------------------

def compute_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Tuple[float, float]:
    """
    Compute the center point of a bounding box.
    """

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    return center_x, center_y


def compute_box_dimensions(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Tuple[float, float, float]:
    """
    Compute bounding-box width, height, and area.
    """

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    area = width * height

    return width, height, area


def compute_proximity_ratio(
    box_area: float,
    frame_area: float
) -> float:
    """
    Compute bounding-box area / frame area.
    """

    if frame_area <= 0:
        return 0.0

    return box_area / frame_area


# ---------------------------------------------------------------------------
# Proximity classification
# ---------------------------------------------------------------------------

def classify_proximity(
    proximity_ratio: float
) -> str:
    """
    Convert proximity ratio into FAR / NEAR / CLOSE.
    """

    if proximity_ratio > PROXIMITY_CLOSE_MIN:
        return Proximity.CLOSE

    elif proximity_ratio >= PROXIMITY_FAR_MAX:
        return Proximity.NEAR

    else:
        return Proximity.FAR


# ---------------------------------------------------------------------------
# Horizontal zone classification
# ---------------------------------------------------------------------------

def classify_zone(
    center_x: float,
    frame_width: float
) -> str:
    """
    Divide the frame into LEFT / CENTER / RIGHT.
    """

    if frame_width <= 0:
        return Zone.CENTER

    one_third = frame_width / 3.0
    two_thirds = 2.0 * frame_width / 3.0

    if center_x < one_third:
        return Zone.LEFT

    elif center_x < two_thirds:
        return Zone.CENTER

    else:
        return Zone.RIGHT


# ---------------------------------------------------------------------------
# Vertical position classification
# ---------------------------------------------------------------------------

def classify_vertical_position(
    center_y: float,
    frame_height: float
) -> str:
    """
    Determine whether an object is visually HIGH, MID, or LOW.

    The calculation is based on the object's vertical center
    relative to the frame height.

    Returns
    -------
    str
        VerticalPosition.HIGH
        VerticalPosition.MID
        VerticalPosition.LOW
    """

    if frame_height <= 0:
        return VerticalPosition.MID

    vertical_ratio = center_y / frame_height

    if vertical_ratio < VERTICAL_HIGH_MAX:
        return VerticalPosition.HIGH

    elif vertical_ratio < VERTICAL_MID_MAX:
        return VerticalPosition.MID

    else:
        return VerticalPosition.LOW


# ---------------------------------------------------------------------------
# Surface classification
# ---------------------------------------------------------------------------

def classify_surface(
    center_y: float,
    frame_height: float
) -> str:
    """
    Estimate whether an object is at floor level or table-like level.

    IMPORTANT
    ---------
    This is only an image-position estimate.

    A LOW object is NOT automatically guaranteed to be on the floor.

    Actual table detection should be performed by comparing the
    target object with a detected "dining table".
    """

    if frame_height <= 0:
        return Surface.UNKNOWN

    vertical_ratio = center_y / frame_height

    # Very low in the image
    if vertical_ratio >= FLOOR_LEVEL_MIN:
        return Surface.FLOOR_LEVEL

    # Middle/lower image region
    elif (
        vertical_ratio >= TABLE_LEVEL_MIN
        and vertical_ratio < TABLE_LEVEL_MAX
    ):
        return Surface.TABLE_LEVEL

    return Surface.UNKNOWN


# ---------------------------------------------------------------------------
# Priority classification
# ---------------------------------------------------------------------------

def classify_priority(
    object_name: str
) -> str:
    """
    Look up obstacle priority for a YOLO class.
    """

    name = object_name.strip().lower()

    return OBSTACLE_PRIORITY.get(
        name,
        Priority.IGNORE
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def is_low_object(
    detection: DetectionResult
) -> bool:
    """
    Return True when the detection is visually low.
    """

    return detection.vertical_position == VerticalPosition.LOW


def is_floor_level(
    detection: DetectionResult
) -> bool:
    """
    Return True when the object is estimated to be at floor level.
    """

    return detection.surface == Surface.FLOOR_LEVEL


def is_table_level(
    detection: DetectionResult
) -> bool:
    """
    Return True when the object is estimated to be at table level.
    """

    return detection.surface == Surface.TABLE_LEVEL


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_detection(
    track_id: int,
    object_name: str,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: float,
    frame_height: float
) -> DetectionResult | None:
    """
    Perform complete spatial analysis on one YOLO detection.

    Returns
    -------
    DetectionResult
        Structured detection information.

    None
        If the bounding box or frame dimensions are invalid.
    """

    # ------------------------------------------------------------
    # Validate frame
    # ------------------------------------------------------------

    if frame_width <= 0 or frame_height <= 0:
        return None

    # ------------------------------------------------------------
    # Compute bounding-box dimensions
    # ------------------------------------------------------------

    box_width, box_height, box_area = compute_box_dimensions(
        x1,
        y1,
        x2,
        y2
    )

    if box_area <= 0:
        return None

    # ------------------------------------------------------------
    # Frame area
    # ------------------------------------------------------------

    frame_area = frame_width * frame_height

    # ------------------------------------------------------------
    # Center
    # ------------------------------------------------------------

    center_x, center_y = compute_center(
        x1,
        y1,
        x2,
        y2
    )

    # ------------------------------------------------------------
    # Proximity
    # ------------------------------------------------------------

    proximity_ratio = compute_proximity_ratio(
        box_area,
        frame_area
    )

    proximity = classify_proximity(
        proximity_ratio
    )

    # ------------------------------------------------------------
    # Horizontal zone
    # ------------------------------------------------------------

    zone = classify_zone(
        center_x,
        frame_width
    )

    # ------------------------------------------------------------
    # Vertical position
    # ------------------------------------------------------------

    vertical_position = classify_vertical_position(
        center_y,
        frame_height
    )

    # ------------------------------------------------------------
    # Surface estimate
    # ------------------------------------------------------------

    surface = classify_surface(
        center_y,
        frame_height
    )

    # ------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------

    priority = classify_priority(
        object_name
    )

    is_obstacle = (
        priority != Priority.IGNORE
    )

    # ------------------------------------------------------------
    # Return structured result
    # ------------------------------------------------------------

    return DetectionResult(
        track_id=track_id,

        object_name=object_name,

        confidence=confidence,

        center_x=center_x,
        center_y=center_y,

        box_width=box_width,
        box_height=box_height,
        box_area=box_area,

        frame_area=frame_area,

        proximity_ratio=round(
            proximity_ratio,
            4
        ),

        proximity=proximity,

        zone=zone,

        vertical_position=vertical_position,

        surface=surface,

        priority=priority,

        is_obstacle=is_obstacle,
    )
