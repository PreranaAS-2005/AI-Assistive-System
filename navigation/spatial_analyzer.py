"""
navigation/spatial_analyzer.py

Pure spatial-analysis logic for Phase 3 of the AI-Based Assistive Spatial
Memory and Guidance System.

This module takes raw YOLO detection data (bounding-box coordinates, frame
dimensions, object metadata) and produces structured spatial information:
zone, proximity, priority, and obstacle classification.

No camera, no YOLO model, no voice output — only geometry and classification.
"""

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Priority levels
# ---------------------------------------------------------------------------

class Priority:
    """String constants for obstacle priority levels."""
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"
    IGNORE = "IGNORE"


# ---------------------------------------------------------------------------
# Spatial zones
# ---------------------------------------------------------------------------

class Zone:
    """String constants for spatial zones within the camera frame."""
    LEFT   = "LEFT"
    CENTER = "CENTER"
    RIGHT  = "RIGHT"


# ---------------------------------------------------------------------------
# Proximity tiers
# ---------------------------------------------------------------------------

class Proximity:
    """
    String constants for approximate proximity tiers.

    IMPORTANT: These are approximations only.
    The system has no depth sensor (RGB-D, LiDAR, or stereo camera).
    Proximity is estimated from bounding-box area relative to frame area.
    Larger boxes suggest closer objects, but this assumption can break for
    objects of very different real-world sizes (e.g. a distant truck vs.
    a nearby cup).
    """
    FAR   = "FAR"    # box occupies < 5% of frame area
    NEAR  = "NEAR"   # box occupies 5% – 15% of frame area
    CLOSE = "CLOSE"  # box occupies > 15% of frame area


# ---------------------------------------------------------------------------
# Proximity thresholds (tunable)
# ---------------------------------------------------------------------------

PROXIMITY_FAR_MAX   = 0.05   # ratio below this → FAR
PROXIMITY_CLOSE_MIN = 0.15   # ratio above this → CLOSE
# between FAR_MAX and CLOSE_MIN → NEAR


# ---------------------------------------------------------------------------
# Obstacle priority map
# ---------------------------------------------------------------------------

# Maps YOLO class names (lowercase) to a priority level.
# Objects not listed here default to IGNORE.
OBSTACLE_PRIORITY: dict[str, str] = {
    # HIGH — moving or large objects, collision risk
    "person":       Priority.HIGH,
    "bicycle":      Priority.HIGH,
    "motorcycle":   Priority.HIGH,
    "car":          Priority.HIGH,
    "truck":        Priority.HIGH,
    "bus":          Priority.HIGH,
    "dog":          Priority.HIGH,
    "cat":          Priority.HIGH,

    # MEDIUM — stationary objects at body or leg level
    "chair":        Priority.MEDIUM,
    "bench":        Priority.MEDIUM,
    "couch":        Priority.MEDIUM,
    "dining table": Priority.MEDIUM,
    "toilet":       Priority.MEDIUM,
    "bed":          Priority.MEDIUM,
    "potted plant": Priority.MEDIUM,
    "suitcase":     Priority.MEDIUM,
    "backpack":     Priority.MEDIUM,
    "handbag":      Priority.MEDIUM,

    # LOW — small objects, tripping hazard only if on floor
    "bottle":       Priority.LOW,
    "cup":          Priority.LOW,
    "book":         Priority.LOW,
    "laptop":       Priority.LOW,
    "keyboard":     Priority.LOW,
    "remote":       Priority.LOW,
    "cell phone":   Priority.LOW,

    # IGNORE — wall-mounted or irrelevant objects
    "tv":              Priority.IGNORE,
    "clock":           Priority.IGNORE,
    "painting":        Priority.IGNORE,
    "framed artwork":  Priority.IGNORE,
}


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Structured spatial analysis result for a single YOLO detection.

    Attributes
    ----------
    track_id : int
        Persistent tracking ID assigned by the YOLO tracker.
    object_name : str
        Human-readable class label from the YOLO model (e.g. "person").
    confidence : float
        Detection confidence score between 0.0 and 1.0.
    center_x : float
        Horizontal center of the bounding box in pixels.
    center_y : float
        Vertical center of the bounding box in pixels.
    box_width : float
        Width of the bounding box in pixels.
    box_height : float
        Height of the bounding box in pixels.
    box_area : float
        Area of the bounding box in square pixels.
    frame_area : float
        Total frame area in square pixels (frame_width × frame_height).
    proximity_ratio : float
        box_area / frame_area — used for approximate proximity estimation.
    proximity : str
        Proximity tier: FAR, NEAR, or CLOSE.
    zone : str
        Spatial zone: LEFT, CENTER, or RIGHT.
    priority : str
        Obstacle priority: HIGH, MEDIUM, LOW, or IGNORE.
    is_obstacle : bool
        True if priority is HIGH, MEDIUM, or LOW (i.e. not IGNORE).
    """
    track_id       : int
    object_name    : str
    confidence     : float
    center_x       : float
    center_y       : float
    box_width      : float
    box_height     : float
    box_area       : float
    frame_area     : float
    proximity_ratio: float
    proximity      : str
    zone           : str
    priority       : str
    is_obstacle    : bool


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_center(
    x1: float, y1: float,
    x2: float, y2: float
) -> Tuple[float, float]:
    """
    Compute the center point of a bounding box.

    Parameters
    ----------
    x1, y1 : float
        Top-left corner coordinates.
    x2, y2 : float
        Bottom-right corner coordinates.

    Returns
    -------
    (center_x, center_y) : Tuple[float, float]
    """
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return center_x, center_y


def compute_box_dimensions(
    x1: float, y1: float,
    x2: float, y2: float
) -> Tuple[float, float, float]:
    """
    Compute the width, height, and area of a bounding box.

    Parameters
    ----------
    x1, y1 : float
        Top-left corner coordinates.
    x2, y2 : float
        Bottom-right corner coordinates.

    Returns
    -------
    (width, height, area) : Tuple[float, float, float]
        All values are non-negative. Returns (0, 0, 0) for degenerate boxes.
    """
    width  = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area   = width * height
    return width, height, area


def compute_proximity_ratio(
    box_area: float,
    frame_area: float
) -> float:
    """
    Compute the ratio of bounding-box area to total frame area.

    This ratio is the basis for approximate proximity estimation.

    Parameters
    ----------
    box_area : float
        Area of the bounding box in square pixels.
    frame_area : float
        Total frame area in square pixels.

    Returns
    -------
    float
        A value between 0.0 and 1.0. Returns 0.0 if frame_area is zero.
    """
    if frame_area <= 0:
        return 0.0
    return box_area / frame_area


def classify_proximity(proximity_ratio: float) -> str:
    """
    Convert a proximity ratio into a named proximity tier.

    Thresholds (approximate — no depth sensor is present):
      - FAR   : ratio < 0.05  (box occupies less than 5% of the frame)
      - NEAR  : ratio 0.05 – 0.15
      - CLOSE : ratio > 0.15  (box occupies more than 15% of the frame)

    Parameters
    ----------
    proximity_ratio : float

    Returns
    -------
    str
        One of Proximity.FAR, Proximity.NEAR, Proximity.CLOSE.
    """
    if proximity_ratio > PROXIMITY_CLOSE_MIN:
        return Proximity.CLOSE
    elif proximity_ratio >= PROXIMITY_FAR_MAX:
        return Proximity.NEAR
    else:
        return Proximity.FAR


def classify_zone(center_x: float, frame_width: float) -> str:
    """
    Determine which horizontal zone the object occupies.

    The frame is divided into equal thirds:
      LEFT   : center_x < frame_width / 3
      CENTER : frame_width / 3 <= center_x < 2 * frame_width / 3
      RIGHT  : center_x >= 2 * frame_width / 3

    Parameters
    ----------
    center_x : float
        Horizontal center of the bounding box in pixels.
    frame_width : float
        Width of the camera frame in pixels.

    Returns
    -------
    str
        One of Zone.LEFT, Zone.CENTER, Zone.RIGHT.
        Returns Zone.CENTER if frame_width is zero (safe fallback).
    """
    if frame_width <= 0:
        return Zone.CENTER

    one_third  = frame_width / 3.0
    two_thirds = 2.0 * frame_width / 3.0

    if center_x < one_third:
        return Zone.LEFT
    elif center_x < two_thirds:
        return Zone.CENTER
    else:
        return Zone.RIGHT


def classify_priority(object_name: str) -> str:
    """
    Look up the obstacle priority for a given YOLO class name.

    Uses the OBSTACLE_PRIORITY dictionary. Object names not in the
    dictionary default to IGNORE.

    Parameters
    ----------
    object_name : str
        Class label from the YOLO model (case-insensitive).

    Returns
    -------
    str
        One of Priority.HIGH, Priority.MEDIUM, Priority.LOW, Priority.IGNORE.
    """
    return OBSTACLE_PRIORITY.get(object_name.lower(), Priority.IGNORE)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_detection(
    track_id    : int,
    object_name : str,
    confidence  : float,
    x1          : float,
    y1          : float,
    x2          : float,
    y2          : float,
    frame_width : float,
    frame_height: float
) -> DetectionResult | None:
    """
    Perform full spatial analysis on a single YOLO detection.

    Takes raw detection data and returns a DetectionResult containing
    zone, proximity, priority, and all intermediate geometry values.

    Returns None if the bounding box is invalid (zero or negative area)
    or if frame dimensions are invalid.

    Parameters
    ----------
    track_id : int
        Persistent tracking ID from the YOLO tracker.
    object_name : str
        Human-readable class label (e.g. "person", "chair").
    confidence : float
        Detection confidence between 0.0 and 1.0.
    x1, y1 : float
        Top-left corner of the bounding box in pixels.
    x2, y2 : float
        Bottom-right corner of the bounding box in pixels.
    frame_width : float
        Width of the camera frame in pixels.
    frame_height : float
        Height of the camera frame in pixels.

    Returns
    -------
    DetectionResult or None
    """
    # Validate frame dimensions
    if frame_width <= 0 or frame_height <= 0:
        return None

    # Compute box dimensions and validate
    box_width, box_height, box_area = compute_box_dimensions(x1, y1, x2, y2)
    if box_area <= 0:
        return None

    # Compute frame area
    frame_area = frame_width * frame_height

    # Compute center
    center_x, center_y = compute_center(x1, y1, x2, y2)

    # Compute proximity
    proximity_ratio = compute_proximity_ratio(box_area, frame_area)
    proximity       = classify_proximity(proximity_ratio)

    # Compute zone
    zone = classify_zone(center_x, frame_width)

    # Classify priority
    priority    = classify_priority(object_name)
    is_obstacle = priority != Priority.IGNORE

    return DetectionResult(
        track_id        = track_id,
        object_name     = object_name,
        confidence      = confidence,
        center_x        = center_x,
        center_y        = center_y,
        box_width       = box_width,
        box_height      = box_height,
        box_area        = box_area,
        frame_area      = frame_area,
        proximity_ratio = round(proximity_ratio, 4),
        proximity       = proximity,
        zone            = zone,
        priority        = priority,
        is_obstacle     = is_obstacle,
    )
