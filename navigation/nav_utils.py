"""
navigation/nav_utils.py

Lightweight utility functions used by navigation_main.py and the test suite.
No YOLO, no camera, no voice — pure Python.
"""

from __future__ import annotations

from navigation.spatial_analyzer import Zone


def steps_text(distance_m: float) -> str:
    """
    Convert metres to a grammatically-correct steps phrase.
    1 step ≈ 0.5 m.  Minimum 1 step.

    >>> steps_text(0.4)
    '1 step'
    >>> steps_text(1.0)
    '2 steps'
    >>> steps_text(3.5)
    '7 steps'
    """
    n = max(1, round(distance_m / 0.5))
    unit = "step" if n == 1 else "steps"
    return f"{n} {unit}"


def direction_phrase(zone: str) -> tuple[str, str]:
    """
    Return (positional_text, movement_command) for a zone.

    Returns
    -------
    ("ahead",          "Move forward.")   for CENTER
    ("on your left",   "Move left.")      for LEFT
    ("on your right",  "Move right.")     for RIGHT
    """
    if zone == Zone.LEFT:
        return "on your left",  "Move left."
    if zone == Zone.RIGHT:
        return "on your right", "Move right."
    return "ahead",             "Move forward."
