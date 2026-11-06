"""
navigation/target_search.py

Target object search logic for the AI-Based Assistive Spatial Memory
and Guidance System.

This module receives DetectionResult objects (produced by spatial_analyzer.py)
and performs a one-shot search: collect a short frame window, verify consistent
detection, announce the result once, then stop.

No camera, no YOLO model, no OpenCV, no voice output — only search logic.
"""

import re
import time
from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import DetectionResult, Zone
from navigation.object_memory import ObjectMemory


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Consecutive frames the target must appear in to be confirmed
REQUIRED_FRAMES: int = 3

# Total frames to search before reporting "not currently visible"
SEARCH_WINDOW_FRAMES: int = 15


# ---------------------------------------------------------------------------
# YOLO class aliases — maps user-friendly names to YOLO class labels
# ---------------------------------------------------------------------------

SEARCH_ALIASES: dict[str, str] = {
    "phone":        "cell phone",
    "mobile":       "cell phone",
    "cellphone":    "cell phone",
    "table":        "dining table",
    "tv":           "tv",
    "television":   "tv",
    "laptop":       "laptop",
    "computer":     "laptop",
    "bottle":       "bottle",
    "cup":          "cup",
    "mug":          "cup",
    "chair":        "chair",
    "book":         "book",
    "backpack":     "backpack",
    "bag":          "handbag",
    "handbag":      "handbag",
    "suitcase":     "suitcase",
    "remote":       "remote",
    "keyboard":     "keyboard",
    "clock":        "clock",
}


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class TargetSearchResult:
    """
    Result of a single target-search frame evaluation.

    Attributes
    ----------
    found : bool
        True when the target object is detected this frame.
    object_name : str
        Normalized YOLO class name of the target (e.g. "bottle").
    zone : str
        Spatial zone: LEFT, CENTER, or RIGHT (empty when not found).
    confidence : float
        Detection confidence of the best matching box (0.0 when not found).
    message : str or None
        A spoken message when the search completes this frame, else None.
    distance : None
        Placeholder for future distance estimation integration.
    """
    found      : bool
    object_name: str
    zone       : str
    confidence : float
    message    : Optional[str]
    distance   : None = None


# ---------------------------------------------------------------------------
# Main search class
# ---------------------------------------------------------------------------

class TargetSearch:
    """
    One-shot target object search engine.

    Usage
    -----
    search = TargetSearch()
    search.set_target("find my bottle")

    # Each frame while searching, pass all DetectionResult objects:
    result = search.process_frame(all_detections)

    # result.message is set exactly once when the search completes,
    # then the search automatically stops.
    """

    # Regex patterns for parsing natural-language search requests
    _FIND_PATTERNS = [
        re.compile(r"^find\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$", re.IGNORECASE),
        re.compile(r"^search\s+(?:for\s+)?(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$", re.IGNORECASE),
        re.compile(r"^locate\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$", re.IGNORECASE),
        re.compile(r"^where\s+is\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$", re.IGNORECASE),
    ]

    def __init__(self, object_memory: Optional[ObjectMemory] = None) -> None:
        self._target_name      : Optional[str] = None
        self._active           : bool = False
        self._window_frames    : int  = 0
        self._consecutive_hits : int  = 0
        self._error_message    : Optional[str] = None

        # Shared object memory populated by navigation_main.py.
        # Keeping this optional preserves compatibility with existing code.
        self.object_memory = object_memory

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True while a one-shot search is in progress."""
        return self._active and self._target_name is not None

    @property
    def target_name(self) -> Optional[str]:
        """The normalized YOLO class name being searched for, or None."""
        return self._target_name

    def parse_request(self, text: str) -> Optional[str]:
        """
        Parse a natural-language search request into a normalized object name.

        Examples
        --------
        "Find my bottle."  -> "bottle"
        "find the phone"   -> "cell phone"
        "bottle"           -> "bottle"  (bare name also accepted)
        """
        text = text.strip()
        if not text:
            return None

        for pattern in self._FIND_PATTERNS:
            match = pattern.match(text)
            if match:
                return self._normalize_name(match.group(1).strip())

        return self._normalize_name(text)

    def set_target(self, name_or_phrase: str) -> Optional[str]:
        """
        Start a one-shot search for the given object.

        Returns the normalized target name if recognized, else None.
        """
        normalized = self.parse_request(name_or_phrase)

        if normalized is None:
            self._error_message = "Could not understand that search request."
            return None

        if normalized not in self._all_valid_classes():
            self._error_message = (
                f"I don't recognize '{normalized}' as a searchable object type."
            )
            return None

        self._target_name      = normalized
        self._active           = True
        self._window_frames    = 0
        self._consecutive_hits = 0
        self._error_message    = None

        return normalized

    def clear_target(self) -> None:
        """Cancel an in-progress search."""
        self._target_name      = None
        self._active           = False
        self._window_frames    = 0
        self._consecutive_hits = 0

    def get_error_message(self) -> Optional[str]:
        """Return the last error message from set_target(), or None."""
        return self._error_message

    def process_frame(
        self,
        detections: list[DetectionResult],
    ) -> TargetSearchResult:
        """
        Process one frame of a one-shot search.

        While the search is active, accumulates detections across frames.
        Returns a message exactly once when the search completes (found or
        not visible), then automatically stops.
        """
        if not self.is_active or self._target_name is None:
            return TargetSearchResult(
                found=False, object_name="", zone="",
                confidence=0.0, message=None,
            )

        target = self._target_name
        self._window_frames += 1

        match   = self._find_best_match(detections, target)
        message = None
        found   = False
        zone    = ""
        conf    = 0.0

        if match is not None:
            found = True
            zone  = match.zone
            conf  = match.confidence
            self._consecutive_hits += 1

            if self._consecutive_hits >= REQUIRED_FRAMES:
                message = self.build_message(
                    found=True,
                    object_name=target,
                    zone=zone,
                )
                self._finish_search()
        else:
            self._consecutive_hits = 0

        if self.is_active and self._window_frames >= SEARCH_WINDOW_FRAMES:
            message = self.build_message(
                found=False,
                object_name=target,
                zone="",
            )
            found = False
            zone  = ""
            conf  = 0.0
            self._finish_search()

        return TargetSearchResult(
            found       = found,
            object_name = target if self._target_name or message else "",
            zone        = zone,
            confidence  = conf,
            message     = message,
        )

    def build_message(
        self,
        found      : bool,
        object_name: str,
        zone       : str,
    ) -> Optional[str]:
        """Build a spoken message for a completed search.

        If the object is not visible in the current search window, fall back
        to the latest sighting stored in ObjectMemory.
        """
        if not found:
            return self._build_memory_fallback_message(object_name)

        if zone == Zone.LEFT or str(zone).upper() == "LEFT":
            return f"Your {object_name} is on your left."
        elif zone == Zone.RIGHT or str(zone).upper() == "RIGHT":
            return f"Your {object_name} is on your right."
        else:
            return f"Your {object_name} is ahead."

    def _build_memory_fallback_message(self, object_name: str) -> str:
        """Build a last-seen message when the target is not in the current frame."""
        if self.object_memory is None:
            return f"Your {object_name} is not currently visible."

        memory = self.object_memory.get_last_seen(object_name)

        if memory is None:
            return f"I haven't detected your {object_name} yet."

        elapsed = max(0.0, time.time() - memory.get("timestamp", time.time()))

        if elapsed < 30:
            time_text = "just now"
        elif elapsed < 120:
            time_text = "about a minute ago"
        elif elapsed < 600:
            time_text = "a few minutes ago"
        else:
            time_text = "more than 10 minutes ago"

        direction = str(memory.get("direction", "")).upper()
        distance = str(memory.get("distance", "")).upper()

        direction_text = {
            "LEFT": "on your left",
            "CENTER": "in front of you",
            "RIGHT": "on your right",
        }.get(direction, "in an unknown direction")

        distance_text = {
            "CLOSE": "very close",
            "NEAR": "nearby",
            "FAR": "far away",
        }.get(distance, "")

        if distance_text:
            return (
                f"I cannot see your {object_name} right now. "
                f"I last saw it {time_text}, {distance_text}, {direction_text}."
            )

        return (
            f"I cannot see your {object_name} right now. "
            f"I last saw it {time_text}, {direction_text}."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finish_search(self) -> None:
        """End the current search and reset internal counters."""
        self._active           = False
        self._window_frames    = 0
        self._consecutive_hits = 0
        self._target_name      = None

    @staticmethod
    def _all_valid_classes() -> set[str]:
        return set(SEARCH_ALIASES.values())

    @staticmethod
    def _normalize_name(raw: str) -> Optional[str]:
        key = raw.strip().lower()
        if not key:
            return None

        if key in SEARCH_ALIASES:
            return SEARCH_ALIASES[key]

        if key in SEARCH_ALIASES.values():
            return key

        return key

    @staticmethod
    def _find_best_match(
        detections: list[DetectionResult],
        target    : str,
    ) -> Optional[DetectionResult]:
        """Find the best matching detection for the target class name."""
        target_lower = target.lower()
        candidates = [
            d for d in detections
            if d.object_name.lower() == target_lower
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda d: (d.confidence, d.box_area),
        )