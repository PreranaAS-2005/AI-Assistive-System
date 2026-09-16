"""
navigation/target_search.py

Target object search logic for the AI-Based Assistive Spatial
Memory and Guidance System.

Searches continuously for a target for a configurable amount of time.
The target must be detected in consecutive frames before being confirmed.

No camera, no YOLO model, no OpenCV, no voice output.
"""

import re
import time
from dataclasses import dataclass
from typing import Optional

from navigation.spatial_analyzer import DetectionResult, Zone


# ============================================================
# CONFIGURATION
# ============================================================

# Target must be detected in this many consecutive frames
# before it is considered confirmed.
REQUIRED_FRAMES: int = 3

# Maximum amount of time to search for the target.
# This replaces the old frame-based search window.
SEARCH_TIMEOUT_SECONDS: float = 30.0


# ============================================================
# YOLO CLASS ALIASES
# ============================================================

# Friendly names -> YOLO class names
SEARCH_ALIASES: dict[str, str] = {

    "phone": "cell phone",
    "mobile": "cell phone",
    "cellphone": "cell phone",

    "table": "dining table",

    "tv": "tv",
    "television": "tv",

    "laptop": "laptop",
    "computer": "laptop",

    "bottle": "bottle",

    "cup": "cup",
    "mug": "cup",

    "chair": "chair",

    "book": "book",

    "backpack": "backpack",

    "bag": "handbag",
    "handbag": "handbag",

    "suitcase": "suitcase",

    "remote": "remote",

    "keyboard": "keyboard",

    "clock": "clock",
}


# ============================================================
# OUTPUT DATA STRUCTURE
# ============================================================

@dataclass
class TargetSearchResult:

    found: bool
    object_name: str
    zone: str
    confidence: float
    message: Optional[str]
    distance: Optional[float] = None


# ============================================================
# TARGET SEARCH CLASS
# ============================================================

class TargetSearch:

    _FIND_PATTERNS = [

        re.compile(
            r"^find\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^search\s+(?:for\s+)?(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^locate\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^where\s+is\s+(?:my\s+|the\s+|a\s+)?(.+?)[\.\!\?]*$",
            re.IGNORECASE,
        ),
    ]


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self) -> None:

        self._target_name: Optional[str] = None
        self._active: bool = False
        self._consecutive_hits: int = 0
        self._search_start_time: Optional[float] = None
        self._error_message: Optional[str] = None


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
    # PARSE REQUEST
    # ========================================================

    def parse_request(
        self,
        text: str,
    ) -> Optional[str]:

        text = text.strip()

        if not text:
            return None

        for pattern in self._FIND_PATTERNS:

            match = pattern.match(text)

            if match:

                return self._normalize_name(
                    match.group(1).strip()
                )

        # Bare object name
        return self._normalize_name(text)


    # ========================================================
    # COMPATIBILITY ALIAS
    # ========================================================

    def parse_command(
        self,
        text: str,
    ) -> Optional[str]:
        """
        Compatibility alias for older code/tests.

        Both parse_command() and parse_request()
        perform the same operation.
        """

        return self.parse_request(text)


    # ========================================================
    # SET TARGET
    # ========================================================

    def set_target(
        self,
        name_or_phrase: str,
    ) -> Optional[str]:

        normalized = self.parse_request(
            name_or_phrase
        )

        if normalized is None:

            self._error_message = (
                "Could not understand that search request."
            )

            return None

        if normalized not in self._all_valid_classes():

            self._error_message = (
                f"I don't recognize "
                f"'{normalized}' as a searchable object type."
            )

            return None

        # Start new search
        self._target_name = normalized
        self._active = True
        self._consecutive_hits = 0
        self._search_start_time = time.monotonic()
        self._error_message = None

        print(
            f"[SEARCH] Target set to: {normalized}"
        )

        print(
            f"[SEARCH] Searching for up to "
            f"{SEARCH_TIMEOUT_SECONDS:.0f} seconds."
        )

        return normalized


    # ========================================================
    # CLEAR TARGET
    # ========================================================

    def clear_target(self) -> None:

        self._target_name = None
        self._active = False
        self._consecutive_hits = 0
        self._search_start_time = None


    # ========================================================
    # CANCEL
    # ========================================================

    def cancel(self) -> None:
        """
        Cancel the current search.
        """

        self.clear_target()


    # ========================================================
    # ERROR MESSAGE
    # ========================================================

    def get_error_message(self) -> Optional[str]:

        return self._error_message


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        detections: list[DetectionResult],
    ) -> TargetSearchResult:

        # ----------------------------------------------------
        # No active search
        # ----------------------------------------------------

        if (
            not self.is_active
            or self._target_name is None
        ):

            return TargetSearchResult(
                found=False,
                object_name="",
                zone="",
                confidence=0.0,
                message=None,
            )

        target = self._target_name


        # ----------------------------------------------------
        # Calculate elapsed search time
        # ----------------------------------------------------

        elapsed = 0.0

        if self._search_start_time is not None:

            elapsed = (
                time.monotonic()
                - self._search_start_time
            )


        # ----------------------------------------------------
        # Find target in current frame
        # ----------------------------------------------------

        match = self._find_best_match(
            detections,
            target,
        )

        found = False
        zone = ""
        confidence = 0.0
        message = None


        # ====================================================
        # TARGET FOUND IN CURRENT FRAME
        # ====================================================

        if match is not None:

            found = True

            zone = match.zone
            confidence = match.confidence

            self._consecutive_hits += 1

            print(
                f"[TARGET] {target} detected | "
                f"confidence={confidence:.2f} | "
                f"zone={zone} | "
                f"confirmation="
                f"{self._consecutive_hits}/"
                f"{REQUIRED_FRAMES}"
            )


            # ------------------------------------------------
            # CONFIRM TARGET
            # ------------------------------------------------

            if (
                self._consecutive_hits
                >= REQUIRED_FRAMES
            ):

                message = self.build_message(
                    found=True,
                    object_name=target,
                    zone=zone,
                )

                self._finish_search()


        # ====================================================
        # TARGET NOT FOUND IN CURRENT FRAME
        # ====================================================

        else:

            if self._consecutive_hits > 0:

                print(
                    f"[TARGET] {target} detection "
                    f"interrupted. "
                    f"Resetting confirmation."
                )

            self._consecutive_hits = 0


        # ====================================================
        # SEARCH TIMEOUT
        # ====================================================

        if (
            self.is_active
            and elapsed >= SEARCH_TIMEOUT_SECONDS
        ):

            print(
                f"[SEARCH] Search timeout after "
                f"{elapsed:.1f} seconds."
            )

            message = self.build_message(
                found=False,
                object_name=target,
                zone="",
            )

            found = False
            zone = ""
            confidence = 0.0

            self._finish_search()


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return TargetSearchResult(

            found=found,

            object_name=(
                target
                if (
                    self._target_name is not None
                    or message is not None
                )
                else ""
            ),

            zone=zone,

            confidence=confidence,

            message=message,
        )


    # ============================================================
    # BUILD SPOKEN MESSAGE
    # ============================================================

    def build_message(
        self,
        found: bool,
        object_name: str,
        zone: str,
    ) -> Optional[str]:

        if not found:

            return (
                f"Your {object_name} "
                f"is not currently visible."
            )

        if zone == Zone.LEFT:

            return (
                f"Your {object_name} "
                f"is on your left."
            )

        elif zone == Zone.RIGHT:

            return (
                f"Your {object_name} "
                f"is on your right."
            )

        else:

            return (
                f"Your {object_name} "
                f"is ahead."
            )


    # ============================================================
    # FINISH SEARCH
    # ============================================================

    def _finish_search(self) -> None:

        self._active = False
        self._consecutive_hits = 0
        self._search_start_time = None
        self._target_name = None


    # ============================================================
    # VALID YOLO CLASSES
    # ============================================================

    @staticmethod
    def _all_valid_classes() -> set[str]:

        return set(
            SEARCH_ALIASES.values()
        )


    # ============================================================
    # NORMALIZE OBJECT NAME
    # ============================================================

    @staticmethod
    def _normalize_name(
        raw: str,
    ) -> Optional[str]:

        key = raw.strip().lower()

        if not key:
            return None

        if key in SEARCH_ALIASES:

            return SEARCH_ALIASES[key]

        if key in SEARCH_ALIASES.values():

            return key

        return key


    # ============================================================
    # FIND BEST TARGET MATCH
    # ============================================================

    @staticmethod
    def _find_best_match(
        detections: list[DetectionResult],
        target: str,
    ) -> Optional[DetectionResult]:

        target_lower = (
            target.lower().strip()
        )

        candidates = [

            d

            for d in detections

            if (
                d.object_name
                and
                d.object_name.lower().strip()
                == target_lower
            )

        ]

        if not candidates:

            return None

        # If multiple target objects are visible,
        # select the highest-confidence one.
        return max(

            candidates,

            key=lambda d: (
                d.confidence,
                d.box_area,
            ),

        )
