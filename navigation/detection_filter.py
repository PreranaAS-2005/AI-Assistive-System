"""Temporal reliability filter for YOLO DetectionResult objects."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Optional

from navigation.spatial_analyzer import DetectionResult


SUPPORTED_OBJECTS = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "dog",
    "cat",
    "chair",
    "bench",
    "couch",
    "dining table",
    "toilet",
    "bed",
    "potted plant",
    "suitcase",
    "backpack",
    "handbag",
    "bottle",
    "cup",
    "book",
    "laptop",
    "keyboard",
    "remote",
    "cell phone",
    "tv",
}

MIN_CONFIDENCE = 0.60
STABILITY_FRAMES = 3
HISTORY_SIZE = 5
MISSING_FRAMES_TO_REMOVE = 5


@dataclass(frozen=True)
class StableDetection:
    object_name: str
    confidence: float
    detection: DetectionResult
    observations: int


class DetectionFilter:

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        stability_frames: int = STABILITY_FRAMES,
        history_size: int = HISTORY_SIZE,
        supported_objects: Optional[Iterable[str]] = None,
        missing_frames_to_remove: int = MISSING_FRAMES_TO_REMOVE,
    ) -> None:

        if stability_frames < 1:
            raise ValueError(
                "stability_frames must be >= 1"
            )

        if history_size < stability_frames:
            raise ValueError(
                "history_size must be >= stability_frames"
            )

        if missing_frames_to_remove < 1:
            raise ValueError(
                "missing_frames_to_remove must be >= 1"
            )

        self.min_confidence = float(min_confidence)

        self.stability_frames = int(
            stability_frames
        )

        self.history_size = int(
            history_size
        )

        self.missing_frames_to_remove = int(
            missing_frames_to_remove
        )

        self.supported_objects = {
            name.lower()
            for name in (
                supported_objects
                if supported_objects is not None
                else SUPPORTED_OBJECTS
            )
        }

        self._history = defaultdict(
            lambda: deque(
                maxlen=self.history_size
            )
        )

        self._missing_frames = defaultdict(int)

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self) -> None:
        """Clear all detection history."""

        self._history.clear()
        self._missing_frames.clear()

    # ---------------------------------------------------------
    # Process one camera frame
    # ---------------------------------------------------------

    def process_frame(
        self,
        detections: Iterable[DetectionResult],
    ) -> list[StableDetection]:
        """
        Process one frame.

        An object becomes stable after STABILITY_FRAMES
        consecutive valid detections.

        If the object disappears, its missing-frame counter
        increases. After MISSING_FRAMES_TO_REMOVE frames,
        its history is removed completely.
        """

        current_best: dict[str, DetectionResult] = {}

        # -----------------------------------------------------
        # Select the best detection for each object in this frame
        # -----------------------------------------------------

        for detection in detections:

            name = detection.object_name.lower()

            # Ignore unsupported classes
            if name not in self.supported_objects:
                continue

            # Ignore weak detections
            if detection.confidence < self.min_confidence:
                continue

            previous = current_best.get(name)

            if previous is None:

                current_best[name] = detection

            elif self._score(detection) > self._score(previous):

                current_best[name] = detection

        # -----------------------------------------------------
        # Update history
        # -----------------------------------------------------

        tracked_names = (
            set(self._history.keys())
            | set(current_best.keys())
        )

        for name in tracked_names:

            # Object detected in current frame
            if name in current_best:

                self._history[name].append(
                    current_best[name]
                )

                # Object is visible again
                self._missing_frames[name] = 0

            # Object missing from current frame
            else:

                self._missing_frames[name] += 1

                # Remove stale object
                if (
                    self._missing_frames[name]
                    >= self.missing_frames_to_remove
                ):

                    self._history.pop(
                        name,
                        None,
                    )

                    self._missing_frames.pop(
                        name,
                        None,
                    )

        # -----------------------------------------------------
        # Build stable detections
        # -----------------------------------------------------

        stable: list[StableDetection] = []

        for name, history in self._history.items():

            # Not enough consecutive observations
            if len(history) < self.stability_frames:
                continue

            recent = list(history)[
                -self.stability_frames:
            ]

            best = max(
                recent,
                key=self._score,
            )

            stable.append(
                StableDetection(
                    object_name=name,
                    confidence=best.confidence,
                    detection=best,
                    observations=len(recent),
                )
            )

        # Strongest detections first
        stable.sort(
            key=lambda item: (
                item.confidence,
                item.detection.box_area,
            ),
            reverse=True,
        )

        return stable

    # ---------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------

    def get_history_count(
        self,
        object_name: str,
    ) -> int:

        return len(
            self._history.get(
                object_name.lower(),
                (),
            )
        )

    def get_missing_count(
        self,
        object_name: str,
    ) -> int:

        return self._missing_frames.get(
            object_name.lower(),
            0,
        )

    # ---------------------------------------------------------
    # Detection scoring
    # ---------------------------------------------------------

    @staticmethod
    def _score(
        detection: DetectionResult,
    ) -> tuple[float, float]:

        return (
            detection.confidence,
            detection.box_area,
        )