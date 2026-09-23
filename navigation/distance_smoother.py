"""Stable distance reporting for assistive navigation.

This module smooths noisy frame-by-frame distance estimates and only reports
when the distance band changes. It does not run YOLO or speak audio.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional

from navigation.distance_estimator import DistanceEstimate


@dataclass(frozen=True)
class DistanceState:
    object_name: str
    distance_m: float
    band: str


class DistanceSmoother:
    """Reduce distance jitter and suppress repeated announcements."""

    BANDS = (
        (0.0, 0.75, "very close"),
        (0.75, 1.50, "close"),
        (1.50, 3.00, "moderate distance"),
        (3.00, float("inf"), "far"),
    )

    def __init__(
        self,
        window_size: int = 5,
        change_threshold_m: float = 0.30,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if change_threshold_m < 0:
            raise ValueError("change_threshold_m must be >= 0")

        self.window_size = int(window_size)
        self.change_threshold_m = float(change_threshold_m)

        self._values: dict[str, deque[float]] = {}
        self._last_reported: dict[str, DistanceState] = {}

    def reset(self) -> None:
        self._values.clear()
        self._last_reported.clear()

    def update(
        self,
        estimate: DistanceEstimate,
    ) -> tuple[DistanceState, bool]:
        """Return smoothed state and whether a new announcement is needed."""

        name = estimate.object_name.lower()

        values = self._values.setdefault(
            name,
            deque(maxlen=self.window_size),
        )
        values.append(float(estimate.distance_m))

        smoothed = sum(values) / len(values)
        band = self._band(smoothed)

        state = DistanceState(
            object_name=name,
            distance_m=smoothed,
            band=band,
        )

        previous = self._last_reported.get(name)

        if previous is None:
            should_report = True
        elif previous.band != state.band:
            should_report = True
        elif abs(previous.distance_m - state.distance_m) >= self.change_threshold_m:
            should_report = True
        else:
            should_report = False

        if should_report:
            self._last_reported[name] = state

        return state, should_report

    def remove(self, object_name: str) -> None:
        name = object_name.lower()
        self._values.pop(name, None)
        self._last_reported.pop(name, None)

    @classmethod
    def _band(cls, distance_m: float) -> str:
        for low, high, label in cls.BANDS:
            if low <= distance_m < high:
                return label
        return "far"

    @staticmethod
    def spoken_message(state: DistanceState) -> str:
        if state.band == "very close":
            return f"{state.object_name} is very close, about {state.distance_m:.1f} meters."
        if state.band == "close":
            return f"{state.object_name} is close, about {state.distance_m:.1f} meters."
        if state.band == "moderate distance":
            return f"{state.object_name} is about {state.distance_m:.1f} meters away."
        return f"{state.object_name} is far, about {state.distance_m:.1f} meters away."
