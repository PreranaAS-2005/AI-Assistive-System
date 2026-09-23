"""Select one relevant obstacle guidance decision.

Phase 3 goal:
- Avoid switching rapidly between several detected people/obstacles.
- Prefer objects that are closer.
- Prefer objects in CENTER over distant side objects when they are similarly close.
- Keep the selected track stable for a short period.
"""

from __future__ import annotations

from typing import Optional

from navigation.obstacle_guidance import GuidanceDecision
from navigation.spatial_analyzer import Priority


_PRIORITY_RANK = {
    Priority.IGNORE: 0,
    Priority.LOW: 1,
    Priority.MEDIUM: 2,
    Priority.HIGH: 3,
}


class GuidanceSelector:
    """Choose one obstacle decision for the voice layer."""

    def __init__(
        self,
        switch_distance_margin_m: float = 0.75,
    ) -> None:

        if switch_distance_margin_m < 0:
            raise ValueError(
                "switch_distance_margin_m must be >= 0"
            )

        self.switch_distance_margin_m = float(
            switch_distance_margin_m
        )

        self._selected_track_id: Optional[int] = None

    def reset(self) -> None:
        """Forget the currently selected obstacle."""

        self._selected_track_id = None

    def select(
        self,
        decisions: list[GuidanceDecision],
    ) -> Optional[GuidanceDecision]:
        """
        Select at most one decision.

        Rules:
        1. Ignore decisions that should not announce.
        2. If the currently selected track is present, keep it unless
           another candidate is substantially more relevant.
        3. Otherwise choose the strongest candidate.
        """

        candidates = [
            decision
            for decision in decisions
            if decision.should_announce
        ]

        if not candidates:
            return None

        # ---------------------------------------------------------
        # Keep the current focus when it is still present.
        # ---------------------------------------------------------

        if self._selected_track_id is not None:

            current = next(
                (
                    decision
                    for decision in candidates
                    if decision.detection.track_id
                    == self._selected_track_id
                ),
                None,
            )

            if current is not None:

                stronger = [
                    decision
                    for decision in candidates
                    if decision.detection.track_id
                    != self._selected_track_id
                    and self._is_substantially_better(
                        decision,
                        current,
                    )
                ]

                if not stronger:

                    return current

                best = max(
                    stronger,
                    key=self._score,
                )

                self._selected_track_id = (
                    best.detection.track_id
                )

                return best

        # ---------------------------------------------------------
        # No current focus.
        # ---------------------------------------------------------

        best = max(
            candidates,
            key=self._score,
        )

        self._selected_track_id = (
            best.detection.track_id
        )

        return best

    def _is_substantially_better(
        self,
        candidate: GuidanceDecision,
        current: GuidanceDecision,
    ) -> bool:
        """Return True only when switching focus is justified."""

        candidate_priority = _PRIORITY_RANK.get(
            candidate.detection.priority,
            0,
        )

        current_priority = _PRIORITY_RANK.get(
            current.detection.priority,
            0,
        )

        # A clearly higher-priority obstacle can take focus.
        if candidate_priority > current_priority:
            return True

        # Same priority: use proximity first.
        candidate_proximity = self._proximity_rank(
            candidate.detection.proximity
        )

        current_proximity = self._proximity_rank(
            current.detection.proximity
        )

        if candidate_proximity > current_proximity:
            return True

        # Same proximity: prefer CENTER.
        if (
            candidate.detection.zone == "CENTER"
            and current.detection.zone != "CENTER"
        ):
            return True

        # Otherwise don't switch simply because confidence changed.
        return False

    def _score(
        self,
        decision: GuidanceDecision,
    ) -> tuple[int, int, int, float, float]:
        """Score a candidate for initial focus selection."""

        detection = decision.detection

        priority = _PRIORITY_RANK.get(
            detection.priority,
            0,
        )

        proximity = self._proximity_rank(
            detection.proximity
        )

        center_bonus = (
            1
            if detection.zone == "CENTER"
            else 0
        )

        return (
            priority,
            proximity,
            center_bonus,
            detection.confidence,
            detection.box_area,
        )

    @staticmethod
    def _proximity_rank(
        proximity: str,
    ) -> int:

        return {
            "FAR": 0,
            "NEAR": 1,
            "CLOSE": 2,
        }.get(
            str(proximity).upper(),
            0,
        )