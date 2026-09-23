"""
navigation/direction_assistant.py

Converts a GuidanceDecision (from obstacle_guidance.py) into a short,
clear text message that can be passed to VoiceManager.speak().

This module is purely rule-based. It contains no camera, no YOLO model,
no OpenCV, and no voice output. It only builds strings.

Responsibilities:
  - Normalize object names for natural speech ("dining table" → "table").
  - Apply message rules based on priority, zone, and proximity.
  - Return None when no announcement is needed.
  - Return "Path ahead is clear." for clear-path notices.
"""

from typing import Optional

from navigation.obstacle_guidance import GuidanceDecision
from navigation.spatial_analyzer import Priority, Zone, Proximity


# ---------------------------------------------------------------------------
# Object name normalization map
# ---------------------------------------------------------------------------

# Maps YOLO class names that are awkward when spoken to friendlier versions.
# Only entries that need normalization are listed here.
# Names not in this map are used as-is.
NAME_MAP: dict[str, str] = {
    "cell phone":   "phone",
    "dining table": "table",
}


# ---------------------------------------------------------------------------
# DirectionAssistant
# ---------------------------------------------------------------------------

class DirectionAssistant:
    """
    Converts a GuidanceDecision into a short spoken guidance message.

    Usage
    -----
    assistant = DirectionAssistant()

    message = assistant.build_message(decision)

    if message:
        voice.speak(message)

    Rules summary
    -------------
    - If should_announce is False  → return None.
    - If is_clear_notice is True   → return "Path ahead is clear."
    - HIGH priority obstacles get urgency-scaled messages
      (e.g. "Stop." for CENTER+CLOSE, "Be careful" for CENTER+NEAR).
    - MEDIUM / LOW priority obstacles get simple positional messages.
    - FAR objects get a quiet positional message only (no "Stop", no "close").
    - Object names are normalized before being inserted into messages.
    - No technical data (IDs, ratios, coordinates) ever appears in a message.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_message(self, decision: GuidanceDecision) -> Optional[str]:
        """
        Convert a GuidanceDecision into a spoken guidance string.

        Parameters
        ----------
        decision : GuidanceDecision
            The output from ObstacleGuidance.process_frame().

        Returns
        -------
        str or None
            A short guidance message ready for VoiceManager.speak(),
            or None if no announcement is required.
        """
        # Nothing to announce
        if not decision.should_announce:
            return None

        # Path is now clear — a previously blocking obstacle has gone
        if decision.is_clear_notice:
            return "Path ahead is clear."

        det  = decision.detection
        name = self._normalize_name(det.object_name)

        return self._select_message(
            name      = name,
            zone      = det.zone,
            proximity = det.proximity,
            priority  = det.priority,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_name(object_name: str) -> str:
        """
        Return a speech-friendly version of the YOLO class name.

        Uses NAME_MAP for known awkward names; returns the original
        name (lowercased) for everything else.

        Parameters
        ----------
        object_name : str
            Raw class label from the YOLO model.

        Returns
        -------
        str
        """
        return NAME_MAP.get(object_name.lower(), object_name.lower())

    @staticmethod
    def _select_message(
        name    : str,
        zone    : str,
        proximity: str,
        priority: str,
    ) -> Optional[str]:
        """
        Apply the rule table to select the appropriate guidance message.

        Parameters
        ----------
        name : str
            Normalized object name (already speech-friendly).
        zone : str
            Zone.LEFT, Zone.CENTER, or Zone.RIGHT.
        proximity : str
            Proximity.FAR, Proximity.NEAR, or Proximity.CLOSE.
        priority : str
            Priority.HIGH, Priority.MEDIUM, Priority.LOW, or Priority.IGNORE.

        Returns
        -------
        str or None
        """
        # IGNORE objects should never reach here, but handle defensively
        if priority == Priority.IGNORE:
            return None

        # ------------------------------------------------------------------
        # HIGH priority — urgency scales with proximity
        # ------------------------------------------------------------------
        if priority == Priority.HIGH:

            if proximity == Proximity.CLOSE:
                if zone == Zone.CENTER:
                    return f"Stop. {name} directly ahead."
                elif zone == Zone.LEFT:
                    return f"{name} close on your left."
                else:  # RIGHT
                    return f"{name} close on your right."

            elif proximity == Proximity.NEAR:
                if zone == Zone.CENTER:
                    return f"{name} ahead. Be careful."
                elif zone == Zone.LEFT:
                    return f"{name} on your left."
                else:  # RIGHT
                    return f"{name} on your right."

            else:  # FAR — low urgency, simple positional only
                if zone == Zone.CENTER:
                    return f"{name} ahead."
                elif zone == Zone.LEFT:
                    return f"{name} on your left."
                else:  # RIGHT
                    return f"{name} on your right."

        # ------------------------------------------------------------------
        # MEDIUM or LOW priority — simple positional messages
        # Proximity does not change the message template for these tiers,
        # but FAR MEDIUM/LOW objects are still announced if the guidance
        # system decided they were worth announcing (e.g. zone change).
        # ------------------------------------------------------------------
        if priority in (Priority.MEDIUM, Priority.LOW):

            if zone == Zone.CENTER:
                return f"{name} ahead."
            elif zone == Zone.LEFT:
                return f"{name} on your left."
            else:  # RIGHT
                return f"{name} on your right."

        # Fallback — should not be reached
        return None
