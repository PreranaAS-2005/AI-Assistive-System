"""Tests for navigation/guidance_selector.py."""

import sys

from navigation.obstacle_guidance import GuidanceDecision
from navigation.spatial_analyzer import (
    DetectionResult,
    Priority,
    Zone,
)
from navigation.guidance_selector import GuidanceSelector


passed = 0
failed = 0


def make_detection(
    track_id,
    name="person",
    zone=Zone.CENTER,
    proximity="NEAR",
    priority=Priority.HIGH,
    confidence=0.85,
    box_area=5000.0,
):
    return DetectionResult(
        track_id=track_id,
        object_name=name,
        confidence=confidence,
        center_x=320.0,
        center_y=240.0,
        box_width=100.0,
        box_height=100.0,
        box_area=box_area,
        frame_area=307200.0,
        proximity_ratio=0.016,
        proximity=proximity,
        zone=zone,
        priority=priority,
        is_obstacle=True,
    )


def make_decision(detection):
    return GuidanceDecision(
        should_announce=True,
        detection=detection,
        is_clear_notice=False,
    )


def check(name, condition, detail=""):
    global passed, failed

    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(
            f"  FAIL  {name}"
            + (f" - {detail}" if detail else "")
        )
        failed += 1


print("\n=== Guidance Selector Unit Tests ===\n")


# ---------------------------------------------------------
# 1. Select one candidate
# ---------------------------------------------------------

selector = GuidanceSelector()

person = make_decision(
    make_detection(
        track_id=1,
        proximity="NEAR",
    )
)

result = selector.select([person])

check(
    "selects one decision",
    result is person,
)


# ---------------------------------------------------------
# 2. Prefer CLOSE over NEAR
# ---------------------------------------------------------

selector.reset()

far_person = make_decision(
    make_detection(
        track_id=1,
        proximity="FAR",
    )
)

close_person = make_decision(
    make_detection(
        track_id=2,
        proximity="CLOSE",
    )
)

result = selector.select(
    [far_person, close_person]
)

check(
    "prefers closer obstacle",
    result is close_person,
)


# ---------------------------------------------------------
# 3. Keep current focus
# ---------------------------------------------------------

selector.reset()

person_a = make_decision(
    make_detection(
        track_id=1,
        proximity="NEAR",
        zone=Zone.LEFT,
    )
)

person_b = make_decision(
    make_detection(
        track_id=2,
        proximity="NEAR",
        zone=Zone.RIGHT,
    )
)

first = selector.select(
    [person_a, person_b]
)

second = selector.select(
    [person_a, person_b]
)

check(
    "keeps selected track",
    first is second,
)


# ---------------------------------------------------------
# 4. CENTER can become more relevant
# ---------------------------------------------------------

selector.reset()

side = make_decision(
    make_detection(
        track_id=1,
        zone=Zone.LEFT,
        proximity="NEAR",
    )
)

center = make_decision(
    make_detection(
        track_id=2,
        zone=Zone.CENTER,
        proximity="NEAR",
    )
)

first = selector.select([side])

second = selector.select(
    [side, center]
)

check(
    "center obstacle can take focus",
    second is center,
)


# ---------------------------------------------------------
# 5. Higher priority can take focus
# ---------------------------------------------------------

selector.reset()

medium = make_decision(
    make_detection(
        track_id=1,
        priority=Priority.MEDIUM,
        proximity="NEAR",
    )
)

high = make_decision(
    make_detection(
        track_id=2,
        priority=Priority.HIGH,
        proximity="NEAR",
    )
)

selector.select([medium])

result = selector.select(
    [medium, high]
)

check(
    "higher priority can take focus",
    result is high,
)


# ---------------------------------------------------------
# 6. Empty input
# ---------------------------------------------------------

selector.reset()

result = selector.select([])

check(
    "empty decisions return none",
    result is None,
)


# ---------------------------------------------------------
# 7. Ignore non-announcing decisions
# ---------------------------------------------------------

selector.reset()

ignored = GuidanceDecision(
    should_announce=False,
    detection=make_detection(track_id=10),
)

result = selector.select([ignored])

check(
    "non-announcing decision ignored",
    result is None,
)


# ---------------------------------------------------------
# 8. Reset allows a new focus
# ---------------------------------------------------------

selector.reset()

a = make_decision(
    make_detection(track_id=1)
)

b = make_decision(
    make_detection(track_id=2)
)

first = selector.select([a])

selector.reset()

second = selector.select([b])

check(
    "reset allows new focus",
    first is a and second is b,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    f"\n=== Results: {passed} passed, {failed} failed ===\n"
)

if failed:
    sys.exit(1)