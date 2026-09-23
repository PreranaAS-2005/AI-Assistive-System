"""
target_search_test.py

Offline unit tests for navigation/target_search.py.

Run with:  python target_search_test.py
No camera or YOLO model required.
"""

import sys

from navigation.spatial_analyzer import DetectionResult, Zone, Priority
from navigation.target_search import (
    TargetSearch,
    REQUIRED_FRAMES,
    SEARCH_WINDOW_FRAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(
    object_name : str,
    zone        : str = Zone.CENTER,
    confidence  : float = 0.85,
    track_id    : int = 1,
    box_area    : float = 5000.0,
) -> DetectionResult:
    """Build a minimal DetectionResult for testing."""
    return DetectionResult(
        track_id        = track_id,
        object_name     = object_name,
        confidence      = confidence,
        center_x        = 320.0,
        center_y        = 240.0,
        box_width       = 100.0,
        box_height      = 100.0,
        box_area        = box_area,
        frame_area      = 307200.0,
        proximity_ratio = 0.016,
        proximity       = "NEAR",
        zone            = zone,
        priority        = Priority.LOW,
        is_obstacle     = True,
    )


def _run_frames(search: TargetSearch, detections_list: list[list]) -> list:
    """Run process_frame for each frame's detection list; collect messages."""
    messages = []
    for dets in detections_list:
        result = search.process_frame(dets)
        if result.message:
            messages.append(result.message)
    return messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f"  -  {detail}" if detail else ""))
        failed += 1


print("\n=== Target Search Unit Tests ===\n")

# ------------------------------------------------------------------
# 1. Phrase parsing
# ------------------------------------------------------------------
print("1. Phrase parsing")
search = TargetSearch()

check("Find my bottle.",   search.parse_request("Find my bottle.")   == "bottle")
check("find the phone",    search.parse_request("find the phone")    == "cell phone")
check("search for a cup",  search.parse_request("search for a cup")  == "cup")
check("where is my chair", search.parse_request("where is my chair") == "chair")
check("bottle (bare)",     search.parse_request("bottle")            == "bottle")
check("empty string",      search.parse_request("")                  is None)
check("table alias",       search.parse_request("find my table")     == "dining table")

# ------------------------------------------------------------------
# 2. set_target validation
# ------------------------------------------------------------------
print("\n2. set_target validation")
search = TargetSearch()

check("valid target",     search.set_target("find my bottle") == "bottle")
check("is_active",        search.is_active)
check("target_name",      search.target_name == "bottle")

search.clear_target()
check("unknown object",   search.set_target("find my keys") is None)
check("error message",    search.get_error_message() is not None)

search.clear_target()
check("phone alias",      search.set_target("phone") == "cell phone")

# ------------------------------------------------------------------
# 3. Matching - ignores unrelated objects during window
# ------------------------------------------------------------------
print("\n3. Matching - ignores unrelated objects")
search = TargetSearch()
search.set_target("bottle")

wrong_frames = [[_make_detection("chair")]] * SEARCH_WINDOW_FRAMES
messages = _run_frames(search, wrong_frames)
check("not visible after window", any("not currently visible" in m for m in messages))
check("exactly one message",      len(messages) == 1, f"got: {messages}")
check("search stopped",           not search.is_active)

# ------------------------------------------------------------------
# 4. Found message - zone CENTER
# ------------------------------------------------------------------
print("\n4. Found message - zone CENTER")
search = TargetSearch()
search.set_target("bottle")
bottle_center = [_make_detection("bottle", zone=Zone.CENTER)]
messages = _run_frames(search, [bottle_center] * REQUIRED_FRAMES)
check("CENTER message after confirmation", any("ahead" in m for m in messages), f"got: {messages}")
check("exactly one message",               len(messages) == 1, f"got: {messages}")
check("search stopped after found",        not search.is_active)

# ------------------------------------------------------------------
# 5. Found message - zone LEFT / RIGHT
# ------------------------------------------------------------------
print("\n5. Found message - zone LEFT / RIGHT")
search = TargetSearch()
search.set_target("bottle")

left_det  = _make_detection("bottle", zone=Zone.LEFT)
right_det = _make_detection("bottle", zone=Zone.RIGHT)

left_msgs = _run_frames(search, [[left_det]] * REQUIRED_FRAMES)
search.set_target("bottle")
right_msgs = _run_frames(search, [[right_det]] * REQUIRED_FRAMES)

check("LEFT message",  any("left"  in m for m in left_msgs),  f"got: {left_msgs}")
check("RIGHT message", any("right" in m for m in right_msgs), f"got: {right_msgs}")

# ------------------------------------------------------------------
# 6. Not-visible message wording
# ------------------------------------------------------------------
print("\n6. Not-visible message wording")
search = TargetSearch()
search.set_target("bottle")

empty_frames = [[]] * SEARCH_WINDOW_FRAMES
messages = _run_frames(search, empty_frames)
check(
    "not currently visible",
    any("not currently visible" in m for m in messages),
    f"got: {messages}",
)
check(
    "does NOT say 'does not exist'",
    not any("does not exist" in m for m in messages),
)

# ------------------------------------------------------------------
# 7. No announcement before REQUIRED_FRAMES consecutive hits
# ------------------------------------------------------------------
print("\n7. No early announcement")
search = TargetSearch()
search.set_target("bottle")

bottle_det = _make_detection("bottle")
early = [[bottle_det]] * (REQUIRED_FRAMES - 1)
messages = _run_frames(search, early)
check("no message before confirmation", len(messages) == 0, f"got: {messages}")
check("still searching",                search.is_active)

# ------------------------------------------------------------------
# 8. Announces once - no repeat on extra frames
# ------------------------------------------------------------------
print("\n8. Announces once - no repeat")
search = TargetSearch()
search.set_target("bottle")

bottle_det = _make_detection("bottle", zone=Zone.CENTER)
all_frames = [[bottle_det]] * (REQUIRED_FRAMES + 5)
messages = _run_frames(search, all_frames)
check("exactly one announcement", len(messages) == 1, f"got: {messages}")
check("search inactive after",    not search.is_active)

# Extra frames after search ended produce nothing
extra = _run_frames(search, [[bottle_det]] * 3)
check("no message after search ended", len(extra) == 0)

# ------------------------------------------------------------------
# 9. Multiple instances - picks best match
# ------------------------------------------------------------------
print("\n9. Multiple instances - picks highest confidence")
search = TargetSearch()
search.set_target("bottle")

multi = [
    _make_detection("bottle", zone=Zone.LEFT,  confidence=0.70, track_id=1),
    _make_detection("bottle", zone=Zone.RIGHT, confidence=0.95, track_id=2),
]
result = search.process_frame(multi)
check(
    "picks highest confidence zone",
    result.zone == Zone.RIGHT,
    f"zone={result.zone}",
)

# ------------------------------------------------------------------
# 10. clear_target cancels search
# ------------------------------------------------------------------
print("\n10. clear_target")
search = TargetSearch()
search.set_target("bottle")
search.clear_target()
check("not active after clear", not search.is_active)
result = search.process_frame([_make_detection("bottle")])
check("no message after clear", result.message is None)

# ------------------------------------------------------------------
# 11. Interrupted consecutive hits reset confirmation
# ------------------------------------------------------------------
print("\n11. Interrupted hits reset confirmation")
search = TargetSearch()
search.set_target("bottle")

bottle_det = _make_detection("bottle")
# Two hits, one miss, then not enough frames left to confirm
frames = (
    [[bottle_det], [bottle_det], [], [bottle_det]]
    + [[]] * (SEARCH_WINDOW_FRAMES - 4)
)
messages = _run_frames(search, frames)
check(
    "not visible when hits not consecutive",
    any("not currently visible" in m for m in messages),
    f"got: {messages}",
)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n=== Results: {passed} passed, {failed} failed ===\n")

if failed > 0:
    sys.exit(1)
