"""
target_search_test.py

Offline unit tests for navigation/target_search.py.

Run with:  python target_search_test.py
No camera or YOLO model required.
"""

import sys

from navigation.spatial_analyzer import analyze_detection
from navigation.target_search import TargetSearch, REQUIRED_FRAMES, SEARCH_TIMEOUT_SECONDS


def make_detection(name, confidence=0.80, x1=250, y1=100, x2=390, y2=300):
    return analyze_detection(
        track_id=1,
        object_name=name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        frame_width=640,
        frame_height=480,
    )


def test_parse_command():
    search = TargetSearch()
    assert search.parse_command("Find my bottle.") == "bottle"


def test_set_target():
    search = TargetSearch()
    target = search.set_target("find bottle")

    assert target == "bottle"
    assert search.target_name == "bottle"
    assert search.is_active


def test_unknown_object():
    search = TargetSearch()
    assert search.set_target("find helicopter") is None


def test_bottle_confirmation():
    search = TargetSearch()
    search.set_target("find bottle")

    detection = make_detection("bottle")

    result = search.process_frame([detection])
    assert result.found
    assert result.message is None

    result = search.process_frame([detection])
    assert result.found
    assert result.message is None

    result = search.process_frame([detection])

    assert result.found
    assert result.message == "Your bottle is ahead."
    assert not search.is_active


def test_left_direction():
    search = TargetSearch()
    search.set_target("find bottle")

    detection = make_detection("bottle", x1=20, x2=150)

    for _ in range(REQUIRED_FRAMES):
        result = search.process_frame([detection])

    assert result.found
    assert "left" in result.message.lower()


def test_right_direction():
    search = TargetSearch()
    search.set_target("find bottle")

    detection = make_detection("bottle", x1=490, x2=620)

    for _ in range(REQUIRED_FRAMES):
        result = search.process_frame([detection])

    assert result.found
    assert "right" in result.message.lower()


def test_unrelated_objects_do_not_find_target():
    search = TargetSearch()
    search.set_target("find bottle")

    detection = make_detection("chair")

    result = None

    # Stay well inside the 30-second timeout.
    for _ in range(10):
        result = search.process_frame([detection])

    assert not result.found
    assert result.message is None
    assert search.is_active


def test_multiple_instances_highest_confidence():
    search = TargetSearch()
    search.set_target("find bottle")

    low = make_detection(
        "bottle",
        confidence=0.65,
        x1=20,
        x2=150,
    )

    high = make_detection(
        "bottle",
        confidence=0.95,
        x1=490,
        x2=620,
    )

    for _ in range(REQUIRED_FRAMES):
        result = search.process_frame([low, high])

    assert result.found
    assert "right" in result.message.lower()


def test_clear_target():
    search = TargetSearch()
    search.set_target("find bottle")

    assert search.is_active

    search.clear_target()

    assert not search.is_active
    assert search.target_name is None


def test_cancel():
    search = TargetSearch()
    search.set_target("find bottle")

    assert search.is_active

    search.cancel()

    assert not search.is_active
    assert search.target_name is None


def test_timeout_value():
    assert SEARCH_TIMEOUT_SECONDS == 30.0


if __name__ == "__main__":
    tests = [
        test_parse_command,
        test_set_target,
        test_unknown_object,
        test_bottle_confirmation,
        test_left_direction,
        test_right_direction,
        test_unrelated_objects_do_not_find_target,
        test_multiple_instances_highest_confidence,
        test_clear_target,
        test_cancel,
        test_timeout_value,
    ]

    passed = 0

    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")