"""
path_checker_test.py

Offline unit tests for:
  - navigation/path_checker.py
  - steps_text() helper in navigation_main.py
  - direction_phrase() helper in navigation_main.py
  - voice/voice_input.py  (command filtering, no microphone required)

Run with:
    python path_checker_test.py

No camera, no YOLO model, no microphone required.
"""

import sys

from navigation.spatial_analyzer import DetectionResult, Zone, Priority
from navigation.path_checker import PathChecker, PathStatus

# Import helpers from the lightweight utils module (no YOLO/camera side-effects)
from navigation.nav_utils import steps_text, direction_phrase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(
    name      : str,
    zone      : str  = Zone.CENTER,
    proximity : str  = "NEAR",
    priority  : str  = Priority.MEDIUM,
    box_area  : float = 5000.0,
    track_id  : int  = 1,
    confidence: float = 0.80,
) -> DetectionResult:
    return DetectionResult(
        track_id        = track_id,
        object_name     = name,
        confidence      = confidence,
        center_x        = 320.0,
        center_y        = 240.0,
        box_width       = 80.0,
        box_height      = 80.0,
        box_area        = box_area,
        frame_area      = 307200.0,
        proximity_ratio = box_area / 307200.0,
        proximity       = proximity,
        zone            = zone,
        priority        = priority,
        is_obstacle     = priority != Priority.IGNORE,
    )


passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  PASS  {label}")
        passed += 1
    else:
        msg = f"  FAIL  {label}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)
        failed += 1


# ===========================================================
# 1. steps_text()
# ===========================================================

print("\n=== 1. steps_text() ===")

check("0.1 m -> 1 step",   steps_text(0.1)  == "1 step")
check("0.4 m -> 1 step",   steps_text(0.4)  == "1 step")
check("0.5 m -> 1 step",   steps_text(0.5)  == "1 step")
check("0.6 m -> 1 step",   steps_text(0.6)  == "1 step",  steps_text(0.6))
check("0.75 m -> 2 steps",  steps_text(0.75) == "2 steps", steps_text(0.75))
check("1.0 m -> 2 steps",   steps_text(1.0)  == "2 steps", steps_text(1.0))
check("1.5 m -> 3 steps",   steps_text(1.5)  == "3 steps", steps_text(1.5))
check("2.0 m -> 4 steps",   steps_text(2.0)  == "4 steps")
check("3.5 m -> 7 steps",   steps_text(3.5)  == "7 steps")
check("0.0 m -> 1 step",    steps_text(0.0)  == "1 step",  steps_text(0.0))
check("grammar: 1 (no s)", "step" in steps_text(0.5) and "steps" not in steps_text(0.5))
check("grammar: 2 (s)",    "steps" in steps_text(1.0))


# ===========================================================
# 2. direction_phrase()
# ===========================================================

print("\n=== 2. direction_phrase() ===")

pos, move = direction_phrase(Zone.CENTER)
check("CENTER pos",  pos  == "ahead")
check("CENTER move", move == "Move forward.")

pos, move = direction_phrase(Zone.LEFT)
check("LEFT pos",  pos  == "on your left")
check("LEFT move", move == "Move left.")

pos, move = direction_phrase(Zone.RIGHT)
check("RIGHT pos",  pos  == "on your right")
check("RIGHT move", move == "Move right.")


# ===========================================================
# 3. PathChecker — clear path
# ===========================================================

print("\n=== 3. PathChecker — clear path ===")

checker = PathChecker()

bottle = _det("bottle", zone=Zone.CENTER, box_area=4000.0, track_id=10)

# No other objects -> clear
status = checker.check(bottle, [bottle], "bottle")
check("no obstacles -> clear",        status.is_clear)
check("no blocker object",            status.blocker is None)
check("avoidance direction empty",    status.avoidance_direction == "")

# Only same-object instances -> clear (target not a blocker of itself)
bottle2 = _det("bottle", zone=Zone.CENTER, box_area=3000.0, track_id=11)
status = checker.check(bottle, [bottle, bottle2], "bottle")
check("two bottles -> clear",          status.is_clear)

# IGNORE-priority object in corridor -> not a blocker
tv = _det("tv", zone=Zone.CENTER, box_area=9000.0, priority=Priority.IGNORE, track_id=20)
status = checker.check(bottle, [bottle, tv], "bottle")
check("IGNORE object not a blocker",  status.is_clear)

# Obstacle FARTHER than target (smaller box) -> not a blocker
far_chair = _det("chair", zone=Zone.CENTER, box_area=1000.0, track_id=30)
status = checker.check(bottle, [bottle, far_chair], "bottle")
check("farther object not a blocker", status.is_clear,
      f"chair area={far_chair.box_area} bottle area={bottle.box_area}")


# ===========================================================
# 4. PathChecker — blocked path
# ===========================================================

print("\n=== 4. PathChecker — blocked path ===")

# Chair in CENTER corridor, closer than the bottle
bottle_ctr = _det("bottle", zone=Zone.CENTER, box_area=2000.0, track_id=10)
chair_close = _det(
    "chair",
    zone=Zone.CENTER,
    box_area=8000.0,   # larger = closer
    priority=Priority.MEDIUM,
    track_id=40,
)

status = checker.check(bottle_ctr, [bottle_ctr, chair_close], "bottle")
check("chair blocking center target",  not status.is_clear)
check("blocker is chair",              status.blocker is not None)
check("blocker name is chair",
      status.blocker.object_name.lower() == "chair" if status.blocker else False)
check("avoidance message non-empty",   bool(status.avoidance_message))
check("'blocking' in message",
      "blocking" in status.avoidance_message.lower())
check("'Move' in message",
      "Move" in status.avoidance_message)


# ===========================================================
# 5. PathChecker — target LEFT, blocker in corridor
# ===========================================================

print("\n=== 5. PathChecker — target LEFT, blocker same corridor ===")

bottle_left = _det("bottle", zone=Zone.LEFT, box_area=2000.0, track_id=10)
person_left = _det(
    "person",
    zone=Zone.LEFT,
    box_area=9000.0,
    priority=Priority.HIGH,
    track_id=50,
)

status = checker.check(bottle_left, [bottle_left, person_left], "bottle")
check("person blocking left path",     not status.is_clear)
check("avoidance direction given",     status.avoidance_direction in ("left", "right"))
check("message mentions person",
      "person" in status.avoidance_message.lower())


# ===========================================================
# 6. PathChecker — obstacle NOT in corridor is ignored
# ===========================================================

print("\n=== 6. PathChecker — obstacle outside corridor ignored ===")

bottle_right = _det("bottle", zone=Zone.RIGHT, box_area=2000.0, track_id=10)
chair_left   = _det(
    "chair",
    zone=Zone.LEFT,    # different corridor
    box_area=9000.0,
    priority=Priority.MEDIUM,
    track_id=60,
)

status = checker.check(bottle_right, [bottle_right, chair_left], "bottle")
check("obstacle in wrong zone ignored", status.is_clear)


# ===========================================================
# 7. PathChecker — CENTER blocker also blocks LEFT/RIGHT targets
# ===========================================================

print("\n=== 7. PathChecker — CENTER obstacle blocks any target zone ===")

bottle_rt2 = _det("bottle", zone=Zone.RIGHT, box_area=2000.0, track_id=10)
chair_ctr2 = _det(
    "chair",
    zone=Zone.CENTER,   # center always in corridor
    box_area=8000.0,
    priority=Priority.MEDIUM,
    track_id=70,
)

status = checker.check(bottle_rt2, [bottle_rt2, chair_ctr2], "bottle")
check("center blocker blocks right-target route", not status.is_clear)


# ===========================================================
# 8. PathChecker — multiple blockers: highest priority wins
# ===========================================================

print("\n=== 8. PathChecker — multiple blockers, highest priority selected ===")

target      = _det("bottle", zone=Zone.CENTER, box_area=1000.0, track_id=10)
low_blocker = _det("cup",    zone=Zone.CENTER, box_area=5000.0,
                   priority=Priority.LOW,    track_id=80)
high_blocker = _det("person", zone=Zone.CENTER, box_area=5000.0,
                    priority=Priority.HIGH,   track_id=81)

status = checker.check(target, [target, low_blocker, high_blocker], "bottle")
check("blocked",                         not status.is_clear)
check("highest-priority blocker chosen",
      status.blocker is not None
      and status.blocker.object_name.lower() == "person",
      str(status.blocker.object_name if status.blocker else None))


# ===========================================================
# 9. VoiceInput command filtering (no microphone)
# ===========================================================

print("\n=== 9. VoiceInput._is_relevant() ===")

from voice.voice_input import VoiceInput

check("find bottle",          VoiceInput._is_relevant("find bottle"))
check("find my phone",        VoiceInput._is_relevant("find my phone"))
check("search for a chair",   VoiceInput._is_relevant("search for a chair"))
check("where is my bottle",   VoiceInput._is_relevant("where is my bottle"))
check("locate the table",     VoiceInput._is_relevant("locate the table"))
check("clear",                VoiceInput._is_relevant("clear"))
check("stop",                 VoiceInput._is_relevant("stop"))
check("help",                 VoiceInput._is_relevant("help"))
check("random noise",    not  VoiceInput._is_relevant("random background noise"))
check("empty string",    not  VoiceInput._is_relevant(""))


# ===========================================================
# 10. TargetSearch — parse_request for new aliases
# ===========================================================

print("\n=== 10. TargetSearch — parse_request generality ===")

from navigation.target_search import TargetSearch

ts = TargetSearch()
check("find bottle",        ts.parse_request("find bottle")        == "bottle")
check("find chair",         ts.parse_request("find chair")         == "chair")
check("find phone",         ts.parse_request("find phone")         == "cell phone")
check("find laptop",        ts.parse_request("find laptop")        == "laptop")
check("find my cup",        ts.parse_request("find my cup")        == "cup")
check("find backpack",      ts.parse_request("find backpack")      == "backpack")
check("find remote",        ts.parse_request("find remote")        == "remote")
check("find keyboard",      ts.parse_request("find keyboard")      == "keyboard")
check("find a book",        ts.parse_request("find a book")        == "book")
check("find the couch",     ts.parse_request("find the couch")     is not None)


# ===========================================================
# 11. Path-clear message integration
# ===========================================================

print("\n=== 11. Path-clear spoken message format ===")

def build_clear_msg(target: str, zone: str, dist_m: float) -> str:
    pos, move = direction_phrase(zone)
    steps = steps_text(dist_m)
    return (
        f"Path is clear. "
        f"Your {target} is {pos}, "
        f"about {steps} away. "
        f"{move}"
    )

msg = build_clear_msg("bottle", Zone.LEFT, 1.0)
check("left clear msg contains 'left'",      "left" in msg)
check("left clear msg contains 'Move left'", "Move left" in msg)
check("left clear msg contains steps",       "2 steps" in msg)

msg = build_clear_msg("phone", Zone.CENTER, 0.5)
check("center clear msg contains 'ahead'",   "ahead" in msg)
check("center clear msg: 1 step (not 1 steps)",
      "1 step" in msg and "1 steps" not in msg)

msg = build_clear_msg("chair", Zone.RIGHT, 3.5)
check("right clear msg contains 'right'",    "right" in msg)
check("right clear msg: 7 steps",            "7 steps" in msg)

# Blocked message format
def build_blocked_msg(blocker: str, direction: str) -> str:
    return (
        f"Path is not clear. "
        f"{blocker.capitalize()} is blocking your path. "
        f"Move {direction}."
    )

msg = build_blocked_msg("chair", "right")
check("blocked msg format correct",  "Path is not clear" in msg)
check("blocker name in message",     "Chair" in msg)
check("avoidance in message",        "Move right" in msg)


# ===========================================================
# Summary
# ===========================================================

print(f"\n=== Results: {passed} passed, {failed} failed ===\n")

if failed:
    sys.exit(1)
