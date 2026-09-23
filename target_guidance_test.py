"""Offline tests for navigation/target_guidance.py."""

import sys

from navigation.spatial_analyzer import DetectionResult, Zone, Priority
from navigation.target_guidance import (
    TargetGuidance,
    REQUIRED_ZONE_FRAMES,
    TARGET_LOST_FRAMES,
)


def make_detection(name="bottle", zone=Zone.CENTER, confidence=0.85):
    return DetectionResult(
        track_id=1,
        object_name=name,
        confidence=confidence,
        center_x=320.0,
        center_y=240.0,
        box_width=100.0,
        box_height=100.0,
        box_area=5000.0,
        frame_area=307200.0,
        proximity_ratio=0.016,
        proximity="NEAR",
        zone=zone,
        priority=Priority.LOW,
        is_obstacle=True,
    )


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


print("\n=== Target Guidance Unit Tests ===\n")

g = TargetGuidance()
g.start("bottle", Zone.CENTER)

# Same initial zone does not repeat the search confirmation.
msgs = []
for _ in range(5):
    r = g.process_frame([make_detection(zone=Zone.CENTER)])
    if r.message:
        msgs.append(r.message)
check("no duplicate CENTER announcement", len(msgs) == 0)

# Three stable RIGHT frames cause one announcement.
msgs = []
for _ in range(REQUIRED_ZONE_FRAMES):
    r = g.process_frame([make_detection(zone=Zone.RIGHT)])
    if r.message:
        msgs.append(r.message)
check("RIGHT after stable change", msgs == ["Your bottle is on your right."])

# Extra RIGHT frames do not repeat.
for _ in range(5):
    r = g.process_frame([make_detection(zone=Zone.RIGHT)])
    if r.message:
        msgs.append(r.message)
check("no repeated RIGHT", msgs.count("Your bottle is on your right.") == 1)

# CENTER then LEFT.
for _ in range(REQUIRED_ZONE_FRAMES):
    r = g.process_frame([make_detection(zone=Zone.CENTER)])
check("CENTER transition", r.message == "Your bottle is ahead.")

for _ in range(REQUIRED_ZONE_FRAMES):
    r = g.process_frame([make_detection(zone=Zone.LEFT)])
check("LEFT transition", r.message == "Your bottle is on your left.")

# Unrelated object is ignored.
r = g.process_frame([make_detection(name="person", zone=Zone.RIGHT)])
check("unrelated object ignored", r.message is None)

# Lost target eventually ends guidance.
g2 = TargetGuidance()
g2.start("bottle", Zone.CENTER)
lost_message = None
for _ in range(TARGET_LOST_FRAMES):
    r = g2.process_frame([])
    if r.message:
        lost_message = r.message
check("lost target message", lost_message is not None)
check("guidance stops after loss", not g2.is_active)

# Stop explicitly.
g3 = TargetGuidance()
g3.start("bottle", Zone.CENTER)
g3.stop()
check("explicit stop", not g3.is_active)

print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
if failed:
    sys.exit(1)
