"""Offline tests for navigation/detection_filter.py."""
import sys
from navigation.spatial_analyzer import DetectionResult, Zone, Priority
from navigation.detection_filter import DetectionFilter, MIN_CONFIDENCE, STABILITY_FRAMES

def make_detection(name="chair", confidence=.85, area=5000.0):
    return DetectionResult(1, name, confidence, 320.0, 240.0, 100.0, 100.0,
                           area, 307200.0, .016, "NEAR", Zone.CENTER,
                           Priority.LOW, True)

passed = failed = 0
def check(name, condition):
    global passed, failed
    print(("  PASS  " if condition else "  FAIL  ") + name)
    passed += condition
    failed += not condition

print("\n=== Detection Filter Unit Tests ===\n")
f = DetectionFilter()
check("rejects low-confidence detection", f.process_frame([make_detection("chair", MIN_CONFIDENCE-.01)]) == [])
f.reset(); check("rejects unsupported object", f.process_frame([make_detection("refrigerator", .95)]) == [])
f.reset(); check("one frame is not stable", f.process_frame([make_detection()]) == [])
f.reset()
for _ in range(STABILITY_FRAMES): result = f.process_frame([make_detection()])
check("chair becomes stable", len(result) == 1)
check("stable class is chair", result[0].object_name == "chair")
f.reset()
for _ in range(STABILITY_FRAMES): result = f.process_frame([make_detection("bottle", .70, 2000), make_detection("bottle", .92, 3000)])
check("highest-confidence instance selected", result[0].confidence == .92)
f.reset()
for _ in range(STABILITY_FRAMES): result = f.process_frame([make_detection("person", .80), make_detection("tv", .90)])
check("multiple classes supported", {x.object_name for x in result} == {"person", "tv"})
f.reset(); check("reset clears history", f.get_history_count("chair") == 0)
print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
if failed: sys.exit(1)
