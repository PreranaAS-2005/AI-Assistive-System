"""
Offline tests for navigation/distance_estimator.py.

Run:
    python distance_estimator_test.py
"""

import sys

from navigation.spatial_analyzer import DetectionResult, Zone, Priority
from navigation.distance_estimator import DistanceEstimator


def make_detection(
    name="person",
    pixel_width=350.0,
    confidence=0.90,
):
    return DetectionResult(
        track_id=1,
        object_name=name,
        confidence=confidence,
        center_x=320.0,
        center_y=240.0,
        box_width=pixel_width,
        box_height=200.0,
        box_area=pixel_width * 200.0,
        frame_area=307200.0,
        proximity_ratio=0.10,
        proximity="NEAR",
        zone=Zone.CENTER,
        priority=Priority.LOW,
        is_obstacle=True,
    )


passed = 0
failed = 0


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


print("\n=== Distance Estimator Unit Tests ===\n")

# Use focal length 700 px and a 0.45 m person width.
estimator = DistanceEstimator(focal_length_px=700.0)

# 1. Wider bounding box means closer object.
near = estimator.estimate(
    make_detection("person", pixel_width=350.0)
)
far = estimator.estimate(
    make_detection("person", pixel_width=175.0)
)

check(
    "near object gives smaller distance",
    near is not None and far is not None
    and near.distance_m < far.distance_m,
)

# 2. Formula check:
# distance = 0.45 * 700 / 350 = 0.9 m
check(
    "distance formula",
    near is not None and abs(near.distance_m - 0.9) < 0.001,
    f"got {near.distance_m if near else None}",
)

# 3. Different supported object.
laptop = estimator.estimate(
    make_detection("laptop", pixel_width=224.0)
)
check(
    "laptop supported",
    laptop is not None and laptop.object_name == "laptop",
)

# 4. Unknown object returns None.
unknown = estimator.estimate(
    make_detection("cylinder", pixel_width=100.0)
)
check(
    "unknown object returns None",
    unknown is None,
)

# 5. Tiny bounding box is rejected.
tiny = estimator.estimate(
    make_detection("person", pixel_width=2.0)
)
check(
    "tiny bounding box rejected",
    tiny is None,
)

# 6. Distance is capped.
very_small = estimator.estimate(
    make_detection("person", pixel_width=1.0)
)
check(
    "maximum distance cap applied",
    very_small is None or very_small.distance_m <= 20.0,
)

# 7. Spoken formatting.
text = estimator.format_distance(
    make_detection_result := estimator.estimate(
        make_detection("person", pixel_width=350.0)
    )
)
check(
    "spoken distance format",
    text == "person is less than 1 meter away.",
    f"got: {text}",
)

# 8. One-metre-plus wording.
text2 = estimator.format_distance(
    estimator.estimate(
        make_detection("person", pixel_width=175.0)
    )
)
check(
    "meter wording",
    "approximately 1.8 meters away" in text2,
    f"got: {text2}",
)

print(
    f"\n=== Results: {passed} passed, {failed} failed ===\n"
)

if failed:
    sys.exit(1)
