"""Offline tests for navigation/distance_smoother.py."""

import sys

from navigation.distance_estimator import DistanceEstimate
from navigation.distance_smoother import DistanceSmoother


passed = 0
failed = 0


def estimate(name, distance):
    return DistanceEstimate(
        object_name=name,
        distance_m=distance,
        confidence=0.90,
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


print("\n=== Distance Smoother Unit Tests ===\n")


# ---------------------------------------------------------
# 1. First detection
# ---------------------------------------------------------

s = DistanceSmoother()

state, report = s.update(
    estimate("person", 1.20)
)

check(
    "first distance is reported",
    report,
)


# ---------------------------------------------------------
# 2. Small jitter should not repeatedly report
# ---------------------------------------------------------

reports = []

for value in [
    1.21,
    1.25,
    1.18,
    1.27,
    1.24,
]:

    _, should_report = s.update(
        estimate("person", value)
    )

    reports.append(
        should_report
    )

check(
    "small distance jitter is suppressed",
    not any(reports),
    f"reports={reports}",
)


# ---------------------------------------------------------
# 3. Large movement
#
# Because the smoother uses a 5-frame window, one sudden
# measurement should not immediately trigger a report.
# Repeated measurements of the new position should.
# ---------------------------------------------------------

s.reset()

s.update(
    estimate("person", 1.20)
)

reported = False

for _ in range(5):

    _, should_report = s.update(
        estimate("person", 0.60)
    )

    if should_report:
        reported = True
        break

check(
    "large sustained movement is reported",
    reported,
)


# ---------------------------------------------------------
# 4. Distance-band change
# ---------------------------------------------------------

s.reset()

s.update(
    estimate("person", 1.20)
)

_, report = s.update(
    estimate("person", 2.20)
)

check(
    "distance-band change is reported",
    report,
)


# ---------------------------------------------------------
# 5. Repeated stable distance
# ---------------------------------------------------------

s.reset()

s.update(
    estimate("chair", 2.00)
)

_, report = s.update(
    estimate("chair", 2.01)
)

check(
    "repeated stable distance is suppressed",
    not report,
)


# ---------------------------------------------------------
# 6. Different objects have independent state
# ---------------------------------------------------------

s.reset()

_, first_person = s.update(
    estimate("person", 2.0)
)

_, first_chair = s.update(
    estimate("chair", 2.0)
)

check(
    "objects have independent state",
    first_person and first_chair,
)


# ---------------------------------------------------------
# 7. Spoken wording
# ---------------------------------------------------------

state, _ = s.update(
    estimate("bottle", 0.60)
)

message = s.spoken_message(state)

check(
    "spoken message contains close distance",
    "bottle" in message
    and "very close" in message,
    message,
)


# ---------------------------------------------------------
# 8. Reset
# ---------------------------------------------------------

s.reset()

_, report = s.update(
    estimate("person", 2.0)
)

check(
    "reset allows a new report",
    report,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(
    f"\n=== Results: {passed} passed, {failed} failed ===\n"
)

if failed:
    sys.exit(1)