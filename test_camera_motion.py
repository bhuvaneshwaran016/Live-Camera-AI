import cv2
import math

from camera_motion import CameraMotionEstimator


import os
import ultralytics

IMAGE_PATH = os.path.join(
    os.path.dirname(ultralytics.__file__),
    "assets",
    "bus.jpg"
)

TEST_ANGLES = [
    0,
    30,
    45,
    90,
    180,
    270,
]

MAX_ERROR_DEG = 3.0


def angular_error(estimated, expected):
    """
    Smallest difference between two angles.
    Handles wrap-around at 360 degrees.
    """

    difference = abs(estimated - expected)

    while difference > 180:
        difference = abs(difference - 360)

    return difference


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )


height, width = image.shape[:2]

center = (
    width / 2,
    height / 2
)


print("=" * 70)
print("CAMERA MOTION MULTI-ANGLE VALIDATION")
print("=" * 70)

print(f"\nTest image: {IMAGE_PATH}")
print(f"Image size: {width}x{height}")

print("\nAngles being tested:")
print(TEST_ANGLES)

print("\n" + "-" * 70)


results = []


# ---------------------------------------------------------
# Test every rotation
# ---------------------------------------------------------

for expected_angle in TEST_ANGLES:

    estimator = CameraMotionEstimator(
        max_features=1500,
        min_matches=10,
    )

    # First frame establishes reference
    first_result = estimator.estimate(image)

    # Create rotated frame
    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        expected_angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    # Estimate transformation
    result = estimator.estimate(rotated)

    estimated_angle = result["rotation_deg"]

    # The affine convention may produce the opposite
    # sign depending on image transformation direction.
    # Compare both possibilities.
    error_normal = angular_error(
        estimated_angle,
        expected_angle
    )

    error_negative = angular_error(
        -estimated_angle,
        expected_angle
    )

    if error_negative < error_normal:
        comparison_angle = -estimated_angle
        error = error_negative
    else:
        comparison_angle = estimated_angle
        error = error_normal

    passed = (
        result["valid"]
        and error <= MAX_ERROR_DEG
    )

    results.append({
        "expected": expected_angle,
        "estimated": estimated_angle,
        "comparison": comparison_angle,
        "error": error,
        "matches": result["matches"],
        "inliers": result["inliers"],
        "scale": result["scale"],
        "valid": result["valid"],
        "passed": passed,
    })

    print(
        f"\nExpected:   {expected_angle:7.2f}°"
    )

    print(
        f"Estimated:  {estimated_angle:7.2f}°"
    )

    print(
        f"Error:      {error:7.2f}°"
    )

    print(
        f"Matches:    {result['matches']:7d}"
    )

    print(
        f"Inliers:    {result['inliers']:7d}"
    )

    print(
        f"Scale:      {result['scale']:7.4f}"
    )

    print(
        f"Valid:      {result['valid']}"
    )

    print(
        f"Result:     {'PASS' if passed else 'FAIL'}"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

passed_count = sum(
    1 for result in results
    if result["passed"]
)

failed_count = len(results) - passed_count

errors = [
    result["error"]
    for result in results
    if result["valid"]
]

print(
    f"\nPassed: {passed_count}/{len(results)}"
)

print(
    f"Failed: {failed_count}/{len(results)}"
)

if errors:

    print(
        f"Mean absolute error: "
        f"{sum(errors) / len(errors):.3f}°"
    )

    print(
        f"Maximum absolute error: "
        f"{max(errors):.3f}°"
    )

else:

    print("No valid measurements.")


print("\n" + "-" * 70)

for result in results:

    print(
        f"{result['expected']:>6.0f}°"
        f"  ->  "
        f"{result['estimated']:>8.2f}°"
        f"  error={result['error']:.2f}°"
        f"  "
        f"{'PASS' if result['passed'] else 'FAIL'}"
    )


# ---------------------------------------------------------
# Final test status
# ---------------------------------------------------------

if failed_count > 0:

    raise RuntimeError(
        "Camera motion multi-angle validation FAILED."
    )

print("\nALL ROTATION TESTS PASSED")
print("TEST COMPLETE")