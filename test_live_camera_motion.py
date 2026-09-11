import time
import cv2

from camera_motion import CameraMotionEstimator


CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


print("=" * 70)
print("LIVE CAMERA MOTION TEST")
print("=" * 70)

print(f"\nOpening camera: /dev/video{CAMERA_INDEX}")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open /dev/video{CAMERA_INDEX}"
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print(
    f"Camera resolution: "
    f"{actual_width}x{actual_height}"
)

estimator = CameraMotionEstimator(
    max_features=1500,
    min_matches=10,
)

print("\nStarting live camera-motion estimation.")
print("Move the camera slowly in different directions.")
print("Press Q to quit.\n")


previous_time = time.perf_counter()
frame_count = 0

fps = 0.0


while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("[ERROR] Could not read camera frame.")
        break

    result = estimator.estimate(frame)

    current_time = time.perf_counter()

    dt = current_time - previous_time

    if dt > 0:
        fps = 1.0 / dt

    previous_time = current_time

    frame_count += 1

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    display = frame.copy()

    if result["valid"]:

        rotation = result["rotation_deg"]
        tx = result["translation_x"]
        ty = result["translation_y"]
        scale = result["scale"]
        matches = result["matches"]
        inliers = result["inliers"]

        status = "VALID"

    else:

        rotation = 0.0
        tx = 0.0
        ty = 0.0
        scale = 1.0
        matches = result["matches"]
        inliers = result["inliers"]

        status = "NO RELIABLE MOTION"

    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    y = 30

    lines = [
        f"Camera Motion: {status}",
        f"Frame: {frame_count}",
        f"FPS: {fps:.1f}",
        f"Rotation: {rotation:+.2f} deg",
        f"Translation X: {tx:+.1f}",
        f"Translation Y: {ty:+.1f}",
        f"Scale: {scale:.4f}",
        f"Matches: {matches}",
        f"RANSAC inliers: {inliers}",
        (
            "Cumulative rotation: "
            f"{result['cumulative_rotation_deg']:+.2f} deg"
        ),
    ]

    for line in lines:

        cv2.putText(
            display,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        y += 30

    cv2.imshow(
        "Live Camera Motion",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\nCamera released.")
print("=" * 70)
print("LIVE CAMERA MOTION TEST COMPLETE")
print("=" * 70)
