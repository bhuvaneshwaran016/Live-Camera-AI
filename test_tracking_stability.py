import time
import cv2
from collections import defaultdict
from ultralytics import YOLO

from camera_motion import CameraMotionEstimator


CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODEL_PATH = "yolo11m.pt"
IMG_SIZE = 640
CONF = 0.25
IOU = 0.45


def main():
    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    motion = CameraMotionEstimator(
        max_features=1500,
        min_matches=10,
    )

    # Track how often each YOLO ID appears.
    id_frames = defaultdict(int)

    # Count IDs seen overall.
    all_ids = set()

    # Count IDs that disappear/reappear.
    previous_ids = set()
    reappeared_ids = set()

    frame_number = 0
    last_time = time.perf_counter()
    fps = 0.0

    print()
    print("=" * 70)
    print("TRACKING STABILITY TEST")
    print("=" * 70)
    print("1. Keep camera STILL for ~10 seconds.")
    print("2. Slowly pan left/right for ~10 seconds.")
    print("3. Slowly tilt/move camera for ~10 seconds.")
    print("4. Press Q to quit.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame_number += 1

        camera_state = motion.estimate(frame)

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            verbose=False,
        )

        result = results[0]

        current_ids = set()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for bbox, confidence, class_id, track_id in zip(
                xyxy,
                confs,
                classes,
                track_ids,
            ):
                track_id = int(track_id)

                current_ids.add(track_id)
                all_ids.add(track_id)
                id_frames[track_id] += 1

                x1, y1, x2, y2 = map(int, bbox)

                class_name = model.names[int(class_id)]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id} {class_name} {confidence:.2f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # IDs that were previously seen and then appear again.
        for track_id in current_ids:
            if track_id not in previous_ids and track_id in all_ids:
                reappeared_ids.add(track_id)

        previous_ids = current_ids

        # FPS
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------
        visible = len(current_ids)
        unique = len(all_ids)

        if id_frames:
            average_lifetime = sum(id_frames.values()) / len(id_frames)
            longest_lifetime = max(id_frames.values())
        else:
            average_lifetime = 0
            longest_lifetime = 0

        info = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Visible IDs: {visible}",
            f"Total IDs: {unique}",
            f"Avg ID lifetime: {average_lifetime:.1f} frames",
            f"Longest ID lifetime: {longest_lifetime} frames",
            f"Camera valid: {camera_state['valid']}",
            f"Rotation: {camera_state['rotation_deg']:.2f} deg",
            f"Cumulative: {camera_state['cumulative_rotation_deg']:.2f} deg",
            f"Matches: {camera_state['matches']}",
            f"Inliers: {camera_state['inliers']}",
        ]

        y = 30

        for text in info:
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 27

        cv2.imshow("Tracking Stability Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("TRACKING STABILITY RESULTS")
    print("=" * 70)
    print(f"Frames processed:       {frame_number}")
    print(f"Total unique IDs:       {len(all_ids)}")
    print(f"Reappeared IDs:         {len(reappeared_ids)}")
    print(f"Average ID lifetime:    {sum(id_frames.values()) / len(id_frames) if id_frames else 0:.2f} frames")
    print(f"Longest ID lifetime:    {max(id_frames.values()) if id_frames else 0} frames")
    print()

    if id_frames:
        lifetimes = sorted(id_frames.values(), reverse=True)

        print("Top 20 longest-lived IDs:")
        for track_id, lifetime in sorted(
            id_frames.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]:
            print(f"  ID {track_id:4d}: {lifetime:4d} frames")

    print("=" * 70)


if __name__ == "__main__":
    main()
