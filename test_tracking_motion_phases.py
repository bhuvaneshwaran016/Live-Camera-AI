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

PHASE_SECONDS = 10


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

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------
    phase_stats = {
        "STATIONARY_1": {
            "frames": 0,
            "new_ids": 0,
            "visible_ids": set(),
            "rotation_sum": 0.0,
            "valid_motion_frames": 0,
        },
        "MOVING": {
            "frames": 0,
            "new_ids": 0,
            "visible_ids": set(),
            "rotation_sum": 0.0,
            "valid_motion_frames": 0,
        },
        "STATIONARY_2": {
            "frames": 0,
            "new_ids": 0,
            "visible_ids": set(),
            "rotation_sum": 0.0,
            "valid_motion_frames": 0,
        },
    }

    all_ids = set()
    previous_ids = set()

    id_first_frame = {}
    id_last_frame = {}
    id_frame_count = defaultdict(int)

    phase_names = [
        "STATIONARY_1",
        "MOVING",
        "STATIONARY_2",
    ]

    phase_index = 0
    phase_start = time.perf_counter()

    frame_number = 0
    fps = 0.0
    last_frame_time = time.perf_counter()

    print()
    print("=" * 70)
    print("CAMERA-MOTION TRACKING STABILITY TEST")
    print("=" * 70)
    print()
    print("PHASE 1: Keep camera completely STILL for 10 seconds.")
    print("PHASE 2: Slowly move/pan/tilt the camera for 10 seconds.")
    print("PHASE 3: Return camera to a stable position for 10 seconds.")
    print()
    print("The program will advance phases automatically.")
    print("Press Q to quit.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera frame read failed.")
            break

        frame_number += 1

        # -----------------------------------------------------
        # Phase handling
        # -----------------------------------------------------
        elapsed_phase = time.perf_counter() - phase_start

        if elapsed_phase >= PHASE_SECONDS:
            if phase_index < len(phase_names) - 1:
                phase_index += 1
                phase_start = time.perf_counter()

                print(
                    f"\n>>> ENTERING PHASE: "
                    f"{phase_names[phase_index]}"
                )
            else:
                # Test finished automatically.
                break

        phase = phase_names[phase_index]
        stats = phase_stats[phase]

        # -----------------------------------------------------
        # Camera motion
        # -----------------------------------------------------
        camera_state = motion.estimate(frame)

        if camera_state["valid"]:
            stats["valid_motion_frames"] += 1
            stats["rotation_sum"] += abs(
                camera_state["rotation_deg"]
            )

        stats["frames"] += 1

        # -----------------------------------------------------
        # YOLO + ByteTrack
        # -----------------------------------------------------
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

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
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

                stats["visible_ids"].add(track_id)

                # Correct new-ID detection.
                if track_id not in all_ids:
                    all_ids.add(track_id)
                    stats["new_ids"] += 1
                    id_first_frame[track_id] = frame_number

                id_last_frame[track_id] = frame_number
                id_frame_count[track_id] += 1

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
                    f"ID {track_id} "
                    f"{class_name} "
                    f"{confidence:.2f}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------
        now = time.perf_counter()
        dt = now - last_frame_time
        last_frame_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            fps = (
                instant_fps
                if fps == 0
                else 0.9 * fps + 0.1 * instant_fps
            )

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------
        rotation = camera_state["rotation_deg"]
        inliers = camera_state["inliers"]
        matches = camera_state["matches"]

        elapsed = time.perf_counter() - phase_start
        remaining = max(0.0, PHASE_SECONDS - elapsed)

        info = [
            f"PHASE: {phase}",
            f"Phase time remaining: {remaining:.1f}s",
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Visible IDs: {len(current_ids)}",
            f"Total IDs: {len(all_ids)}",
            f"New IDs this phase: {stats['new_ids']}",
            f"Rotation: {rotation:.2f} deg",
            f"Matches: {matches}",
            f"RANSAC inliers: {inliers}",
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

        cv2.imshow(
            "Tracking Motion Phases",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("CAMERA-MOTION TRACKING RESULTS")
    print("=" * 70)

    for phase in phase_names:
        stats = phase_stats[phase]

        avg_rotation = (
            stats["rotation_sum"]
            / stats["valid_motion_frames"]
            if stats["valid_motion_frames"]
            else 0.0
        )

        print()
        print(f"{phase}")
        print("-" * 40)
        print(f"Frames:              {stats['frames']}")
        print(f"Unique IDs observed: {len(stats['visible_ids'])}")
        print(f"New IDs:             {stats['new_ids']}")
        print(f"Avg |rotation|:      {avg_rotation:.3f} deg/frame")

    print()
    print("=" * 70)
    print("GLOBAL ID STATISTICS")
    print("=" * 70)
    print(f"Total frames:         {frame_number}")
    print(f"Total unique IDs:     {len(all_ids)}")

    if id_frame_count:
        lifetimes = list(id_frame_count.values())

        print(
            f"Average ID lifetime:  "
            f"{sum(lifetimes) / len(lifetimes):.2f} frames"
        )

        print(
            f"Longest ID lifetime:  "
            f"{max(lifetimes)} frames"
        )

        print()
        print("Longest-lived IDs:")

        for track_id, lifetime in sorted(
            id_frame_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]:
            print(
                f"  ID {track_id:4d}: "
                f"{lifetime:4d} frames "
                f"(frame "
                f"{id_first_frame[track_id]} -> "
                f"{id_last_frame[track_id]})"
            )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
