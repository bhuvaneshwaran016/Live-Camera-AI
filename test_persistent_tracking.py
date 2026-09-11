import time
import cv2
from ultralytics import YOLO

from camera_motion import CameraMotionEstimator
from tracking.persistent_tracker import PersistentTracker
from geometry.spatial import bbox_center


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
        raise RuntimeError(
            "Could not open camera"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    motion = CameraMotionEstimator(
        max_features=1500,
        min_matches=10,
    )

    tracker = PersistentTracker(
        max_missing_frames=30,
        max_center_distance=180,
        min_iou=0.05,
    )

    frame_number = 0

    last_time = time.perf_counter()
    fps = 0.0

    print()
    print("=" * 70)
    print("PERSISTENT CAMERA-AWARE TRACKING")
    print("=" * 70)
    print()
    print("Keep the camera still first.")
    print("Then slowly pan and tilt it.")
    print("Watch PIDs, not YOLO IDs.")
    print()
    print("Press Q to quit.")
    print()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print(
                "Camera frame read failed."
            )
            break

        frame_number += 1

        # -----------------------------------------------------
        # Camera motion
        # -----------------------------------------------------

        camera_state = motion.estimate(
            frame
        )

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

        detections = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(
                int
            )
            ids = boxes.id.cpu().numpy().astype(
                int
            )

            for (
                bbox,
                confidence,
                class_id,
                detector_id,
            ) in zip(
                xyxy,
                confs,
                classes,
                ids,
            ):
                bbox = tuple(
                    map(float, bbox)
                )

                detections.append({
                    "track_id": int(
                        detector_id
                    ),
                    "class_id": int(
                        class_id
                    ),
                    "class_name": model.names[
                        int(class_id)
                    ],
                    "confidence": float(
                        confidence
                    ),
                    "bbox": bbox,
                    "center": bbox_center(
                        bbox
                    ),
                })

        # -----------------------------------------------------
        # Persistent tracking
        # -----------------------------------------------------

        visible_tracks = tracker.update(
            detections,
            camera_state=camera_state,
            timestamp=time.time(),
        )

        # -----------------------------------------------------
        # Draw persistent tracks
        # -----------------------------------------------------

        display = frame.copy()

        for track in visible_tracks:
            x1, y1, x2, y2 = map(
                int,
                track.bbox,
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"PID {track.persistent_id} | "
                f"{track.class_name} | "
                f"{track.confidence:.2f} | "
                f"YID {track.last_detector_id}"
            )

            cv2.putText(
                display,
                label,
                (
                    x1,
                    max(25, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cx, cy = track.center

            cv2.circle(
                display,
                (int(cx), int(cy)),
                5,
                (0, 255, 255),
                -1,
            )

        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------

        now = time.perf_counter()

        dt = now - last_time
        last_time = now

        if dt > 0:
            instant_fps = 1.0 / dt

            fps = (
                instant_fps
                if fps == 0
                else 0.9 * fps + 0.1 * instant_fps
            )

        # -----------------------------------------------------
        # Camera statistics
        # -----------------------------------------------------

        camera_valid = camera_state[
            "valid"
        ]

        rotation = camera_state[
            "rotation_deg"
        ]

        cumulative = camera_state[
            "cumulative_rotation_deg"
        ]

        matches = camera_state[
            "matches"
        ]

        inliers = camera_state[
            "inliers"
        ]

        inlier_ratio = camera_state[
            "inlier_ratio"
        ]

        # -----------------------------------------------------
        # Overlay
        # -----------------------------------------------------

        info = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Visible PIDs: {tracker.get_visible_count()}",
            f"Total PIDs: {tracker.get_unique_count()}",
            f"Camera: {'VALID' if camera_valid else 'INVALID'}",
            f"Rotation: {rotation:.2f} deg",
            f"Cumulative: {cumulative:.2f} deg",
            f"Matches: {matches}",
            f"Inliers: {inliers}",
            f"Inlier ratio: {inlier_ratio:.2%}",
        ]

        y = 30

        for text in info:
            cv2.putText(
                display,
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
            "Persistent Camera-Aware Tracking",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("PERSISTENT TRACKING RESULTS")
    print("=" * 70)
    print(
        f"Frames processed: "
        f"{frame_number}"
    )
    print(
        f"Total persistent IDs: "
        f"{tracker.get_unique_count()}"
    )
    print(
        f"Currently visible: "
        f"{tracker.get_visible_count()}"
    )
    print()

    for track in tracker.get_all_tracks():
        print(
            f"PID {track.persistent_id:4d} | "
            f"{track.class_name:15s} | "
            f"frames={track.frames_seen:4d} | "
            f"detector IDs="
            f"{sorted(track.detector_ids)}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
