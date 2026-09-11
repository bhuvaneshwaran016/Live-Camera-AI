import time
import cv2
from ultralytics import YOLO

from camera_motion import CameraMotionEstimator
from tracking.object_map import ObjectMap
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
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_width}x{actual_height}")

    motion = CameraMotionEstimator(
        max_features=1500,
        min_matches=10,
    )

    object_map = ObjectMap()

    print()
    print("=" * 70)
    print("LIVE YOLO + BYTETRACK + CAMERA MOTION")
    print("=" * 70)
    print("Move the camera slowly in different directions.")
    print("Watch whether track IDs remain stable.")
    print("Press Q to quit.")
    print()

    frame_number = 0
    fps = 0.0
    last_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera frame read failed.")
            break

        frame_number += 1

        # ---------------------------------------------------------
        # Camera motion
        # ---------------------------------------------------------
        camera_state = motion.estimate(frame)

        # ---------------------------------------------------------
        # YOLO + ByteTrack
        # ---------------------------------------------------------
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

        if result.boxes is not None:
            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = [-1] * len(xyxy)

            for bbox, confidence, class_id, track_id in zip(
                xyxy,
                confs,
                classes,
                track_ids,
            ):
                x1, y1, x2, y2 = map(float, bbox)

                bbox_tuple = (x1, y1, x2, y2)

                # ObjectMap requires the center explicitly.
                center = bbox_center(bbox_tuple)

                class_name = model.names[int(class_id)]

                detections.append({
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": bbox_tuple,
                    "center": center,
                })

        # ---------------------------------------------------------
        # Update persistent object map
        # ---------------------------------------------------------
        object_map.update(detections)

        # ---------------------------------------------------------
        # FPS
        # ---------------------------------------------------------
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            if fps == 0:
                fps = instant_fps
            else:
                fps = 0.9 * fps + 0.1 * instant_fps

        # ---------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------
        display = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = map(int, detection["bbox"])

            track_id = detection["track_id"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"ID {track_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            cv2.putText(
                display,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Draw object center.
            cx, cy = detection["center"]

            cv2.circle(
                display,
                (int(cx), int(cy)),
                5,
                (0, 255, 255),
                -1,
            )

        # ---------------------------------------------------------
        # Camera state
        # ---------------------------------------------------------
        valid = camera_state["valid"]
        rotation = camera_state["rotation_deg"]
        cumulative = camera_state["cumulative_rotation_deg"]
        tx = camera_state["translation_x"]
        ty = camera_state["translation_y"]
        scale = camera_state["scale"]
        matches = camera_state["matches"]
        inliers = camera_state["inliers"]

        visible_count = object_map.visible_object_count()
        unique_count = object_map.unique_object_count()

        y = 30

        info = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Visible objects: {visible_count}",
            f"Unique objects: {unique_count}",
            f"Camera motion: {'VALID' if valid else 'INVALID'}",
            f"Rotation: {rotation:.2f} deg",
            f"Cumulative: {cumulative:.2f} deg",
            f"Translation: X={tx:.1f} Y={ty:.1f}",
            f"Scale: {scale:.4f}",
            f"Matches: {matches}",
            f"RANSAC inliers: {inliers}",
        ]

        for text in info:
            cv2.putText(
                display,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 27

        cv2.imshow(
            "YOLO + ByteTrack + Camera Motion",
            display,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Frames processed: {frame_number}")
    print(f"Unique objects observed: {object_map.unique_object_count()}")
    print(f"Currently visible: {object_map.visible_object_count()}")
    print()


if __name__ == "__main__":
    main()
