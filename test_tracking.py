import time
import cv2
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11m.pt"

CAMERA_INDEX = 0

IMG_SIZE = 640
CONF = 0.25
IOU = 0.45

WINDOW_NAME = "YOLO11m Live Tracking Test"


# ============================================================
# GPU
# ============================================================

print("=" * 60)
print("YOLO11m LIVE CAMERA + TRACKING TEST")
print("=" * 60)

print("\nCUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 60)
print("LOADING YOLO11m")
print("-" * 60)

model = YOLO(MODEL_PATH)

print("YOLO11m loaded successfully.")


# ============================================================
# OPEN CAMERA
# ============================================================

print("\n" + "-" * 60)
print("OPENING CAMERA")
print("-" * 60)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera index {CAMERA_INDEX}"
    )

# Request a reasonable camera resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(
    f"Camera resolution: "
    f"{actual_width}x{actual_height}"
)

print("\nStarting live tracking.")
print("Press Q to quit.")


# ============================================================
# FPS / PERFORMANCE STATE
# ============================================================

frame_count = 0

fps_start = time.perf_counter()
fps = 0.0

latencies = []


# ============================================================
# TRACKING LOOP
# ============================================================

try:

    while True:

        loop_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Camera frame read failed.")
            break

        frame_count += 1


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMG_SIZE,
            conf=CONF,
            iou=IOU,
            verbose=False,
        )


        # ----------------------------------------------------
        # PROCESS RESULTS
        # ----------------------------------------------------

        result = results[0]

        boxes = result.boxes

        object_count = 0

        if boxes is not None and len(boxes) > 0:

            object_count = len(boxes)

            # Class names.
            names = result.names

            for i in range(len(boxes)):

                box = boxes[i]

                # Bounding box.
                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = xyxy

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)


                # Confidence.
                confidence = float(
                    box.conf[0].cpu().item()
                )


                # Class.
                class_id = int(
                    box.cls[0].cpu().item()
                )

                class_name = names[class_id]


                # Track ID.
                track_id = None

                if box.id is not None:
                    track_id = int(
                        box.id[0].cpu().item()
                    )


                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )


                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                if track_id is not None:

                    label = (
                        f"{class_name} "
                        f"ID:{track_id} "
                        f"{confidence:.2f}"
                    )

                else:

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )


        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        loop_time = time.perf_counter() - loop_start

        latency_ms = loop_time * 1000.0

        latencies.append(latency_ms)

        # Keep memory bounded.
        if len(latencies) > 300:
            latencies.pop(0)


        # FPS calculation.
        elapsed = time.perf_counter() - fps_start

        if elapsed >= 1.0:

            fps = frame_count / elapsed

            frame_count = 0
            fps_start = time.perf_counter()


        # ----------------------------------------------------
        # GPU MEMORY
        # ----------------------------------------------------

        allocated_gb = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Objects: {object_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Latency: {latency_ms:.1f} ms",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"VRAM: {allocated_gb:.2f} GB",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(WINDOW_NAME, frame)


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    print("\nCamera released.")

    print("=" * 60)
    print("LIVE TRACKING TEST COMPLETE")
    print("=" * 60)
