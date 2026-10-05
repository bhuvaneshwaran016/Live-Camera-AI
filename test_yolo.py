import time
from pathlib import Path

import torch
from ultralytics import YOLO


import ultralytics
import os

MODEL_PATH = "yolo11m.pt"
IMAGE_PATH = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")

IMG_SIZE = 640
CONF = 0.25
IOU = 0.45


def main():
    print("=" * 60)
    print("YOLO11m GPU INFERENCE TEST")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Image : {IMAGE_PATH}")
    print(f"CUDA  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = 0

    model = YOLO(MODEL_PATH)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    # Warmup
    print("Warming up GPU...")
    model.predict(
        source=IMAGE_PATH,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=device,
        verbose=False,
    )

    torch.cuda.synchronize()

    # Reset CUDA peak-memory statistics
    torch.cuda.reset_peak_memory_stats(device)

    # Timed inference
    start = time.perf_counter()

    results = model.predict(
        source=IMAGE_PATH,
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        device=device,
        verbose=False,
    )

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    result = results[0]

    print()
    print("-" * 60)
    print("INFERENCE")
    print("-" * 60)

    print(f"Total inference time : {elapsed * 1000:.2f} ms")
    print(f"Approx FPS           : {1.0 / elapsed:.2f}")

    print()
    print("-" * 60)
    print("IMAGE")
    print("-" * 60)

    if result.orig_img is not None:
        h, w = result.orig_img.shape[:2]
        print(f"Original resolution  : {w} x {h}")

    print()
    print("-" * 60)
    print("DETECTIONS")
    print("-" * 60)

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections.")
    else:
        names = result.names

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            label = names[class_id]

            print(f"[{i}]")
            print(f"  class_id   : {class_id}")
            print(f"  label      : {label}")
            print(f"  confidence : {confidence:.4f}")
            print(f"  bbox       : ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
            print(f"  center     : ({cx:.2f}, {cy:.2f})")
            print(f"  size       : {width:.2f} x {height:.2f}")

    print()
    print("-" * 60)
    print("GPU MEMORY")
    print("-" * 60)

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print(f"Allocated VRAM      : {allocated:.3f} GB")
    print(f"Reserved VRAM       : {reserved:.3f} GB")
    print(f"Peak allocated VRAM : {peak:.3f} GB")

    print()
    print("=" * 60)
    print("YOLO11m GPU TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
