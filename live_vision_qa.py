import os
import sys
import warnings

warnings.filterwarnings("ignore")

import cv2
import json
import time
import threading
import traceback
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ultralytics import YOLO

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info

from tracking.persistent_tracker import PersistentTracker
from geometry.spatial import relative_geometry

# Camera motion estimator.
# Your tested implementation is expected at the project root.
try:
    from camera_motion import CameraMotionEstimator
except ImportError:
    CameraMotionEstimator = None


import yaml

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_DIR / "config.yaml"
CONFIG = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Warning: Could not read config.yaml ({exc}). Using defaults.")

cam_cfg = CONFIG.get("camera", {})
det_cfg = CONFIG.get("detection", {})
qwen_cfg = CONFIG.get("qwen", {})
app_cfg = CONFIG.get("application", {})

YOLO_MODEL = det_cfg.get("model", "yolo11s.pt")

qwen_model_setting = qwen_cfg.get("model", "models/Qwen2.5-VL-3B-Instruct")
if Path(qwen_model_setting).is_absolute():
    QWEN_MODEL = Path(qwen_model_setting)
else:
    QWEN_MODEL = PROJECT_DIR / qwen_model_setting

QWEN_PRECISION = str(qwen_cfg.get("precision", "4bit")).lower()
QWEN_DEVICE = str(qwen_cfg.get("device", "cuda")).lower()

CAMERA_INDEX = cam_cfg.get("index", 0)

CAMERA_WIDTH = cam_cfg.get("width", 1280)
CAMERA_HEIGHT = cam_cfg.get("height", 720)

YOLO_IMGSZ = det_cfg.get("image_size", 640)
YOLO_CONF = det_cfg.get("confidence", 0.25)
YOLO_IOU = det_cfg.get("iou", 0.45)

WINDOW_NAME = app_cfg.get("window_name", "Live Vision AI")

# Maximum number of tokens generated for an answer.
MAX_NEW_TOKENS = qwen_cfg.get("max_new_tokens", 256)

# Keep a few recent questions/results for diagnostics.
QUESTION_HISTORY_SIZE = 20

# Only one Qwen generation may happen at once.
QWEN_LOCK = threading.Lock()


# ============================================================
# GLOBAL LIVE STATE
# ============================================================

class LiveState:
    def __init__(self):
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_frame_number = 0
        self.latest_timestamp = 0.0

        self.latest_detections = []
        self.latest_tracks = []

        self.target_classes = None  # None means detect ALL objects
        self.target_filter_name = None

        self.camera_state = {
            "valid": False,
            "rotation_deg": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "scale": 1.0,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "cumulative_rotation_deg": 0.0,
            "affine_matrix": None,
        }

        self.processing_qwen = False
        self.last_question = ""
        self.last_answer = ""
        self.last_qwen_time = 0.0
        self.last_qwen_frame = 0

        self.question_history = deque(
            maxlen=QUESTION_HISTORY_SIZE
        )

        self.running = True

    def set_target_classes(self, classes, filter_name=None):
        with self.lock:
            self.target_classes = list(classes) if classes is not None else None
            self.target_filter_name = filter_name

    def get_target_classes(self):
        with self.lock:
            return list(self.target_classes) if self.target_classes is not None else None

    def set_frame(
        self,
        frame,
        frame_number,
        timestamp,
    ):
        with self.lock:
            self.latest_frame = frame.copy()
            self.latest_frame_number = frame_number
            self.latest_timestamp = timestamp

    def get_frame_snapshot(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0, 0.0

            return (
                self.latest_frame.copy(),
                self.latest_frame_number,
                self.latest_timestamp,
            )

    def set_scene_state(
        self,
        detections,
        tracks,
        camera_state,
    ):
        with self.lock:
            self.latest_detections = list(detections)
            self.latest_tracks = list(tracks)

            if camera_state is not None:
                self.camera_state = dict(camera_state)

    def get_scene_snapshot(self):
        with self.lock:
            return (
                list(self.latest_detections),
                list(self.latest_tracks),
                dict(self.camera_state),
            )


LIVE = LiveState()


# ============================================================
# UTILITIES
# ============================================================

def gpu_memory():
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    free, total = torch.cuda.mem_get_info()

    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free / 1024**3,
        "total": total / 1024**3,
    }


def clean_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# QWEN LOADING
# ============================================================

def load_qwen():
    print()
    print("=" * 70)
    print(f"Loading local Qwen2.5-VL 3B (Precision: {QWEN_PRECISION.upper()})...")
    print("=" * 70)

    model_path = QWEN_MODEL.resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Qwen model directory does not exist:\n{model_path}\n"
            f"Please run setup.bat or download the model first."
        )

    config_file = model_path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Qwen config.json not found in:\n{config_file}"
        )

    weight_files = list(
        model_path.glob("*.safetensors")
    )

    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors model weights found in:\n{model_path}"
        )

    total_weight_gb = sum(
        p.stat().st_size for p in weight_files
    ) / 1024**3

    print("Model:", model_path)
    print(f"Weights: {total_weight_gb:.2f} GB")

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    print("Processor loaded.")

    print(f"\nLoading model into GPU/RAM ({QWEN_PRECISION} precision mode)...")

    t0 = time.perf_counter()

    kwargs = {
        "local_files_only": True,
    }

    if QWEN_PRECISION == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = "auto"
            print("Mode: 4-bit NF4 Quantization (Laptop GPU ~2.2GB VRAM requirement)")
        except Exception as exc:
            print(f"WARNING: 4-bit quantization error ({exc}). Falling back to float16.")
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
    elif QWEN_PRECISION == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = "auto"
            print("Mode: 8-bit Quantization (~3.5GB VRAM requirement)")
        except Exception as exc:
            print(f"WARNING: 8-bit quantization error ({exc}). Falling back to float16.")
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
    elif QWEN_PRECISION == "cpu":
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = "cpu"
        print("Mode: System CPU / RAM")
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
        print("Mode: standard float16")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        **kwargs
    )

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 70)
    print("QWEN READY")
    print("=" * 70)
    print(f"Load time: {elapsed:.2f} sec")

    mem = gpu_memory()

    print(f"GPU allocated: {mem['allocated']:.2f} GB")
    print(f"GPU reserved:  {mem['reserved']:.2f} GB")
    print(f"GPU free:      {mem['free']:.2f} GB")

    return model, processor


# ============================================================
# YOLO LOADING
# ============================================================

def load_yolo():
    print()
    print("=" * 70)
    print(f"Loading YOLO model ({YOLO_MODEL})...")
    print("=" * 70)

    model = YOLO(YOLO_MODEL)

    print(f"YOLO ({YOLO_MODEL}) loaded.")

    return model


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    print()
    print("=" * 70)
    print(f"Opening camera (Index {CAMERA_INDEX})...")
    print("=" * 70)

    if sys.platform == "win32":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    # Verify actual frame.
    ok, frame = cap.read()

    if not ok or frame is None:
        cap.release()

        raise RuntimeError(
            "Camera opened but did not return a valid frame."
        )

    h, w = frame.shape[:2]

    print(f"Camera resolution: {w}x{h}")

    return cap



# ============================================================
# DETECTION CONVERSION
# ============================================================

def yolo_results_to_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    boxes = result.boxes

    xyxy = boxes.xyxy.detach().cpu().numpy()

    confs = boxes.conf.detach().cpu().numpy()

    classes = boxes.cls.detach().cpu().numpy()

    ids = None

    if boxes.id is not None:
        ids = (
            boxes.id.detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]

        class_id = int(classes[i])

        if isinstance(names, dict):
            class_name = names.get(
                class_id,
                str(class_id),
            )
        else:
            class_name = str(
                names[class_id]
            )

        detector_id = None

        if ids is not None:
            detector_id = int(ids[i])

        detections.append(
            {
                "track_id": detector_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "bbox": (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ),
            }
        )

    return detections


# ============================================================
# CAMERA MOTION
# ============================================================

def create_motion_estimator():
    if CameraMotionEstimator is None:
        print(
            "WARNING: camera_motion.py not available. "
            "Camera motion compensation disabled."
        )

        return None

    try:
        estimator = CameraMotionEstimator()

        print("Camera motion estimator loaded.")

        return estimator

    except Exception as exc:
        print(
            "WARNING: could not initialize camera motion:",
            exc,
        )

        return None


def update_camera_motion(
    estimator,
    frame,
):
    if estimator is None:
        return {
            "valid": False,
            "rotation_deg": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "scale": 1.0,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "cumulative_rotation_deg": 0.0,
            "affine_matrix": None,
        }

    try:
        state = estimator.estimate(frame)

        if state is None:
            return {
                "valid": False,
                "rotation_deg": 0.0,
                "translation_x": 0.0,
                "translation_y": 0.0,
                "scale": 1.0,
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "cumulative_rotation_deg": 0.0,
                "affine_matrix": None,
            }

        return state

    except Exception as exc:
        print(
            "Camera motion error:",
            repr(exc),
        )

        return {
            "valid": False,
            "rotation_deg": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "scale": 1.0,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "cumulative_rotation_deg": 0.0,
            "affine_matrix": None,
        }


# ============================================================
# OBJECT CONTEXT
# ============================================================

def track_to_dict(track):
    bbox = tuple(
        float(v)
        for v in track.bbox
    )

    center = tuple(
        float(v)
        for v in track.center
    )

    return {
        "persistent_id": int(
            track.persistent_id
        ),
        "class_id": int(
            track.class_id
        ),
        "class_name": str(
            track.class_name
        ),
        "confidence": round(
            float(track.confidence),
            4,
        ),
        "bbox": [
            round(v, 2)
            for v in bbox
        ],
        "center": [
            round(v, 2)
            for v in center
        ],
        "visible": bool(
            track.visible
        ),
        "state": str(
            track.state
        ),
        "confirmed": bool(
            track.confirmed
        ),
        "frames_seen": int(
            track.frames_seen
        ),
        "missed_frames": int(
            track.missed_frames
        ),
    }


def build_scene_context(
    detections,
    tracks,
    camera_state,
    frame_shape,
):
    h, w = frame_shape[:2]

    visible_tracks = [
        t for t in tracks
        if getattr(t, "visible", False)
    ]

    objects = [
        track_to_dict(t)
        for t in visible_tracks
    ]

    # Deterministic spatial relationships.
    relationships = []

    for i in range(len(visible_tracks)):
        for j in range(i + 1, len(visible_tracks)):
            a = visible_tracks[i]
            b = visible_tracks[j]

            try:
                geometry = relative_geometry(
                    a.bbox,
                    b.bbox,
                )

                relationships.append(
                    {
                        "object_a": int(
                            a.persistent_id
                        ),
                        "object_a_class": str(
                            a.class_name
                        ),
                        "object_b": int(
                            b.persistent_id
                        ),
                        "object_b_class": str(
                            b.class_name
                        ),
                        "geometry": geometry,
                    }
                )

            except Exception:
                # Geometry is supplementary.
                pass

    class_counts = {}

    for t in visible_tracks:
        name = str(t.class_name)

        class_counts[name] = (
            class_counts.get(name, 0) + 1
        )

    context = {
        "frame": {
            "width": int(w),
            "height": int(h),
        },

        "detection_authority": (
            "YOLO11m is authoritative for "
            "object class, confidence, bbox, "
            "and current visible-object count."
        ),

        "visible_object_count": len(
            visible_tracks
        ),

        "unique_persistent_object_count": len(
            tracks
        ),

        "class_counts": class_counts,

        "objects": objects,

        "spatial_relationships": relationships,

        "camera": {
            "valid": bool(
                camera_state.get(
                    "valid",
                    False,
                )
            ),
            "image_rotation_deg": round(
                safe_float(
                    camera_state.get(
                        "rotation_deg",
                        0.0,
                    )
                ),
                3,
            ),
            "translation_x": round(
                safe_float(
                    camera_state.get(
                        "translation_x",
                        0.0,
                    )
                ),
                2,
            ),
            "translation_y": round(
                safe_float(
                    camera_state.get(
                        "translation_y",
                        0.0,
                    )
                ),
                2,
            ),
            "scale": round(
                safe_float(
                    camera_state.get(
                        "scale",
                        1.0,
                    )
                ),
                5,
            ),
            "matches": int(
                camera_state.get(
                    "matches",
                    0,
                )
            ),
            "inliers": int(
                camera_state.get(
                    "inliers",
                    0,
                )
            ),
            "inlier_ratio": round(
                safe_float(
                    camera_state.get(
                        "inlier_ratio",
                        0.0,
                    )
                ),
                3,
            ),
        },

        # Do NOT claim physical yaw/pitch/roll.
        "world_orientation": {
            "yaw_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "reason": (
                "No calibrated camera/IMU/world "
                "reference supplied."
            ),
        },
    }

    return context


# ============================================================
# QWEN PROMPT
# ============================================================

def make_qwen_prompt(
    question,
    scene_context,
):
    context_json = json.dumps(
        scene_context,
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are the semantic visual reasoning layer of a live-camera
vision system.

The supplied image is the CURRENT camera frame.

Answer the user's question using what is actually visible
in this image.

AUTHORITATIVE DETECTION RULES:
- YOLO11m is authoritative for detected object classes.
- YOLO11m bounding boxes are authoritative for localization.
- YOLO11m confidence values are authoritative.
- Persistent IDs are supplied by the tracking system.
- Do not invent objects that are not visible.
- Do not change YOLO object classes or bounding boxes.
- Do not invent persistent IDs.
- If the requested information cannot be determined from
  the current image, say that it cannot be determined.

COUNTING:
- For questions asking how many detected objects of a class
  are visible, use the supplied YOLO-derived object context.
- Do not replace the YOLO count with your own visual count.

SPATIAL REASONING:
- Use the supplied deterministic spatial relationships
  when relevant.
- You may interpret the scene semantically, but do not
  fabricate geometric facts.

CAMERA ORIENTATION:
- image_rotation_deg describes apparent image motion.
- It is NOT necessarily physical camera yaw/pitch/roll.
- Never claim a physical world angle unless the supplied
  information actually supports it.

TEMPORAL RULE:
- Answer about THIS supplied frame.
- Do not assume something remains true from previous frames.

USER QUESTION:
{question}

CURRENT SCENE CONTEXT:
{context_json}

Give a concise, direct answer to the user.
""".strip()


# ============================================================
# QWEN INFERENCE
# ============================================================

def run_qwen(
    model,
    processor,
    frame,
    question,
    scene_context,
):
    if frame is None:
        return (
            "I don't have a valid current camera frame."
        )

    prompt = make_qwen_prompt(
        question,
        scene_context,
    )

    # Convert OpenCV BGR -> RGB PIL image.
    # qwen_vl_utils/Transformers handles the actual vision
    # preprocessing.
    from PIL import Image

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # With device_map="auto", model.device is the appropriate
    # input destination for this single-GPU deployment.
    if hasattr(model, "device"):
        device = model.device

        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Remove prompt tokens.
    generated_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output_text = output_text.strip()

    if not output_text:
        return "Qwen returned an empty answer."

    return output_text


# ============================================================
# QUESTION WORKER
# ============================================================

def answer_question(
    model,
    processor,
    question,
):
    if not question.strip():
        return

    if not QWEN_LOCK.acquire(
        blocking=False
    ):
        print(
            "\nQwen is still processing the previous "
            "question. Please wait."
        )

        return

    try:
        LIVE.processing_qwen = True
        LIVE.last_question = question

        # Capture the newest available frame.
        frame, frame_number, timestamp = (
            LIVE.get_frame_snapshot()
        )

        if frame is None:
            answer = (
                "No camera frame is currently available."
            )

            LIVE.last_answer = answer

            return

        detections, tracks, camera_state = (
            LIVE.get_scene_snapshot()
        )

        scene_context = build_scene_context(
            detections,
            tracks,
            camera_state,
            frame.shape,
        )

        print()
        print("=" * 70)
        print("QUESTION")
        print("=" * 70)
        print(question)

        print(
            f"Using live frame: {frame_number}"
        )

        print(
            f"Visible tracked objects: "
            f"{len([t for t in tracks if t.visible])}"
        )

        mem_before = gpu_memory()

        print(
            f"GPU before Qwen: "
            f"{mem_before['allocated']:.2f} GB allocated, "
            f"{mem_before['free']:.2f} GB free"
        )

        t0 = time.perf_counter()

        try:
            answer = run_qwen(
                model,
                processor,
                frame,
                question,
                scene_context,
            )

        except torch.cuda.OutOfMemoryError:
            clean_gpu()

            answer = (
                "Qwen ran out of GPU memory while "
                "analyzing the current frame."
            )

            print(
                "\nWARNING: CUDA out of memory."
            )

        except Exception as exc:
            answer = (
                "I could not analyze the current "
                f"camera frame: {exc}"
            )

            print(
                "\nQwen error:"
            )
            traceback.print_exc()

        elapsed = time.perf_counter() - t0

        LIVE.last_answer = answer
        LIVE.last_qwen_time = elapsed
        LIVE.last_qwen_frame = frame_number

        LIVE.question_history.append(
            {
                "question": question,
                "answer": answer,
                "frame": frame_number,
                "time": elapsed,
            }
        )

        print()
        print("=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)

        print(
            f"\nQwen time: {elapsed:.2f} sec"
        )

        mem_after = gpu_memory()

        print(
            f"GPU after Qwen: "
            f"{mem_after['allocated']:.2f} GB allocated, "
            f"{mem_after['free']:.2f} GB free"
        )

        clean_gpu()

    finally:
        LIVE.processing_qwen = False
        QWEN_LOCK.release()


# ============================================================
# TARGET CLASS DYNAMIC MATCHING (OPEN VOCABULARY SUPPORT)
# ============================================================

def match_target_classes(yolo_names, query_text):
    """
    Dynamically matches query_text against yolo_names without hardcoded COCO dictionaries.
    """
    if not query_text:
        return None
    query_clean = str(query_text).lower().strip()

    if query_clean in ["all", "clear", "reset", "everything", "none", "off", "all objects"]:
        return None

    # Dynamic synonym normalization helper
    synonyms = {
        "human": ["person"],
        "humans": ["person"],
        "people": ["person"],
        "man": ["person"],
        "woman": ["person"],
        "guy": ["person"],
        "phone": ["cell phone"],
        "phones": ["cell phone"],
        "mobile": ["cell phone"],
        "cellphone": ["cell phone"],
        "laptop": ["laptop"],
        "laptops": ["laptop"],
        "computer": ["laptop", "tv"],
        "computers": ["laptop", "tv"],
        "screen": ["tv"],
        "monitor": ["tv"],
        "bottle": ["bottle"],
        "bottles": ["bottle"],
        "cup": ["cup"],
        "cups": ["cup"],
        "chair": ["chair"],
        "chairs": ["chair"],
    }

    words = query_clean.split()
    target_words = set(words)
    target_words.add(query_clean)

    for w in words:
        singular = w[:-1] if w.endswith("s") and len(w) > 2 else w
        target_words.add(singular)
        if w in synonyms:
            target_words.update(synonyms[w])
        if singular in synonyms:
            target_words.update(synonyms[singular])

    matched_ids = set()
    if isinstance(yolo_names, dict):
        for cid, cname in yolo_names.items():
            cname_lower = str(cname).lower()
            for tw in target_words:
                if tw == cname_lower or tw in cname_lower or cname_lower in tw:
                    matched_ids.add(int(cid))

    if matched_ids:
        return sorted(list(matched_ids))
    return None


# ============================================================
# TERMINAL INPUT
# ============================================================

def terminal_input_loop(
    model,
    processor,
    yolo_names=None,
):
    print()
    print("=" * 70)
    print("LIVE QUESTION & OPEN-VOCABULARY DETECTION INPUT")
    print("=" * 70)
    print("• Count Any Object: 'count humans', 'count phone', 'count laptops', etc.")
    print("• Open Vision Questions: 'where is the charger', 'what am I holding', etc.")
    print("• Reset Detection Filter: 'clear' or 'detect all'")
    print("• Type 'quit' or 'exit' to stop.")
    print("=" * 70)
    print()

    while LIVE.running:
        try:
            question = input(
                "\nQuestion / Command > "
            )

        except (EOFError, KeyboardInterrupt):
            LIVE.running = False
            break

        question = question.strip()

        if not question:
            continue

        if question.lower() in {
            "quit",
            "exit",
        }:
            LIVE.running = False
            break

        q_lower = question.lower()

        # Check for count / filter commands
        count_triggers = ["count ", "how many ", "detect ", "filter ", "only ", "show ", "target "]
        is_count_cmd = any(t in q_lower for t in count_triggers) or q_lower in ["count", "filter clear", "detect all", "clear", "all"]

        if is_count_cmd:
            target_str = q_lower
            for t in count_triggers:
                target_str = target_str.replace(t, "")
            target_str = target_str.replace("the ", "").replace("in live video", "").replace("live", "").strip()

            if target_str in ["clear", "all", "reset", "everything", "off", "none", ""]:
                LIVE.set_target_classes(None, None)
                print()
                print("=" * 70)
                print("LIVE VIDEO FILTER RESET")
                print("=" * 70)
                print("Live camera is now detecting and tracking ALL objects.")
                print("=" * 70)
            else:
                matched_ids = match_target_classes(yolo_names, target_str)
                LIVE.set_target_classes(matched_ids, target_str)

                # Compute instant count from live snapshot
                _, tracks, _ = LIVE.get_scene_snapshot()
                visible_tracks = [t for t in tracks if getattr(t, "visible", False)]
                if matched_ids:
                    target_count = len([t for t in visible_tracks if int(t.class_id) in matched_ids])
                    matched_names = [yolo_names.get(cid, str(cid)) for cid in matched_ids] if isinstance(yolo_names, dict) else matched_ids
                else:
                    target_count = len(visible_tracks)
                    matched_names = ["Open-Vocabulary / Vision LLM"]

                print()
                print("=" * 70)
                print(f"INSTANT LIVE COUNT (0.01s)")
                print("=" * 70)
                print(f"Target Object : {target_str.title()} ({matched_names})")
                print(f"Current Count : {target_count} visible in live video")
                print(f"Live Status   : Monitoring every 2 seconds continuously in terminal.")
                print("=" * 70)

        # Spawns Qwen Vision-LLM worker for visual analysis/reasoning on the live camera frame
        worker = threading.Thread(
            target=answer_question,
            args=(
                model,
                processor,
                question,
            ),
            daemon=True,
        )

        worker.start()


# ============================================================
# TARGET OBJECT LIVE 2-SECOND CONTINUOUS COUNT MONITOR
# ============================================================

def smart_target_monitor_loop():
    """
    When a target object or prompt is active, this monitors and prints
    the live camera count EVERY 2 SECONDS continuously in the terminal.
    """
    last_check_time = 0.0

    while LIVE.running:
        time.sleep(0.4)
        now = time.time()
        if now - last_check_time < 2.0:
            continue
        last_check_time = now

        target_classes = LIVE.get_target_classes()
        current_filter_name = LIVE.target_filter_name

        if target_classes is None and current_filter_name is None:
            continue

        detections, tracks, _ = LIVE.get_scene_snapshot()
        visible_tracks = [t for t in tracks if getattr(t, "visible", False)]

        if target_classes is not None:
            matching_tracks = [t for t in visible_tracks if int(t.class_id) in target_classes]
            count = len(matching_tracks)
        else:
            count = len(visible_tracks)

        time_str = time.strftime("%H:%M:%S")
        target_disp = (current_filter_name or "ALL OBJECTS").upper()
        print(f"\n[LIVE 2s DETECT {time_str}] Target: '{target_disp}' | Visible Count: {count}")



# ============================================================
# VISUALIZATION
# ============================================================

def draw_overlay(
    frame,
    tracks,
    camera_state,
    fps,
):
    output = frame.copy()

    # Draw persistent tracks.
    for track in tracks:
        if not track.visible:
            continue

        x1, y1, x2, y2 = [
            int(round(v))
            for v in track.bbox
        ]

        pid = int(track.persistent_id)

        name = str(track.class_name)

        conf = float(track.confidence)

        confirmed = (
            "C"
            if track.confirmed
            else "?"
        )

        label = (
            f"ID {pid} | {name} "
            f"{conf:.2f} | {confirmed}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cx, cy = [
            int(round(v))
            for v in track.center
        ]

        cv2.circle(
            output,
            (cx, cy),
            4,
            (0, 255, 255),
            -1,
        )

    visible = sum(
        1
        for t in tracks
        if t.visible
    )

    unique = len(tracks)

    rotation = safe_float(
        camera_state.get(
            "rotation_deg",
            0.0,
        )
    )

    cumulative = safe_float(
        camera_state.get(
            "cumulative_rotation_deg",
            0.0,
        )
    )

    matches = int(
        camera_state.get(
            "matches",
            0,
        )
    )

    inliers = int(
        camera_state.get(
            "inliers",
            0,
        )
    )

    qwen_status = (
        "PROCESSING"
        if LIVE.processing_qwen
        else "READY"
    )

    lines = [
        f"FPS: {fps:.1f}",
        f"Visible: {visible}",
        f"Unique IDs: {unique}",
        (
            f"Camera rot: {rotation:+.2f} deg "
            f"cum: {cumulative:+.2f}"
        ),
        (
            f"Motion matches/inliers: "
            f"{matches}/{inliers}"
        ),
        f"Qwen: {qwen_status}",
    ]

    y = 25

    for line in lines:
        cv2.putText(
            output,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 25

    # Last answer preview.
    if LIVE.last_answer:
        answer = LIVE.last_answer.replace(
            "\n",
            " ",
        )

        if len(answer) > 110:
            answer = answer[:107] + "..."

        cv2.putText(
            output,
            "Answer:",
            (10, output.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            answer,
            (10, output.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("LIVE VISION AI")
    print("=" * 70)

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA is unavailable."
        )

    else:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load models.
    # --------------------------------------------------------

    yolo = load_yolo()

    qwen, processor = load_qwen()

    # --------------------------------------------------------
    # Camera.
    # --------------------------------------------------------

    cap = open_camera()

    # --------------------------------------------------------
    # Persistent tracker.
    # --------------------------------------------------------

    tracker = PersistentTracker(
        max_missing_frames=30,
        max_center_distance=180.0,
        min_iou=0.05,
        min_confirmed_hits=3,
        distance_weight=0.40,
        iou_weight=0.30,
        appearance_weight=0.30,
        appearance_gate=0.15,
        max_reid_frames=30,
    )

    # --------------------------------------------------------
    # Camera motion.
    # --------------------------------------------------------

    motion_estimator = (
        create_motion_estimator()
    )

    # --------------------------------------------------------
    # Main camera loop setup.
    # --------------------------------------------------------

    frame_count = 0
    fps_start = time.perf_counter()
    fps_frames = 0
    fps = 0.0

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    print()
    print("=" * 70)
    print("CAMERA LOOP STARTED")
    print("=" * 70)
    print("The camera is now continuously running.")
    print("Press 'q' in the camera window to quit.")
    print("=" * 70)

    # Start terminal input thread and smart target monitor thread.
    input_thread = threading.Thread(
        target=terminal_input_loop,
        args=(
            qwen,
            processor,
            getattr(yolo, "names", None),
        ),
        daemon=True,
    )
    input_thread.start()

    monitor_thread = threading.Thread(
        target=smart_target_monitor_loop,
        daemon=True,
    )
    monitor_thread.start()

    try:
        while LIVE.running:
            loop_start = time.perf_counter()

            # ------------------------------------------------
            # Capture newest frame.
            # ------------------------------------------------

            ok, frame = cap.read()

            if not ok or frame is None:
                print(
                    "WARNING: camera frame read failed."
                )

                time.sleep(0.05)

                continue

            frame_count += 1

            timestamp = time.time()

            # Always retain newest frame.
            LIVE.set_frame(
                frame,
                frame_count,
                timestamp,
            )

            # ------------------------------------------------
            # Camera motion.
            # ------------------------------------------------

            camera_state = (
                update_camera_motion(
                    motion_estimator,
                    frame,
                )
            )

            # ------------------------------------------------
            # YOLO + ByteTrack.
            #
            # This remains authoritative for detection.
            # ------------------------------------------------

            yolo_start = time.perf_counter()

            try:
                results = yolo.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    verbose=False,
                )

                yolo_result = results[0]

                detections = (
                    yolo_results_to_detections(
                        yolo_result
                    )
                )

            except torch.cuda.OutOfMemoryError:
                clean_gpu()

                print(
                    "YOLO CUDA OOM."
                )

                detections = []

            except Exception:
                print(
                    "YOLO error:"
                )

                traceback.print_exc()

                detections = []

            yolo_time = (
                time.perf_counter()
                - yolo_start
            )

            # ------------------------------------------------
            # Persistent tracking.
            # ------------------------------------------------

            try:
                tracks = tracker.update(
                    detections=detections,
                    timestamp=timestamp,
                    affine_matrix=(
                        camera_state.get(
                            "affine_matrix"
                        )
                    ),
                    frame=frame,
                )

            except Exception:
                print(
                    "Persistent tracker error:"
                )

                traceback.print_exc()

                tracks = []

            LIVE.set_scene_state(
                detections,
                tracks,
                camera_state,
            )

            # ------------------------------------------------
            # FPS.
            # ------------------------------------------------

            fps_frames += 1

            fps_elapsed = (
                time.perf_counter()
                - fps_start
            )

            if fps_elapsed >= 1.0:
                fps = (
                    fps_frames
                    / fps_elapsed
                )

                fps_frames = 0
                fps_start = (
                    time.perf_counter()
                )

            # ------------------------------------------------
            # Display.
            # ------------------------------------------------

            visible_tracks = [
                t for t in tracks
                if t.visible
            ]

            display = draw_overlay(
                frame,
                visible_tracks,
                camera_state,
                fps,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # ------------------------------------------------
            # Keyboard.
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                LIVE.running = False
                break

            loop_time = (
                time.perf_counter()
                - loop_start
            )

            # Don't artificially sleep.
            # Camera/YOLO should run as fast as available.

    except KeyboardInterrupt:
        print(
            "\nKeyboard interrupt."
        )

    finally:
        LIVE.running = False

        cap.release()

        cv2.destroyAllWindows()

        clean_gpu()

        print()
        print("=" * 70)
        print("LIVE VISION AI STOPPED")
        print("=" * 70)


if __name__ == "__main__":
    main()
