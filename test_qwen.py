import os
import time
from pathlib import Path
import torch
import ultralytics

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_DIR / "models" / "Qwen2.5-VL-3B-Instruct"

MODEL_ID = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "Qwen/Qwen2.5-VL-3B-Instruct"

IMAGE_PATH = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg")

USER_PROMPT = (
    "Look carefully at this image. "
    "Describe what you see, including the main objects, "
    "how many people are visible, and any important spatial "
    "relationships between the objects."
)

MAX_NEW_TOKENS = 256


# ============================================================
# START
# ============================================================

print("=" * 60)
print("QWEN2.5-VL-3B IMAGE UNDERSTANDING TEST (LAPTOP OPTIMIZED)")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD IMAGE
# ============================================================

print("\n" + "-" * 60)
print("LOADING IMAGE")
print("-" * 60)

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image:", IMAGE_PATH)
print("Image size:", image.size)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\n" + "-" * 60)
print("LOADING PROCESSOR")
print("-" * 60)

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Processor loaded successfully.")


# ============================================================
# LOAD MODEL (4-BIT QUANTIZED FOR LAPTOP)
# ============================================================

print("\n" + "-" * 60)
print("LOADING QWEN MODEL (4-bit NF4 Quantization)")
print("-" * 60)

load_start = time.perf_counter()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",
)

load_time = time.perf_counter() - load_start

print("Model loaded successfully in 4-bit mode.")
print(f"Load time: {load_time:.2f} seconds")


print("\nModel device:")
print(next(model.parameters()).device)

allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

print("\nGPU MEMORY AFTER MODEL LOAD")
print(f"Allocated VRAM : {allocated_gb:.3f} GB")
print(f"Reserved VRAM  : {reserved_gb:.3f} GB")


# ============================================================
# BUILD VISION MESSAGE
# ============================================================

print("\n" + "-" * 60)
print("BUILDING VISION REQUEST")
print("-" * 60)

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
                "text": USER_PROMPT,
            },
        ],
    }
]

print("Prompt:")
print(USER_PROMPT)


# ============================================================
# PREPARE INPUT
# ============================================================

print("\n" + "-" * 60)
print("PREPARING MODEL INPUT")
print("-" * 60)

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move tensor inputs to GPU.
inputs = {
    key: value.to("cuda")
    if isinstance(value, torch.Tensor)
    else value
    for key, value in inputs.items()
}

print("Inputs prepared successfully.")

allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

print("\nGPU MEMORY BEFORE GENERATION")
print(f"Allocated VRAM : {allocated_gb:.3f} GB")
print(f"Reserved VRAM  : {reserved_gb:.3f} GB")


# ============================================================
# RESET PEAK MEMORY COUNTER
# ============================================================

torch.cuda.reset_peak_memory_stats()


# ============================================================
# GENERATE
# ============================================================

print("\n" + "-" * 60)
print("RUNNING QWEN VISION INFERENCE")
print("-" * 60)

torch.cuda.synchronize()

inference_start = time.perf_counter()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

torch.cuda.synchronize()

inference_time = time.perf_counter() - inference_start

print(f"Inference time: {inference_time:.3f} seconds")


# ============================================================
# REMOVE INPUT TOKENS
# ============================================================

input_token_length = inputs["input_ids"].shape[1]

generated_ids_trimmed = [
    output_ids[input_token_length:]
    for output_ids in generated_ids
]


# ============================================================
# DECODE
# ============================================================

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# ============================================================
# DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 60)
print("QWEN VISUAL ANSWER")
print("=" * 60)

print(output_text)


# ============================================================
# GPU MEMORY AFTER GENERATION
# ============================================================

print("\n" + "=" * 60)
print("GPU MEMORY AFTER GENERATION")
print("=" * 60)

allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

print(f"Allocated VRAM : {allocated_gb:.3f} GB")
print(f"Reserved VRAM  : {reserved_gb:.3f} GB")
print(f"Peak VRAM      : {peak_gb:.3f} GB")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IMAGE UNDERSTANDING TEST COMPLETE")
print("=" * 60)

print(f"Image             : {IMAGE_PATH}")
print(f"Image resolution  : {image.size[0]}x{image.size[1]}")
print(f"Inference latency : {inference_time:.3f} seconds")
print(f"Generated tokens  : {generated_ids_trimmed[0].shape[0]}")
print(f"Peak VRAM         : {peak_gb:.3f} GB")

print("=" * 60)