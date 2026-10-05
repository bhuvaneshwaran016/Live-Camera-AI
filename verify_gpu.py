#!/usr/bin/env python3

import importlib
import platform
import subprocess
import sys
from typing import Optional


def get_package_version(package_name: str) -> Optional[str]:
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def get_torch_info():
    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return {
                "torch_version": torch.__version__,
                "cuda_available": False,
                "cuda_version": torch.version.cuda,
                "gpu_name": None,
                "vram_gb": None,
                "compute_capability": None,
                "fp16_support": False,
            }

        device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)

        props = torch.cuda.get_device_properties(device_index)

        vram_gb = props.total_memory / (1024 ** 3)
        compute_capability = f"{props.major}.{props.minor}"

        # NVIDIA GPUs of this class support FP16, but we still report
        # the actual CUDA device capability rather than assuming it.
        fp16_support = props.major >= 5

        return {
            "torch_version": torch.__version__,
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "compute_capability": compute_capability,
            "fp16_support": fp16_support,
        }

    except ImportError:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu_name": None,
            "vram_gb": None,
            "compute_capability": None,
            "fp16_support": False,
        }

    except Exception as exc:
        print(f"ERROR: Failed to inspect PyTorch/CUDA: {exc}")
        sys.exit(1)


def get_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except FileNotFoundError:
        return None


def get_tensor_rt_version():
    candidates = [
        "tensorrt",
        "tensorrt_bindings",
    ]

    for package in candidates:
        try:
            module = importlib.import_module(package)

            version = getattr(module, "__version__", None)

            if version:
                return version

            try:
                import importlib.metadata

                return importlib.metadata.version(package)
            except Exception:
                pass

        except Exception:
            continue

    return None


def print_status(label: str, value):
    if value is None:
        value = "NOT AVAILABLE"

    print(f"{label:<24}: {value}")


def main():
    print("=" * 50)
    print("GPU VERIFICATION")
    print("=" * 50)

    print()

    print_status("OS", platform.platform())
    print_status("Python", platform.python_version())

    torch_info = get_torch_info()

    print()
    print_status("GPU", torch_info["gpu_name"])

    if torch_info["vram_gb"] is not None:
        print_status("VRAM", f"{torch_info['vram_gb']:.2f} GB")
    else:
        print_status("VRAM", None)

    print_status("CUDA", torch_info["cuda_version"])
    print_status("PyTorch", torch_info["torch_version"])
    print_status("CUDA available", torch_info["cuda_available"])
    print_status("TensorRT", get_tensor_rt_version())
    print_status("Compute capability", torch_info["compute_capability"])
    print_status("FP16 support", torch_info["fp16_support"])

    print()

    print("NVIDIA-SMI")
    print("-" * 50)

    smi = get_nvidia_smi()

    if smi:
        print(smi)
    else:
        print("nvidia-smi not available")

    print()

    print("=" * 50)
    print("VERIFICATION RESULT")
    print("=" * 50)

    if not torch_info["cuda_available"]:
        print("FAIL")
        print()
        print("CUDA is not available through PyTorch.")
        print("Do NOT continue to TensorRT/model optimization yet.")
        sys.exit(1)

    if torch_info["gpu_name"] is None:
        print("FAIL")
        print("No CUDA GPU detected.")
        sys.exit(1)

    print("PASS")
    print()
    print("CUDA GPU detected successfully.")

    vram = torch_info["vram_gb"] or 0.0
    print()
    print("=" * 50)
    print("LAPTOP OPTIMIZATION RECOMMENDATION")
    print("=" * 50)

    try:
        import bitsandbytes
        print_status("bitsandbytes", bitsandbytes.__version__)
    except Exception:
        print_status("bitsandbytes", "Not installed")

    if vram <= 7.0:
        print(f"\n[Laptop Profile Detected: {torch_info['gpu_name']} ({vram:.2f} GB VRAM)]")
        print("  - Qwen precision : '4bit' (uses ~2.2 GB VRAM)")
        print("  - YOLO model     : 'yolo11s.pt' (fast & lightweight)")
        print("  - Status         : READY for laptop execution in config.yaml")
    else:
        print(f"\n[High-VRAM GPU Detected: {torch_info['gpu_name']} ({vram:.2f} GB VRAM)]")
        print("  - Qwen precision : '4bit' or 'float16'")
        print("  - YOLO model     : 'yolo11s.pt' or 'yolo11m.pt'")

    print()
    print("Next step:")
    print("Run `run.bat` or `python live_vision_qa.py` to start Live Vision AI.")

if __name__ == "__main__":
    main()