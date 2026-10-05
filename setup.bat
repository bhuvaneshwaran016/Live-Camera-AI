@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo.
echo ============================================================
echo              LIVE VISION AI - WINDOWS SETUP
echo ============================================================
echo.

echo [1/8] Checking Python...

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python 3.12 and enable "Add Python to PATH".
    pause
    exit /b 1
)

python --version
if errorlevel 1 (
    echo ERROR: Python is not working correctly.
    pause
    exit /b 1
)

echo.
echo [2/8] Checking Python version...

python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.12 or newer is required.
    pause
    exit /b 1
)

echo.
echo [3/8] Creating virtual environment...

if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

call ".venv\Scripts\activate.bat"

echo.
echo [4/8] Upgrading pip...

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    pause
    exit /b 1
)

echo.
echo [5/8] Installing dependencies...

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [6/8] Checking NVIDIA GPU...

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo WARNING: nvidia-smi was not found.
    echo A supported NVIDIA GPU is recommended.
) else (
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
)

echo.
echo [7/8] Downloading YOLO models (yolo11s.pt & yolo11m.pt)...

python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); YOLO('yolo11m.pt'); print('YOLO models ready.')"
if errorlevel 1 (
    echo ERROR: YOLO model setup failed.
    pause
    exit /b 1
)

echo.
echo [8/8] Downloading Qwen2.5-VL 3B...

python -c "from pathlib import Path; from huggingface_hub import snapshot_download; p=Path('models/Qwen2.5-VL-3B-Instruct'); p.mkdir(parents=True, exist_ok=True); snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', local_dir=str(p)); print('Qwen2.5-VL 3B ready.')"
if errorlevel 1 (
    echo ERROR: Qwen model setup failed.
    pause
    exit /b 1
)


echo.
echo ============================================================
echo                    SETUP COMPLETE
echo ============================================================
echo.
echo Run:
echo.
echo     run.bat
echo.
pause
exit /b 0
