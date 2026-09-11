@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found.
    echo.
    echo Run:
    echo     setup.bat
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo ============================================================
echo                    LIVE VISION AI
echo ============================================================
echo.

python live_vision_qa.py

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
echo                 LIVE VISION AI STOPPED
echo ============================================================
echo.

if not "%EXIT_CODE%"=="0" (
    echo Process exited with code %EXIT_CODE%.
    echo.
)

pause
exit /b %EXIT_CODE%
