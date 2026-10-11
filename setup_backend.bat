@echo off
REM ============================================================
REM  AI-Assisted Multi-Sensor Surveillance System
REM  Backend setup script — Windows
REM  Run ONCE to create virtualenv and install Python deps.
REM ============================================================

echo.
echo ============================================================
echo  Setting up backend Python environment...
echo ============================================================
echo.

cd /d "%~dp0backend"

REM Create virtualenv if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.10+ is installed and on PATH.
        pause & exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate and install dependencies
echo Installing Python dependencies (this may take several minutes)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Some packages failed to install.
    echo Check the output above for details.
    echo NOTE: torch/torchvision may require a specific install command
    echo       if you have a GPU. See: https://pytorch.org/get-started/locally/
    pause & exit /b 1
)

REM Create data directories
echo.
echo Creating data directories...
if not exist "data\uploads"      mkdir data\uploads
if not exist "data\checkpoints"  mkdir data\checkpoints
if not exist "data\datasets"     mkdir data\datasets
if not exist "data\reports"      mkdir data\reports
if not exist "data\replay_data"  mkdir data\replay_data
if not exist "data\radar_data"   mkdir data\radar_data

echo.
echo Initialising database and downloading YOLOv8n pretrained weights (needs internet once)...
python scripts\init_backend.py

echo.
echo ============================================================
echo  Backend setup complete!
echo  Run start_backend.bat to start the API server.
echo ============================================================
echo.
pause
