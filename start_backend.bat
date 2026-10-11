@echo off
REM ============================================================
REM  AI-Assisted Multi-Sensor Surveillance System
REM  Start the FastAPI backend server
REM ============================================================

echo.
echo ============================================================
echo  Starting backend API server on http://localhost:8000
echo  API docs available at http://localhost:8000/docs
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

cd /d "%~dp0backend"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Run setup_backend.bat first.
    pause & exit /b 1
)

call venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
