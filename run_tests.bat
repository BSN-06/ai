@echo off
REM ============================================================
REM  AI-Assisted Multi-Sensor Surveillance System
REM  Run the automated pytest test suite
REM ============================================================

echo.
echo ============================================================
echo  Running backend pytest suite...
echo ============================================================
echo.

cd /d "%~dp0backend"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Run setup_backend.bat first.
    pause & exit /b 1
)

call venv\Scripts\activate.bat
REM Pass extra pytest args, e.g.:  run_tests.bat -k "not real_models"
python -m pytest tests/ -v --tb=short %* 2>&1

echo.
echo ============================================================
echo  Tests complete. See output above for results.
echo ============================================================
echo.
pause
