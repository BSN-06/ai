@echo off
REM ============================================================
REM  AI-Assisted Multi-Sensor Surveillance System
REM  Start the React/Vite frontend dev server
REM ============================================================

echo.
echo ============================================================
echo  Starting frontend dev server on http://localhost:5173
echo  Make sure the backend is running on port 8000 first.
echo  Press Ctrl+C to stop.
echo ============================================================
echo.

cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo node_modules not found. Installing dependencies...
    npm install
    if errorlevel 1 (
        echo ERROR: npm install failed.
        pause & exit /b 1
    )
)

npm run dev
