@echo off
setlocal enabledelayedexpansion

echo [BioDynamics Agent] One-click dev startup
echo.

cd /d "%~dp0"

if not exist "backend\venv\Scripts\uvicorn.exe" (
    echo [ERROR] backend\venv\Scripts\uvicorn.exe not found.
    echo Please install backend dependencies first:
    echo   cd backend
    echo   python -m venv venv
    echo   .\venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found.
    pause
    exit /b 1
)

echo [BioDynamics Agent] Starting backend at http://localhost:8000 ...
start "Backend - BioDynamics Agent" cmd /k "cd /d backend && .\venv\Scripts\uvicorn.exe app.main:app --reload --port 8000"

echo [BioDynamics Agent] Starting frontend at http://localhost:3000 ...
start "Frontend - BioDynamics Agent" cmd /k "cd /d frontend && npm run dev"

echo.
echo [BioDynamics Agent] Waiting for services to start...
ping -n 9 127.0.0.1 >nul

echo [BioDynamics Agent] Opening browser...
start http://localhost:3000

echo [BioDynamics Agent] Startup complete. Keep backend/frontend windows running.
pause
