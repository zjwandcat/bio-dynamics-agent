@echo off
setlocal enabledelayedexpansion

REM One-click dev startup for BioDynamics Agent
REM Compatible with both .venv and venv virtual environment directories.

cd /d "%~dp0\.."

set VENV_DIR=""
if exist "backend\.venv\Scripts\uvicorn.exe" (
    set VENV_DIR=backend\.venv
) else if exist "backend\venv\Scripts\uvicorn.exe" (
    set VENV_DIR=backend\venv
)

if "!VENV_DIR!"=="" (
    echo [ERROR] backend virtual environment not found.
    echo Please install backend dependencies first:
    echo   cd backend
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found.
    pause
    exit /b 1
)

REM Step 1: detect model connectivity
echo [BioDynamics Agent] Step 1/3: Detecting model connectivity...
"!VENV_DIR!\Scripts\python.exe" backend\scripts\test_model_connectivity.py
if errorlevel 1 (
    echo [WARN] Some models failed connectivity check, continuing startup.
)
echo.

REM Step 2: backend will clear LangGraph MemorySaver during lifespan
echo [BioDynamics Agent] Step 2/3: Starting backend (context memory will be cleared automatically)...
echo.
start "BioDynamics-Backend" cmd /k "cd /d backend && ..\!VENV_DIR!\Scripts\uvicorn.exe app.main:app --reload --port 8000"

echo [BioDynamics Agent] Step 3/3: Starting frontend http://localhost:3000 ...
start "BioDynamics-Frontend" cmd /k "cd /d frontend && npm run dev"

echo.
echo [BioDynamics Agent] Waiting for services to be ready...
ping -n 9 127.0.0.1 >nul

echo [BioDynamics Agent] Opening browser (Edge preferred)...
set "EDGE_EXE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if exist "!EDGE_EXE!" (
    start "" "!EDGE_EXE!" "http://localhost:3000"
) else (
    start http://localhost:3000
)

echo [BioDynamics Agent] Startup complete. Keep backend/frontend windows running.
pause
