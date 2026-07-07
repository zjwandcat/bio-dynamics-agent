﻿﻿@echo off
setlocal enabledelayedexpansion

chcp 65001 >nul

echo [BioDynamics Agent] One-click dev startup
echo.

cd /d "%~dp0\.."

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

REM Step 1: detect model connectivity
echo [BioDynamics Agent] Step 1/3: Detecting model connectivity...
backend\venv\Scripts\python.exe backend\scripts\test_model_connectivity.py
if %%errorlevel%% neq 0 (
    echo [WARN] Some models failed connectivity check, continuing startup.
)
echo.

REM Step 2: backend will clear LangGraph MemorySaver during lifespan
echo [BioDynamics Agent] Step 2/3: Starting backend (context memory will be cleared automatically)...
echo.
start "BioDynamics-Backend" cmd /k "cd /d backend && .\venv\Scripts\uvicorn.exe app.main:app --reload --port 8000"

echo [BioDynamics Agent] Step 3/3: Starting frontend http://localhost:3000 ...
start "BioDynamics-Frontend" cmd /k "cd /d frontend && npm run dev"

echo.
echo [BioDynamics Agent] Waiting for services to be ready...
ping -n 9 127.0.0.1 >nul

echo [BioDynamics Agent] Opening browser...
start http://localhost:3000

echo [BioDynamics Agent] Startup complete. Keep backend/frontend windows running.
pause
