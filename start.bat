@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM Switch to script directory
chdir /d "%~dp0"

REM Python 3.14 executable path (system-level, used by this project)
set "PYTHON_EXE=C:\Users\27553\AppData\Local\Python\pythoncore-3.14-64\python.exe"

REM Check Python 3.14
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 3.14 not found at %PYTHON_EXE%
    echo Please install Python 3.14 and ensure the py launcher is available.
    pause
    exit /b 1
)

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to run Python 3.14 at %PYTHON_EXE%
    pause
    exit /b 1
)

REM Install frontend dependencies if missing
if not exist "frontend\node_modules\.package-lock.json" (
    echo [INFO] Installing frontend dependencies...
    pushd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
    popd
)

REM Start backend (use pushd to avoid quote-nesting issues with %PYTHON_EXE%)
pushd backend
start "BioDynamics Backend" cmd /k "%PYTHON_EXE%" -m uvicorn app.main:app --reload --port 8000
popd

REM Start frontend
pushd frontend
start "BioDynamics Frontend" cmd /k npm run dev
popd

echo [INFO] Waiting for backend (8000) and frontend (3000) to be ready...

REM Poll ports using PowerShell for up to 120 seconds
set "BACKEND_READY=0"
set "FRONTEND_READY=0"
set "MAX_WAIT=120"
set /a "ELAPSED=0"

:WAIT_LOOP
if "%BACKEND_READY%"=="1" if "%FRONTEND_READY%"=="1" goto :OPEN_BROWSER

if %ELAPSED% geq %MAX_WAIT% (
    echo [WARN] Timed out waiting for services. Opening browser anyway.
    goto :OPEN_BROWSER
)

if "%BACKEND_READY%"=="0" (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/models/status' -UseBasicParsing -TimeoutSec 2; exit [int]$r.StatusCode } catch { exit 999 }" >nul 2>&1
    if not errorlevel 1 set "BACKEND_READY=1"
)

if "%FRONTEND_READY%"=="0" (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3000/' -UseBasicParsing -TimeoutSec 2; exit [int]$r.StatusCode } catch { exit 999 }" >nul 2>&1
    if not errorlevel 1 set "FRONTEND_READY=1"
)

if %ELAPSED% lss 1 (
    echo [INFO] Polling services...
) else if %ELAPSED% equ 10 (
    echo [INFO] Still waiting for backend, this may take a while on first run...
) else if %ELAPSED% equ 30 (
    echo [INFO] Backend is still loading dependencies (torch, transformers, etc.)...
)

ping -n 2 127.0.0.1 >nul
set /a "ELAPSED+=1"
goto :WAIT_LOOP

:OPEN_BROWSER
set "EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if exist "%EDGE%" (
    start "" "%EDGE%" "http://localhost:3000"
) else (
    start "" "http://localhost:3000"
)

echo [OK] BioDynamics Agent started. Keep backend/frontend windows running.
pause
