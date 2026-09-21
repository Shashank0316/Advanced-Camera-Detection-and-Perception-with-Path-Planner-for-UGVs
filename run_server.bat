@echo off
title Vision-Based Autonomous Navigation for UGV - Local Server
echo ======================================================================
echo  VISION-BASED AUTONOMOUS NAVIGATION (UGV) - LOCAL SIMULATION SERVER
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Detecting Python environment...
set PYTHON_CMD=
python --version >nul 2>&1 && set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" (
    py --version >nul 2>&1 && set PYTHON_CMD=py
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python is not recognized in your PATH!
    echo.
    echo Quick fixes:
    echo 1. Try running: py -m pip install -r requirements.txt
    echo 2. Or reinstall Python from python.org and CHECK the box:
    echo    "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo Found Python launcher: %PYTHON_CMD%
%PYTHON_CMD% --version

echo [2/3] Checking dependencies using %PYTHON_CMD% -m pip...
%PYTHON_CMD% -m pip show fastapi opencv-python numpy matplotlib websockets >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [NOTICE] Installing missing dependencies from requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies via %PYTHON_CMD% -m pip.
        pause
        exit /b 1
    )
)

echo [3/3] Starting Local Server and Launching Dashboard...
echo Open your browser at: http://127.0.0.1:8000
%PYTHON_CMD% run_server.py

pause
