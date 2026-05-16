@echo off
REM Plant Timelapse System - Windows Installer/Launcher
REM This script handles installation with admin privileges and runs the application
cd /d "%~dp0"

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    REM Re-launch as admin
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%~dp0\" && \"%~f0\"' -Verb RunAs"
    exit /b
)

REM We're now running as admin
echo.
echo Plant Timelapse System
echo ======================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if ADB is installed
adb version >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: ADB not found
    echo Please install Android Debug Bridge and add it to PATH
    echo Download: https://developer.android.com/studio/command-line/adb
    pause
)

REM Menu
echo.
echo Options:
echo 1. Install package (pip install -e .)
echo 2. Install dependencies only (pip install -r requirements.txt)
echo 3. Run application (python main.py)
echo 4. Run as module (python -m scripts)
echo 5. Exit
echo.

:menu
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Installing plant-timelapse package...
    pip install -e .
    echo Installation complete!
    goto menu
)
if "%choice%"=="2" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Dependencies installed!
    goto menu
)
if "%choice%"=="3" (
    echo Starting Plant Timelapse System...
    echo Navigate to http://localhost:5000
    python main.py
    goto menu
)
if "%choice%"=="4" (
    echo Starting Plant Timelapse System (module mode)...
    echo Navigate to http://localhost:5000
    python -m scripts
    goto menu
)
if "%choice%"=="5" (
    exit /b 0
)

echo Invalid choice. Please try again.
goto menu
