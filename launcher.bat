@echo off
REM AI-OS Launcher for Windows

setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM Check for Visual C++ Redistributable (required for PyTorch)
REM Check if vcruntime140.dll exists in System32
if not exist "%SystemRoot%\System32\vcruntime140.dll" (
    echo.
    echo [WARNING] Microsoft Visual C++ Redistributable is not installed.
    echo This is REQUIRED for AI-OS to run properly.
    echo.
    echo Please download and install it from:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo After installing, please restart this application.
    echo.
    pause
    exit /b 1
)

REM Check for vcruntime140_1.dll as well (needed for newer PyTorch versions)
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    echo.
    echo [WARNING] Microsoft Visual C++ Redistributable may be outdated.
    echo AI-OS requires a recent version to run properly.
    echo.
    echo Please download and install the latest version from:
    echo   https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo After installing, please restart this application.
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at: %VENV_DIR%
    echo Please run the installer first.
    pause
    exit /b 1
)

REM Activate virtual environment and run aios
cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" -m aios.cli.aios %*

REM Keep window open if run without arguments
if "%~1"=="" (
    echo.
    echo Press any key to exit...
    pause > nul
)

endlocal
