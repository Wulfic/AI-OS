@echo off
REM AI-OS Launcher for Windows

setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

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
