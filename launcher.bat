@echo off
REM AI-OS Launcher for Windows

setlocal EnableDelayedExpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM Check if we're running in GUI mode (first argument is "gui")
set "IS_GUI_MODE=0"
if /i "%~1"=="gui" set "IS_GUI_MODE=1"

REM Check for Visual C++ Redistributable (required for PyTorch) - even in GUI mode
REM Missing VC++ will cause c10.dll to fail to load
if not exist "%SystemRoot%\System32\vcruntime140.dll" goto :vcpp_missing
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" goto :vcpp_missing
goto :vcpp_ok

:vcpp_missing
if "%IS_GUI_MODE%"=="1" (
    REM In GUI mode, show a popup message
    powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Microsoft Visual C++ Redistributable is required but not installed.`n`nPlease download and install it from:`nhttps://aka.ms/vs/17/release/vc_redist.x64.exe`n`nAfter installing, restart AI-OS.', 'AI-OS - Missing Dependency', 'OK', 'Error')" >nul 2>&1
    exit /b 1
) else (
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

:vcpp_ok
REM For GUI mode, skip the rest of the visual checks
if "%IS_GUI_MODE%"=="1" goto :run_gui

REM Check if virtual environment exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at: %VENV_DIR%
    echo Please run the installer first.
    pause
    exit /b 1
)

REM Run aios with all arguments
cd /d "%SCRIPT_DIR%"
"%PYTHON_EXE%" -m aios.cli.aios %*

REM Keep window open if run without arguments (interactive menu mode)
if "%~1"=="" (
    echo.
    echo Press any key to exit...
    pause > nul
)
goto :end

:run_gui
REM GUI mode - minimal checks, silent operation
if not exist "%PYTHON_EXE%" exit /b 1

cd /d "%SCRIPT_DIR%"
REM Use pythonw.exe for GUI mode if available (no console window)
set "PYTHONW_EXE=%VENV_DIR%\Scripts\pythonw.exe"
if exist "%PYTHONW_EXE%" (
    start "" "%PYTHONW_EXE%" -m aios.cli.aios %*
) else (
    start "" "%PYTHON_EXE%" -m aios.cli.aios %*
)
goto :end

:end
endlocal
