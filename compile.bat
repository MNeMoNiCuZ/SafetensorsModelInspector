@echo off
setlocal enabledelayedexpansion

echo ------------------------------------------------------------------------------
echo Safetensors Model Inspector - Compilation Script (PyInstaller)
echo ------------------------------------------------------------------------------

:: Check for virtual environment
set VENV_NAME=venv
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment '%VENV_NAME%' not found.
    echo Please run 'venv_create.bat' first to set up the environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call "%VENV_NAME%\Scripts\activate.bat"

:: Ensure pyinstaller is installed
echo Verifying PyInstaller installation...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] PyInstaller not found in venv. Installing...
    pip install pyinstaller
)

:: Run PyInstaller
echo Starting compilation...
pyinstaller --noconfirm --onefile --windowed ^
    --name "SafetensorsModelInspector" ^
    --clean ^
    "gui.py"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------------------------
echo SUCCESS: Compilation complete.
echo The executable can be found in the 'dist' folder.
echo ------------------------------------------------------------------------------
pause
