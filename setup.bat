@echo off
:: Change to the directory where this script lives
cd /d "%~dp0"

echo Setting up Kokoro TTS Studio...

:: If venv already exists, skip Python install entirely
if exist venv\Scripts\python.exe (
    echo Existing venv found, skipping Python setup.
    goto :install_deps
)

:: Try system-wide Python 3.12 first (via py launcher)
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Python 3.12 found via py launcher.
    set "USE_PY_LAUNCHER=1"
    goto :create_venv
)

set "LOCAL_PY=%~dp0python312\python.exe"

:: If local Python 3.12 already installed, skip download
if exist "%LOCAL_PY%" (
    echo Python 3.12 found locally.
    goto :create_venv
)

echo Python 3.12 not found. Downloading installer...
set "INSTALLER=python-3.12.10-amd64.exe"
set "URL=https://www.python.org/ftp/python/3.12.10/%INSTALLER%"

curl -L -o "%INSTALLER%" "%URL%"
if not exist "%INSTALLER%" (
    echo ERROR: Failed to download Python 3.12 installer.
    echo Please download manually from https://www.python.org/downloads/release/python-31210/
    pause
    exit /b 1
)

echo Installing Python 3.12 locally (this may take a minute)...
"%INSTALLER%" /passive TargetDir="%~dp0python312" InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_launcher=0 AssociateFiles=0 Shortcuts=0 Include_test=0 Include_doc=0

:: Clean up installer
del "%INSTALLER%" >nul 2>&1

if not exist "%LOCAL_PY%" (
    echo ERROR: Python 3.12 installation failed.
    pause
    exit /b 1
)

echo Python 3.12 installed locally!

:create_venv
if not exist venv (
    echo Creating virtual environment...
    if defined USE_PY_LAUNCHER (
        py -3.12 -m venv venv
    ) else (
        "%LOCAL_PY%" -m venv venv
    )
)

:install_deps
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete! Run runner.bat to start.
pause
