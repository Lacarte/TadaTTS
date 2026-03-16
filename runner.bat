@echo off
:: Change to the directory where this script lives
cd /d "%~dp0"

echo Starting TADA TTS Studio...

call venv\Scripts\activate.bat

:: Find available port starting from 5000
set PORT=5000
:check_port
netstat -an | find ":%PORT%" >nul
if %ERRORLEVEL% equ 0 (
    set /a PORT+=1
    goto check_port
)

echo Found available port: %PORT%

:: Start backend — use cmd /k so the window stays open if it crashes
start "TADA TTS Backend" cmd /k "venv\Scripts\python.exe backend.py --port %PORT%"

:: Wait for server to respond (up to 60 seconds — TADA models take longer to load)
echo Waiting for server to start...
set TRIES=0
:wait_loop
if %TRIES% geq 60 (
    echo ERROR: Server did not start within 60 seconds.
    echo Check the TADA TTS Backend window for errors.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul
curl -s http://localhost:%PORT%/api/health >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set /a TRIES+=1
    goto wait_loop
)

echo Server is ready!

:: Open browser
start http://localhost:%PORT%

echo TADA TTS Studio is running at http://localhost:%PORT%
echo Press any key to stop...
pause

:: Cleanup
taskkill /FI "WINDOWTITLE eq TADA TTS Backend*" /F >nul 2>&1
