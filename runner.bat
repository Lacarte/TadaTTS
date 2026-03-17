@echo off
:: Change to the directory where this script lives
cd /d "%~dp0"

echo Starting TADA TTS Studio...

call venv\Scripts\activate.bat

:: Kill any leftover zombie servers on ports 5000-5009
for /L %%p in (5000,1,5009) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%%p "') do (
        taskkill /PID %%a /F >nul 2>&1
    )
)

:: Find available port starting from 5000
set PORT=5000
:check_port
netstat -an | find ":%PORT%" >nul
if %ERRORLEVEL% equ 0 (
    set /a PORT+=1
    goto check_port
)

echo Found available port: %PORT%
echo.
echo   TADA TTS Studio
echo   http://localhost:%PORT%
echo.
echo   Close this window to stop the server.
echo.

:: Open browser after a short delay (in background)
start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:%PORT%"

:: Run server — restart automatically on crash
:run_server
venv\Scripts\python.exe backend.py --port %PORT%
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% neq 0 (
    echo.
    echo ========================================
    echo   Server crashed (exit code %EXIT_CODE%)
    echo   Restarting in 3 seconds...
    echo   Press Ctrl+C to stop.
    echo ========================================
    echo.
    timeout /t 3 /nobreak >nul
    goto run_server
)

echo.
echo Server stopped.
pause
