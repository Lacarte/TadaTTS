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

:: Run server in THIS console — closing the window kills it
venv\Scripts\python.exe backend.py --port %PORT%
