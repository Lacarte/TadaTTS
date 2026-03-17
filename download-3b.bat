@echo off
echo ============================================
echo   TADA 3B Model Download (aria2 - 16x speed)
echo ============================================
echo.

set ARIA2=C:\Users\Admin\Downloads\aria2-1.37.0-win-64bit-build1\aria2c.exe
set DEST=C:\Users\Admin\.cache\huggingface\hub\models--HumeAI--tada-3b-ml\snapshots\1861fbabe8c6f9163d20c8579fce42400719eb2a

echo Downloading model-00001-of-00002.safetensors (~5GB)...
echo Destination: %DEST%
echo.

"%ARIA2%" -x 16 -s 16 -k 1M --console-log-level=notice --summary-interval=5 -d "%DEST%" -o "model-00001-of-00002.safetensors" "https://huggingface.co/HumeAI/tada-3b-ml/resolve/main/model-00001-of-00002.safetensors"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================
    echo   Download complete! TADA 3B is ready.
    echo   Restart the server and select TADA 3B.
    echo ============================================
) else (
    echo.
    echo Download failed or interrupted. Run this script again to resume.
)

pause
