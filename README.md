# Kokoro TTS Studio

A web studio for [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) -- high-quality open-source text-to-speech with 50+ multilingual voices. Generate realistic speech from text in your browser, no GPU required.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![kokoro-onnx](https://img.shields.io/badge/kokoro--onnx-v1.0-coral)
![Flask](https://img.shields.io/badge/flask-backend-teal)

## Features

- **Browser-based UI** -- dark-themed single-page app served by Flask, no build step
- **50+ multilingual voices** -- American/British English, Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese
- **Real-time download progress** -- SSE streams per-file progress when fetching model from GitHub Releases
- **Breathing-block chunking** -- long texts split into 150-200 char blocks with merge logic for natural pacing
- **Async generation with abort** -- chunked generation runs in background threads; cancel anytime
- **Audio enhancement** -- LavaSR upscaling from 24kHz to 48kHz
- **Silence removal** -- Silero VAD trims dead air with configurable threshold (0.2s-1.0s)
- **Loudness normalization** -- ffmpeg loudnorm for consistent volume
- **Karaoke word highlighting** -- stable-ts forced alignment with real-time word tracking and click-to-seek
- **Text normalization** -- expands numbers, currency, abbreviations, dates, and symbols; auto-formats into breathing blocks
- **Speed control** -- adjustable playback from 0.5x to 2.0x (persisted)
- **MP3 export** -- WAV to MP3 conversion with live SSE progress
- **3-version player** -- switch between Original, Enhanced, and Cleaned audio; preserves seek position
- **Standalone force alignment** -- upload any audio + transcript for word-level timestamps
- **Unified library** -- TTS and alignment history merged with type tags and filter tabs
- **Per-generation subfolders** -- each job saved in its own directory under `generated_assets/tts/`
- **Soft delete** -- files moved to TRASH folder, with delete-all option and confirmation modal
- **Sidebar navigation** -- collapsible sidebar with 4 pages (TTS, Alignment, Library, Settings)
- **Dark theme** -- always-dark UI with navy/teal/coral palette
- **Keyboard shortcut** -- Ctrl+Enter to generate
- **Open folder** -- reveal generation output in OS file explorer
- **One-click startup** -- `runner.bat` finds a free port, starts the server, opens the browser

## Quick Start (Windows)

### 1. Setup

```
setup.bat
```

Creates a Python 3.12 venv and installs dependencies. If Python 3.12 isn't found on your system, the script downloads and installs it locally.

### 2. Run

```
runner.bat
```

Starts the backend on an available port (default 5000), waits for the health check to pass, then opens your browser.

### Manual Start

```bash
# activate venv
venv\Scripts\activate

# run the server
python backend.py --port 5000
```

Then open `http://localhost:5000`.

## Requirements

- **Python 3.12**
- **ffmpeg** (optional, for MP3 conversion and loudnorm) -- place `ffmpeg.exe` in `bin/` or install system-wide

Python dependencies:

```
kokoro-onnx             (pulls onnxruntime, numpy, soundfile, etc.)
flask, flask-cors
loguru                  (structured logging with rotation)
openai-whisper, stable-ts   (forced alignment / karaoke)
LavaSR                      (audio enhancement)
num2words                   (text normalization)
```

## Processing Pipeline

```
User clicks Generate
  |
  +-- Step 1: Generate audio (Kokoro) -> WAV + JSON metadata
  |     +-- Start alignment thread (stable-ts, parallel)
  |     +-- Start enhancement thread (LavaSR)
  |
  +-- Step 2: Enhancement (LavaSR) -> 48kHz upscaled WAV
  |     +-- Chains into VAD when done
  |
  +-- Step 3: Silence Removal (Silero VAD) -> cleaned WAV
  |     +-- Uses enhanced audio if available, preserves gaps <= max_silence_ms
  |
  +-- Step 4: Loudnorm (ffmpeg) -> overwrites cleaned WAV in-place
  |     +-- Runs inside VAD background thread after silence removal
  |
  +-- Step 5: MP3 Conversion (ffmpeg, frontend-driven) -> final MP3
```

For long texts, the breathing-block chunker splits into 150-200 char blocks before generation, then crossfade-concatenates the resulting audio chunks.

## Project Structure

```
KokoroTTS-Studio/
+-- main.py             # Standalone CLI script
+-- backend.py          # Flask API server (~2,100 lines)
+-- requirements.txt    # Python dependencies
+-- setup.bat           # Environment setup (auto-downloads Python 3.12 if needed)
+-- runner.bat          # One-click launcher with health-check polling
+-- CLAUDE.md           # AI assistant project brief
+-- PLAN.md             # Architecture notes and session log
+-- models/             # Kokoro model files (auto-downloaded, gitignored)
+-- bin/                # Local ffmpeg (optional, gitignored)
+-- logs/               # Loguru rotating logs (gitignored)
+-- generated_assets/   # All generated output (gitignored)
|   +-- tts/            # Per-generation subfolders
|   |   +-- some-text_20260221_143052/
|   |   |   +-- some-text_20260221_143052.wav          # Original (24kHz)
|   |   |   +-- some-text_20260221_143052.json         # Metadata
|   |   |   +-- some-text_20260221_143052_enhanced.wav  # Enhanced (48kHz)
|   |   |   +-- some-text_20260221_143052_cleaned.wav   # Silence removed + loudnorm
|   |   |   +-- some-text_20260221_143052_cleaned.mp3   # MP3 of cleaned
|   |   +-- TRASH/      # Soft-deleted TTS files
|   +-- force-alignment/ # Standalone alignment results
|       +-- TRASH/       # Soft-deleted alignment files
+-- frontend/
    +-- index.html      # Single-file UI (~2,800 lines, inline CSS/JS, Tailwind CDN)
```

## Model

**Kokoro v1.0** -- 82M parameters, ~373MB total
- `kokoro-v1.0.onnx` (~326MB) + `voices-v1.0.bin` (~47MB)
- Auto-downloaded from [GitHub Releases](https://github.com/thewh1teagle/kokoro-onnx/releases) on first use
- Sample rate: 24,000 Hz

## Credits

Built on [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle, based on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
