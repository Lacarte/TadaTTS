# TADA TTS Studio

A web studio for [TADA TTS](https://github.com/HumeAI/tada) -- high-quality open-source text-to-speech with voice cloning. Clone any voice from a reference audio and generate natural speech in your browser.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![TADA TTS](https://img.shields.io/badge/TADA-1B%20%7C%203B-coral)
![Flask](https://img.shields.io/badge/flask-backend-teal)

## Features

- **Voice cloning** -- upload a reference audio (5-30s) to clone any voice
- **Two models** -- TADA-1B (English, ~4GB) and TADA-3B (10 languages, ~9GB)
- **CPU/GPU support** -- toggle between CPU and CUDA in the UI
- **Browser-based UI** -- dark-themed single-page app served by Flask, no build step
- **Real-time download progress** -- SSE streams progress when fetching models from HuggingFace
- **Breathing-block chunking** -- long texts split into 150-200 char blocks for natural pacing
- **Async generation with abort** -- chunked generation runs in background threads; cancel anytime
- **Audio enhancement** -- LavaSR upscaling from 24kHz to 48kHz
- **Silence removal** -- Silero VAD trims dead air with configurable threshold (0.2s-1.0s)
- **Loudness normalization** -- ffmpeg loudnorm for consistent volume
- **Karaoke word highlighting** -- stable-ts forced alignment with real-time word tracking
- **Text normalization** -- expands numbers, currency, abbreviations, dates, symbols
- **Speed control** -- adjustable from 0.5x to 2.0x (persisted)
- **MP3 export** -- WAV to MP3 conversion with live progress
- **3-version player** -- Original, Enhanced, and Cleaned audio
- **Standalone force alignment** -- upload any audio + transcript for word-level timestamps
- **Unified library** -- TTS and alignment history with filter tabs
- **Soft delete** -- files moved to TRASH folder
- **Dark theme** -- always-dark UI with navy/teal/coral palette

## Quick Start

```bash
# 1. Setup (creates venv, installs dependencies)
setup.bat

# 2. Run (starts server, opens browser)
runner.bat
```

Or manually:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python backend.py
```

## How It Works

### Voice Cloning
1. Upload a reference audio file (WAV, MP3, FLAC, OGG) -- 5-30 seconds recommended
2. Optionally provide the transcript of the reference audio
3. The TADA encoder processes the reference to create a voice prompt
4. Use the voice profile for all future generations

### Generation Pipeline

```
Step 1: Generate audio (TADA) -> WAV + JSON metadata
Step 2: Enhance (LavaSR) -> 48kHz upscaled WAV
Step 3: Clean (Silero VAD) -> silence removed
Step 4: Normalize (ffmpeg loudnorm) -> consistent volume
Step 5: Convert (ffmpeg) -> MP3
```

Each step is optional and runs in the background. Steps 2-5 chain automatically.

## Project Structure

```
TadaTTS-Studio/
+-- backend.py          # Flask API server
+-- frontend/
|   +-- index.html      # Single-file UI (inline CSS/JS, Tailwind CDN)
+-- voices/             # Voice profiles (reference audio + metadata)
+-- generated_assets/   # All generated output
+-- models/             # (gitignored) cached model files
+-- logs/               # (gitignored) rotating log files
+-- bin/                # (optional) local ffmpeg binary
```

## Models

**TADA-1B** -- ~2B parameters, English only, ~4GB
- HuggingFace: `HumeAI/tada-1b`

**TADA-3B Multilingual** -- ~4B parameters, 10 languages, ~9GB
- HuggingFace: `HumeAI/tada-3b-ml`
- Languages: English, Arabic, Chinese, German, Spanish, French, Italian, Japanese, Polish, Portuguese

Models auto-download from HuggingFace on first use.

## Dependencies

```
hume-tada (pulls torch, torchaudio, transformers)
flask, flask-cors
loguru
openai-whisper, stable-ts (word alignment)
LavaSR (audio enhancement)
num2words (text normalization)
```

## Credits

Built on [TADA TTS](https://github.com/HumeAI/tada) by Hume AI.
