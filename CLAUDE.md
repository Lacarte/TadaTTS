## Kokoro TTS Studio -- Project Brief

### Goal
A complete web application for Kokoro TTS text-to-speech generation with a Flask backend, dark-themed single-page frontend, real-time model download progress, multi-step audio processing pipeline, history management, and one-click startup.

---

### Project Structure
```
KokoroTTS-Studio/
+-- main.py                 # Original CLI script
+-- backend.py              # Flask API server (~2,100 lines)
+-- requirements.txt        # Python dependencies
+-- runner.bat              # Windows one-click launcher (health-check polling)
+-- setup.bat               # Environment setup (auto-downloads Python 3.12 if needed)
+-- PLAN.md                 # Architecture notes and session log
+-- models/                 # Kokoro model files (auto-downloaded, gitignored)
|   +-- kokoro-v1.0.onnx    # TTS model (~326MB)
|   +-- voices-v1.0.bin     # Voice embeddings (~47MB)
+-- bin/                    # Local ffmpeg (optional, gitignored)
+-- logs/                   # Loguru rotating logs (gitignored)
+-- generated_assets/       # All generated output (gitignored)
|   +-- tts/                # Per-generation subfolders
|   |   +-- <basename>/     # One subfolder per generation
|   |   |   +-- <basename>.wav          # Original (24kHz)
|   |   |   +-- <basename>.json         # Metadata
|   |   |   +-- <basename>_enhanced.wav # Enhanced (48kHz, LavaSR)
|   |   |   +-- <basename>_cleaned.wav  # Silence removed + loudnorm
|   |   |   +-- <basename>_cleaned.mp3  # MP3 of cleaned version
|   |   +-- TRASH/          # Soft-deleted TTS files
|   +-- force-alignment/    # Standalone alignment results
|       +-- TRASH/          # Soft-deleted alignment files
+-- frontend/
    +-- index.html          # Single-file UI (~2,800 lines, inline CSS/JS, Tailwind CDN)
```

---

### 1. BACKEND (`backend.py`)

**Framework:** Flask + Flask-CORS + Loguru logging

**Core Logic:** Based on `main.py`:
```python
from kokoro_onnx import Kokoro
import soundfile as sf
import time

kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

start = time.perf_counter()
audio, sr = kokoro.create(prompt, voice="af_bella", speed=1.0, lang="en-us")
end = time.perf_counter()

duration_generated = len(audio) / sr
inference_time = end - start
rtf = inference_time / duration_generated
```

**Text Processing Pipeline:**
- `clean_for_tts()` -- strips markdown, replaces URLs, collapses whitespace
- `normalize_for_tts()` -- expands contractions, abbreviations, currency, units, dates, ordinals, numbers via num2words
- `tts_breathing_blocks()` -- splits long texts into 150-200 char blocks with short-fragment merging, wraps in `[...]` brackets

**Generation:**
- Single-block fast path for short texts (synchronous)
- Chunked async generation for long texts via background thread + job queue
- SSE progress streaming at `/api/generate-progress/<job_id>`
- Abort support at `/api/generate-abort/<job_id>`
- Audio padding (`pad_audio`) and crossfade concatenation (`concatenate_chunks`)

**Post-Processing (all lazy-loaded, gracefully skipped if unavailable):**
- Force alignment via stable-ts (Whisper tiny.en) -- word-level timestamps
- Audio enhancement via LavaSR -- 24kHz to 48kHz upscaling
- Silence removal via Silero VAD (torch.hub) -- with configurable threshold
- Loudness normalization via ffmpeg loudnorm
- MP3 conversion via ffmpeg with SSE progress

Each post-processor uses its own background thread with per-basename metadata locks for atomic JSON read-modify-write. Enhancement chains into VAD when complete.

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve `frontend/index.html` |
| `/api/health` | GET | Status + feature flags (ffmpeg, alignment, enhance, VAD) |
| `/api/models` | GET | List models with specs |
| `/api/voices` | GET | All available voices |
| `/api/normalize` | POST | Text normalization + breathing-block formatting |
| `/api/model-status/<id>` | GET | Check if model files are present |
| `/api/download-model/<id>` | GET (SSE) | Stream download progress in real-time |
| `/api/generate` | POST | `{model, voice, prompt, speed, max_silence_ms}` |
| `/api/generate-progress/<job_id>` | GET (SSE) | Stream chunked generation progress |
| `/api/generate-abort/<job_id>` | POST | Abort running generation |
| `/api/generation` | GET | List all generated files with metadata |
| `/api/generation` | DELETE | Delete all files (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single file (move to TRASH) |
| `/api/generation/<file>/alignment` | GET | Word alignment data (triggers retroactively) |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status (triggers retroactively) |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-check` | GET | Check if MP3 exists |
| `/api/generation/<file>/mp3` | GET | Serve cached MP3 |
| `/api/generation/<file>/mp3-convert` | GET (SSE) | Convert WAV to MP3 with progress |
| `/api/generation/alignments` | GET | List alignment data (TTS + standalone) |
| `/api/generation/force-alignment` | GET | List standalone force-alignment results |
| `/api/generation/alignment/<folder>` | DELETE | Soft-delete alignment folder |
| `/api/force-align` | POST | Standalone force alignment (audio + text upload) |
| `/api/open-generation-folder` | POST | Open OS file explorer at job folder |
| `/generation/<file>` | GET | Serve audio file |
| `/generation/force-alignment/<file>` | GET | Serve alignment audio |

**SSE Download Progress Format:**
```
data: {"phase": "checking", "model": "kokoro"}
data: {"phase": "downloading", "file": "kokoro-v1.0.onnx", "progress": 78, "total_mb": 326.0, "downloaded_mb": 254.3, "speed": "87.6MB/s"}
data: {"phase": "downloading", "file": "voices-v1.0.bin", "progress": 100, "total_mb": 47.0, "downloaded_mb": 47.0, "speed": "92.1MB/s"}
data: {"phase": "ready", "message": "Model ready"}
```

**File Naming Convention:**
```python
import re
from datetime import datetime

def generate_filename(prompt):
    excerpt = re.sub(r'[^a-zA-Z0-9]+', '-', prompt[:30].lower()).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{excerpt}_{timestamp}"
```

**Metadata JSON Structure:**
```json
{
  "filename": "reports-of-my-death_20250220_143052.wav",
  "prompt": "The reports of my death are greatly exaggerated...",
  "model": "kokoro-v1.0",
  "model_id": "kokoro",
  "voice": "af_bella",
  "timestamp": "2025-02-20T14:30:52",
  "inference_time": 1.234,
  "rtf": 0.15,
  "duration_seconds": 8.2,
  "sample_rate": 24000,
  "word_alignment": [{"word": "The", "start": 0.0, "end": 0.15}, ...],
  "enhanced": true,
  "cleaned": true
}
```

**Port Configuration:** Auto-detect starting from 5000, increment until available.

**Logging:** Loguru with daily rotation, 7-day retention, gzip compression, clean console output with level icons.

---

### 2. FRONTEND (`frontend/index.html`)

**Architecture:** Single file (~2,800 lines), inline CSS/JS, served by Flask

**Tech Stack:**
- Tailwind CSS v3 via CDN (custom dark palette, NO default blue/indigo)
- Vanilla JavaScript (ES6+)
- Native Web Audio API for playback
- EventSource for SSE download/generation progress
- Fonts: Space Grotesk (display), DM Sans (body), JetBrains Mono (code/numbers)

**Design System (Always-Dark):**
```css
/* Navy surface hierarchy */
--bg-base: #0a0e13;
--bg-surface: #0f1520;
--bg-card: #161d2a;

/* Accent colors */
--brand-primary: #FF6B6B;      /* Coral red */
--brand-secondary: #4ECDC4;    /* Teal */
--brand-accent: #FFE66D;       /* Yellow */
--brand-purple: #A78BFA;       /* Purple */
```

**Layout:**
- Fixed collapsible sidebar (220px / 64px collapsed, persisted to localStorage)
- Mobile bottom tab bar (< 768px, sidebar hidden)
- Fixed bottom player bar (80px)

**4 Pages (sidebar navigation):**

1. **TTS (Generate)**
   - Single model display (Kokoro v1.0) with size badge
   - 50+ clickable voice chips grouped by language with gender indicator dots (F/M)
   - Prompt textarea with word/token counter
   - Format button (calls `/api/normalize`) + Copy button
   - Ctrl+Enter shortcut hint
   - Generate button (changes color per pipeline step)
   - 4-step processing stepper (Generate -> Enhance -> Clean -> Normalize)

2. **Alignment**
   - Drag-and-drop audio file upload zone
   - Transcript textarea
   - Submit button
   - Karaoke results with word-level highlighting and click-to-seek

3. **Library**
   - Unified history list (TTS + force-alignment results merged by timestamp)
   - Filter tabs: ALL / #TTS / #ALIGNMENT
   - Per-item: play button, text expand, metadata, alignment button, delete
   - Delete-all button with confirmation modal

4. **Settings**
   - Speed selector (0.5x-2.0x, persisted)
   - Max silence duration (200ms-1000ms, persisted)
   - Feature availability status panel (ffmpeg, alignment, enhance, VAD)
   - About section

**Player (fixed footer):**
- Seek bar, play/pause, time display
- Karaoke text with real-time word highlighting, auto-scroll, click-to-seek
- 3-version selector: Original / Enhanced / Cleaned (preserves seek position)
- MP3 download button (cleaned version)
- Delete button
- EQ animation bars

**Interactive States:**
- Hover: Scale 1.02, shadow increase, color shift
- Focus: Ring-2 ring-brand-primary outline-none
- Active: Scale 0.98, darker background
- Disabled: Opacity 50%, cursor not-allowed
- Loading: Skeleton shimmer or spinner

**Persistence (localStorage):**
- Selected model, voice, speed, silence threshold
- Sidebar collapsed state
- All restored on page load

---

### 3. SETUP & RUNNER SCRIPTS

**`requirements.txt`:**
```
kokoro-onnx
flask
flask-cors
loguru
openai-whisper
stable-ts
git+https://github.com/ysharma3501/LavaSR.git
num2words
```

**`setup.bat`:**
- Tries system-wide `py -3.12` launcher first
- Falls back to downloading Python 3.12.10 installer from python.org into local `python312/` subfolder
- Skips Python setup entirely if venv already exists
- Installs pip dependencies from requirements.txt

**`runner.bat`:**
- Finds available port starting from 5000 (netstat check)
- Starts backend with `cmd /k` (keeps window open on crash)
- Health-check polling via `curl` to `/api/health` (up to 30 retries)
- Opens browser only after server confirms healthy
- Cleanup: taskkill on exit

---

### 4. MODEL CONFIGURATION

**Model:** Kokoro v1.0 (82M params, ~373MB total)
- Model file: `kokoro-v1.0.onnx` (~326MB) — downloaded from GitHub Releases
- Voices file: `voices-v1.0.bin` (~47MB) — downloaded from GitHub Releases
- Source: [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- Sample rate: 24,000 Hz

**Voices (50+ multilingual):**

| Prefix | Language | Gender | Voices |
|--------|----------|--------|--------|
| `af_` | American English | Female | alloy, aoede, bella, heart, jessica, kore, nicole, nova, river, sarah, sky |
| `am_` | American English | Male | adam, echo, eric, fenrir, liam, michael, onyx, puck |
| `bf_` | British English | Female | alice, emma, isabella, lily |
| `bm_` | British English | Male | daniel, fable, george, lewis |
| `jf_` | Japanese | Female | alpha, gongitsune, nezumi, tebukuro |
| `jm_` | Japanese | Male | kumo |
| `zf_` | Chinese | Female | xiaobei, xiaoni, xiaoxuan, xiaoyi |
| `zm_` | Chinese | Male | yunjian, yunxi, yunxia, yunyang |
| `ef_` | Spanish | Female | dora |
| `em_` | Spanish | Male | alex, santa |
| `ff_` | French | Female | siwis |
| `hf_` | Hindi | Female | alpha, beta |
| `hm_` | Hindi | Male | omega, psi |
| `if_` | Italian | Female | sara |
| `im_` | Italian | Male | nicola |
| `pf_` | Portuguese | Female | dora |
| `pm_` | Portuguese | Male | alex, santa |

> Language is auto-derived from voice prefix via `_voice_to_lang()`. The `lang` parameter is passed to `kokoro.create()` automatically.

---

### 5. IMPLEMENTED FEATURES

- [x] `main.py` works as standalone CLI tool
- [x] `backend.py` serves API and frontend on auto-detected port
- [x] Frontend is single `index.html` with inline Tailwind CSS (custom dark colors, no blue/indigo)
- [x] Mobile-first responsive (320px to 1440px+) with sidebar + bottom tab bar
- [x] Model download shows real-time progress via SSE (from GitHub Releases)
- [x] Generation shows 4-step processing stepper with token count estimation
- [x] Audio auto-plays on completion with native player
- [x] Files saved to `generated_assets/tts/<basename>/` with metadata JSON
- [x] Unified library lists TTS + alignment history with filter tabs
- [x] `setup.bat` creates venv and installs dependencies (auto-downloads Python 3.12)
- [x] `runner.bat` auto-detects port, launches backend with health polling, opens browser
- [x] All interactive elements have hover/focus/active states
- [x] Layered shadows and intentional spacing throughout
- [x] Dark theme (always-dark)
- [x] Keyboard shortcut: Ctrl+Enter to generate
- [x] Drag-and-drop (audio files on alignment, text on prompt)
- [x] Karaoke word highlighting with click-to-seek
- [x] Audio enhancement (LavaSR 24kHz -> 48kHz)
- [x] Silence removal (Silero VAD) with configurable threshold
- [x] Loudness normalization (ffmpeg loudnorm)
- [x] MP3 conversion with live progress
- [x] 3-version player (Original/Enhanced/Cleaned)
- [x] Standalone force alignment page
- [x] Breathing-block text chunking for long texts
- [x] Async generation with abort support
- [x] Per-generation subfolders
- [x] Soft delete with TRASH folder and confirmation modal
- [x] Text normalization (numbers, currency, abbreviations, dates)
- [x] Speed control (0.5x-2.0x, persisted)
- [x] Collapsible sidebar navigation (persisted)
- [x] Retroactive processing (trigger alignment/enhancement/VAD on old files)
- [x] Open generation folder in OS file explorer
- [x] Copy prompt to clipboard
- [x] 50+ multilingual voices grouped by language in UI

---

### 6. UPCOMING / TODO

- [ ] Prosody (Expression) in Audio via Parselmouth -- pitch shift, pitch range, rate adjustment sliders
- [ ] Audio waveform visualization during playback
- [ ] Batch generation queue for automation pipelines
- [ ] Favicon
