## TADA TTS Studio -- Project Brief

### Goal
A complete web application for TADA TTS text-to-speech generation with voice cloning, Flask backend, dark-themed single-page frontend, real-time model download progress, multi-step audio processing pipeline, history management, and one-click startup.

---

### Project Structure
```
TadaTTS-Studio/
+-- main.py                 # CLI demo script
+-- backend.py              # Flask API server (~2,600 lines)
+-- requirements.txt        # Python dependencies
+-- runner.bat              # Windows one-click launcher (health-check polling)
+-- setup.bat               # Environment setup (auto-downloads Python 3.12 if needed)
+-- PLAN.md                 # Architecture notes and session log
+-- voices/                 # Voice profiles (reference audio + transcripts)
|   +-- <voice-id>/         # One folder per voice profile
|       +-- profile.json    # Voice metadata (name, transcript, audio_file)
|       +-- reference.wav   # Reference audio for voice cloning
|       +-- prompt_cache.pt # Cached encoder output (auto-generated)
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
    +-- index.html          # Single-file UI (~4,500 lines, inline CSS/JS, Tailwind CDN)
```

---

### 1. BACKEND (`backend.py`)

**Framework:** Flask + Flask-CORS + Loguru logging

**Core Logic:** Based on TADA TTS by HumeAI:
```python
import torch
import torchaudio
from tada.modules.tada import TadaForCausalLM
from tada.modules.encoder import Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b").to(device)

# Voice cloning: encode reference audio
audio, sr = torchaudio.load("reference.wav")
prompt = encoder(audio.to(device), text=["transcript"], sample_rate=sr)

# Generate speech
output = model.generate(prompt=prompt, text="Hello world")
generated_audio = output.audio[0].cpu()
torchaudio.save("output.wav", generated_audio.unsqueeze(0), 24000)
```

**Voice Cloning System:**
- Users upload reference audio (5-30s) + optional transcript
- Encoder processes reference → cached prompt tensor (`.pt` file)
- Prompt is reused for all generations with that voice profile
- Voice profiles stored in `voices/` directory

**Two Models:**
- **TADA-1B** (`HumeAI/tada-1b`): English only, ~4GB, faster
- **TADA-3B** (`HumeAI/tada-3b-ml`): 10 languages, ~9GB

**Device Selection:**
- CPU mode (default): Works everywhere, slower
- GPU/CUDA mode: Requires NVIDIA GPU with sufficient VRAM

**Text Processing Pipeline:**
- `clean_for_tts()` -- strips markdown, replaces URLs, collapses whitespace
- `normalize_for_tts()` -- expands contractions, abbreviations, currency, units, dates, ordinals, numbers via num2words
- `tts_breathing_blocks()` -- splits long texts into 150-200 char blocks

**Generation:**
- Single-block fast path for short texts (synchronous)
- Chunked async generation for long texts via background thread + job queue
- SSE progress streaming at `/api/generate-progress/<job_id>`
- Abort support at `/api/generate-abort/<job_id>`

**Post-Processing (all lazy-loaded, gracefully skipped if unavailable):**
- Force alignment via stable-ts (Whisper tiny.en)
- Audio enhancement via LavaSR (24kHz to 48kHz)
- Silence removal via Silero VAD
- Loudness normalization via ffmpeg loudnorm
- MP3 conversion via ffmpeg with SSE progress

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve `frontend/index.html` |
| `/api/health` | GET | Status + feature flags + GPU info + device |
| `/api/device` | GET/POST | Get/set compute device (cpu/cuda) |
| `/api/models` | GET | List models (tada-1b, tada-3b) |
| `/api/voices` | GET | List voice profiles |
| `/api/voices/upload` | POST | Upload reference audio to create voice profile |
| `/api/voices/<id>` | DELETE | Delete voice profile |
| `/api/voices/<id>/audio` | GET | Serve reference audio |
| `/api/normalize` | POST | Text normalization + breathing-block formatting |
| `/api/model-status/<id>` | GET | Check if model is cached in HuggingFace hub |
| `/api/download-model/<id>` | GET (SSE) | Download model from HuggingFace with progress |
| `/api/generate` | POST | `{model, voice, prompt, speed, max_silence_ms}` |
| `/api/generate-progress/<job_id>` | GET (SSE) | Stream chunked generation progress |
| `/api/generate-abort/<job_id>` | POST | Abort running generation |
| `/api/stream` | POST (SSE) | Stream TTS audio chunk-by-chunk (listen mode) |
| `/api/generation` | GET | List all generated files with metadata |
| `/api/generation` | DELETE | Delete all files (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single file (move to TRASH) |
| `/api/generation/<file>/alignment` | GET | Word alignment data |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-check` | GET | Check if MP3 exists |
| `/api/generation/<file>/mp3` | GET | Serve cached MP3 |
| `/api/generation/<file>/mp3-convert` | GET (SSE) | Convert WAV to MP3 with progress |
| `/api/force-align` | POST | Standalone force alignment |
| `/api/open-generation-folder` | POST | Open OS file explorer |

**Metadata JSON Structure:**
```json
{
  "filename": "hello-world_20250220_143052.wav",
  "prompt": "Hello world...",
  "model": "TADA 1B",
  "model_id": "tada-1b",
  "voice": "My Narrator",
  "voice_id": "my-narrator_20250220_142000",
  "device": "cpu",
  "timestamp": "2025-02-20T14:30:52",
  "inference_time": 5.234,
  "rtf": 0.65,
  "duration_seconds": 8.2,
  "sample_rate": 24000
}
```

**Port Configuration:** Auto-detect starting from 5000.

---

### 2. FRONTEND (`frontend/index.html`)

**Architecture:** Single file (~4,500 lines), inline CSS/JS, served by Flask

**Tech Stack:**
- Tailwind CSS v3 via CDN (custom dark palette)
- Vanilla JavaScript (ES6+)
- Native Web Audio API for playback
- EventSource for SSE progress
- Fonts: Space Grotesk, DM Sans, JetBrains Mono

**4 Pages (sidebar navigation):**

1. **TTS (Generate)**
   - Model selector: TADA-1B (English) / TADA-3B (Multilingual)
   - Voice profile management: upload reference audio, select profile
   - Prompt textarea with word/token counter
   - Generate/Listen mode toggle
   - 4-step processing stepper

2. **Alignment** - Drag-and-drop force alignment

3. **Library** - Unified history with filter tabs

4. **Settings**
   - Speed selector (0.5x-2.0x)
   - Max silence duration
   - **Device toggle (CPU/GPU)** with GPU info display
   - Feature availability status
   - About section

**Player (fixed footer):** Seek bar, play/pause, karaoke, 3-version selector, MP3 download

---

### 3. MODEL CONFIGURATION

**TADA-1B** (English only)
- HuggingFace: `HumeAI/tada-1b`
- Size: ~4GB
- Languages: English

**TADA-3B Multilingual**
- HuggingFace: `HumeAI/tada-3b-ml`
- Size: ~9GB
- Languages: English, Arabic, Chinese, German, Spanish, French, Italian, Japanese, Polish, Portuguese

**TADA Codec Encoder** (shared)
- HuggingFace: `HumeAI/tada-codec` (subfolder: encoder)
- Used to encode reference audio for voice cloning

**Output:** 24,000 Hz WAV

---

### 4. REQUIREMENTS

```
hume-tada
flask
flask-cors
loguru
openai-whisper
stable-ts
git+https://github.com/ysharma3501/LavaSR.git
num2words
```

Note: `hume-tada` pulls in `torch`, `torchaudio`, `transformers`, etc.
