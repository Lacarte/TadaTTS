# KittenTTS Studio -- Project Status & Architecture

**Repository:** https://github.com/Lacarte/KittenTTS-Studio
**Branch:** main

## Current State (2026-02-22)

### Latest Commit
- `2a4319a` -- `feat: UI/UX audit improvements and force-alignment karaoke`

---

## Architecture Overview

### Processing Pipeline (per generation)
```
User clicks Generate
  |
  +-- Step 1: Generate audio (KittenTTS) -> saves WAV + JSON
  |     +-- Breathing-block chunking for long texts (150-200 char blocks)
  |     +-- Crossfade concatenation of audio chunks
  |     +-- Start alignment thread (stable-ts, parallel)
  |     +-- Start enhancement thread (LavaSR)
  |
  +-- Step 2: Enhancement (LavaSR) -> saves _enhanced.wav (48kHz)
  |     +-- Chains into VAD when done
  |
  +-- Step 3: Silence Removal (Silero VAD) -> saves _cleaned.wav
  |     +-- Uses enhanced audio if available, preserves gaps <= max_silence_ms
  |
  +-- Step 4: Loudnorm (ffmpeg) -> overwrites _cleaned.wav in-place
  |     +-- Runs inside VAD background thread after silence removal
  |
  +-- Step 5: MP3 Conversion (ffmpeg, frontend-driven) -> saves _cleaned.mp3
```

### Audio File Variants (per-generation subfolder)
```
generated_assets/
  tts/
    some-text_20260221_143052/
      some-text_20260221_143052.wav          # Original (24kHz)
      some-text_20260221_143052.json         # Metadata
      some-text_20260221_143052_enhanced.wav # Enhanced (48kHz, LavaSR)
      some-text_20260221_143052_cleaned.wav  # Silence removed + loudnorm
      some-text_20260221_143052_cleaned.mp3  # MP3 of cleaned version
    TRASH/                                   # Soft-deleted TTS files
  force-alignment/
    audio-name_20260222_alignment.json       # Standalone alignment results
    TRASH/                                   # Soft-deleted alignment files
```

### Player Version Selector
3-button group in player footer: **Original** / **Enhanced** / **Cleaned**
- Preserves playback position when switching
- Polls backend for readiness, enables buttons as versions become available

---

## Commit History

| Commit | Feature |
|--------|---------|
| `1794108` | Initial web app: Flask backend, single-file frontend, model download SSE, generation, history |
| `d88891a` | Project renamed to KittenTTS-Studio |
| `61b1b98` | Redesign: sidebar navigation + always-dark theme |
| `ef59b7c` | Abort button + concurrent generation OOM prevention |
| `3408a03` | Atomic metadata management + alignment pipeline step |
| `2f72fd0` | Audio padding, open-folder button, alignment progress UX |
| `941dcbc` | Loguru logging, infinite polling fix, loudnorm channel error fix |
| `4a9e51f` | History sorting by time, copy button, polling fixes |
| `d4d3acc` | Breathing-block TTS chunking, short-block merge, UI refinements |
| `c4d3e1f` | Force alignment page, library tabs, output folder restructuring |
| `4b2b799` | Per-generation subfolders, unified history with type tags and filters |
| `2a4319a` | UI/UX audit improvements, force-alignment karaoke |

---

## Backend Components (`backend.py`, ~2,100 lines)

### Globals & Models
- `alignment_model`, `alignment_lock` -- stable-ts/whisper (lazy-loaded)
- `enhance_model`, `enhance_lock` -- LavaSR (lazy-loaded)
- `vad_model`, `vad_utils`, `vad_lock` -- Silero VAD (lazy-loaded via torch.hub)
- Each has `*_tasks` dict + `*_tasks_lock` for tracking background threads
- Each has `_check_*_available()` with cached result
- `_get_metadata_lock(basename)` -- per-generation lock for atomic JSON read-modify-write

### Text Processing
- `clean_for_tts()` -- strips markdown `*_#\`~`, replaces URLs with "link", collapses whitespace
- `normalize_for_tts()` -- full pipeline: symbols, contractions, abbreviations, currency, units, dates, time, ordinals, numbers (user-triggered via Format button)
- `tts_breathing_blocks()` -- splits long texts into 150-200 char blocks with short-fragment merging

### Generation
- Chunked async generation via background thread + job queue
- SSE progress streaming at `/api/generate-progress/<job_id>`
- Abort support at `/api/generate-abort/<job_id>`
- Single-block fast path for short texts (synchronous response)
- Audio padding (`pad_audio`) and crossfade concatenation (`concatenate_chunks`)

### Logging
- Loguru with rotating daily file logs (7-day retention, gzip compressed)
- Clean console output with level icons

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend |
| `/api/health` | GET | Status + feature flags (ffmpeg, alignment, enhance, VAD) |
| `/api/models` | GET | List 5 models with specs |
| `/api/voices` | GET | 8 voices |
| `/api/normalize` | POST | Text normalization + breathing-block formatting |
| `/api/model-status/<id>` | GET | Check if model is cached |
| `/api/download-model/<id>` | GET (SSE) | Stream download progress |
| `/api/generate` | POST | Generate audio (accepts model, voice, prompt, speed, max_silence_ms) |
| `/api/generate-progress/<job_id>` | GET (SSE) | Stream chunked generation progress |
| `/api/generate-abort/<job_id>` | POST | Abort running generation |
| `/api/generation` | GET | List history |
| `/api/generation` | DELETE | Delete all (move to TRASH) |
| `/api/generation/<file>` | DELETE | Delete single (move to TRASH) |
| `/generation/<file>` | GET | Serve audio file |
| `/api/generation/<file>/alignment` | GET | Word alignment data (retroactive) |
| `/api/generation/<file>/enhance-status` | GET | Enhancement status (retroactive) |
| `/api/generation/<file>/vad-status` | GET | Silence removal + loudnorm status |
| `/api/generation/<file>/mp3-check` | GET | Check if MP3 exists |
| `/api/generation/<file>/mp3` | GET | Serve cached MP3 |
| `/api/generation/<file>/mp3-convert` | GET (SSE) | Convert WAV to MP3 with progress |
| `/api/generation/alignments` | GET | List alignment data (TTS + standalone) |
| `/api/generation/force-alignment` | GET | List standalone force-alignment results |
| `/api/generation/alignment/<folder>` | DELETE | Soft-delete alignment folder |
| `/api/force-align` | POST | Standalone force alignment (audio + text upload) |
| `/api/open-generation-folder` | POST | Open OS file explorer at job folder |
| `/generation/force-alignment/<file>` | GET | Serve alignment audio |

---

## Frontend Components (`frontend/index.html`, ~2,700 lines)

### STATE Object
```javascript
{
  selectedModel, selectedVoice, history, nowPlaying,
  isGenerating, darkMode: true, downloadEventSource, eqInterval,
  ffmpeg, alignment, alignmentAvailable, alignmentPollTimer, activeWordIndex,
  enhanceAvailable, enhancePollTimer,
  vadAvailable, vadPollTimer,
  activeVersion,      // 'original' | 'enhanced' | 'cleaned'
  processingStep,     // 0=idle, 1-5=pipeline steps
  histFilter,         // 'all' | 'tts' | 'align'
  sidebarCollapsed,   // persisted to localStorage
}
```

### Design System
- Always-dark theme: navy surfaces (#0a0e13 / #0f1520 / #161d2a)
- Teal accent (#4ECDC4), coral (#FF6B6B), gold (#FFE66D), purple (#A78BFA)
- Fonts: Space Grotesk (display), DM Sans (body), JetBrains Mono (code/numbers)
- Fixed collapsible sidebar (220px / 64px collapsed)
- Mobile bottom tab bar (< 768px)
- Fixed bottom player bar (80px)

### 4 Pages (sidebar navigation)
1. **TTS** -- model select, 8 voice chips (F/M gender dots), prompt textarea, word/token counter, Format/Copy buttons, Ctrl+Enter hint, Generate button, 4-step processing stepper
2. **Alignment** -- standalone force alignment: drag-and-drop file upload zone, transcript textarea, submit, karaoke results with word highlighting
3. **Library** -- unified history with filter tabs (ALL / #TTS / #ALIGNMENT), play/metadata/delete per item, delete-all button
4. **Settings** -- speed selector (0.5x-2.0x), max silence duration (200ms-1000ms), feature status panel, About

### Key JS Functions
- `handleGenerate()` -- orchestrates multi-step pipeline with sequential polling
- `pollUntilDone(url, done, pending, interval, onProgress)` -- generic status poller
- `switchAudioVersion(ver)` -- swap player src, preserve position
- `pollVersionStatuses(filename)` -- parallel polling for enhance + VAD readiness
- `normalizePrompt()` -- calls `/api/normalize`, replaces prompt text
- `autoConvertMp3(wavFilename)` -- SSE-driven MP3 conversion
- `downloadCleanedMp3()` -- fetch + blob download of cleaned MP3
- `renderKaraokeText()` -- word-level karaoke with requestAnimationFrame, binary search, click-to-seek

### Key Frontend Features
- Karaoke word highlighting with auto-scroll and click-to-seek
- 3-version player (Original/Enhanced/Cleaned) with position preservation
- 4-step processing stepper with color-coded progress
- Retroactive processing triggers (alignment/enhancement/VAD on old files)
- Breathing-block text formatting via Format button
- Drag-and-drop (text files on prompt, audio files on alignment upload)
- EQ animation bars in player
- Toast notifications (teal/coral/dark variants)
- All selections persisted to localStorage (model, voice, speed, silence, sidebar)
- Confirmation modal for bulk delete

---

## Dependencies (`requirements.txt`)
```
kittentts-0.8.0 (wheel from GitHub release)
flask, flask-cors
loguru                          # structured logging
openai-whisper, stable-ts       # alignment
git+LavaSR                      # enhancement
num2words                       # text normalization
```
Implicit: torch, numpy, soundfile, huggingface-hub, onnxruntime, spacy (pulled by kittentts/whisper/LavaSR)

### Optional System Dependencies
- **ffmpeg** -- MP3 conversion, loudnorm. Place in `bin/` or install system-wide.

---

## Upcoming / TODO

- [ ] Prosody (Expression) in Audio via Parselmouth -- pitch shift, pitch range, rate adjustment sliders
- [ ] Long text chunking improvements for better prosody on complex prompts
- [ ] Batch generation queue for automation pipelines
- [ ] Audio waveform visualization during playback
- [ ] Favicon


////////////////// new features

in the future i want to add a stream button that will play the audio as it generates 


Generate vs Stream in Kokoro TTS
Based on the screenshot, there are two modes for generating speech:
ModeCharacter LimitHow It WorksGenerate~500 charsProcesses all text at once, returns complete audio fileStream~5000 charsProcesses in chunks, sends audio as it's generated

Visual Explanation
GENERATE MODE (500 chars max)
─────────────────────────────
Input: "Hello world, this is a test."
         │
         ▼
    ┌─────────┐
    │ Process │  ← Waits until complete
    │   ALL   │
    └─────────┘
         │
         ▼
    [Complete WAV file]
    
    ⏱️ Wait... wait... wait... ✓ Done! Play entire audio


STREAM MODE (5000 chars max)
─────────────────────────────
Input: "Hello world, this is a test. More text here..."
         │
         ▼
    ┌─────────┐
    │ Chunk 1 │ ──► 🔊 Play immediately
    └─────────┘
    ┌─────────┐
    │ Chunk 2 │ ──► 🔊 Play next
    └─────────┘
    ┌─────────┐
    │ Chunk 3 │ ──► 🔊 Continue...
    └─────────┘
    
    ⏱️ Starts playing almost instantly!

When to Use Each
Use CaseModeWhyShort text (<500 chars)GenerateSimpler, single file outputLong text (articles, books)StreamFaster first response, handles more textReal-time applicationsStreamLower latency, progressive playbackDownloading audio fileGenerateComplete file at once

In Code (kokoro-onnx)
pythonfrom kokoro_onnx import Kokoro
import soundfile as sf

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# GENERATE MODE - returns complete audio
samples, sr = kokoro.create(
    text="Short text here",
    voice="af_heart",
    speed=1.0
)
sf.write("output.wav", samples, sr)


# STREAM MODE - yields chunks as generated
for chunk in kokoro.create_stream(
    text="Very long text that could be thousands of characters...",
    voice="af_heart",
    speed=1.0
):
    # Process/play each chunk immediately
    play_audio(chunk)

Why Stream Allows More Characters
FactorGenerateStreamMemoryHolds entire audio in RAMOnly holds current chunkGPU/CPU timeOne long operationMany short operationsUser experienceWait for allHear immediately
Stream breaks the work into pieces, so it can handle longer text without running out of memory or timing out.

TL;DR:

Generate = Wait for complete audio (shorter text)
Stream = Play as it generates (longer text, faster start)