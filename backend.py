"""TADA TTS Studio — Flask API Server"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import threading
import uuid
from datetime import datetime
from queue import Queue

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import urllib.request
from loguru import logger

# ---------------------------------------------------------------------------
# Loguru configuration
# ---------------------------------------------------------------------------
logger.remove()  # Remove default stderr handler
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_LEVEL_ICONS = {
    "TRACE": ".",
    "DEBUG": "-",
    "INFO": "*",
    "SUCCESS": "+",
    "WARNING": "!",
    "ERROR": "x",
    "CRITICAL": "X",
}


def _console_format(record):
    icon = _LEVEL_ICONS.get(record["level"].name, "\u25cf")
    msg = (
        f"<dim>{record['time']:HH:mm:ss}</dim> "
        f"<level>{icon}</level> "
        f"<level>{record['message']}</level>\n"
    )
    if record["exception"]:
        msg += "{exception}"
    return msg


# Console: INFO and above, clean minimal format
logger.add(sys.stderr, level="INFO", format=_console_format, colorize=True)

# File: DEBUG and above, rotated daily, kept 7 days
logger.add(os.path.join(LOG_DIR, "tada_{time:YYYY-MM-DD}.log"),
           level="DEBUG", rotation="1 day", retention="7 days", compression="zip",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} - {message}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GENERATION_DIR = os.path.join(os.path.dirname(__file__), "generated_assets")
AUDIO_DIR = os.path.join(GENERATION_DIR, "tts")
TRASH_DIR = os.path.join(AUDIO_DIR, "TRASH")
ALIGN_DIR = os.path.join(GENERATION_DIR, "force-alignment")
ALIGN_TRASH_DIR = os.path.join(ALIGN_DIR, "TRASH")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
SAMPLE_VOICES_DIR = os.path.join(os.path.dirname(__file__), "sample voices")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)
os.makedirs(ALIGN_DIR, exist_ok=True)
os.makedirs(ALIGN_TRASH_DIR, exist_ok=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "tada-1b": {
        "name": "TADA 1B",
        "size": "~4GB",
        "hf_model_id": "HumeAI/tada-1b",
        "languages": ["en"],
        "params": "~2B",
    },
    "tada-3b": {
        "name": "TADA 3B Multilingual",
        "size": "~9GB",
        "hf_model_id": "HumeAI/tada-3b-ml",
        "languages": ["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
        "params": "~4B",
    },
}

# TADA codec encoder (shared across models)
ENCODER_HF_ID = "HumeAI/tada-codec"

# Language code mapping for TADA 3B multilingual
TADA_LANGUAGES = {
    "en": "English", "ar": "Arabic", "zh": "Chinese", "de": "German",
    "es": "Spanish", "fr": "French", "it": "Italian", "ja": "Japanese",
    "pl": "Polish", "pt": "Portuguese",
}

# Voice profiles directory
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Default device selection
_current_device = "cpu"
_current_model_id = None

# Cached TADA instances
tada_encoder = None
tada_model = None
tada_lock = threading.Lock()

# Alignment model (stable-ts / Whisper) — optional feature
alignment_model = None
alignment_lock = threading.Lock()
alignment_available = None  # None = not checked yet, True/False after first check
alignment_tasks = {}        # {basename: threading.Thread}
alignment_tasks_lock = threading.Lock()

# Enhancement model (LavaSR) — optional feature
enhance_model = None
enhance_lock = threading.Lock()
enhance_available = None  # None = not checked yet, True/False after first check
enhance_tasks = {}        # {basename: threading.Thread}
enhance_tasks_lock = threading.Lock()

# Silence removal model (Silero VAD) — optional feature
vad_model = None
vad_utils = None
vad_lock = threading.Lock()
vad_available = None  # None = not checked yet, True/False after first check
vad_tasks = {}        # {basename: threading.Thread}
vad_tasks_lock = threading.Lock()

# Chunked generation jobs: {job_id: {"queue": Queue, "status": str, "metadata": dict, "created": float}}
generation_jobs = {}
generation_jobs_lock = threading.Lock()
generation_inference_lock = threading.Lock()  # Serialize TADA inference (not thread-safe)

# Per-basename locks for metadata JSON read-modify-write (prevents race conditions
# between alignment, enhancement, and VAD threads overwriting each other's fields)
_metadata_locks = {}
_metadata_locks_lock = threading.Lock()


def _get_metadata_lock(basename):
    """Get or create a per-basename lock for metadata JSON access."""
    with _metadata_locks_lock:
        if basename not in _metadata_locks:
            _metadata_locks[basename] = threading.Lock()
        return _metadata_locks[basename]


def _tts_job_dir(basename):
    """Return the per-generation subfolder for a TTS job."""
    return os.path.join(AUDIO_DIR, basename)


def _folder_for_file(filename):
    """Derive the TTS job folder name from any variant filename (original, enhanced, cleaned).
    Strips all known suffixes so e.g. 'foo_enhanced_cleaned.wav' → 'foo'."""
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    changed = True
    while changed:
        changed = False
        for suffix in ("_cleaned", "_enhanced"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
    return base


def _update_metadata(basename, updates):
    """Atomically read-modify-write metadata JSON fields.
    Uses per-basename lock + atomic write (temp file → rename) to prevent
    both race conditions and corruption from interrupted writes."""
    lock = _get_metadata_lock(basename)
    json_path = os.path.join(_tts_job_dir(basename), basename + ".json")
    tmp_path = json_path + ".tmp"
    with lock:
        with open(json_path, "r") as f:
            metadata = json.load(f)
        metadata.update(updates)
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_path, json_path)  # atomic on same filesystem
    return metadata


def _read_metadata(basename):
    """Read metadata JSON safely through the per-basename lock."""
    lock = _get_metadata_lock(basename)
    json_path = os.path.join(_tts_job_dir(basename), basename + ".json")
    with lock:
        with open(json_path, "r") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_filename(prompt: str) -> str:
    excerpt = re.sub(r"[^a-zA-Z0-9]+", "-", prompt[:30].lower()).strip("-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{excerpt}_{timestamp}"


def clean_for_tts(text: str) -> str:
    """Strip markdown formatting, URLs, brackets, and excess whitespace before TTS."""
    text = re.sub(r"[*_#`~]", "", text)       # markdown chars
    text = re.sub(r"https?://\S+", "link", text)  # URLs
    text = re.sub(r"[\[\]]", "", text)         # strip bracket wrappers
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# TTS Text Normalization
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
    "I'd": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "it'd": "it would", "it'll": "it will", "it's": "it is",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
    "that's": "that is", "that'd": "that would", "that'll": "that will",
    "who's": "who is", "who'd": "who would", "who'll": "who will",
    "what's": "what is", "what'd": "what did", "what'll": "what will",
    "where's": "where is", "when's": "when is", "why's": "why is", "how's": "how is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "mightn't": "might not", "needn't": "need not",
    "let's": "let us", "there's": "there is", "here's": "here is", "o'clock": "of the clock",
}

_ORDINALS = {
    r'\b1st\b': 'first', r'\b2nd\b': 'second', r'\b3rd\b': 'third',
    r'\b4th\b': 'fourth', r'\b5th\b': 'fifth', r'\b6th\b': 'sixth',
    r'\b7th\b': 'seventh', r'\b8th\b': 'eighth', r'\b9th\b': 'ninth',
    r'\b10th\b': 'tenth', r'\b11th\b': 'eleventh', r'\b12th\b': 'twelfth',
    r'\b13th\b': 'thirteenth', r'\b14th\b': 'fourteenth', r'\b15th\b': 'fifteenth',
    r'\b20th\b': 'twentieth', r'\b21st\b': 'twenty first', r'\b22nd\b': 'twenty second',
    r'\b23rd\b': 'twenty third', r'\b30th\b': 'thirtieth', r'\b31st\b': 'thirty first',
}

_ABBREVIATIONS = {
    r'\bDr\.\b': 'Doctor', r'\bMr\.\b': 'Mister', r'\bMrs\.\b': 'Missus', r'\bMs\.\b': 'Miss',
    r'\bProf\.\b': 'Professor', r'\bSt\.\b': 'Saint', r'\bAve\.\b': 'Avenue',
    r'\bBlvd\.\b': 'Boulevard', r'\bDept\.\b': 'Department', r'\bEst\.\b': 'Estimated',
    r'\betc\.\b': 'et cetera', r'\be\.g\.\b': 'for example', r'\bi\.e\.\b': 'that is',
    r'\bvs\.\b': 'versus', r'\bapprox\.\b': 'approximately',
    r'\bmin\.\b': 'minutes', r'\bmax\.\b': 'maximum', r'\bno\.\b': 'number',
    r'\bAPI\b': 'A P I', r'\bURL\b': 'U R L', r'\bHTTP\b': 'H T T P',
    r'\bHTML\b': 'H T M L', r'\bCSS\b': 'C S S', r'\bSQL\b': 'S Q L',
    r'\bRBQ\b': 'R B Q', r'\bID\b': 'I D', r'\bPIN\b': 'pin',
    r'\bOTP\b': 'O T P', r'\bSMS\b': 'S M S', r'\bPDF\b': 'P D F',
}

_UNITS = {
    r'(\d+)\s?km\b': r'\1 kilometers', r'(\d+)\s?m\b': r'\1 meters',
    r'(\d+)\s?cm\b': r'\1 centimeters', r'(\d+)\s?mm\b': r'\1 millimeters',
    r'(\d+)\s?kg\b': r'\1 kilograms', r'(\d+)\s?g\b': r'\1 grams',
    r'(\d+)\s?mg\b': r'\1 milligrams', r'(\d+)\s?lb\b': r'\1 pounds',
    r'(\d+)\s?oz\b': r'\1 ounces', r'(\d+)\s?mph\b': r'\1 miles per hour',
    r'(\d+)\s?kph\b': r'\1 kilometers per hour',
    r'(\d+)\s?°C\b': r'\1 degrees Celsius', r'(\d+)\s?°F\b': r'\1 degrees Fahrenheit',
    r'(\d+)\s?%': r'\1 percent',
    r'(\d+)\s?MB\b': r'\1 megabytes', r'(\d+)\s?GB\b': r'\1 gigabytes',
    r'(\d+)\s?TB\b': r'\1 terabytes', r'(\d+)\s?ms\b': r'\1 milliseconds',
    r'(\d+)\s?fps\b': r'\1 frames per second',
}

_SYMBOLS = {
    '&': 'and', '@': 'at', '#': 'number', '+': 'plus', '=': 'equals',
    '>': 'greater than', '<': 'less than', '~': 'approximately',
    '|': '', '\\': '', '/': ' or ',
    '\u2013': '-', '\u2014': ',', '\u2026': '...',
    '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
}

_DATE_MONTHS = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December',
}


def _expand_symbols(text):
    for sym, rep in _SYMBOLS.items():
        text = text.replace(sym, rep)
    return text

def _expand_contractions(text):
    for c, e in _CONTRACTIONS.items():
        text = re.sub(re.escape(c), e, text, flags=re.IGNORECASE)
    return text

def _expand_abbreviations(text):
    for p, r in _ABBREVIATIONS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_currency(text):
    text = re.sub(r'\$(\d+)', r'\1 dollars', text)
    text = re.sub(r'€(\d+)', r'\1 euros', text)
    text = re.sub(r'£(\d+)', r'\1 pounds', text)
    text = re.sub(r'¥(\d+)', r'\1 yen', text)
    text = re.sub(r'HTG\s?(\d+)', r'\1 Haitian gourdes', text)
    return text

def _expand_units(text):
    for p, r in _UNITS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_dates(text):
    def _replace(m):
        y, mo, d = m.group(1), m.group(2), m.group(3)
        return f"{_DATE_MONTHS.get(mo, mo)} {int(d)}, {y}"
    return re.sub(r'\b(\d{4})-(\d{2})-(\d{2})\b', _replace, text)

def _expand_time(text):
    text = re.sub(
        r'\b(\d{1,2}):(\d{2})\s?(am|pm)\b',
        lambda m: f"{m.group(1)} {m.group(2)} {m.group(3).replace('am','a m').replace('pm','p m')}",
        text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(\d{1,2})\s?(am|pm)\b',
        lambda m: f"{m.group(1)} {m.group(2).replace('am','a m').replace('pm','p m')}",
        text, flags=re.IGNORECASE)
    return text

def _expand_ordinals(text):
    for p, r in _ORDINALS.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def _expand_numbers(text):
    try:
        from num2words import num2words
        # Floats first (3.14 -> three point one four)
        def _float_repl(m):
            whole = num2words(int(m.group(1)))
            decimals = " ".join(num2words(int(d)) for d in m.group(2))
            return f"{whole} point {decimals}"
        text = re.sub(r'\b(\d+)\.(\d+)\b', _float_repl, text)
        # Integers (42 -> forty-two)
        text = re.sub(r'\b(\d+)\b', lambda m: num2words(int(m.group(1))), text)
    except ImportError:
        pass  # num2words not installed — skip number expansion
    return text


def normalize_for_tts(text: str) -> str:
    """Full TTS normalization pipeline. Order matters."""
    text = _expand_symbols(text)
    text = _expand_contractions(text)
    text = _expand_abbreviations(text)
    text = _expand_currency(text)
    text = _expand_units(text)
    text = _expand_dates(text)
    text = _expand_time(text)
    text = _expand_ordinals(text)
    text = _expand_numbers(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting & audio concatenation for chunked generation
# ---------------------------------------------------------------------------

def tts_breathing_blocks(
    text: str,
    min_chars: int = 150,
    max_chars: int = 200,
) -> list[str]:
    """Split text into breathing-sized blocks for TTS chunked generation.

    Each block is wrapped in [...] brackets which act as pacing cues for the
    TTS model. Blocks aim for *min_chars*–*max_chars*, preferring sentence
    boundaries, then comma/semicolon boundaries, then word boundaries.
    """
    if not text or not text.strip():
        return []

    # 1) Normalize quotes, dashes, ellipses, and whitespace
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", ". ").replace("\u2013", "-").replace("\u2026", "...")
    text = re.sub(r"\s+", " ", text).strip()

    # 2) Split into sentences (keeps punctuation)
    sentences = re.findall(r".+?(?:\.{3}|[.!?])(?:\s+|$)", text)
    if not sentences:
        sentences = [text]

    # 3) Build blocks aiming for min..max chars, preferring sentence boundaries
    blocks: list[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur.strip():
            blocks.append(cur.strip())
        cur = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if not cur:
            cur = s
            continue

        if len(cur) + 1 + len(s) <= max_chars:
            cur = f"{cur} {s}"
            continue

        if len(cur) >= min_chars:
            flush()
            cur = s
            continue

        # Split on commas/semicolons/colons to fill the block without exceeding max
        parts = re.split(r"(?<=[,;:])\s+", s)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not cur:
                cur = p
                continue
            if len(cur) + 1 + len(p) <= max_chars:
                cur = f"{cur} {p}"
            else:
                flush()
                cur = p

    flush()

    # 4) Post-process: merge any block shorter than min_block with its neighbor.
    #    Prevents tiny blocks that cause TTS cut-off and pop artifacts.
    min_block = 80
    hard_limit = max_chars + min_block  # allow slight overflow vs tiny blocks

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(blocks):
            if len(blocks[i]) >= min_block:
                i += 1
                continue
            # Try forward merge (into next block)
            if i + 1 < len(blocks):
                merged = f"{blocks[i]} {blocks[i + 1]}"
                if len(merged) <= hard_limit:
                    blocks[i] = merged
                    blocks.pop(i + 1)
                    changed = True
                    if len(blocks[i]) >= min_block:
                        i += 1
                    continue
            # Try backward merge (into previous block)
            if i > 0:
                merged = f"{blocks[i - 1]} {blocks[i]}"
                if len(merged) <= hard_limit:
                    blocks[i - 1] = merged
                    blocks.pop(i)
                    changed = True
                    continue
            i += 1  # unmergeable, leave as-is

    return blocks


def format_breathing_blocks(text: str, min_chars: int = 150, max_chars: int = 200) -> str:
    """Format text into bracket-wrapped breathing blocks for display/preview."""
    blocks = tts_breathing_blocks(text, min_chars, max_chars)
    if not blocks:
        return text.strip()
    if len(blocks) == 1:
        return blocks[0]  # single block: return plain text, no brackets
    return "\n\n".join(f"[{b}]" for b in blocks)


def _validate_brackets(text: str) -> str:
    """Check bracket formatting quality.

    Returns:
        "none"         — no brackets at all
        "well_formed"  — all brackets matched, not nested, no empty pairs
        "malformed"    — anything else (unmatched, nested, empty, stray content)
    """
    if "[" not in text and "]" not in text:
        return "none"
    # Quick structural checks
    if text.count("[") != text.count("]"):
        return "malformed"
    if "[[" in text or "]]" in text or "[]" in text:
        return "malformed"
    # Walk through: ensure no nesting and no significant content outside brackets
    depth = 0
    outside_chars = []
    for ch in text:
        if ch == "[":
            depth += 1
            if depth > 1:
                return "malformed"
        elif ch == "]":
            depth -= 1
            if depth < 0:
                return "malformed"
        elif depth == 0:
            outside_chars.append(ch)
    if depth != 0:
        return "malformed"
    # Check that content outside brackets is only whitespace
    outside = "".join(outside_chars).strip()
    if outside:
        return "malformed"
    return "well_formed"


def pad_audio(audio, sample_rate=24000, pad_ms=50):
    """Prepend/append short silence to prevent clipping on hard consonants."""
    pad = np.zeros(int(sample_rate * pad_ms / 1000), dtype=np.float32)
    return np.concatenate([pad, audio, pad])



def concatenate_chunks(chunks: list, sample_rate: int = 24000,
                       gap_ms: int = 80, crossfade_ms: int = 20) -> np.ndarray:
    """Concatenate audio chunks with silence gaps and crossfade.

    Args:
        chunks: List of 1-D or 2-D numpy audio arrays.
        sample_rate: Audio sample rate (default 24 kHz).
        gap_ms: Silence between sentences in milliseconds.
        crossfade_ms: Linear crossfade at boundaries in milliseconds.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    # Flatten any 2-D arrays (some ONNX models return [1, N])
    flat = [c.squeeze() for c in chunks]
    if len(flat) == 1:
        return flat[0]

    gap_samples = int(sample_rate * gap_ms / 1000)
    xfade_samples = int(sample_rate * crossfade_ms / 1000)
    silence = np.zeros(gap_samples, dtype=np.float32)

    parts = []
    for i, chunk in enumerate(flat):
        if i == 0:
            parts.append(chunk)
            continue
        prev = parts[-1]
        # Apply crossfade if both chunks are long enough
        if xfade_samples > 0 and len(prev) >= xfade_samples and len(chunk) >= xfade_samples:
            fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)
            tail = prev[-xfade_samples:] * fade_out
            head = chunk[:xfade_samples] * fade_in
            # Replace tail of previous part with crossfaded overlap
            parts[-1] = prev[:-xfade_samples]
            parts.append(tail + head)
            parts.append(silence)
            parts.append(chunk[xfade_samples:])
        else:
            parts.append(silence)
            parts.append(chunk)

    return np.concatenate(parts)


def find_available_port(start: int = 5000) -> int:
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start


def _model_files_present(model_id=None) -> bool:
    """Check if TADA model is cached in HuggingFace hub."""
    try:
        from huggingface_hub import try_to_load_from_cache
        if model_id is None:
            model_id = _current_model_id or "tada-1b"
        cfg = MODELS.get(model_id, MODELS["tada-1b"])
        # Check if model config exists in cache
        result = try_to_load_from_cache(cfg["hf_model_id"], "config.json")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


def _get_device():
    """Return the current device string."""
    return _current_device


def _set_device(device: str):
    """Set the device and reload models if needed."""
    global _current_device, tada_encoder, tada_model
    device = device.lower().strip()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"
    if device != _current_device:
        _current_device = device
        # Force reload on next generation
        with tada_lock:
            tada_encoder = None
            tada_model = None
        logger.info("Device set to {}", device)


def _check_gpu_info():
    """Return GPU info dict or None if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "vram_mb": round(props.total_mem / 1024 / 1024),
                "available": True,
            }
    except Exception:
        pass
    return {"name": None, "vram_mb": 0, "available": False}


def load_encoder():
    """Load (or return cached) TADA encoder."""
    global tada_encoder
    if tada_encoder is not None:
        return tada_encoder

    import torch
    from tada.modules.encoder import Encoder

    device = _get_device()
    with tada_lock:
        if tada_encoder is None:
            logger.info("Loading TADA encoder on {} ...", device)
            tada_encoder = Encoder.from_pretrained(
                ENCODER_HF_ID, subfolder="encoder"
            ).to(device)
            logger.success("TADA encoder ready on {}", device)
    return tada_encoder


def load_model(model_id=None):
    """Load (or return cached) TADA model."""
    global tada_model, _current_model_id
    if model_id is None:
        model_id = _current_model_id or "tada-1b"

    # If switching models, unload previous
    if _current_model_id and _current_model_id != model_id:
        with tada_lock:
            tada_model = None
            _current_model_id = None

    if tada_model is not None:
        return tada_model

    import torch
    from tada.modules.tada import TadaForCausalLM

    cfg = MODELS.get(model_id, MODELS["tada-1b"])
    device = _get_device()
    # Use float16 on CPU (float32 OOMs on 3B model), bfloat16 on CUDA
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    with tada_lock:
        if tada_model is None:
            logger.info("Loading {} on {} ({}) ...", cfg["name"], device, dtype)
            tada_model = TadaForCausalLM.from_pretrained(
                cfg["hf_model_id"], dtype=dtype
            ).to(device)
            _current_model_id = model_id
            logger.success("{} ready on {}", cfg["name"], device)
    return tada_model


def _load_voice_profile(voice_id):
    """Load a voice profile and return the encoded prompt."""
    import torch
    import torchaudio
    from tada.modules.encoder import EncoderOutput

    profile_dir = os.path.join(VOICES_DIR, voice_id)
    if not os.path.isdir(profile_dir):
        return None

    # Check for cached prompt
    cache_path = os.path.join(profile_dir, "prompt_cache.pt")
    if os.path.exists(cache_path):
        try:
            return EncoderOutput.load(cache_path, device=_get_device())
        except Exception:
            pass  # Re-encode if cache invalid

    # Find audio file
    meta_path = os.path.join(profile_dir, "profile.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        meta = json.load(f)

    audio_file = meta.get("audio_file", "")
    transcript = meta.get("transcript", "")
    audio_path = os.path.join(profile_dir, audio_file)
    if not os.path.exists(audio_path):
        return None

    # Encode (cap at 15s to avoid OOM in aligner's 128K-vocab CTC logits)
    encoder = load_encoder()
    audio, sr = torchaudio.load(audio_path)
    max_samples = sr * 15
    if audio.shape[-1] > max_samples:
        audio = audio[:, :max_samples]
        logger.info("Trimmed reference audio to 15s for voice {}", voice_id)
    audio = audio.to(_get_device())
    text_arg = [transcript] if transcript else None
    prompt = encoder(audio, text=text_arg, sample_rate=sr)

    # Cache for future use
    try:
        prompt.save(cache_path)
    except Exception:
        pass

    return prompt


def _list_voice_profiles():
    """List all saved voice profiles."""
    profiles = []
    if not os.path.exists(VOICES_DIR):
        return profiles
    for entry in os.listdir(VOICES_DIR):
        profile_dir = os.path.join(VOICES_DIR, entry)
        if not os.path.isdir(profile_dir):
            continue
        meta_path = os.path.join(profile_dir, "profile.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["id"] = entry
            profiles.append(meta)
        except (json.JSONDecodeError, OSError):
            pass
    profiles.sort(key=lambda x: x.get("created", ""), reverse=True)
    return profiles


# ---------------------------------------------------------------------------
# HTTP download with SSE progress
# ---------------------------------------------------------------------------


def _download_file_with_progress(url: str, dest_path: str, queue: Queue, label: str):
    """Download a file from URL, pushing SSE progress events to a queue."""
    tmp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TadaTTS-Studio/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 256 * 1024
            start_time = time.time()

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    speed = downloaded / max(elapsed, 0.001)
                    progress = int((downloaded / total) * 100) if total else 0

                    if speed >= 1_000_000:
                        speed_str = f"{speed / 1_000_000:.1f}MB/s"
                    elif speed >= 1_000:
                        speed_str = f"{speed / 1_000:.1f}KB/s"
                    else:
                        speed_str = f"{speed:.0f}B/s"

                    queue.put({
                        "phase": "downloading",
                        "file": label,
                        "progress": progress,
                        "downloaded_mb": round(downloaded / 1_000_000, 2),
                        "total_mb": round(total / 1_000_000, 2),
                        "size": f"{total / 1_000_000:.1f}MB",
                        "speed": speed_str,
                    })

        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


# --- Serve frontend ---
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# --- Health ---
@app.route("/api/health")
def health():
    gpu_info = _check_gpu_info()
    return jsonify({
        "status": "ok",
        "port": request.host.split(":")[-1],
        "ffmpeg": _find_ffmpeg() is not None,
        "alignment": _check_alignment_available(),
        "enhance": _check_enhance_available(),
        "vad": _check_vad_available(),
        "device": _get_device(),
        "gpu": gpu_info,
    })


# --- Normalize text ---
@app.route("/api/normalize", methods=["POST"])
def normalize_text():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    validity = _validate_brackets(text)

    if validity == "well_formed":
        # Extract blocks, normalize each individually, re-wrap
        blocks = re.findall(r'\[([^\[\]]+)\]', text)
        normalized_blocks = [normalize_for_tts(b) for b in blocks if b.strip()]
        if len(normalized_blocks) <= 1:
            formatted = normalized_blocks[0] if normalized_blocks else text.strip()
        else:
            formatted = "\n\n".join(f"[{b}]" for b in normalized_blocks)
    else:
        # "none" or "malformed": strip all brackets, normalize, re-chunk
        stripped = re.sub(r'[\[\]]', '', text)
        normalized = normalize_for_tts(stripped)
        formatted = format_breathing_blocks(normalized)

    return jsonify({"original": text, "normalized": formatted})


# --- Device ---
@app.route("/api/device", methods=["GET"])
def get_device():
    gpu_info = _check_gpu_info()
    return jsonify({"device": _get_device(), "gpu": gpu_info})


@app.route("/api/device", methods=["POST"])
def set_device():
    data = request.get_json(force=True)
    device = data.get("device", "cpu")
    _set_device(device)
    return jsonify({"device": _get_device()})


# --- Models ---
@app.route("/api/models")
def models():
    out = []
    for mid, m in MODELS.items():
        out.append({
            "id": mid, "name": m["name"], "size": m["size"],
            "params": m["params"], "languages": m["languages"],
        })
    return jsonify(out)


# --- Voices (voice profiles) ---
@app.route("/api/voices")
def voices():
    return jsonify(_list_voice_profiles())


@app.route("/api/voices/upload", methods=["POST"])
def upload_voice():
    """Upload a reference audio + transcript to create a voice profile."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    name = request.form.get("name", "").strip()
    transcript = request.form.get("transcript", "").strip()

    if not name:
        name = os.path.splitext(audio_file.filename)[0][:40]

    ext = os.path.splitext(audio_file.filename)[1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Create profile directory
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', name[:40].lower()).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_id = f"{safe_name}_{timestamp}"
    profile_dir = os.path.join(VOICES_DIR, voice_id)
    os.makedirs(profile_dir, exist_ok=True)

    # Save audio
    audio_filename = f"reference{ext}"
    audio_path = os.path.join(profile_dir, audio_filename)
    audio_file.save(audio_path)

    # Convert to WAV if needed
    if ext != ".wav":
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            wav_path = os.path.join(profile_dir, "reference.wav")
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-y", "-i", audio_path, "-ar", "24000", "-ac", "1", wav_path],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0:
                audio_filename = "reference.wav"

    # Save profile metadata
    profile = {
        "name": name,
        "audio_file": audio_filename,
        "transcript": transcript,
        "original_filename": audio_file.filename,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(profile_dir, "profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    profile["id"] = voice_id
    logger.success("Voice profile created: {} ({})", name, voice_id)
    return jsonify(profile)


@app.route("/api/voices/<voice_id>", methods=["DELETE"])
def delete_voice(voice_id):
    voice_id = os.path.basename(voice_id)
    profile_dir = os.path.join(VOICES_DIR, voice_id)
    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir)
        return jsonify({"status": "deleted", "id": voice_id})
    return jsonify({"error": "Voice not found"}), 404


@app.route("/api/voices/<voice_id>/audio")
def serve_voice_audio(voice_id):
    voice_id = os.path.basename(voice_id)
    profile_dir = os.path.join(VOICES_DIR, voice_id)
    meta_path = os.path.join(profile_dir, "profile.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "Voice not found"}), 404
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return send_from_directory(profile_dir, meta.get("audio_file", "reference.wav"))


# --- Sample voices (presets) ---
@app.route("/api/samplevoices")
def list_sample_voices():
    """List all sample voice files organized by language subfolder."""
    samples = []
    if not os.path.exists(SAMPLE_VOICES_DIR):
        return jsonify(samples)
    for lang_dir in sorted(os.listdir(SAMPLE_VOICES_DIR)):
        lang_path = os.path.join(SAMPLE_VOICES_DIR, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        for fname in sorted(os.listdir(lang_path)):
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                continue
            # Derive a display name from filename
            name = fname
            for prefix in ("voice_preview_", "ElevenLabs_"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            name = os.path.splitext(name)[0]
            # Clean up timestamps and metadata suffixes
            name = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}_?', '', name)
            name = re.sub(r'_?(pre|pvc)_sp\d+.*$', '', name)
            name = re.sub(r'\s*\(\d+\)\s*$', '', name)
            name = name.strip(' _-')
            samples.append({
                "lang": lang_dir,
                "file": fname,
                "name": name or fname,
                "path": f"{lang_dir}/{fname}",
            })
    return jsonify(samples)


@app.route("/api/sample-voices/file/<path:filepath>")
def serve_sample_voice(filepath):
    """Serve a sample voice audio file."""
    return send_from_directory(SAMPLE_VOICES_DIR, filepath)


@app.route("/api/voices/from-sample", methods=["POST"])
def create_voice_from_sample():
    """Create a voice profile from a sample voice file."""
    data = request.get_json(force=True)
    sample_path = data.get("path", "")
    name = data.get("name", "")
    transcript = data.get("transcript", "")

    if not sample_path:
        return jsonify({"error": "No sample path provided"}), 400

    src = os.path.join(SAMPLE_VOICES_DIR, sample_path)
    if not os.path.isfile(src):
        return jsonify({"error": "Sample file not found"}), 404

    if not name:
        name = os.path.splitext(os.path.basename(sample_path))[0][:40]

    ext = os.path.splitext(sample_path)[1].lower()
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', name[:40].lower()).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_id = f"{safe_name}_{timestamp}"
    profile_dir = os.path.join(VOICES_DIR, voice_id)
    os.makedirs(profile_dir, exist_ok=True)

    # Copy sample file
    audio_filename = f"reference{ext}"
    shutil.copy2(src, os.path.join(profile_dir, audio_filename))

    # Convert to WAV if needed
    if ext != ".wav":
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            wav_path = os.path.join(profile_dir, "reference.wav")
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-y", "-i", os.path.join(profile_dir, audio_filename),
                 "-ar", "24000", "-ac", "1", wav_path],
                capture_output=True, timeout=60,
            )
            if result.returncode == 0:
                audio_filename = "reference.wav"

    profile = {
        "name": name,
        "audio_file": audio_filename,
        "transcript": transcript,
        "original_filename": os.path.basename(sample_path),
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "sample",
    }
    with open(os.path.join(profile_dir, "profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    profile["id"] = voice_id
    logger.success("Voice profile from sample: {} ({})", name, voice_id)
    return jsonify(profile)


# --- Model status ---
@app.route("/api/model-status/<model_id>")
def model_status(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    cached = _model_files_present(model_id)
    return jsonify({"model_id": model_id, "cached": cached})


# --- Download model with SSE progress ---
@app.route("/api/download-model/<model_id>")
def download_model(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404

    model_cfg = MODELS[model_id]

    def stream():
        yield f"data: {json.dumps({'phase': 'checking', 'model': model_id})}\n\n"

        try:
            # Step 1: Download encoder from HuggingFace
            yield f"data: {json.dumps({'phase': 'downloading', 'file': 'TADA Encoder', 'progress': 0, 'speed': 'downloading from HuggingFace...'})}\n\n"
            load_result = {}

            def _download_encoder():
                try:
                    load_encoder()
                except Exception as e:
                    load_result["error"] = e

            t = threading.Thread(target=_download_encoder)
            t.start()
            tick = 0
            while t.is_alive():
                t.join(timeout=2.0)
                tick += 1
                yield f"data: {json.dumps({'phase': 'downloading', 'file': 'TADA Encoder', 'progress': min(90, tick * 5), 'speed': 'downloading from HuggingFace...'})}\n\n"
            if "error" in load_result:
                raise load_result["error"]
            yield f"data: {json.dumps({'phase': 'downloading', 'file': 'TADA Encoder', 'progress': 100, 'speed': 'complete'})}\n\n"

            # Step 2: Download and load TADA model
            yield f"data: {json.dumps({'phase': 'downloading', 'file': model_cfg['name'], 'progress': 0, 'speed': 'downloading from HuggingFace...'})}\n\n"
            load_result = {}

            def _download_model():
                try:
                    load_model(model_id)
                except Exception as e:
                    load_result["error"] = e

            t = threading.Thread(target=_download_model)
            t.start()
            tick = 0
            while t.is_alive():
                t.join(timeout=2.0)
                tick += 1
                yield f"data: {json.dumps({'phase': 'downloading', 'file': model_cfg['name'], 'progress': min(90, tick * 3), 'speed': 'downloading from HuggingFace...'})}\n\n"
            if "error" in load_result:
                raise load_result["error"]
            yield f"data: {json.dumps({'phase': 'downloading', 'file': model_cfg['name'], 'progress': 100, 'speed': 'complete'})}\n\n"

            yield f"data: {json.dumps({'phase': 'ready', 'message': 'Model ready'})}\n\n"

        except Exception as e:
            logger.exception("Model download/load failed")
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Chunked generation background worker ---

def _background_chunked_generate(job_id, voice_id, voice_prompt, sentences, speed,
                                  max_silence_ms, prompt, basename, model_id,
                                  voice_name=None, skip_enhance=False, skip_clean=False):
    """Generate audio for each sentence chunk, concatenate, loudnorm, and save."""
    with generation_jobs_lock:
        job = generation_jobs[job_id]
    q = job["queue"]
    try:
        import torch
        import torchaudio
        model = load_model(model_id)

        audio_chunks = []
        total = len(sentences)
        total_inference = 0.0

        for i, block in enumerate(sentences):
            # Check if abort was requested
            if job.get("abort"):
                q.put({"phase": "aborted"})
                with generation_jobs_lock:
                    job["status"] = "aborted"
                return

            q.put({"phase": "generating", "chunk": i + 1, "total": total,
                    "sentence": block})

            start = time.perf_counter()
            with generation_inference_lock:
                output = model.generate(prompt=voice_prompt, text=block)
                chunk_audio = output.audio[0].cpu().numpy()
            elapsed = time.perf_counter() - start
            total_inference += elapsed
            audio_chunks.append(chunk_audio)

        # Concatenate with crossfade and silence gaps
        q.put({"phase": "concatenating"})
        audio = concatenate_chunks(audio_chunks, sample_rate=24000, gap_ms=80, crossfade_ms=20)
        audio = pad_audio(audio, sample_rate=24000)

        job_dir = _tts_job_dir(basename)
        os.makedirs(job_dir, exist_ok=True)
        wav_path = os.path.join(job_dir, basename + ".wav")
        sf.write(wav_path, audio, 24000)

        # Apply loudnorm
        q.put({"phase": "normalizing"})
        _run_loudnorm(wav_path)

        # Re-read to get accurate duration after loudnorm
        info = sf.info(wav_path)
        duration_generated = info.duration
        rtf = total_inference / duration_generated if duration_generated > 0 else 0
        logger.success("Generated  {:.1f}s audio in {:.2f}s | RTF {:.2f} | {} chunks", duration_generated, total_inference, rtf, total)

        clean_prompt = re.sub(r'[\[\]]', '', prompt).strip()
        words = len(clean_prompt.split())
        approx_tokens = int(words * 1.3)

        cfg = MODELS.get(model_id, MODELS["tada-1b"])
        metadata = {
            "filename": basename + ".wav",
            "folder": basename,
            "prompt": clean_prompt,
            "model": cfg["name"],
            "model_id": model_id,
            "voice": voice_name or voice_id,
            "voice_id": voice_id,
            "device": _get_device(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inference_time": round(total_inference, 3),
            "rtf": round(rtf, 4),
            "duration_seconds": round(duration_generated, 2),
            "sample_rate": 24000,
            "speed": speed,
            "max_silence_ms": max_silence_ms,
            "words": words,
            "approx_tokens": approx_tokens,
            "chunked": True,
            "num_chunks": total,
        }
        # Set post-processing statuses before writing to disk
        metadata["alignment_status"] = "pending" if _check_alignment_available() else "unavailable"
        metadata["enhance_status"] = "skipped" if skip_enhance else ("pending" if _check_enhance_available() else "unavailable")
        metadata["vad_status"] = "skipped" if skip_clean else ("pending" if _check_vad_available() else "unavailable")
        json_path = os.path.join(job_dir, basename + ".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Now kick off background post-processing (respecting skip flags)
        _start_alignment(basename)
        if not skip_enhance:
            _start_enhancement(basename)
        if not skip_clean:
            if skip_enhance or not _check_enhance_available():
                _start_vad(basename, max_silence_ms)

        q.put({"phase": "done", "metadata": metadata})
        with generation_jobs_lock:
            job["status"] = "done"
            job["metadata"] = metadata

    except Exception as e:
        logger.exception("Chunked generation failed")
        q.put({"phase": "error", "message": str(e)})
        with generation_jobs_lock:
            job["status"] = "error"


def _cleanup_old_jobs(max_age_s=300):
    """Remove generation jobs older than max_age_s seconds."""
    now = time.time()
    with generation_jobs_lock:
        expired = [jid for jid, job in generation_jobs.items()
                   if now - job.get("created", 0) > max_age_s]
        for jid in expired:
            del generation_jobs[jid]


# --- Generate audio ---
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model_id = data.get("model", "tada-1b")
    voice_id = data.get("voice", "")
    prompt = data.get("prompt", "")
    speed = float(data.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))  # clamp to 0.5–2.0
    max_silence_ms = int(data.get("max_silence_ms", 500))
    max_silence_ms = max(200, min(1000, max_silence_ms))  # clamp to 200–1000
    skip_enhance = data.get("skip_enhance", False)
    skip_clean = data.get("skip_clean", False)

    # TADA inference parameters
    inference_params = data.get("inference", {})

    if not prompt.strip():
        return jsonify({"error": "Prompt is required"}), 400
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    if not voice_id:
        return jsonify({"error": "Voice profile is required. Upload a reference audio first."}), 400

    # Load voice profile
    voice_prompt = _load_voice_profile(voice_id)
    if voice_prompt is None:
        return jsonify({"error": f"Voice profile not found: {voice_id}"}), 404

    # Get voice name for metadata
    voice_name = voice_id
    profile_meta_path = os.path.join(VOICES_DIR, voice_id, "profile.json")
    if os.path.exists(profile_meta_path):
        with open(profile_meta_path) as f:
            voice_name = json.load(f).get("name", voice_id)

    # Reject if another generation or stream is already running
    if _stream_active.is_set():
        return jsonify({"error": "A stream is already in progress. Please wait."}), 429
    with generation_jobs_lock:
        for job in generation_jobs.values():
            if job.get("status") == "running":
                return jsonify({"error": "A generation is already in progress. Please wait or abort."}), 429

    tada = load_model(model_id)
    logger.info("Generate  \033[1m{}\033[0m | {} | {} chars", model_id, voice_name, len(prompt))

    # Clean the prompt text (strip markdown, URLs, brackets, whitespace)
    pre_blocks = re.findall(r'\[([^\[\]]+)\]', prompt)
    if pre_blocks:
        # Strip brackets, join all blocks into one continuous text
        cleaned_parts = []
        for b in pre_blocks:
            cleaned = re.sub(r"[*_#`~]", "", b)
            cleaned = re.sub(r"https?://\S+", "link", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                cleaned_parts.append(cleaned)
        tts_prompt = " ".join(cleaned_parts)
    else:
        tts_prompt = clean_for_tts(prompt)

    # --- Send the whole text as one block to TADA (no chunk splitting) ---
    import torch
    _cleanup_old_jobs()
    single_block = tts_prompt
    start = time.perf_counter()
    try:
        # Build InferenceOptions from request params
        from tada.modules.tada import InferenceOptions
        inf_opts = InferenceOptions(
            acoustic_cfg_scale=float(inference_params.get("acoustic_cfg_scale", 1.6)),
            duration_cfg_scale=float(inference_params.get("duration_cfg_scale", 1.0)),
            noise_temperature=float(inference_params.get("noise_temperature", 0.9)),
            num_flow_matching_steps=int(inference_params.get("num_flow_matching_steps", 10)),
            num_acoustic_candidates=int(inference_params.get("num_acoustic_candidates", 1)),
            scorer=inference_params.get("scorer", "likelihood"),
            spkr_verification_weight=float(inference_params.get("spkr_verification_weight", 1.0)),
            speed_up_factor=float(inference_params["speed_up_factor"]) if inference_params.get("speed_up_factor") else None,
            negative_condition_source=inference_params.get("negative_condition_source", "negative_step_output"),
            text_only_logit_scale=float(inference_params.get("text_only_logit_scale", 0.0)),
        )
        gen_kwargs = {
            "prompt": voice_prompt,
            "text": single_block,
            "num_extra_steps": int(inference_params.get("num_extra_steps", 0)),
            "normalize_text": inference_params.get("normalize_text", True),
            "inference_options": inf_opts,
        }
        with generation_inference_lock:
            output = tada.generate(**gen_kwargs)
            audio = output.audio[0].cpu().numpy()
    except Exception as e:
        logger.exception("TTS inference failed")
        return jsonify({"error": f"Generation failed: {e}"}), 500
    end = time.perf_counter()

    audio = pad_audio(audio, sample_rate=24000)
    duration_generated = len(audio) / 24000
    inference_time = end - start
    rtf = inference_time / duration_generated

    basename = generate_filename(prompt)
    job_dir = _tts_job_dir(basename)
    os.makedirs(job_dir, exist_ok=True)
    wav_name = f"{basename}.wav"
    json_name = f"{basename}.json"

    sf.write(os.path.join(job_dir, wav_name), audio, 24000)
    logger.success("Generated  {:.1f}s audio in {:.2f}s | RTF {:.2f}", duration_generated, inference_time, rtf)

    clean_prompt = re.sub(r'[\[\]]', '', prompt).strip()
    cfg = MODELS.get(model_id, MODELS["tada-1b"])
    metadata = {
        "filename": wav_name,
        "folder": basename,
        "prompt": clean_prompt,
        "model": cfg["name"],
        "model_id": model_id,
        "voice": voice_name,
        "voice_id": voice_id,
        "device": _get_device(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inference_time": round(inference_time, 3),
        "rtf": round(rtf, 4),
        "duration_seconds": round(duration_generated, 2),
        "sample_rate": 24000,
        "speed": speed,
        "max_silence_ms": max_silence_ms,
        "words": len(clean_prompt.split()),
        "approx_tokens": int(len(clean_prompt.split()) * 1.3),
    }
    with open(os.path.join(job_dir, json_name), "w") as f:
        json.dump(metadata, f, indent=2)

    # Kick off background alignment and post-processing (respecting skip flags)
    _start_alignment(basename)
    metadata["alignment_status"] = "pending" if _check_alignment_available() else "unavailable"
    if not skip_enhance:
        _start_enhancement(basename)
        metadata["enhance_status"] = "pending" if _check_enhance_available() else "unavailable"
    else:
        metadata["enhance_status"] = "skipped"
    if not skip_clean:
        # VAD is chained from enhancement; if enhance unavailable/skipped, start VAD directly
        if skip_enhance or not _check_enhance_available():
            _start_vad(basename, max_silence_ms)
        metadata["vad_status"] = "pending" if _check_vad_available() else "unavailable"
    else:
        metadata["vad_status"] = "skipped"

    return jsonify(metadata)


# --- Chunked generation SSE progress ---
@app.route("/api/generate-progress/<job_id>")
def generate_progress(job_id):
    with generation_jobs_lock:
        job = generation_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job ID"}), 404

    def stream():
        # If job already finished (e.g. EventSource reconnect after completion),
        # return the final status immediately instead of blocking on the queue.
        status = job.get("status")
        if status == "done":
            yield f"data: {json.dumps({'phase': 'done', 'metadata': job.get('metadata')})}\n\n"
            return
        if status in ("error", "aborted"):
            yield f"data: {json.dumps({'phase': status})}\n\n"
            return

        q = job["queue"]
        while True:
            try:
                event = q.get(timeout=10)
            except Exception:
                # Queue read timed out — check if job finished while we were waiting
                # (handles race where another SSE consumer drained the "done" event)
                with generation_jobs_lock:
                    cur_status = job.get("status")
                if cur_status == "done":
                    yield f"data: {json.dumps({'phase': 'done', 'metadata': job.get('metadata')})}\n\n"
                    break
                if cur_status in ("error", "aborted"):
                    yield f"data: {json.dumps({'phase': cur_status})}\n\n"
                    break
                continue  # Still running — keep waiting
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("phase") in ("done", "error", "aborted"):
                break

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/generate-abort/<job_id>", methods=["POST"])
def abort_generation(job_id):
    with generation_jobs_lock:
        job = generation_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job ID"}), 404
        job["abort"] = True
    return jsonify({"status": "aborting"})


# --- Stream audio (listen-only, no save) ---

# Global flag so /api/generate can also reject while streaming is active
_stream_active = threading.Event()


@app.route("/api/stream", methods=["POST"])
def stream_audio():
    """Stream TTS audio via SSE using TADA generate().

    TADA doesn't support native streaming, so we split text into sentences,
    generate each, and send as base64-encoded float32 PCM chunks.
    Nothing is saved to disk — this is listen-only mode.
    """
    data = request.get_json()
    model_id = data.get("model", "tada-1b")
    voice_id = data.get("voice", "")
    prompt = data.get("prompt", "")
    speed = float(data.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))
    inference_params = data.get("inference", {})

    if not prompt.strip():
        return jsonify({"error": "Prompt is required"}), 400
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    if not voice_id:
        return jsonify({"error": "Voice profile is required."}), 400

    # Load voice profile
    voice_prompt = _load_voice_profile(voice_id)
    if voice_prompt is None:
        return jsonify({"error": f"Voice profile not found: {voice_id}"}), 404

    # Reject if another generation/stream is already running
    with generation_jobs_lock:
        for job in generation_jobs.values():
            if job.get("status") == "running":
                return jsonify({"error": "A generation is already in progress. Please wait or abort."}), 429
    if _stream_active.is_set():
        return jsonify({"error": "A stream is already in progress."}), 429

    tada = load_model(model_id)
    tts_prompt = clean_for_tts(prompt)

    logger.info("Stream  \033[1m{}\033[0m | {} | {} chars", model_id, voice_id, len(prompt))

    # Build InferenceOptions
    from tada.modules.tada import InferenceOptions
    inf_opts = InferenceOptions(
        acoustic_cfg_scale=float(inference_params.get("acoustic_cfg_scale", 1.6)),
        duration_cfg_scale=float(inference_params.get("duration_cfg_scale", 1.0)),
        noise_temperature=float(inference_params.get("noise_temperature", 0.9)),
        num_flow_matching_steps=int(inference_params.get("num_flow_matching_steps", 10)),
        num_acoustic_candidates=int(inference_params.get("num_acoustic_candidates", 1)),
        scorer=inference_params.get("scorer", "likelihood"),
        speed_up_factor=float(inference_params["speed_up_factor"]) if inference_params.get("speed_up_factor") else None,
        negative_condition_source=inference_params.get("negative_condition_source", "negative_step_output"),
        text_only_logit_scale=float(inference_params.get("text_only_logit_scale", 0.0)),
    )

    q = Queue()

    def _run_stream():
        import torch
        _stream_active.set()
        try:
            with generation_inference_lock:
                output = tada.generate(
                    prompt=voice_prompt, text=tts_prompt,
                    num_extra_steps=int(inference_params.get("num_extra_steps", 0)),
                    normalize_text=inference_params.get("normalize_text", True),
                    inference_options=inf_opts,
                )
                audio = output.audio[0].cpu().numpy().astype(np.float32)
            q.put(("audio", audio, 24000))
            q.put(("done", None, None))
        except Exception as exc:
            logger.exception("Stream generation failed")
            q.put(("error", str(exc), None))
        finally:
            _stream_active.clear()

    t = threading.Thread(target=_run_stream, daemon=True)
    t.start()

    def _sse():
        chunk_num = 0
        while True:
            try:
                kind, payload, sr = q.get(timeout=120)
            except Exception:
                yield f"data: {json.dumps({'phase': 'error', 'message': 'Stream timed out'})}\n\n"
                break
            if kind == "audio":
                chunk_num += 1
                pcm_bytes = payload.tobytes()
                b64 = base64.b64encode(pcm_bytes).decode("ascii")
                yield f"data: {json.dumps({'phase': 'audio', 'chunk': chunk_num, 'samples': b64, 'sample_rate': sr})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'phase': 'done', 'total_chunks': chunk_num})}\n\n"
                break
            elif kind == "error":
                yield f"data: {json.dumps({'phase': 'error', 'message': payload})}\n\n"
                break

    return Response(
        _sse(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- List audio files ---
@app.route("/api/generation")
def list_audio():
    files = []
    if not os.path.exists(AUDIO_DIR):
        return jsonify(files)
    for entry in os.listdir(AUDIO_DIR):
        entry_path = os.path.join(AUDIO_DIR, entry)
        if not os.path.isdir(entry_path) or entry == "TRASH":
            continue
        json_path = os.path.join(entry_path, entry + ".json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    files.append(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skipping corrupt/partial metadata {}: {}", entry, e)
    files.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(files)


# --- List alignment files ---
@app.route("/api/generation/alignments")
def list_alignments():
    """List all TTS files that have alignment data."""
    items = []
    if not os.path.exists(AUDIO_DIR):
        return jsonify(items)
    for entry in os.listdir(AUDIO_DIR):
        entry_path = os.path.join(AUDIO_DIR, entry)
        if not os.path.isdir(entry_path) or entry == "TRASH":
            continue
        json_path = os.path.join(entry_path, entry + ".json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            status = meta.get("alignment_status", "pending")
            if status not in ("ready", "aligning", "failed"):
                continue
            alignment = meta.get("word_alignment", [])
            items.append({
                "source_audio": meta.get("filename", entry),
                "status": status,
                "version": "original",
                "word_count": len(alignment),
                "timestamp": meta.get("timestamp", ""),
            })
            # Also check enhanced alignment
            if meta.get("enhanced_alignment_status") == "ready":
                enh_alignment = meta.get("enhanced_word_alignment", [])
                items.append({
                    "source_audio": meta.get("filename", entry),
                    "status": "ready",
                    "version": "enhanced",
                    "word_count": len(enh_alignment),
                    "timestamp": meta.get("timestamp", ""),
                })
        except (json.JSONDecodeError, OSError):
            pass
    # Also include standalone force-alignment results (subfolders with alignment.json)
    if os.path.exists(ALIGN_DIR):
        for entry in os.listdir(ALIGN_DIR):
            entry_path = os.path.join(ALIGN_DIR, entry)
            if not os.path.isdir(entry_path) or entry == "TRASH":
                continue
            json_path = os.path.join(entry_path, "alignment.json")
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
                items.append({
                    "source_audio": meta.get("source_file", entry),
                    "status": "ready",
                    "version": "standalone",
                    "folder": entry,
                    "word_count": meta.get("word_count", 0),
                    "timestamp": meta.get("timestamp", ""),
                })
            except (json.JSONDecodeError, OSError):
                pass
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(items)


# --- List force-alignment items ---
@app.route("/api/generation/force-alignment")
def list_force_alignments():
    """List all standalone force-alignment results for the history view."""
    items = []
    if not os.path.exists(ALIGN_DIR):
        return jsonify(items)
    for entry in os.listdir(ALIGN_DIR):
        entry_path = os.path.join(ALIGN_DIR, entry)
        if not os.path.isdir(entry_path) or entry == "TRASH":
            continue
        json_path = os.path.join(entry_path, "alignment.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r") as f:
                meta = json.load(f)
            words = meta.get("alignment", [])
            duration = round(words[-1]["end"], 2) if words else 0
            items.append({
                "type": "force-alignment",
                "folder": meta.get("folder", entry),
                "source_file": meta.get("source_file", ""),
                "transcript": meta.get("transcript", ""),
                "word_count": meta.get("word_count", len(words)),
                "word_alignment": words,
                "duration_seconds": duration,
                "inference_time": meta.get("inference_time", 0),
                "timestamp": meta.get("timestamp", ""),
            })
        except (json.JSONDecodeError, OSError, IndexError, KeyError):
            pass
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(items)


# --- Standalone force alignment ---
@app.route("/api/force-align", methods=["POST"])
def force_align():
    """Run force alignment on an uploaded audio file with transcript text.

    Creates a named subfolder under force-alignment/ containing:
      - The original audio file
      - alignment.json with word-level timestamps
    """
    if not _check_alignment_available():
        return jsonify({"error": "Force alignment not available (stable-ts not installed)"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No transcript text provided"}), 400

    audio_file = request.files["audio"]
    original_name = audio_file.filename
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg"):
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    # Create a named subfolder: <audio-name>_<YYYYMMDD_HHMMSS>/
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', os.path.splitext(original_name)[0][:40]).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{safe_name}_{timestamp}"
    job_dir = os.path.join(ALIGN_DIR, folder_name)
    os.makedirs(job_dir, exist_ok=True)

    # Save the original audio file into the subfolder
    audio_path = os.path.join(job_dir, original_name)
    audio_file.save(audio_path)

    # Convert to WAV if needed (alignment needs WAV)
    wav_path = audio_path
    conv_path = None
    try:
        if ext != ".wav":
            ffmpeg = _find_ffmpeg()
            if not ffmpeg:
                return jsonify({"error": "ffmpeg required for non-WAV files"}), 400
            conv_path = os.path.join(job_dir, os.path.splitext(original_name)[0] + "_conv.wav")
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-y", "-i", audio_path, "-ar", "24000", "-ac", "1", conv_path],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                return jsonify({"error": "Audio conversion failed"}), 500
            wav_path = conv_path

        start = time.perf_counter()
        alignment = _run_alignment(wav_path, text)
        elapsed = time.perf_counter() - start

        if not alignment:
            return jsonify({"error": "Alignment produced no results"}), 500

        # Save alignment JSON into the subfolder
        result_data = {
            "source_file": original_name,
            "folder": folder_name,
            "transcript": text,
            "alignment": alignment,
            "word_count": len(alignment),
            "inference_time": round(elapsed, 3),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(job_dir, "alignment.json"), "w") as f:
            json.dump(result_data, f, indent=2)

        logger.success("Force-aligned  {} | {} words in {:.2f}s -> {}", original_name, len(alignment), elapsed, folder_name)
        return jsonify(result_data)

    finally:
        # Clean up temp conversion file (keep the original audio)
        if conv_path:
            try:
                os.unlink(conv_path)
            except OSError:
                pass


# --- Delete audio file (move to TRASH) ---
@app.route("/api/generation/<filename>", methods=["DELETE"])
def delete_audio(filename):
    basename = filename.rsplit(".", 1)[0]
    job_dir = _tts_job_dir(basename)
    if os.path.isdir(job_dir):
        shutil.move(job_dir, os.path.join(TRASH_DIR, basename))
        return jsonify({"status": "deleted", "filename": filename})
    return jsonify({"error": "File not found"}), 404


# --- Delete alignment folder (move to TRASH) ---
@app.route("/api/generation/alignment/<folder>", methods=["DELETE"])
def delete_alignment(folder):
    folder = os.path.basename(folder)  # sanitize
    job_dir = os.path.join(ALIGN_DIR, folder)
    if os.path.isdir(job_dir):
        shutil.move(job_dir, os.path.join(ALIGN_TRASH_DIR, folder))
        return jsonify({"status": "deleted", "folder": folder})
    return jsonify({"error": "Folder not found"}), 404


# --- Open audio folder in OS file manager (with file selected if provided) ---
@app.route("/api/open-generation-folder", methods=["POST"])
def open_audio_folder():
    import sys
    data = request.get_json(silent=True) or {}
    item_type = data.get("type", "")
    if item_type == "force-alignment":
        align_folder = os.path.basename(data.get("folder", ""))
        align_file = os.path.basename(data.get("filename", ""))
        job_dir = os.path.abspath(os.path.join(ALIGN_DIR, align_folder)) if align_folder else ""
        file_path = os.path.join(job_dir, align_file) if job_dir and align_file else ""
        folder = os.path.abspath(ALIGN_DIR)
    else:
        filename = data.get("filename", "")
        filename = os.path.basename(filename) if filename else ""
        basename = filename.rsplit(".", 1)[0] if filename else ""
        job_dir = os.path.abspath(_tts_job_dir(basename)) if basename else ""
        file_path = os.path.join(job_dir, filename) if job_dir and filename else ""
        folder = os.path.abspath(AUDIO_DIR)
    try:
        if sys.platform == "win32":
            if file_path and os.path.exists(file_path):
                subprocess.Popen(["explorer", "/select,", file_path])
            else:
                os.startfile(folder)
        elif sys.platform == "darwin":
            if file_path and os.path.exists(file_path):
                subprocess.Popen(["open", "-R", file_path])
            else:
                subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return jsonify({"status": "ok"})
    except Exception as e:
        logger.error("Failed to open folder: {}", e)
        return jsonify({"error": str(e)}), 500


# --- Delete all audio files (move to TRASH) ---
@app.route("/api/generation", methods=["DELETE"])
def delete_all_audio():
    count = 0
    for entry in os.listdir(AUDIO_DIR):
        entry_path = os.path.join(AUDIO_DIR, entry)
        if os.path.isdir(entry_path) and entry != "TRASH":
            shutil.move(entry_path, os.path.join(TRASH_DIR, entry))
            count += 1
    return jsonify({"status": "deleted", "count": count})


# --- Word alignment for karaoke ---
@app.route("/api/generation/<filename>/alignment")
def get_alignment(filename):
    """Return alignment data, triggering retroactive alignment for old files.
    Query param ?version=enhanced returns alignment for the enhanced file."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(_tts_job_dir(basename), basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_alignment_available():
        return jsonify({"status": "unavailable"})

    try:
        metadata = _read_metadata(basename)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Alignment metadata read failed for {} (likely being written): {}", basename, e)
        return jsonify({"status": "aligning"})  # file being written, try again later

    version = request.args.get("version", "original")

    if version == "cleaned":
        cl_status = metadata.get("cleaned_alignment_status")
        if cl_status == "ready":
            return jsonify({
                "status": "ready",
                "word_alignment": metadata.get("cleaned_word_alignment", []),
            })
        if cl_status == "aligning":
            return jsonify({"status": "aligning"})
        # Cleaned file may exist but alignment not started yet — trigger it
        if metadata.get("cleaned_filename"):
            _start_alignment(basename)
            return jsonify({"status": "aligning"})
        return jsonify({"status": "unavailable"})

    # For original and enhanced, use the same word_alignment
    status = metadata.get("alignment_status")

    if status == "ready":
        return jsonify({
            "status": "ready",
            "word_alignment": metadata.get("word_alignment", []),
        })

    if status == "failed":
        # Retry — previous failure may have been due to a fixable issue
        _start_alignment(basename)
        return jsonify({"status": "aligning"})

    if status == "aligning":
        # Check if thread is actually still running (may have crashed)
        with alignment_tasks_lock:
            if basename not in alignment_tasks:
                _start_alignment(basename)
        resp = {"status": "aligning"}
        started = metadata.get("alignment_started_at")
        duration = metadata.get("duration_seconds")
        if started:
            resp["elapsed"] = round(time.time() - started, 1)
        if duration:
            resp["audio_duration"] = duration
        return jsonify(resp)

    # No alignment attempted yet (old file) — trigger retroactive alignment
    _start_alignment(basename)
    return jsonify({"status": "aligning"})


# --- Audio enhancement status ---
@app.route("/api/generation/<filename>/enhance-status")
def get_enhance_status(filename):
    """Return enhancement status, triggering retroactive enhancement for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(_tts_job_dir(basename), basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_enhance_available():
        return jsonify({"status": "unavailable"})

    try:
        metadata = _read_metadata(basename)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Enhancement metadata read failed for {} (likely being written): {}", basename, e)
        return jsonify({"status": "enhancing"})  # file being written, try again later

    status = metadata.get("enhance_status")

    if status == "ready" and metadata.get("enhanced_filename"):
        enhanced_path = os.path.join(_tts_job_dir(basename), metadata["enhanced_filename"])
        if os.path.exists(enhanced_path):
            return jsonify({
                "status": "ready",
                "enhanced_filename": metadata["enhanced_filename"],
            })
        # File missing — re-enhance
        _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    if status == "failed":
        return jsonify({"status": "failed"})

    if status == "enhancing":
        with enhance_tasks_lock:
            if basename not in enhance_tasks:
                _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    # No enhancement attempted yet (old file) — trigger retroactive
    _start_enhancement(basename)
    return jsonify({"status": "enhancing"})


# --- Silence removal (VAD) status ---
@app.route("/api/generation/<filename>/vad-status")
def get_vad_status(filename):
    """Return silence removal status, triggering retroactive cleaning for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(_tts_job_dir(basename), basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_vad_available():
        return jsonify({"status": "unavailable"})

    try:
        metadata = _read_metadata(basename)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("VAD metadata read failed for {} (likely being written): {}", basename, e)
        return jsonify({"status": "cleaning"})  # file being written, try again later

    status = metadata.get("vad_status")

    if status == "ready" and metadata.get("cleaned_filename"):
        cleaned_path = os.path.join(_tts_job_dir(basename), metadata["cleaned_filename"])
        if os.path.exists(cleaned_path):
            return jsonify({
                "status": "ready",
                "cleaned_filename": metadata["cleaned_filename"],
            })
        _start_vad(basename)
        return jsonify({"status": "cleaning"})

    if status == "failed":
        return jsonify({"status": "failed"})

    if status == "normalizing":
        return jsonify({"status": "normalizing"})

    if status == "cleaning":
        with vad_tasks_lock:
            if basename not in vad_tasks:
                _start_vad(basename)
        return jsonify({"status": "cleaning"})

    # No VAD attempted yet — trigger retroactive
    _start_vad(basename)
    return jsonify({"status": "cleaning"})


# --- Locate ffmpeg helper ---
def _find_ffmpeg():
    bin_dir = os.path.join(os.path.dirname(__file__), "bin")
    local = os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return local if os.path.isfile(local) else shutil.which("ffmpeg")


ALIGNMENT_VERSION = 2  # Bump when alignment logic changes to invalidate cached data

# --- Alignment helpers (stable-ts) ---

def _check_alignment_available():
    """Check if stable-ts is importable. Cached after first call."""
    global alignment_available
    if alignment_available is not None:
        return alignment_available
    try:
        import stable_whisper  # noqa: F401
        alignment_available = True
    except ImportError:
        alignment_available = False
    return alignment_available


def _load_alignment_model():
    """Load (or return cached) stable-ts Whisper tiny.en model."""
    global alignment_model
    if alignment_model is not None:
        return alignment_model
    import stable_whisper
    with alignment_lock:
        if alignment_model is None:
            alignment_model = stable_whisper.load_model("tiny.en")
    return alignment_model


def _run_alignment(wav_path, prompt_text):
    """Run forced alignment. Returns list of {word, begin, end} or None."""
    import warnings
    try:
        model = _load_alignment_model()
        # Load audio as numpy array to avoid stable-ts needing ffmpeg in PATH
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Whisper expects 16kHz — resample if needed (TADA outputs 24kHz)
        if sr != 16000:
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio), dtype=np.float32),
                audio,
            ).astype(np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.align(audio, prompt_text, language="en", fast_mode=True)
        for w in caught:
            msg = str(w.message)
            if "failed to align" in msg:
                logger.warning("Align partial: {}", msg)
            else:
                logger.debug("Align warning: {}", msg)
        alignment = []
        for w in result.all_words():
            word_text = w.word.strip()
            if word_text:
                alignment.append({
                    "word": word_text,
                    "begin": round(w.start, 3),
                    "end": round(w.end, 3),
                })
        return alignment if alignment else None
    except Exception as e:
        logger.exception("Alignment failed for {}", wav_path)
        return None


def _audio_hash(wav_path):
    """Compute SHA-256 hash of audio file for cache validation."""
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _background_align(basename):
    """Run alignment in background thread, update metadata JSON when done.
    Aligns the original WAV and, if available, the enhanced WAV too."""
    job_dir = _tts_job_dir(basename)
    json_path = os.path.join(job_dir, basename + ".json")
    wav_path = os.path.join(job_dir, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        prompt_text = metadata.get("prompt", "")
        if not prompt_text.strip():
            _update_metadata(basename, {"alignment_status": "failed"})
            return

        # --- Align original WAV ---
        current_hash = _audio_hash(wav_path)
        need_original = not (
            metadata.get("alignment_status") == "ready"
            and metadata.get("audio_hash") == current_hash
            and metadata.get("alignment_version") == ALIGNMENT_VERSION
            and metadata.get("word_alignment")
        )

        if need_original:
            _update_metadata(basename, {
                "alignment_status": "aligning",
                "alignment_started_at": time.time(),
            })
            align_start = time.perf_counter()
            alignment = _run_alignment(wav_path, prompt_text)
            align_elapsed = round(time.perf_counter() - align_start, 3)
            if alignment:
                _update_metadata(basename, {
                    "alignment_status": "ready",
                    "alignment_time": align_elapsed,
                    "word_alignment": alignment,
                    "audio_hash": current_hash,
                    "alignment_version": ALIGNMENT_VERSION,
                })
                logger.success("Aligned  {} | {} words | {:.2f}s", basename, len(alignment), align_elapsed)
            else:
                _update_metadata(basename, {"alignment_status": "failed"})
                logger.warning("Alignment produced no results for {}", basename)

        # --- Align cleaned WAV (if it exists) ---
        # Re-read metadata: VAD may have finished since we started
        metadata = _read_metadata(basename)
        cleaned_name = metadata.get("cleaned_filename")
        if cleaned_name:
            cleaned_path = os.path.join(job_dir, cleaned_name)
            if os.path.exists(cleaned_path):
                cleaned_hash = _audio_hash(cleaned_path)
                need_cleaned = not (
                    metadata.get("cleaned_alignment_status") == "ready"
                    and metadata.get("cleaned_audio_hash") == cleaned_hash
                    and metadata.get("cleaned_alignment_version") == ALIGNMENT_VERSION
                    and metadata.get("cleaned_word_alignment")
                )
                if need_cleaned:
                    _update_metadata(basename, {"cleaned_alignment_status": "aligning"})
                    cleaned_alignment = _run_alignment(cleaned_path, prompt_text)
                    if cleaned_alignment:
                        _update_metadata(basename, {
                            "cleaned_alignment_status": "ready",
                            "cleaned_word_alignment": cleaned_alignment,
                            "cleaned_audio_hash": cleaned_hash,
                            "cleaned_alignment_version": ALIGNMENT_VERSION,
                        })
                    else:
                        _update_metadata(basename, {"cleaned_alignment_status": "failed"})

    except Exception as e:
        logger.exception("Background alignment failed for {}", basename)
        try:
            _update_metadata(basename, {"alignment_status": "failed", "error_message": f"Alignment: {e}"})
        except Exception:
            pass
    finally:
        with alignment_tasks_lock:
            alignment_tasks.pop(basename, None)


def _start_alignment(basename):
    """Spawn alignment thread if not already running for this file."""
    if not _check_alignment_available():
        return
    with alignment_tasks_lock:
        if basename in alignment_tasks:
            return
        t = threading.Thread(target=_background_align, args=(basename,), daemon=True)
        alignment_tasks[basename] = t
        t.start()


# --- Enhancement helpers (LavaSR) ---

def _check_enhance_available():
    """Check if LavaSR is importable. Cached after first call."""
    global enhance_available
    if enhance_available is not None:
        return enhance_available
    try:
        from LavaSR.model import LavaEnhance  # noqa: F401
        enhance_available = True
    except ImportError:
        enhance_available = False
    return enhance_available


def _load_enhance_model():
    """Load (or return cached) LavaSR enhancement model."""
    global enhance_model
    if enhance_model is not None:
        return enhance_model
    from LavaSR.model import LavaEnhance
    with enhance_lock:
        if enhance_model is None:
            enhance_model = LavaEnhance("YatharthS/LavaSR", "cpu")
    return enhance_model


def _run_enhance(wav_path):
    """Enhance audio file in chunks to avoid OOM on long audio. Returns enhanced filename or None."""
    import gc
    try:
        gc.collect()
        try:
            import torch
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
        except Exception:
            pass

        model = _load_enhance_model()
        audio, sr = model.load_audio(wav_path)

        # Chunk parameters: 30s segments with 0.5s overlap for crossfade
        CHUNK_SEC = 30
        OVERLAP_SEC = 0.5
        chunk_samples = int(sr * CHUNK_SEC)
        overlap_samples = int(sr * OVERLAP_SEC)
        total_samples = audio.shape[-1]

        # Short audio — process in one shot
        if total_samples <= chunk_samples + overlap_samples:
            enhanced = model.enhance(audio)
            enhanced_np = enhanced.cpu().numpy().squeeze()
            del enhanced
        else:
            # Process in chunks to stay within memory limits
            enhanced_chunks = []
            pos = 0
            while pos < total_samples:
                end = min(pos + chunk_samples + overlap_samples, total_samples)
                chunk = audio[:, pos:end] if audio.dim() == 2 else audio[pos:end].unsqueeze(0)
                enh_chunk = model.enhance(chunk)
                enhanced_chunks.append(enh_chunk.cpu().numpy().squeeze())
                del enh_chunk, chunk
                gc.collect()
                try:
                    import torch
                    if hasattr(torch, 'cuda'):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                pos += chunk_samples  # advance by chunk_samples, leaving overlap for crossfade

            # Crossfade-stitch enhanced chunks (output is 48kHz = 2x input SR)
            enhanced_np = _stitch_enhanced_chunks(enhanced_chunks, overlap_samples * 2)
            del enhanced_chunks
            gc.collect()

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        enhanced_name = f"{basename}_enhanced.wav"
        enhanced_path = os.path.join(os.path.dirname(wav_path), enhanced_name)
        sf.write(enhanced_path, enhanced_np, 48000)
        return enhanced_name
    except Exception as e:
        logger.exception("Enhancement failed for {}", wav_path)
        return None


def _stitch_enhanced_chunks(chunks, overlap_samples):
    """Crossfade-stitch enhanced audio chunks at overlap boundaries."""
    if len(chunks) == 1:
        return chunks[0]
    parts = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            # Keep everything except the trailing overlap half
            parts.append(chunk)
            continue
        prev = parts[-1]
        xfade = min(overlap_samples, len(prev), len(chunk))
        if xfade > 0:
            fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            # Blend the overlap region
            prev[-xfade:] = prev[-xfade:] * fade_out + chunk[:xfade] * fade_in
            parts.append(chunk[xfade:])
        else:
            parts.append(chunk)
    return np.concatenate(parts)


def _background_enhance(basename):
    """Run enhancement in background thread, update metadata JSON when done.
    Automatically chains into silence removal (VAD) when enhancement completes."""
    job_dir = _tts_job_dir(basename)
    json_path = os.path.join(job_dir, basename + ".json")
    wav_path = os.path.join(job_dir, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Skip if already enhanced and file exists
        if (metadata.get("enhance_status") == "ready"
                and metadata.get("enhanced_filename")
                and os.path.exists(os.path.join(job_dir, metadata["enhanced_filename"]))):
            # Still chain VAD if not done yet
            if metadata.get("vad_status") not in ("ready", "cleaning"):
                _start_vad(basename)
            return

        _update_metadata(basename, {"enhance_status": "enhancing"})

        enh_start = time.perf_counter()
        enhanced_name = _run_enhance(wav_path)
        enh_elapsed = round(time.perf_counter() - enh_start, 3)

        if enhanced_name:
            _update_metadata(basename, {
                "enhance_status": "ready",
                "enhance_time": enh_elapsed,
                "enhanced_filename": enhanced_name,
            })
            logger.success("Enhanced  {} | {:.2f}s", basename, enh_elapsed)
        else:
            _update_metadata(basename, {"enhance_status": "failed"})
            logger.warning("Enhancement produced no output for {}", basename)

        # Chain: align the enhanced file + start silence removal
        _start_alignment(basename)
        _start_vad(basename)

    except Exception as e:
        logger.exception("Background enhancement failed for {}", basename)
        try:
            _update_metadata(basename, {"enhance_status": "failed", "error_message": f"Enhancement: {e}"})
        except Exception:
            pass
        # Still try VAD even if enhancement failed
        _start_vad(basename)
    finally:
        with enhance_tasks_lock:
            enhance_tasks.pop(basename, None)


def _start_enhancement(basename):
    """Spawn enhancement thread if not already running for this file."""
    if not _check_enhance_available():
        return
    with enhance_tasks_lock:
        if basename in enhance_tasks:
            return
        t = threading.Thread(target=_background_enhance, args=(basename,), daemon=True)
        enhance_tasks[basename] = t
        t.start()


# --- Silence removal helpers (Silero VAD) ---

def _check_vad_available():
    """Check if torch is importable (Silero VAD needs it). Cached after first call."""
    global vad_available
    if vad_available is not None:
        return vad_available
    try:
        import torch  # noqa: F401
        vad_available = True
    except ImportError:
        vad_available = False
    return vad_available


def _load_vad_model():
    """Load (or return cached) Silero VAD model."""
    global vad_model, vad_utils
    if vad_model is not None:
        return vad_model, vad_utils
    import torch
    with vad_lock:
        if vad_model is None:
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
            vad_model = model
            vad_utils = utils
    return vad_model, vad_utils


def _run_silence_removal(wav_path, max_silence_ms=500):
    """Remove silences longer than max_silence_ms using Silero VAD.
    Short pauses (<= threshold) are kept intact for natural speech."""
    import gc
    try:
        import torch
        model, utils = _load_vad_model()
        get_speech_timestamps = utils[0]

        # Free memory before loading audio (enhancement may have left residual tensors)
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Read audio with soundfile and resample to 16kHz (avoids torchaudio dependency)
        audio_np, sr = sf.read(wav_path, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        orig_sr = sr

        # Resample to 16kHz for VAD; keep reference to original
        if sr != 16000:
            target_len = int(len(audio_np) * 16000 / sr)
            audio_16k = np.interp(
                np.linspace(0, len(audio_np), target_len, endpoint=False),
                np.arange(len(audio_np), dtype=np.float32),
                audio_np,
            ).astype(np.float32)
        else:
            audio_16k = audio_np

        wav_16k = torch.from_numpy(audio_16k)
        timestamps = get_speech_timestamps(
            wav_16k, model,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )
        # Free 16kHz copy
        del wav_16k, audio_16k
        gc.collect()

        if not timestamps:
            return None

        # Map 16kHz sample indices to original sample rate
        ratio = orig_sr / 16000

        # Merge segments, keeping silences <= max_silence_ms intact
        chunks = []
        for i, seg in enumerate(timestamps):
            start = int(seg["start"] * ratio)
            end = int(seg["end"] * ratio)

            if i > 0:
                prev_end = int(timestamps[i - 1]["end"] * ratio)
                gap_ms = ((start - prev_end) / orig_sr) * 1000

                if gap_ms <= max_silence_ms:
                    chunks.append(audio_np[prev_end:end])
                else:
                    chunks.append(audio_np[start:end])
            else:
                chunks.append(audio_np[start:end])

        if not chunks:
            return None

        cleaned = np.concatenate(chunks)
        del audio_np, chunks
        gc.collect()
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        cleaned_name = f"{basename}_cleaned.wav"
        cleaned_path = os.path.join(os.path.dirname(wav_path), cleaned_name)
        sf.write(cleaned_path, cleaned, orig_sr)
        return cleaned_name
    except Exception as e:
        logger.exception("Silence removal failed for {}", wav_path)
        return None


def _run_loudnorm(wav_path):
    """Normalize audio volume using ffmpeg loudnorm. Overwrites the file in-place."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False
    tmp_path = wav_path + ".tmp.wav"
    try:
        # Detect sample rate to avoid unnecessary resampling
        try:
            info = sf.info(wav_path)
            sr = info.samplerate
        except Exception:
            logger.debug("Could not read sample rate from {}, defaulting to 24000", wav_path)
            sr = 24000
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-i", wav_path,
             "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
             "-ar", str(sr), "-ac", "1",
             tmp_path],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, wav_path)
            return True
        else:
            stderr = result.stderr.decode(errors='replace')
            # Extract actual error (skip ffmpeg banner lines)
            err_lines = [l for l in stderr.splitlines() if l.strip() and not l.startswith(('  ', 'ffmpeg version', '(c)', 'built with', 'configuration:', 'lib'))]
            err_msg = '\n'.join(err_lines[-5:]) if err_lines else stderr[-500:]
            logger.error("ffmpeg loudnorm failed (rc={}): {}", result.returncode, err_msg)
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Loudnorm timed out for {}", wav_path)
        return False
    except Exception as e:
        logger.exception("Loudnorm error for {}", wav_path)
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _background_vad(basename, max_silence_ms=500):
    """Run silence removal in background thread, update metadata JSON when done.
    Also runs loudnorm on the cleaned file if ffmpeg is available."""
    job_dir = _tts_job_dir(basename)
    json_path = os.path.join(job_dir, basename + ".json")
    # Prefer enhanced audio if available, otherwise use original
    with open(json_path, "r") as f:
        metadata = json.load(f)

    enhanced_name = metadata.get("enhanced_filename")
    if enhanced_name and os.path.exists(os.path.join(job_dir, enhanced_name)):
        wav_path = os.path.join(job_dir, enhanced_name)
    else:
        wav_path = os.path.join(job_dir, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    # Read max_silence_ms from metadata if stored (from generate request)
    max_silence_ms = metadata.get("max_silence_ms", max_silence_ms)

    try:
        # Skip if already cleaned and file exists
        if (metadata.get("vad_status") == "ready"
                and metadata.get("cleaned_filename")
                and os.path.exists(os.path.join(job_dir, metadata["cleaned_filename"]))):
            return

        _update_metadata(basename, {"vad_status": "cleaning"})

        vad_start = time.perf_counter()
        cleaned_name = _run_silence_removal(wav_path, max_silence_ms=max_silence_ms)

        if cleaned_name:
            vad_elapsed = round(time.perf_counter() - vad_start, 3)
            # Run loudnorm on the cleaned file
            _update_metadata(basename, {"vad_status": "normalizing"})

            loudnorm_start = time.perf_counter()
            cleaned_path = os.path.join(job_dir, cleaned_name)
            if _run_loudnorm(cleaned_path):
                logger.success("Normalized  {}", cleaned_name)
            else:
                logger.warning("Loudnorm skipped for {} (ffmpeg unavailable or failed)", cleaned_name)
            loudnorm_elapsed = round(time.perf_counter() - loudnorm_start, 3)

            _update_metadata(basename, {
                "vad_status": "ready",
                "vad_time": vad_elapsed,
                "loudnorm_time": loudnorm_elapsed,
                "cleaned_filename": cleaned_name,
            })
            logger.success("Cleaned  {} | VAD {:.2f}s | Loudnorm {:.2f}s", basename, vad_elapsed, loudnorm_elapsed)
            # Chain: align the cleaned file for karaoke sync
            _start_alignment(basename)
        else:
            _update_metadata(basename, {"vad_status": "failed"})
            logger.warning("Silence removal produced no output for {}", basename)

    except Exception as e:
        logger.exception("Background silence removal failed for {}", basename)
        try:
            _update_metadata(basename, {"vad_status": "failed", "error_message": f"Silence removal: {e}"})
        except Exception:
            pass
    finally:
        with vad_tasks_lock:
            vad_tasks.pop(basename, None)


def _start_vad(basename, max_silence_ms=500):
    """Spawn silence removal thread if not already running for this file."""
    if not _check_vad_available():
        return
    with vad_tasks_lock:
        if basename in vad_tasks:
            return
        t = threading.Thread(target=_background_vad, args=(basename, max_silence_ms), daemon=True)
        vad_tasks[basename] = t
        t.start()


# --- Check if MP3 exists (returns 200 always, avoids 404 console noise) ---
@app.route("/api/generation/<filename>/mp3-check")
def check_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"exists": False})
    folder = _folder_for_file(filename)
    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(_tts_job_dir(folder), mp3_name)
    return jsonify({"exists": os.path.exists(mp3_path)})


# --- Serve cached MP3 ---
@app.route("/api/generation/<filename>/mp3")
def serve_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    folder = _folder_for_file(filename)
    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    job_dir = _tts_job_dir(folder)
    mp3_path = os.path.join(job_dir, mp3_name)
    if not os.path.exists(mp3_path):
        return jsonify({"error": "MP3 not found — convert first"}), 404
    return send_from_directory(job_dir, mp3_name, as_attachment=True)


# --- Convert WAV to MP3 with SSE progress ---
@app.route("/api/generation/<filename>/mp3-convert")
def convert_to_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    folder = _folder_for_file(filename)
    job_dir = _tts_job_dir(folder)
    wav_path = os.path.join(job_dir, filename)
    if not os.path.exists(wav_path):
        return jsonify({"error": "File not found"}), 404

    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(job_dir, mp3_name)

    # Already converted — instant done
    if os.path.exists(mp3_path):
        def _done():
            yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
        return Response(
            _done(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return jsonify({"error": "ffmpeg not found. Place ffmpeg in bin/ or install it system-wide."}), 501

    # Get total duration for progress calculation
    total_duration = 0.0
    json_path = wav_path.rsplit(".", 1)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            total_duration = json.load(f).get("duration_seconds", 0.0)
    if total_duration <= 0:
        try:
            info = sf.info(wav_path)
            total_duration = info.duration
        except Exception:
            logger.debug("Could not read duration from {} for MP3 progress", wav_path)

    def stream():
        yield f"data: {json.dumps({'phase': 'converting', 'progress': 0})}\n\n"

        proc = subprocess.Popen(
            [ffmpeg, "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2",
             "-progress", "pipe:1", "-nostats", "-y", mp3_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        try:
            last_pct = 0
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        if total_duration > 0:
                            pct = min(99, int((us / 1_000_000) / total_duration * 100))
                            if pct > last_pct:
                                last_pct = pct
                                yield f"data: {json.dumps({'phase': 'converting', 'progress': pct})}\n\n"
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line == "progress=end":
                    break

            proc.wait(timeout=30)

            if proc.returncode == 0:
                yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
            else:
                err = proc.stderr.read()[:200] if proc.stderr else "Unknown error"
                yield f"data: {json.dumps({'phase': 'error', 'message': err})}\n\n"
        except GeneratorExit:
            # Client disconnected — kill the subprocess
            proc.kill()
            proc.wait(timeout=5)
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    return Response(
        stream(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# --- Serve force-alignment audio files ---
@app.route("/generation/force-alignment/<path:filename>")
def serve_alignment_audio(filename):
    return send_from_directory(ALIGN_DIR, filename)


# --- Serve TTS audio files ---
@app.route("/generation/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TADA TTS Studio Backend")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on")
    args = parser.parse_args()

    port = args.port if args.port else find_available_port(5000)

    # Auto-detect GPU
    gpu_info = _check_gpu_info()
    if gpu_info["available"]:
        logger.info("GPU detected: {} ({} MB VRAM)", gpu_info["name"], gpu_info["vram_mb"])

    # Startup banner
    profiles = _list_voice_profiles()
    print()
    print(f"  \033[1mTADA TTS Studio\033[0m")
    print(f"  \033[92m>\033[0m http://localhost:{port}")
    print(f"  \033[90m-\033[0m Models: TADA-1B, TADA-3B-ml")
    print(f"  \033[90m-\033[0m Voice profiles: {len(profiles)}")
    print(f"  \033[90m-\033[0m Device: {_get_device()}" + (f" ({gpu_info['name']})" if gpu_info["available"] else ""))
    print()

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
