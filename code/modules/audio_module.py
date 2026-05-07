"""Audio analysis module.

Detects two events from the video's audio track:
1. Buzzer — the scoring apparatus beep (sustained high-energy tone).
2. Allez  — the referee's start command (speech recognition).

Both are returned as timestamps in seconds, which the caller converts to frames.
"""

import logging
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────
BUZZER_MIN_DURATION_S  = 0.15   # buzzer must last at least this long
BUZZER_ENERGY_SIGMA    = 2.5    # how many std-devs above mean to count as loud
BUZZER_HOP_S           = 0.01   # RMS window hop in seconds


class WordStamp:
    """A single recognised word with its start time."""
    def __init__(self, word: str, start: float, end: float):
        self.word  = word
        self.start = start   # seconds
        self.end   = end     # seconds

    def __repr__(self):
        return f"[{self.start:.2f}s] {self.word}"


class AudioAnalysisResult:
    def __init__(self):
        self.buzzer_time_s: float | None    = None   # first buzzer onset in seconds
        self.allez_time_s:  float | None    = None   # "allez" word onset in seconds
        self.transcript:    str             = ""
        self.words:         list[WordStamp] = []     # every recognised word with timestamp
        self.language:      str             = ""     # detected language code


def analyse(video_path: str | Path) -> AudioAnalysisResult:
    """Extract audio from video and detect buzzer + Allez.

    Returns AudioAnalysisResult with timestamps in seconds.
    """
    result = AudioAnalysisResult()
    video_path = Path(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        _extract_audio(video_path, wav_path)
        result.buzzer_time_s = _detect_buzzer(wav_path)
        (result.allez_time_s,
         result.transcript,
         result.words,
         result.language) = _detect_speech(wav_path)
    except Exception as e:
        logger.warning("Audio analysis failed: %s", e)
    finally:
        if wav_path.exists():
            wav_path.unlink()

    return result


# ── Audio extraction ──────────────────────────────────────────────────────

def _extract_audio(video_path: Path, wav_path: Path) -> None:
    """Use ffmpeg to extract mono 16kHz audio (whisper-compatible)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz sample rate
        "-vn",               # no video
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    logger.debug("Audio extracted to %s", wav_path)


# ── Buzzer detection ──────────────────────────────────────────────────────

def _detect_buzzer(wav_path: Path) -> float | None:
    """Detect the first sustained loud event (buzzer) in the audio.

    Strategy:
    - Compute short-time RMS energy.
    - Find frames above mean + BUZZER_ENERGY_SIGMA * std.
    - Require the high-energy region to be sustained for BUZZER_MIN_DURATION_S.
    - Return onset time of the first qualifying region.
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed — buzzer detection skipped.")
        return None

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    hop  = max(1, int(sr * BUZZER_HOP_S))
    rms  = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    threshold = rms.mean() + BUZZER_ENERGY_SIGMA * rms.std()
    loud = rms > threshold

    # Find first sustained loud region
    min_frames = max(1, int(BUZZER_MIN_DURATION_S / BUZZER_HOP_S))
    count = 0
    onset_frame = None
    for i, is_loud in enumerate(loud):
        if is_loud:
            if count == 0:
                onset_frame = i
            count += 1
            if count >= min_frames:
                t = float(times[onset_frame])
                logger.info("Buzzer detected at %.3fs", t)
                return t
        else:
            count = 0
            onset_frame = None

    logger.warning("Buzzer not detected in audio.")
    return None


# ── Allez detection ───────────────────────────────────────────────────────

_ALLEZ_WORDS = {"allez", "allee", "alez", "ale", "play", "プレイ", "aller"}

def _detect_speech(wav_path: Path) -> tuple[float | None, str, list[WordStamp], str]:
    """Use Whisper to transcribe all speech and find 'Allez'.

    Tries faster-whisper first, falls back to openai-whisper.
    Returns (allez_time_s, full_transcript, word_list, language).
    Language is auto-detected; supports English, French, and Chinese.
    """
    try:
        return _speech_faster_whisper(wav_path)
    except ImportError:
        pass
    try:
        return _speech_openai_whisper(wav_path)
    except ImportError:
        logger.warning("Neither faster-whisper nor openai-whisper installed — speech skipped.")
        return None, "", [], ""


def _speech_faster_whisper(wav_path: Path) -> tuple[float | None, str, list[WordStamp], str]:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(wav_path),
        word_timestamps=True,
        language=None,   # auto-detect: en / fr / zh supported
    )
    language = getattr(info, "language", "")

    words: list[WordStamp] = []
    full_text: list[str]   = []
    allez_time: float | None = None

    for seg in segments:
        full_text.append(seg.text.strip())
        if seg.words:
            for w in seg.words:
                ws = WordStamp(w.word.strip(), float(w.start), float(w.end))
                words.append(ws)
                if w.word.strip().lower().rstrip("!") in _ALLEZ_WORDS and allez_time is None:
                    allez_time = ws.start
                    logger.info("Allez detected at %.3fs (word: '%s')", allez_time, ws.word)

    transcript = " ".join(full_text)
    if allez_time is None:
        logger.warning("'Allez' not found. Transcript: %s", transcript)
    return allez_time, transcript, words, language


def _speech_openai_whisper(wav_path: Path) -> tuple[float | None, str, list[WordStamp], str]:
    import whisper
    model  = whisper.load_model("small")
    result = model.transcribe(str(wav_path), word_timestamps=True, language=None)

    language   = result.get("language", "")
    full_text  = result.get("text", "")
    words: list[WordStamp] = []
    allez_time: float | None = None

    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            ws = WordStamp(w["word"].strip(), float(w["start"]), float(w["end"]))
            words.append(ws)
            if w["word"].strip().lower().rstrip("!") in _ALLEZ_WORDS and allez_time is None:
                allez_time = ws.start
                logger.info("Allez detected at %.3fs (word: '%s')", allez_time, ws.word)

    if allez_time is None:
        logger.warning("'Allez' not found. Transcript: %s", full_text)
    return allez_time, full_text, words, language
