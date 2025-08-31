import os
from typing import Tuple

# Select transcription engine from environment variable
TRANSCRIBE_ENGINE = os.getenv("TRANSCRIBE_ENGINE", "whisper").lower()

# Whisper
try:
    import whisper
except Exception:
    whisper = None

# Google Cloud Speech (optional)
try:
    from google.cloud import speech_v1p1beta1 as speech
except Exception:
    speech = None


def transcribe_whisper(wav_path: str, model_name: str = "base") -> Tuple[str, float]:
    """Transcribe audio using OpenAI Whisper."""
    if whisper is None:
        raise RuntimeError("Whisper not installed.")
    
    model = whisper.load_model(model_name)
    result = model.transcribe(wav_path)
    text = (result.get("text") or "").strip()
    
    # Whisper returns segments with timings; approximate duration
    duration = 0.0
    for seg in result.get("segments", []):
        duration += float(seg.get("end", 0) - seg.get("start", 0))
    
    return text, duration


def transcribe_google(wav_path: str, language_code: str = "en-US") -> Tuple[str, float]:
    """Transcribe audio using Google Cloud Speech-to-Text."""
    if speech is None:
        raise RuntimeError("Google Cloud Speech is not available.")
    
    client = speech.SpeechClient()
    
    with open(wav_path, "rb") as f:
        content = f.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code,
        enable_automatic_punctuation=True,
        audio_channel_count=1,
        sample_rate_hertz=16000,
    )
    
    response = client.recognize(config=config, audio=audio)
    text = " ".join([r.alternatives[0].transcript for r in response.results]).strip()
    
    # Duration not directly available; caller may compute via librosa if needed.
    return text, 0.0


def transcribe(wav_path: str) -> Tuple[str, float]:
    """Unified transcribe function that selects engine dynamically."""
    engine = TRANSCRIBE_ENGINE
    if engine == "google":
        return transcribe_google(wav_path)
    return transcribe_whisper(wav_path)
