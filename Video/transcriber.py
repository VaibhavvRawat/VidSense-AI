"""
transcriber.py
Converts audio to text using OpenAI Whisper (medium model).
Translates to English automatically via the 'translate' task.
"""

import whisper

# Loaded once at import time to avoid reloading on every call
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("medium")
    return _model


def speech_to_text(audio_file: str) -> str:
    """Transcribe (and translate to English) an audio file using Whisper."""
    model = _get_model()
    result = model.transcribe(audio_file, task="translate")
    return result["text"]
