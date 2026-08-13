from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------
# QWEN PARAMETERS
# ---------------------------------------------------

@dataclass(frozen=True)
class QwenParams:
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 1.1

    # Since your current API doesn't provide reference_text,
    # use speaker embedding only (XTTS-like workflow)
    x_vector_only_mode: bool = True


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

_qwen_singleton: Optional[Qwen3TTSModel] = None

_lock = threading.Lock()

# ---------------------------------------------------
# LANGUAGE MAP
# ---------------------------------------------------

LANGUAGE_MAP = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}

# ---------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------

def get_qwen() -> Qwen3TTSModel:

    global _qwen_singleton

    if _qwen_singleton is not None:
        return _qwen_singleton

    with _lock:

        if _qwen_singleton is None:

            print("\nLoading Qwen3-TTS...\n")

            _qwen_singleton = Qwen3TTSModel.from_pretrained(
                _MODEL_NAME
            )

            print("\nQwen3-TTS loaded successfully.\n")

    return _qwen_singleton


# ---------------------------------------------------
# SYNTHESIS
# ---------------------------------------------------

def synthesize(
    text: str,
    language: str,
    speaker_wav_path: str | os.PathLike,
    out_path: str | os.PathLike,
    params: QwenParams = QwenParams(),
) -> Path:

    if not text.strip():
        raise ValueError("Text is empty.")

    speaker_wav_path = Path(speaker_wav_path)

    if not speaker_wav_path.exists():
        raise FileNotFoundError(
            f"Reference audio not found: {speaker_wav_path}"
        )

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = get_qwen()

    language = LANGUAGE_MAP.get(language.lower(), language.lower())

    wavs, sample_rate = model.generate_voice_clone(

        text=text,

        language=language,

        ref_audio=str(speaker_wav_path),

        x_vector_only_mode=params.x_vector_only_mode,

        temperature=params.temperature,

        top_k=params.top_k,

        top_p=params.top_p,

        repetition_penalty=params.repetition_penalty,
    )

    sf.write(
        str(out_path),
        wavs[0],
        sample_rate,
    )

    return out_path
