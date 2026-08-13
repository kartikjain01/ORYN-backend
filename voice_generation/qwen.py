from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------

@dataclass(frozen=True)
class QwenParams:
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 1.1


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

_qwen_singleton: Optional[Qwen3TTSModel] = None
_lock = threading.Lock()


# ---------------------------------------------------
# LANGUAGE MAP
# ---------------------------------------------------

LANGUAGE_MAP = {
    # English
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "english": "english",

    # Hindi (temporary)
    "hi": "auto",
    "hi-in": "auto",

    # Spanish
    "es": "spanish",
    "spanish": "spanish",

    # French
    "fr": "french",
    "french": "french",

    # German
    "de": "german",
    "german": "german",

    # Italian
    "it": "italian",
    "italian": "italian",

    # Portuguese
    "pt": "portuguese",
    "portuguese": "portuguese",

    # Russian
    "ru": "russian",
    "russian": "russian",

    # Japanese
    "ja": "japanese",
    "japanese": "japanese",

    # Korean
    "ko": "korean",
    "korean": "korean",

    # Chinese
    "zh": "chinese",
    "chinese": "chinese",

    "auto": "auto",
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
# CREATE
# ---------------------------------------------------

def create(
    text: str,
    speaker: str,
    language: str,
    instruct: str = "",
    params: QwenParams = QwenParams(),
):
    model = get_qwen()

    # -----------------------------------------------
    # LANGUAGE
    # -----------------------------------------------

    language = LANGUAGE_MAP.get(
        language.lower(),
        language.lower()
    )

    print("Qwen language:", language)
    print("Qwen speaker:", speaker)

    # -----------------------------------------------
    # GENERATE
    # -----------------------------------------------

    print("Generating...")

    wavs, sample_rate = model.generate_custom_voice(
    text=text,
    speaker=speaker,
    language=language,
    instruct=instruct,
    temperature=params.temperature,
    top_k=params.top_k,
    top_p=params.top_p,
    repetition_penalty=params.repetition_penalty,
)

    audio = wavs[0]

    print("Sample rate:", sample_rate)
    print("Original audio length:", len(audio))

    # -----------------------------------------------
    # RETURN
    # -----------------------------------------------

    return audio, sample_rate
