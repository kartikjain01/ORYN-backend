from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from fish_speech.utils.schema import (
    ServeTTSRequest,
    ServeReferenceAudio,
)
from fish_speech.inference_engine import (
    TTSInferenceEngine,
)

# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------

@dataclass(frozen=True)
class FishParams:
    chunk_length: int = 200
    max_new_tokens: int = 1024
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    temperature: float = 0.8


# ---------------------------------------------------
# SINGLETON
# ---------------------------------------------------

_fish_engine: Optional[TTSInferenceEngine] = None
_lock = threading.Lock()


# ---------------------------------------------------
# LOAD ENGINE ONCE
# ---------------------------------------------------

def get_fish() -> TTSInferenceEngine:

    global _fish_engine

    if _fish_engine is not None:
        return _fish_engine

    with _lock:

        if _fish_engine is None:

            print("\nLoading Fish Speech...\n")

            _fish_engine = TTSInferenceEngine()

            print("\nFish Speech loaded successfully.\n")

    return _fish_engine


# ---------------------------------------------------
# SYNTHESIS
# ---------------------------------------------------

def synthesize(
    text: str,
    language: str,
    speaker_wav_path: str | os.PathLike,
    out_path: str | os.PathLike,
    params: FishParams = FishParams(),
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

    engine = get_fish()

    with open(speaker_wav_path, "rb") as f:
        audio_bytes = f.read()

    reference = ServeReferenceAudio(
        audio=audio_bytes,
        text=""
    )

    req = ServeTTSRequest(
        text=text,
        references=[reference],
        chunk_length=params.chunk_length,
        max_new_tokens=params.max_new_tokens,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
        temperature=params.temperature,
        streaming=False,
    )

    final_audio = None
    sample_rate = None

    for result in engine.inference(req):

        if result.code == "error":
            raise RuntimeError(str(result.error))

        if result.code == "final":
            sample_rate, final_audio = result.audio

    if final_audio is None:
        raise RuntimeError(
            "Fish Speech did not return audio."
        )

    sf.write(
        str(out_path),
        np.asarray(final_audio),
        sample_rate,
    )

    return out_path
