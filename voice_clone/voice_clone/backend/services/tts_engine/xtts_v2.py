# backend/services/tts_engine/xtts_v2.py
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from TTS.api import TTS


# ----------------------------
# Torch safe-globals (PyTorch 2.6+ fix)
# ----------------------------
def _torch_safe_globals_for_xtts() -> None:
    """
    PyTorch 2.6+ uses torch.load(weights_only=True) by default.
    Coqui XTTS checkpoints include pickled config objects.
    We allowlist the required TTS/XTTS config classes so loading works.
    """
    try:
        from torch.serialization import add_safe_globals

        # XTTS-specific config objects
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs

        # Shared config objects used inside checkpoints
        from TTS.config.shared_configs import BaseDatasetConfig

        add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
    except Exception:
        pass


# ----------------------------
# Config / Params
# ----------------------------
@dataclass(frozen=True)
class XTTSParams:
    temperature: float = 0.65
    top_k: int = 50
    top_p: float = 0.90
    repetition_penalty: float = 2.0
    gpt_cond_len: int = 20


# ----------------------------
# Singleton loader
# ----------------------------
_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
_tts_singleton: Optional[TTS] = None
_lock = threading.Lock()


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_xtts() -> TTS:
    global _tts_singleton

    if _tts_singleton is not None:
        return _tts_singleton

    with _lock:
        if _tts_singleton is None:
            _torch_safe_globals_for_xtts()
            device = _get_device()
            _tts_singleton = TTS(_MODEL_ID).to(device)

    return _tts_singleton


# ----------------------------
# Synthesis
# ----------------------------
def synthesize(
    text: str,
    language: str,
    speaker_wav_path: str | os.PathLike,
    out_path: str | os.PathLike,
    params: XTTSParams = XTTSParams(),
) -> Path:

    if not text or not text.strip():
        raise ValueError("text is empty")

    speaker_wav_path = Path(speaker_wav_path)
    if not speaker_wav_path.exists():
        raise FileNotFoundError(f"speaker_wav_path not found: {speaker_wav_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tts = get_xtts()

    tts.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav_path),
        language=language,
        file_path=str(out_path),
        temperature=float(params.temperature),
        top_k=int(params.top_k),
        top_p=float(params.top_p),
        repetition_penalty=float(params.repetition_penalty),
        gpt_cond_len=int(params.gpt_cond_len),
    )

    return out_path


# ----------------------------
# Quick CLI test
# ----------------------------
if __name__ == "__main__":
    ref = "data/voices/test_voice/clean/reference_clean.wav"
    out = "data/generations/test_job/out.wav"
    txt = "Hello! This is a short test using XTTS v2."

    result = synthesize(txt, "en", ref, out)
    print("Generated:", result)