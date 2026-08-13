# backend/services/tts_engine/xtts_v2.py

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from TTS.api import TTS


# ---------------------------------------------------
# TORCH SAFE GLOBALS
# ---------------------------------------------------

def _torch_safe_globals_for_xtts() -> None:
    """
    PyTorch 2.6+ compatibility fix
    """

    try:

        from torch.serialization import add_safe_globals

        from TTS.tts.configs.xtts_config import (
            XttsConfig,
        )

        from TTS.tts.models.xtts import (
            XttsAudioConfig,
            XttsArgs,
        )

        from TTS.config.shared_configs import (
            BaseDatasetConfig,
        )

        add_safe_globals([
            XttsConfig,
            XttsAudioConfig,
            XttsArgs,
            BaseDatasetConfig,
        ])

    except Exception:
        pass


# ---------------------------------------------------
# CPU OPTIMIZATION
# ---------------------------------------------------

# Use all CPU cores
torch.set_num_threads(
    os.cpu_count()
)

# Better CPU backend acceleration
torch.backends.mkldnn.enabled = True

# Disable gradients globally
torch.set_grad_enabled(False)


# ---------------------------------------------------
# XTTS PARAMS
# ---------------------------------------------------

@dataclass(frozen=True)
class XTTSParams:

    # Slightly optimized
    # while preserving studio quality

    temperature: float = 0.6

    top_k: int = 30

    top_p: float = 0.85

    repetition_penalty: float = 2.0

    # Reduced from 20
    # faster conditioning
    gpt_cond_len: int = 12


# ---------------------------------------------------
# MODEL SINGLETON
# ---------------------------------------------------

_MODEL_ID = (
    "tts_models/multilingual/multi-dataset/xtts_v2"
)

_tts_singleton: Optional[TTS] = None

_lock = threading.Lock()


# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

def _get_device() -> str:

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ---------------------------------------------------
# LOAD XTTS ONCE
# ---------------------------------------------------

def get_xtts() -> TTS:

    global _tts_singleton

    if _tts_singleton is not None:
        return _tts_singleton

    with _lock:

        if _tts_singleton is None:

            print(
                "\nLoading XTTS model...\n"
            )

            _torch_safe_globals_for_xtts()

            device = _get_device()

            # ----------------------------------------
            # OPTIMIZED MODEL LOAD
            # ----------------------------------------

            tts = TTS(
                model_name=_MODEL_ID,
                progress_bar=False,
                gpu=torch.cuda.is_available(),
            )

            tts.to(device)

            # ----------------------------------------
            # ENABLE DEEPSPEED
            # ----------------------------------------

            try:

                tts.synthesizer.tts_model.use_deepspeed = True

                print(
                    "\nDeepSpeed enabled.\n"
                )

            except Exception as e:

                print(
                    "\nDeepSpeed not available:",
                    e,
                    "\n"
                )

            _tts_singleton = tts

            print(
                f"\nXTTS loaded on: {device}\n"
            )

    return _tts_singleton


# ---------------------------------------------------
# SYNTHESIS
# ---------------------------------------------------

def synthesize(
    text: str,
    language: str,
    speaker_wav_path: str | os.PathLike,
    out_path: str | os.PathLike,
    params: XTTSParams = XTTSParams(),
) -> Path:

    if not text or not text.strip():

        raise ValueError(
            "text is empty"
        )

    speaker_wav_path = Path(
        speaker_wav_path
    )

    if not speaker_wav_path.exists():

        raise FileNotFoundError(
            f"speaker_wav_path not found: "
            f"{speaker_wav_path}"
        )

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tts = get_xtts()

    # ----------------------------------------
    # FAST INFERENCE
    # ----------------------------------------

    with torch.inference_mode():

        tts.tts_to_file(

            text=text,

            speaker_wav=str(
                speaker_wav_path
            ),

            language=language,

            file_path=str(
                out_path
            ),

            # IMPORTANT
            # prevents internal resplitting
            split_sentences=False,

            temperature=float(
                params.temperature
            ),

            top_k=int(
                params.top_k
            ),

            top_p=float(
                params.top_p
            ),

            repetition_penalty=float(
                params.repetition_penalty
            ),

            gpt_cond_len=int(
                params.gpt_cond_len
            ),
        )

    return out_path


# ---------------------------------------------------
# QUICK TEST
# ---------------------------------------------------

if __name__ == "__main__":

    ref = (
        "data/voices/test_voice/"
        "clean/reference_clean.wav"
    )

    out = (
        "data/generations/"
        "test_job/out.wav"
    )

    txt = (
        "Hello! This is a short test "
        "using optimized XTTS v2."
    )

    result = synthesize(
        txt,
        "en",
        ref,
        out,
    )

    print(
        "Generated:",
        result
    )
