# backend/services/audio_preprocess/denoise.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from df.enhance import enhance, init_df  # DeepFilterNet
except Exception:
    enhance = None
    init_df = None


@dataclass(frozen=True)
class DenoiseConfig:
    enabled: bool = True


def denoise_deepfilternet(audio: np.ndarray, sr: int, cfg: DenoiseConfig = DenoiseConfig()) -> np.ndarray:
    """
    Uses DeepFilterNet to reduce noise.
    Expects float32 mono audio in [-1, 1].
    """
    if not cfg.enabled:
        return audio.astype(np.float32)

    if enhance is None or init_df is None:
        raise RuntimeError(
            "DeepFilterNet is not installed. Install with: pip install deepfilternet"
        )

    # init model + state
    model, df_state, _ = init_df()

    # DeepFilterNet enhance expects numpy float array
    y = enhance(model, df_state, audio.astype(np.float32), sr)
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    return y