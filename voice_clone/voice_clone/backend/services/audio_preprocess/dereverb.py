# backend/services/audio_preprocess/dereverb.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DereverbConfig:
    enabled: bool = True
    tail_fade_ms: int = 250   # fade-out at end to reduce reverb tail


def simple_reduce_tail(audio: np.ndarray, sr: int, cfg: DereverbConfig = DereverbConfig()) -> np.ndarray:
    """
    Simple baseline dereverb-ish cleanup:
      - fade out last tail_fade_ms to reduce long room tail
    """
    if not cfg.enabled:
        return audio.astype(np.float32)

    x = np.asarray(audio, dtype=np.float32)
    n_fade = int((cfg.tail_fade_ms / 1000.0) * sr)
    if n_fade <= 0 or x.size < n_fade + 10:
        return x

    y = x.copy()
    fade = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
    y[-n_fade:] = y[-n_fade:] * fade
    return y.astype(np.float32)