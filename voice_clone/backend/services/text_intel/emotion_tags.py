from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class EmotionResult:
    text: str
    # Optional: overrides you can pass into XTTSParams later
    xtts_param_overrides: Dict[str, Any]


# Minimal preset map (tune later)
_STYLE_TO_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "neutral": {},
    "calm": {"temperature": 0.55, "top_p": 0.85},
    "energetic": {"temperature": 0.75, "top_p": 0.92},
    "storytelling": {"temperature": 0.65, "top_p": 0.90},
    "sad": {"temperature": 0.55, "top_p": 0.85},
}


def apply_emotion_style(text: str, style: str | None) -> EmotionResult:
    """
    Day-1 approach:
      - No special tokens.
      - Just returns param overrides so your TTS call can adjust.
    """
    style_key = (style or "neutral").strip().lower()
    overrides = dict(_STYLE_TO_OVERRIDES.get(style_key, {}))
    return EmotionResult(text=text, xtts_param_overrides=overrides)