from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhonemeConfig:
    enabled: bool = False  # keep OFF by default (XTTS usually expects text)
    backend: str = "phonemizer"  # if installed later


def text_to_phonemes(text: str, language: str, cfg: PhonemeConfig = PhonemeConfig()) -> str:
    """
    Day-1:
      - OFF by default.
      - If enabled and phonemizer installed, returns phonemes.
      - Otherwise returns original text.
    NOTE: XTTS v2 typically takes text; phonemes require separate handling.
    """
    if not cfg.enabled:
        return text or ""

    if cfg.backend == "phonemizer":
        try:
            from phonemizer import phonemize  # type: ignore
            # language mapping can be extended later
            lang = "en-us" if language == "en" else "hi"
            return phonemize(text, language=lang, strip=True, preserve_punctuation=True, njobs=1)
        except Exception:
            return text or ""

    return text or ""