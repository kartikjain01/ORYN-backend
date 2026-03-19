from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class NLPResult:
    language: str  # "en" or "hi" (day-1)
    sentences: List[str]


_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def detect_language_simple(text: str) -> str:
    """
    Day-1 heuristic:
      - If Devanagari exists -> hi
      - else -> en
    """
    return "hi" if _DEVANAGARI_RE.search(text or "") else "en"


def split_sentences_simple(text: str) -> List[str]:
    s = (text or "").strip()
    if not s:
        return []
    parts = _SENT_SPLIT_RE.split(s)
    return [p.strip() for p in parts if p.strip()]


def run_nlp_light(text: str) -> NLPResult:
    lang = detect_language_simple(text)
    sents = split_sentences_simple(text)
    return NLPResult(language=lang, sentences=sents)