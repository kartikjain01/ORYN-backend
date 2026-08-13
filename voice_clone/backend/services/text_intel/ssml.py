from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass(frozen=True)
class SSMLParseResult:
    text: str
    # pauses as tokens inserted into text so your chunker/tts can later interpret if needed
    # Example token: [[PAUSE_MS:300]]
    pauses_ms: List[int]
    had_ssml: bool


_BREAK_RE = re.compile(
    r"<\s*break\s+[^>]*time\s*=\s*['\"]\s*([0-9]+(\.[0-9]+)?)\s*(ms|s)\s*['\"][^>]*/\s*>",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"</?[^>]+>")

_EMPH_START_RE = re.compile(r"<\s*emphasis[^>]*>", re.IGNORECASE)
_EMPH_END_RE = re.compile(r"</\s*emphasis\s*>", re.IGNORECASE)


def _to_ms(val: str, unit: str) -> int:
    f = float(val)
    return int(round(f if unit.lower() == "ms" else f * 1000.0))


def ssml_to_text(ssml_or_text: str) -> SSMLParseResult:
    """
    Minimal SSML support (day-1):
      - <break time="300ms"/> -> inserts [[PAUSE_MS:300]]
      - <emphasis>text</emphasis> -> converts to plain text (keeps words)
      - strips all other tags safely
    """
    s = ssml_or_text or ""
    had_ssml = "<" in s and ">" in s

    pauses: List[int] = []

    # Convert <emphasis> tags to nothing (we keep the content)
    s = _EMPH_START_RE.sub("", s)
    s = _EMPH_END_RE.sub("", s)

    # Replace breaks with pause tokens
    def _break_sub(m: re.Match) -> str:
        ms = _to_ms(m.group(1), m.group(3))
        pauses.append(ms)
        return f" [[PAUSE_MS:{ms}]] "

    s = _BREAK_RE.sub(_break_sub, s)

    # Strip all remaining tags
    s = _TAG_RE.sub(" ", s)

    # Cleanup spacing
    s = re.sub(r"\s+", " ", s).strip()

    return SSMLParseResult(text=s, pauses_ms=pauses, had_ssml=had_ssml)