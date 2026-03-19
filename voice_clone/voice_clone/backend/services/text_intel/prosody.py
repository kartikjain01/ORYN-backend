# backend/services/text_intel/prosody.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


# ----------------------------
# Prosody markers (internal only)
# ----------------------------
@dataclass(frozen=True)
class ProsodyMarker:
    type: str  # "pause_ms" | "emphasis"
    value: Any
    span: Optional[Tuple[int, int]] = None  # for emphasis (start,end) in text indices


@dataclass(frozen=True)
class ProsodyConfig:
    # how we convert pauses to punctuation
    pause_comma_ms: int = 200      # >= this -> comma
    pause_period_ms: int = 450     # >= this -> period
    pause_ellipsis_ms: int = 300   # >= this -> "..." (if safe)

    # emphasis conversion (light): surround word/phrase with commas or add "..."
    emphasis_strategy: str = "comma"  # "comma" | "ellipsis"
    # safety
    max_ellipsis_per_chunk: int = 2


# Marker token pattern in text, created by upstream (e.g., SSML -> [[PAUSE_MS:300]])
_PAUSE_TOKEN_RE = re.compile(r"\[\[PAUSE_MS:(\d+)\]\]")
# Emphasis tokens (optional future): [[EMPH:word]] ... [[/EMPH]]
_EMPH_OPEN_RE = re.compile(r"\[\[EMPH:(.*?)\]\]")
_EMPH_CLOSE_RE = re.compile(r"\[\[\/EMPH\]\]")


def build_prosody_plan(text: str) -> List[ProsodyMarker]:
    """
    Build a light prosody plan from internal tokens already present in text.
    For day-1:
      - reads [[PAUSE_MS:xxx]] tokens into markers
      - reads [[EMPH:...]] ... [[/EMPH]] blocks into emphasis markers (optional)
    Returns markers, but does NOT modify text.
    """
    markers: List[ProsodyMarker] = []
    for m in _PAUSE_TOKEN_RE.finditer(text):
        markers.append(ProsodyMarker(type="pause_ms", value=int(m.group(1)), span=(m.start(), m.end())))

    # Optional emphasis blocks
    # NOTE: This is a very light parser (first open..first close after it)
    idx = 0
    while True:
        mo = _EMPH_OPEN_RE.search(text, idx)
        if not mo:
            break
        mc = _EMPH_CLOSE_RE.search(text, mo.end())
        if not mc:
            break
        # emphasis span is inside the tokens (content between)
        markers.append(ProsodyMarker(type="emphasis", value=mo.group(1), span=(mo.start(), mc.end())))
        idx = mc.end()

    return markers


def _is_safe_for_ellipsis(left: str, right: str) -> bool:
    """
    Avoid inserting ellipsis inside abbreviations or mid-number.
    """
    if not left:
        return False
    if left[-1].isdigit() and right[:1].isdigit():
        return False
    if left.endswith(("Mr", "Dr", "Ms", "Mrs")):
        return False
    return True


def apply_prosody_to_text(text: str, cfg: ProsodyConfig = ProsodyConfig()) -> str:
    """
    Convert internal prosody tokens -> punctuation tweaks.
    IMPORTANT:
      - We do NOT pass tokens raw to XTTS.
      - We remove tokens and replace with punctuation or spacing.
    """
    s = text

    # 1) Replace pause tokens with punctuation
    ellipsis_count = 0

    def pause_sub(m: re.Match) -> str:
        nonlocal ellipsis_count
        ms = int(m.group(1))

        # Decide punctuation
        if ms >= cfg.pause_period_ms:
            return ". "
        if ms >= cfg.pause_ellipsis_ms and ellipsis_count < cfg.max_ellipsis_per_chunk:
            # only if safe-ish
            ellipsis_count += 1
            return "... "
        if ms >= cfg.pause_comma_ms:
            return ", "
        return " "  # tiny pause -> just space

    s = _PAUSE_TOKEN_RE.sub(pause_sub, s)

    # 2) Emphasis (optional): convert emphasis tokens into punctuation shaping
    # Strategy: remove tokens and add commas or ellipsis around emphasized phrase label (value)
    # Note: This is intentionally simple; we do not rewrite meaning.
    if cfg.emphasis_strategy == "comma":
        # [[EMPH:word]] ... [[/EMPH]] -> ", ... ,"
        s = _EMPH_OPEN_RE.sub(", ", s)
        s = _EMPH_CLOSE_RE.sub(", ", s)
    else:
        # ellipsis style
        if ellipsis_count < cfg.max_ellipsis_per_chunk:
            s = _EMPH_OPEN_RE.sub("... ", s)
            s = _EMPH_CLOSE_RE.sub(" ... ", s)
            ellipsis_count = min(cfg.max_ellipsis_per_chunk, ellipsis_count + 2)
        else:
            s = _EMPH_OPEN_RE.sub(" ", s)
            s = _EMPH_CLOSE_RE.sub(" ", s)

    # 3) Cleanup spacing/punctuation duplicates
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)        # no space before punct
    s = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", s)  # ensure space after punct
    s = re.sub(r"\.{4,}", "...", s)               # collapse long dots
    s = re.sub(r",\s*,", ", ", s)                 # collapse double commas
    s = re.sub(r"\.\s*\.", ".", s)                # collapse double periods

    return s


if __name__ == "__main__":
    demo = 'Hello [[PAUSE_MS:220]] world [[PAUSE_MS:520]] This is a test [[PAUSE_MS:320]] okay?'
    plan = build_prosody_plan(demo)
    print("markers:", plan)
    print("out:", apply_prosody_to_text(demo))