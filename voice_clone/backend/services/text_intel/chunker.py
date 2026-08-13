# backend/services/text_intel/chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChunkerConfig:
    max_chars: int = 240          # 200–250 recommended
    min_chars: int = 40           # try not to create tiny chunks
    hard_max_chars: int = 320     # absolute safety cap
    keep_quotes_together: bool = True


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SOFT_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")
_WS_RE = re.compile(r"\s+")


def _clean_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _is_inside_quotes(text: str, idx: int) -> bool:
    """
    Returns True if position idx is inside an odd number of quote marks.
    Handles both " and ' (simple heuristic).
    """
    left = text[:idx]
    dq = left.count('"')
    sq = left.count("'")
    return (dq % 2 == 1) or (sq % 2 == 1)


def _split_by_punct(text: str) -> List[str]:
    """
    First pass split by sentence punctuation; fallback to soft punctuation.
    """
    t = _clean_ws(text)
    if not t:
        return []

    parts = _SENT_SPLIT_RE.split(t)
    out: List[str] = []
    for p in parts:
        p = _clean_ws(p)
        if not p:
            continue
        # If still long, split by commas/semicolons/colons
        if len(p) > 400:
            out.extend([_clean_ws(x) for x in _SOFT_SPLIT_RE.split(p) if _clean_ws(x)])
        else:
            out.append(p)
    return out


def _best_split_index(text: str, start: int, end: int, cfg: ChunkerConfig) -> int:
    """
    Pick a split point in [start, end] preferring punctuation then spaces,
    and avoid splitting inside quotes if possible.
    Returns index where we split (end-exclusive).
    """
    window = text[start:end]

    # Candidate split chars in preference order
    candidates = []
    for i, ch in enumerate(window):
        if ch in ".!?":
            candidates.append(start + i + 1)
    for i, ch in enumerate(window):
        if ch in ",;:":
            candidates.append(start + i + 1)
    for i, ch in enumerate(window):
        if ch == " ":
            candidates.append(start + i)

    # Prefer later splits (bigger chunk) but <= end
    candidates = sorted(set(candidates), reverse=True)

    for split_idx in candidates:
        if split_idx <= start:
            continue
        if cfg.keep_quotes_together and _is_inside_quotes(text, split_idx):
            continue
        # Avoid producing too small remainder if possible
        return split_idx

    # Fallback: hard split at end
    return end


def chunk_text(text: str, cfg: ChunkerConfig = ChunkerConfig()) -> List[str]:
    """
    Rules:
      - chunk by punctuation (sentence first, then soft punct)
      - enforce max length (default ~240 chars)
      - avoid breaking inside quotes if possible
    Output: list of chunks (strings)
    """
    t = _clean_ws(text)
    if not t:
        return []

    # 1) Start with punctuation-based segments
    segs = _split_by_punct(t)

    # 2) Merge segments into chunks up to max_chars
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        b = _clean_ws(buf)
        if b:
            chunks.append(b)
        buf = ""

    for seg in segs:
        seg = _clean_ws(seg)
        if not seg:
            continue

        if not buf:
            buf = seg
        elif len(buf) + 1 + len(seg) <= cfg.max_chars:
            buf = f"{buf} {seg}"
        else:
            flush()
            buf = seg

    flush()

    # 3) Enforce hard max by splitting long chunks safely
    final_chunks: List[str] = []
    for ch in chunks:
        ch = _clean_ws(ch)
        if len(ch) <= cfg.hard_max_chars:
            final_chunks.append(ch)
            continue

        # Split repeatedly
        s = 0
        while s < len(ch):
            e = min(len(ch), s + cfg.max_chars)
            # If we're near end, take remaining
            if len(ch) - s <= cfg.hard_max_chars and len(ch) - s <= cfg.max_chars:
                part = _clean_ws(ch[s:])
                if part:
                    final_chunks.append(part)
                break

            split_idx = _best_split_index(ch, s, e, cfg)
            part = _clean_ws(ch[s:split_idx])
            if part:
                final_chunks.append(part)
            s = split_idx

    # 4) Optional: merge tiny chunks with previous if possible
    merged: List[str] = []
    for ch in final_chunks:
        if merged and len(ch) < cfg.min_chars and (len(merged[-1]) + 1 + len(ch) <= cfg.max_chars):
            merged[-1] = _clean_ws(merged[-1] + " " + ch)
        else:
            merged.append(ch)

    return merged


if __name__ == "__main__":
    sample = 'He said, "Don’t break inside quotes, please." This is a longer sentence that should be chunked properly, even if it exceeds the maximum length slightly; we will split it safely.'
    out = chunk_text(sample, ChunkerConfig(max_chars=80))
    for i, c in enumerate(out, 1):
        print(i, len(c), c)