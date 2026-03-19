from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.storage.paths import ensure_generation_dirs
from backend.services.tts_engine.xtts_v2 import synthesize, XTTSParams
from backend.services.text_intel.normalize_text import normalize_text
from backend.services.text_intel.chunker import chunk_text, ChunkerConfig
from backend.services.text_intel.prosody import apply_prosody_to_text, ProsodyConfig


@dataclass(frozen=True)
class GenerateConfig:
    chunker: ChunkerConfig = ChunkerConfig(max_chars=240)
    prosody: ProsodyConfig = ProsodyConfig()
    apply_prosody: bool = True
    apply_normalize_text: bool = True


def generate_chunks(
    job_id: str,
    text: str,
    language: str,
    speaker_wav_path: str | os.PathLike,
    params: XTTSParams = XTTSParams(),
    cfg: GenerateConfig = GenerateConfig(),
) -> List[Path]:
    """
    For each chunk:
      - call XTTS
      - save data/generations/<job_id>/chunks/chunk_###.wav
    Returns list of chunk paths.
    """
    gp = ensure_generation_dirs(job_id)
    chunks_dir = Path(gp.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    t = text or ""
    if cfg.apply_normalize_text:
        t = normalize_text(t)

    if cfg.apply_prosody:
        # Converts internal markers (like [[PAUSE_MS:200]]) into commas/periods/...
        t = apply_prosody_to_text(t, cfg.prosody)

    chunks = chunk_text(t, cfg.chunker)
    if not chunks:
        raise ValueError("No chunks produced from text.")

    out_paths: List[Path] = []
    for i, chunk in enumerate(chunks, start=1):
        out_path = chunks_dir / f"chunk_{i:03d}.wav"
        synthesize(
            text=chunk,
            language=language,
            speaker_wav_path=speaker_wav_path,
            out_path=out_path,
            params=params,
        )
        out_paths.append(out_path)

    return out_paths