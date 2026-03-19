from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class MergeConfig:
    crossfade_ms: int = 80            # 50–120ms recommended
    loudnorm: bool = True             # loudness match across chunks
    target_i: float = -16.0           # integrated loudness (LUFS-ish)
    true_peak: float = -1.5
    lra: float = 11.0
    ffmpeg_path: str = "ffmpeg"


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n\n{p.stderr}")


def merge_chunks(
    chunk_wavs: List[str | Path],
    out_path: str | Path,
    cfg: MergeConfig = MergeConfig(),
) -> Path:
    """
    Merge multiple chunk wavs with:
      - per-chunk loudness normalization (optional)
      - sequential crossfade across all chunks
      - export final wav/mp3 based on out_path extension
    """
    if not chunk_wavs:
        raise ValueError("chunk_wavs is empty")

    chunk_paths = [str(Path(p)) for p in chunk_wavs]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg inputs
    cmd = [cfg.ffmpeg_path, "-y"]
    for p in chunk_paths:
        cmd += ["-i", p]

    # Filters:
    # 1) optional loudnorm per input: [i:a]loudnorm=... [a{i}]
    # 2) chain acrossfade: [a0][a1]acrossfade=d=... [m1]; [m1][a2]acrossfade... [m2] ...
    cf_s = max(0.001, cfg.crossfade_ms / 1000.0)

    filter_parts = []
    labels = []

    for i in range(len(chunk_paths)):
        if cfg.loudnorm:
            filter_parts.append(
                f"[{i}:a]loudnorm=I={cfg.target_i}:TP={cfg.true_peak}:LRA={cfg.lra}[a{i}]"
            )
            labels.append(f"[a{i}]")
        else:
            labels.append(f"[{i}:a]")

    # Chain acrossfades
    if len(labels) == 1:
        final_label = labels[0]
    else:
        prev = labels[0]
        for i in range(1, len(labels)):
            out_lbl = f"[m{i}]"
            filter_parts.append(f"{prev}{labels[i]}acrossfade=d={cf_s}:c1=tri:c2=tri{out_lbl}")
            prev = out_lbl
        final_label = prev

    filter_complex = ";".join(filter_parts)

    cmd += ["-filter_complex", filter_complex, "-map", final_label]

    # Output format based on extension
    ext = out_path.suffix.lower()
    if ext == ".mp3":
        cmd += ["-vn", "-codec:a", "libmp3lame", "-q:a", "2", str(out_path)]
    else:
        # default wav
        cmd += ["-vn", "-c:a", "pcm_s16le", str(out_path)]

    _run(cmd)
    return out_path