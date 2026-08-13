from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class MergeConfig:
    # Reduced for faster processing while
    # keeping premium smooth transitions
    crossfade_ms: int = 35

    # KEEP enabled for studio-quality output
    loudnorm: bool = True

    target_i: float = -16.0
    true_peak: float = -1.5
    lra: float = 11.0

    ffmpeg_path: str = "ffmpeg"


def _run(cmd: List[str]) -> None:

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if p.returncode != 0:

        raise RuntimeError(
            f"ffmpeg failed:\n{' '.join(cmd)}\n\n{p.stderr}"
        )


def merge_chunks(
    chunk_wavs: List[str | Path],
    out_path: str | Path,
    cfg: MergeConfig = MergeConfig(),
) -> Path:
    """
    Optimized premium-quality merge pipeline.

    FLOW:
      1) Merge/crossfade chunks
      2) Apply ONE final loudnorm pass
      3) Export mastered audio

    Benefits:
      - Much faster than per-chunk loudnorm
      - Preserves cinematic quality
      - Lower CPU usage
      - Better loudness consistency
    """

    if not chunk_wavs:
        raise ValueError("chunk_wavs is empty")

    chunk_paths = [str(Path(p)) for p in chunk_wavs]

    out_path = Path(out_path)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # TEMP MERGED FILE
    # ----------------------------------------

    temp_merged = (
        out_path.parent
        / f"{out_path.stem}_merged.wav"
    )

    # ----------------------------------------
    # BUILD FFMPEG COMMAND
    # ----------------------------------------

    cmd = [cfg.ffmpeg_path, "-y"]

    for p in chunk_paths:
        cmd += ["-i", p]

    cf_s = max(
        0.001,
        cfg.crossfade_ms / 1000.0
    )

    filter_parts = []

    labels = []

    # NO per-chunk loudnorm anymore
    for i in range(len(chunk_paths)):
        labels.append(f"[{i}:a]")

    # ----------------------------------------
    # SINGLE CHUNK CASE
    # ----------------------------------------

    if len(labels) == 1:

        cmd += [
            "-map",
            "0:a",
        ]

    # ----------------------------------------
    # MULTI-CHUNK CROSSFADE CASE
    # ----------------------------------------

    else:

        prev = labels[0]

        for i in range(1, len(labels)):

            out_lbl = f"[m{i}]"

            filter_parts.append(
                f"{prev}{labels[i]}"
                f"acrossfade="
                f"d={cf_s}:"
                f"c1=tri:"
                f"c2=tri"
                f"{out_lbl}"
            )

            prev = out_lbl

        final_label = prev

        filter_complex = ";".join(filter_parts)

        cmd += [
            "-filter_complex",
            filter_complex,
            "-map",
            final_label,
        ]

    # ----------------------------------------
    # TEMP OUTPUT
    # ----------------------------------------

    cmd += [
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(temp_merged),
    ]

    # ----------------------------------------
    # MERGE PASS
    # ----------------------------------------

    _run(cmd)

    # ----------------------------------------
    # FINAL SINGLE-PASS LOUDNORM
    # ----------------------------------------

    if cfg.loudnorm:

        loudnorm_cmd = [
            cfg.ffmpeg_path,
            "-y",

            "-i",
            str(temp_merged),

            "-af",
            (
                f"loudnorm="
                f"I={cfg.target_i}:"
                f"TP={cfg.true_peak}:"
                f"LRA={cfg.lra}"
            ),

            "-vn",
        ]

        ext = out_path.suffix.lower()

        # MP3 OUTPUT
        if ext == ".mp3":

            loudnorm_cmd += [
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(out_path),
            ]

        # WAV OUTPUT
        else:

            loudnorm_cmd += [
                "-c:a",
                "pcm_s16le",
                str(out_path),
            ]

        _run(loudnorm_cmd)

        # cleanup
        temp_merged.unlink(
            missing_ok=True
        )

    else:

        temp_merged.rename(out_path)

    return out_path
