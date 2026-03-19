# backend/services/audio_preprocess/vad.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None

try:
    from scipy.io import wavfile  # pip install scipy
except Exception:
    wavfile = None


# ----------------------------
# IO
# ----------------------------
def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    p = str(Path(path))
    if sf is not None:
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x[:, 0]
        return x.astype(np.float32), int(sr)

    if wavfile is None:
        raise RuntimeError("Need either 'soundfile' or 'scipy' installed to read WAVs.")

    sr, x = wavfile.read(p)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    return x, int(sr)


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(audio, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)

    if sf is not None:
        sf.write(str(out), y, sr, subtype="PCM_16")
        return

    if wavfile is None:
        raise RuntimeError("Need either 'soundfile' or 'scipy' installed to write WAVs.")

    wavfile.write(str(out), sr, (y * 32767.0).astype(np.int16))


# ----------------------------
# VAD (energy-based, lightweight)
# ----------------------------
@dataclass(frozen=True)
class VADConfig:
    sr: int = 24000
    frame_ms: int = 30          # analysis window
    hop_ms: int = 10            # step
    threshold_ratio: float = 0.35  # threshold = median + ratio*(p95 - median)
    min_speech_ms: int = 400
    min_silence_ms: int = 200
    pad_ms: int = 120

    # reference selection
    target_min_ref_s: float = 20.0
    target_max_ref_s: float = 30.0
    min_segment_s: float = 0.6


def _frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        pad = frame - len(x)
        x = np.pad(x, (0, pad), mode="constant")
    n = 1 + (len(x) - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = i * hop
        chunk = x[start : start + frame]
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-12)
    return rms


def _mask_to_segments(mask: np.ndarray, hop: int, frame: int, cfg: VADConfig) -> List[Tuple[int, int]]:
    """
    Convert speech/non-speech mask (per hop) into sample-index segments.
    Applies min speech length and merges small gaps.
    """
    min_speech = int((cfg.min_speech_ms / 1000.0) * cfg.sr)
    min_sil = int((cfg.min_silence_ms / 1000.0) * cfg.sr)

    # mask indices are hop-based; convert to sample positions (start of hop)
    speech = np.where(mask)[0]
    if speech.size == 0:
        return []

    # Build raw ranges in hop-index space
    ranges = []
    start = speech[0]
    prev = speech[0]
    for idx in speech[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev))
            start = idx
            prev = idx
    ranges.append((start, prev))

    # Convert to sample ranges
    segs = []
    for a, b in ranges:
        s = a * hop
        e = b * hop + frame
        segs.append((s, e))

    # Merge gaps < min_sil
    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s - cur_e <= min_sil:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Filter out short segments
    final = [(s, e) for (s, e) in merged if (e - s) >= min_speech]
    return final


def vad_segment(
    clean_stage1_wav: str | Path,
    out_dir: str | Path,
    cfg: VADConfig = VADConfig(),
) -> dict:
    """
    Input:  clean_stage1.wav
    Output:
      - segments/seg_001.wav, ...
      - reference_clean.wav (best 20–30s)
    """
    out_dir = Path(out_dir)
    segments_dir = out_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    x, sr = load_wav(clean_stage1_wav)
    if sr != cfg.sr:
        raise RuntimeError(f"Expected sr={cfg.sr}, got sr={sr}. Run stage1 resample first.")

    frame = int(cfg.frame_ms * sr / 1000)
    hop = int(cfg.hop_ms * sr / 1000)

    rms = _frame_rms(x, frame=frame, hop=hop)
    med = float(np.median(rms))
    p95 = float(np.percentile(rms, 95))
    thr = med + cfg.threshold_ratio * max(1e-12, (p95 - med))

    mask = rms >= thr

    # Pad speech regions slightly (in hop units) for more natural cut
    pad_hops = int((cfg.pad_ms / 1000.0) * sr / hop)
    if pad_hops > 0 and mask.any():
        m = mask.copy()
        idxs = np.where(mask)[0]
        for i in idxs:
            a = max(0, i - pad_hops)
            b = min(len(m), i + pad_hops + 1)
            m[a:b] = True
        mask = m

    segs = _mask_to_segments(mask, hop=hop, frame=frame, cfg=cfg)

    # Save segments + compute scores
    seg_infos = []
    for i, (s, e) in enumerate(segs, start=1):
        seg_audio = x[s:e]
        dur = (e - s) / sr
        if dur < cfg.min_segment_s:
            continue
        seg_rms = float(np.sqrt(np.mean(seg_audio * seg_audio) + 1e-12))
        name = f"seg_{i:03d}.wav"
        path = segments_dir / name
        save_wav(path, seg_audio, sr)
        seg_infos.append(
            {
                "name": name,
                "path": str(path),
                "start_s": s / sr,
                "end_s": e / sr,
                "dur_s": dur,
                "rms": seg_rms,
            }
        )

    # Build reference_clean.wav (best 20–30s)
    ref_path = out_dir / "reference_clean.wav"
    if not seg_infos:
        # fallback: just use first 30s of stage1 (still better than nothing)
        max_samples = int(cfg.target_max_ref_s * sr)
        save_wav(ref_path, x[:max_samples], sr)
        return {
            "segments_dir": str(segments_dir),
            "segments": [],
            "reference_clean": str(ref_path),
            "note": "No VAD segments found; used first ~30s fallback.",
        }

    # Prefer a single long, strong segment if available
    seg_infos_sorted = sorted(seg_infos, key=lambda d: (d["dur_s"], d["rms"]), reverse=True)
    best_single = next((d for d in seg_infos_sorted if d["dur_s"] >= cfg.target_min_ref_s), None)

    if best_single is not None:
        # take up to 30s from it
        start = int(best_single["start_s"] * sr)
        end = int(best_single["end_s"] * sr)
        clip = x[start:end]
        clip = clip[: int(cfg.target_max_ref_s * sr)]
        save_wav(ref_path, clip, sr)
    else:
        # concatenate top segments by rms until reaching 20–30s
        seg_by_rms = sorted(seg_infos, key=lambda d: d["rms"], reverse=True)
        target_max = int(cfg.target_max_ref_s * sr)
        target_min = int(cfg.target_min_ref_s * sr)

        parts = []
        total = 0
        for d in seg_by_rms:
            s = int(d["start_s"] * sr)
            e = int(d["end_s"] * sr)
            part = x[s:e]
            if part.size == 0:
                continue
            if total + part.size > target_max:
                part = part[: max(0, target_max - total)]
            if part.size > 0:
                parts.append(part)
                total += part.size
            if total >= target_min:
                break

        ref_audio = np.concatenate(parts) if parts else x[:target_max]
        save_wav(ref_path, ref_audio, sr)

    return {
        "segments_dir": str(segments_dir),
        "segments": seg_infos,
        "reference_clean": str(ref_path),
        "threshold": thr,
        "median_rms": med,
        "p95_rms": p95,
    }


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to clean_stage1.wav")
    ap.add_argument("--out_dir", required=True, help="Output directory (will create segments/ and reference_clean.wav)")
    ap.add_argument("--sr", type=int, default=24000)
    args = ap.parse_args()

    cfg = VADConfig(sr=args.sr)
    result = vad_segment(args.input, args.out_dir, cfg)
    print(result["reference_clean"])
    print(result["segments_dir"])


if __name__ == "__main__":
    _cli()