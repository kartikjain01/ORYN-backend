# backend/services/audio_preprocess/quality.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
# IO helpers
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


def _safe_float(x: float) -> float:
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return float(x)


# ----------------------------
# Metrics
# ----------------------------
def clipping_percent(audio: np.ndarray, threshold: float = 0.999) -> float:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return 0.0
    clipped = np.sum(np.abs(x) >= threshold)
    return float(100.0 * clipped / x.size)


def duration_seconds(num_samples: int, sr: int) -> float:
    return float(num_samples) / float(sr) if sr > 0 else 0.0


def speech_ratio_from_segments(
    clean_stage1_wav: str | Path,
    segments_dir: str | Path,
) -> tuple[float, float, int]:
    """
    Returns: (speech_ratio, speech_seconds, num_segments)
    Speech seconds computed by summing durations of segment wavs in segments_dir.
    """
    x, sr = load_wav(clean_stage1_wav)
    total_s = duration_seconds(len(x), sr)

    seg_dir = Path(segments_dir)
    seg_files = sorted(seg_dir.glob("seg_*.wav"))
    speech_s = 0.0
    for f in seg_files:
        sx, ssr = load_wav(f)
        # allow sr mismatch but compute using file sr
        speech_s += duration_seconds(len(sx), ssr)

    ratio = (speech_s / total_s) if total_s > 1e-9 else 0.0
    return _safe_float(ratio), _safe_float(speech_s), len(seg_files)


def rough_snr_db(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    hop_ms: int = 10,
) -> float:
    """
    Very rough SNR estimate:
      - compute frame RMS
      - noise floor ~ 10th percentile RMS
      - speech level ~ 90th percentile RMS
      - snr_db = 20*log10(speech/noise)
    Not a true SNR, but good enough for intake ranking.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0 or sr <= 0:
        return 0.0

    frame = int(frame_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    if frame <= 0 or hop <= 0:
        return 0.0

    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)), mode="constant")

    n = 1 + (len(x) - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        c = x[s : s + frame]
        rms[i] = np.sqrt(np.mean(c * c) + 1e-12)

    noise = float(np.percentile(rms, 10))
    speech = float(np.percentile(rms, 90))
    if noise < 1e-9:
        return 60.0  # effectively very clean / very quiet noise floor
    snr = 20.0 * np.log10((speech + 1e-12) / (noise + 1e-12))
    return _safe_float(float(snr))


# ----------------------------
# Report
# ----------------------------
@dataclass(frozen=True)
class QualityConfig:
    clip_threshold: float = 0.999
    min_duration_s: float = 8.0
    max_duration_s: float = 1200.0
    min_speech_ratio: float = 0.35
    min_snr_db: float = 10.0


def run_quality_checks(
    clean_stage1_wav: str | Path,
    segments_dir: str | Path | None,
    reference_clean_wav: str | Path | None,
    report_path: str | Path,
    cfg: QualityConfig = QualityConfig(),
) -> Dict[str, Any]:
    clean_stage1_wav = Path(clean_stage1_wav)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    x, sr = load_wav(clean_stage1_wav)

    dur_s = duration_seconds(len(x), sr)
    clip_pct = clipping_percent(x, threshold=cfg.clip_threshold)
    snr_db = rough_snr_db(x, sr)

    speech_ratio = None
    speech_s = None
    num_segments = None
    if segments_dir is not None and Path(segments_dir).exists():
        speech_ratio, speech_s, num_segments = speech_ratio_from_segments(clean_stage1_wav, segments_dir)

    # reference duration (optional)
    ref_dur_s = None
    if reference_clean_wav is not None and Path(reference_clean_wav).exists():
        rx, rsr = load_wav(reference_clean_wav)
        ref_dur_s = duration_seconds(len(rx), rsr)

    # simple pass/fail flags
    checks = {
        "duration_ok": (dur_s >= cfg.min_duration_s) and (dur_s <= cfg.max_duration_s),
        "clipping_ok": (clip_pct <= 0.10),  # keep strict-ish; adjust later
        "speech_ratio_ok": (speech_ratio is None) or (speech_ratio >= cfg.min_speech_ratio),
        "snr_ok": (snr_db >= cfg.min_snr_db),
    }
    overall_ok = bool(all(checks.values()))

    report: Dict[str, Any] = {
        "inputs": {
            "clean_stage1_wav": str(clean_stage1_wav),
            "segments_dir": str(segments_dir) if segments_dir is not None else None,
            "reference_clean_wav": str(reference_clean_wav) if reference_clean_wav is not None else None,
        },
        "audio": {
            "sample_rate": int(sr),
            "duration_s": _safe_float(dur_s),
            "clipping_percent": _safe_float(clip_pct),
            "rough_snr_db": _safe_float(snr_db),
        },
        "vad": {
            "speech_ratio": _safe_float(speech_ratio) if speech_ratio is not None else None,
            "speech_seconds": _safe_float(speech_s) if speech_s is not None else None,
            "num_segments": int(num_segments) if num_segments is not None else None,
            "reference_duration_s": _safe_float(ref_dur_s) if ref_dur_s is not None else None,
        },
        "thresholds": asdict(cfg),
        "checks": checks,
        "overall_ok": overall_ok,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Path to clean_stage1.wav")
    ap.add_argument("--segments", default=None, help="Path to segments/ folder (optional)")
    ap.add_argument("--reference", default=None, help="Path to reference_clean.wav (optional)")
    ap.add_argument("--report", required=True, help="Output path for intake_report.json")
    args = ap.parse_args()

    run_quality_checks(
        clean_stage1_wav=args.clean,
        segments_dir=args.segments,
        reference_clean_wav=args.reference,
        report_path=args.report,
    )
    print(str(Path(args.report)))


if __name__ == "__main__":
    _cli()