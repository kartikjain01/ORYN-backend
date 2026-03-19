from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VoiceProfileConfig:
    profile_filename: str = "voice_profile.json"
    stored_reference_name: str = "reference_clean.wav"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists() and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # Keep it silent + safe: caller will treat as "no report"
            return None
    return None


def _recommended_settings_from_quality(quality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight heuristics. You can tune later.
    Returns settings you can feed into XTTS generation defaults.
    """
    rec: Dict[str, Any] = {}

    if not quality:
        return rec

    audio = (quality.get("audio", {}) or {})
    vad = (quality.get("vad", {}) or {})

    clip_pct = audio.get("clipping_percent", None)
    snr_db = audio.get("rough_snr_db", None)
    speech_ratio = vad.get("speech_ratio", None)

    # If noisy/low SNR, keep temperature lower for stability
    if isinstance(snr_db, (int, float)) and snr_db < 12:
        rec["temperature"] = 0.55
        rec["top_p"] = 0.85

    # If clipping is noticeable, be conservative
    if isinstance(clip_pct, (int, float)) and clip_pct > 0.10:
        rec["temperature"] = min(rec.get("temperature", 0.65), 0.55)

    # If speech ratio is low (too much silence), shorter conditioning helps
    if isinstance(speech_ratio, (int, float)) and speech_ratio < 0.45:
        rec["gpt_cond_len"] = 16

    return rec


def build_voice_profile(
    voice_id: str,
    reference_clean_src: str | Path,
    quality_report_path: str | Path | None = None,
    cfg: VoiceProfileConfig = VoiceProfileConfig(),
) -> Dict[str, Any]:
    """
    Input:
      - reference_clean.wav (src path)
      - optional intake_report.json path
    Output:
      - stored reference at data/voices/<voice_id>/clean/reference_clean.wav
      - voice_profile.json in data/voices/<voice_id>/clean/
    """
    # Import inside function to avoid circular-import issues during FastAPI startup
    from backend.storage.paths import ensure_voice_dirs

    vp = ensure_voice_dirs(voice_id)  # creates raw/ and clean/

    src = Path(reference_clean_src)
    if not src.exists():
        raise FileNotFoundError(f"reference_clean.wav not found: {src}")

    dst_ref = Path(vp.clean_dir) / cfg.stored_reference_name
    dst_ref.parent.mkdir(parents=True, exist_ok=True)

    # Copy reference into voice storage (overwrite allowed)
    shutil.copy2(src, dst_ref)

    quality: Optional[Dict[str, Any]] = None
    qpath: Optional[Path] = None
    if quality_report_path is not None:
        qpath = Path(quality_report_path)
        quality = _read_json_if_exists(qpath)

    profile: Dict[str, Any] = {
        "voice_id": voice_id,
        "created_at": _utc_now_iso(),
        "ref_path": str(dst_ref),
        "quality_report": str(qpath) if qpath is not None else None,
        "recommended_settings": _recommended_settings_from_quality(quality),
    }

    profile_path = Path(vp.clean_dir) / cfg.profile_filename
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return profile


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice_id", required=True)
    ap.add_argument("--reference", required=True, help="Path to reference_clean.wav")
    ap.add_argument("--quality", default=None, help="Path to intake_report.json (optional)")
    args = ap.parse_args()

    profile = build_voice_profile(
        voice_id=args.voice_id,
        reference_clean_src=args.reference,
        quality_report_path=args.quality,
    )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    _cli()