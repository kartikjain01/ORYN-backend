from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VoiceProfile:
    voice_id: str
    ref_path: str
    created_at: str
    meta: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_voice_profile(
    voice_id: str,
    reference_wav_path: str | Path,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    from backend.storage.paths import ensure_voice_dirs

    vp = ensure_voice_dirs(voice_id)

    src = Path(reference_wav_path)
    dst_ref = Path(vp.clean_dir) / "reference_clean.wav"

    if not src.exists():
        raise FileNotFoundError(f"reference wav not found: {src}")

    dst_ref.parent.mkdir(parents=True, exist_ok=True)

    # FIX: avoid copying if file already in correct location
    if src.resolve() != dst_ref.resolve():
        shutil.copy2(src, dst_ref)

    profile = {
        "voice_id": voice_id,
        "created_at": _utc_now_iso(),
        "ref_path": str(dst_ref),
        "meta": meta or {},
    }

    profile_path = Path(vp.clean_dir) / "voice_profile.json"

    profile_path.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    return profile