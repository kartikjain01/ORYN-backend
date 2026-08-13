from __future__ import annotations

import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.storage.paths import ensure_voice_dirs, VoicePaths

from backend.services.audio_preprocess.pipeline import (
    preprocess_stage1,
    preprocess_stage1_5,
    PreprocessConfig,
)

from backend.services.audio_preprocess.vad import vad_segment, VADConfig
from backend.services.audio_preprocess.quality import run_quality_checks, QualityConfig
from backend.services.voice_profile.builder import build_voice_profile


router = APIRouter(prefix="/v1/voices", tags=["voices"])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _voice_profile_path(vp: VoicePaths) -> Path:
    return Path(vp.clean_dir) / "voice_profile.json"


def _intake_report_path(vp: VoicePaths) -> Path:
    return Path(vp.clean_dir) / "intake_report.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


class CreateVoiceResponse(BaseModel):
    voice_id: str
    raw_path: str


@router.post("", response_model=CreateVoiceResponse)
async def create_voice(file: UploadFile = File(...)) -> CreateVoiceResponse:

    voice_id = f"voice_{uuid.uuid4().hex[:12]}"
    vp = ensure_voice_dirs(voice_id)

    raw_path = Path(vp.raw_dir) / "reference"

    suffix = Path(file.filename or "").suffix.lower()
    if suffix:
        raw_path = raw_path.with_suffix(suffix)
    else:
        raw_path = raw_path.with_suffix(".wav")

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    raw_path.write_bytes(data)

    return CreateVoiceResponse(
        voice_id=voice_id,
        raw_path=str(raw_path),
    )


class BuildVoiceRequest(BaseModel):
    raw_path: Optional[str] = None
    target_sr: int = 24000


class BuildVoiceResponse(BaseModel):
    voice_id: str
    reference_clean_path: str
    intake_report_path: str
    voice_profile_path: str


@router.post("/{voice_id}/build", response_model=BuildVoiceResponse)
def build_voice(voice_id: str, body: BuildVoiceRequest) -> BuildVoiceResponse:

    vp = ensure_voice_dirs(voice_id)

    # Select raw input
    if body.raw_path:
        raw_in = Path(body.raw_path)
    else:
        raw_dir = Path(vp.raw_dir)
        candidates = sorted([p for p in raw_dir.glob("*") if p.is_file()])

        if not candidates:
            raise HTTPException(status_code=404, detail="No raw audio found")

        raw_in = candidates[0]

    if not raw_in.exists():
        raise HTTPException(status_code=404, detail=f"raw_path not found: {raw_in}")

    cfg = PreprocessConfig(target_sr=body.target_sr)

    # Stage 1 preprocessing
    stage1_path = Path(vp.clean_dir) / "clean_stage1.wav"
    preprocess_stage1(str(raw_in), str(stage1_path), cfg)

    # Stage 1.5 preprocessing
    stage1_5_path = Path(vp.clean_dir) / "clean_stage1_5.wav"

    try:
        preprocess_stage1_5(str(stage1_path), str(stage1_5_path), cfg)
        vad_input = stage1_5_path
    except Exception:
        vad_input = stage1_path

    # VAD segmentation
    vad_out_dir = Path(vp.clean_dir)

    vad_result = vad_segment(
        str(vad_input),
        str(vad_out_dir),
        VADConfig(sr=body.target_sr),
    )

    reference_clean = Path(vad_result["reference_clean"])

    # Quality checks
    report_path = _intake_report_path(vp)

    run_quality_checks(
        clean_stage1_wav=str(vad_input),
        segments_dir=str(Path(vp.clean_dir) / "segments"),
        reference_clean_wav=str(reference_clean),
        report_path=str(report_path),
        cfg=QualityConfig(),
    )

    # Build voice profile
    profile = build_voice_profile(
        voice_id=voice_id,
        reference_wav_path=str(reference_clean),
        meta={"quality_report": str(report_path)},
    )

    profile_path = _voice_profile_path(vp)

    # Save profile JSON (profile is already a dict)
    profile_path.write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    return BuildVoiceResponse(
        voice_id=voice_id,
        reference_clean_path=str(reference_clean),
        intake_report_path=str(report_path),
        voice_profile_path=str(profile_path),
    )


class GetVoiceResponse(BaseModel):
    voice_profile: Optional[Dict[str, Any]] = None
    quality_report: Optional[Dict[str, Any]] = None


@router.get("/{voice_id}", response_model=GetVoiceResponse)
def get_voice(voice_id: str) -> GetVoiceResponse:

    vp = ensure_voice_dirs(voice_id)

    profile = _load_json(_voice_profile_path(vp))
    quality = _load_json(_intake_report_path(vp))

    if profile is None and quality is None:
        raise HTTPException(status_code=404, detail="No voice data found")

    return GetVoiceResponse(
        voice_profile=profile,
        quality_report=quality,
    )