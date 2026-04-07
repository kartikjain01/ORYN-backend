# backend/api/routes/tts.py
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
# -------------------------
# SUPABASE CONFIG
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
from pydantic import BaseModel

from backend.storage.paths import ensure_generation_dirs, ensure_voice_dirs

from backend.services.tts_engine.xtts_v2 import XTTSParams
from backend.services.tts_engine.generate import generate_chunks, GenerateConfig
from backend.services.postprocess.merge import merge_chunks, MergeConfig

# Phase 7 (RQ)
#from redis import Redis
#from rq import Queue
from backend.workers.tasks import run_tts_job  # RQ worker task (dict-based)

router = APIRouter(prefix="/v1/tts", tags=["tts"])

# ----------------------------
# Settings (choose mode)
# ----------------------------
# USE_RQ=1 -> enqueue to Redis/RQ
# USE_RQ=0 -> use FastAPI BackgroundTasks
USE_RQ = os.getenv("USE_RQ", "0") == "1"
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

if USE_RQ:
    from redis import Redis
    from rq import Queue

    redis = Redis.from_url(REDIS_URL)
    q = Queue("tts", connection=redis)
else:
    redis = None
    q = None


# ----------------------------
# Helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    gp = ensure_generation_dirs(job_id)
    return Path(gp.root)


def _job_json(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _write_job(job_id: str, payload: Dict[str, Any]) -> None:
    p = _job_json(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_job(job_id: str) -> Dict[str, Any]:
    p = _job_json(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="job_id not found")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="job.json unreadable")


def _voice_ref_path(voice_id: str) -> Path:
    vp = ensure_voice_dirs(voice_id)
    ref = Path(vp.clean_dir) / "reference_clean.wav"
    if not ref.exists():
        # fallback: voice_profile.json ref_path
        prof = Path(vp.clean_dir) / "voice_profile.json"
        if prof.exists():
            try:
                d = json.loads(prof.read_text(encoding="utf-8"))
                rp = Path(d.get("ref_path", ""))
                if rp.exists():
                    return rp
            except Exception:
                pass
        raise HTTPException(
            status_code=404,
            detail="reference_clean.wav not found; run /v1/voices/{voice_id}/build first",
        )
    return ref

# ============================
# ✅ Upload Function (NEW)
# ============================
def upload_to_supabase(file_path: Path):
    try:
        file_name = f"voice-clone/{uuid.uuid4()}.wav"

        with open(file_path, "rb") as f:
            res = supabase.storage.from_("outputs").upload(file_name, f)

            if hasattr(res, "error") and res.error:
               raise Exception(res["error"])

        public_url = supabase.storage.from_("outputs").get_public_url(file_name)

        return public_url["publicUrl"]

    except Exception as e:
        print("Supabase upload failed:", e)
        return None


# ----------------------------
# Schemas
# ----------------------------
class TTSRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"

    # light controls (map into XTTS params)
    emotion: Optional[str] = None  # reserved for later; currently not used here
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    gpt_cond_len: Optional[int] = None

    # output format
    output_format: str = "wav"  # "wav" or "mp3"


class TTSResponse(BaseModel):
    job_id: str
    status: str


# ----------------------------
# Background job (Phase 6 mode)
# ----------------------------
def _run_tts_job(job_id: str, body: TTSRequest) -> None:
    """
    BackgroundTasks job:
      - generate chunk wavs
      - merge with crossfade + loudness match
      - update job.json status
    """
    try:
        _write_job(job_id, {"job_id": job_id, "status": "running", "updated_at": _utc_now_iso()})

        ref = _voice_ref_path(body.voice_id)

        # params (apply overrides)
        base_params = XTTSParams()
        p = base_params.__dict__.copy()

        if body.temperature is not None:
            p["temperature"] = body.temperature
        if body.top_k is not None:
            p["top_k"] = body.top_k
        if body.top_p is not None:
            p["top_p"] = body.top_p
        if body.repetition_penalty is not None:
            p["repetition_penalty"] = body.repetition_penalty
        if body.gpt_cond_len is not None:
            p["gpt_cond_len"] = body.gpt_cond_len

        params = XTTSParams(**p)

        # generate chunks
        chunk_paths = generate_chunks(
            job_id=job_id,
            text=body.text,
            language=body.language,
            speaker_wav_path=str(ref),
            params=params,
            cfg=GenerateConfig(),
        )

        # merge
        out_ext = ".mp3" if body.output_format.lower() == "mp3" else ".wav"
        final_path = _job_dir(job_id) / f"final{out_ext}"

        merge_chunks(
            chunk_wavs=chunk_paths,
            out_path=str(final_path),
            cfg=MergeConfig(crossfade_ms=80, loudnorm=True),
        )

        audio_url = upload_to_supabase(final_path)

        _write_job(
           job_id,
          {
           "job_id": job_id,
           "status": "done",
           "updated_at": _utc_now_iso(),
           "audio_url": audio_url,
          },
        )

    except Exception as e:
        _write_job(
            job_id,
            {
                "job_id": job_id,
                "status": "failed",
                "updated_at": _utc_now_iso(),
                "error": str(e),
            },
        )


# ----------------------------
# Endpoints
# ----------------------------
@router.post("", response_model=TTSResponse)
def create_tts(body: TTSRequest, background: BackgroundTasks) -> TTSResponse:
    """
    POST /v1/tts
    - If USE_RQ=1: enqueue to RQ/Redis and return job_id immediately
    - Else: run in BackgroundTasks and return job_id immediately
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    ensure_generation_dirs(job_id)

    # quick fail if voice ref missing
    _ = _voice_ref_path(body.voice_id)

    _write_job(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": _utc_now_iso(),
            "voice_id": body.voice_id,
            "language": body.language,
            "output_format": body.output_format,
            "mode": "rq" if USE_RQ else "background",
        },
    )

    if USE_RQ:
        # RQ task expects dict payload
        # pydantic v1: body.dict()
        # pydantic v2: body.model_dump()
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()

        q.enqueue(
            run_tts_job,
            job_id,
            payload,
            job_timeout=3600,
        )
    else:
        background.add_task(_run_tts_job, job_id, body)

    return TTSResponse(job_id=job_id, status="queued")


@router.get("/{job_id}")
def get_tts_status(job_id: str) -> Dict[str, Any]:
    """
    GET /v1/tts/{job_id} status
    """
    return _read_job(job_id)


@router.get("/{job_id}/download")
def download_tts(job_id: str):
    """
    GET /v1/tts/{job_id}/download
    """
    job = _read_job(job_id)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready. status={job.get('status')}")
    audio_url = job.get("audio_url")
    if not audio_url:
       raise HTTPException(status_code=404, detail="Audio not found")

    return {"download_url": audio_url}
