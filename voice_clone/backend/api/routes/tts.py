# backend/api/routes/tts.py
from __future__ import annotations

import json
import os
import uuid
import re  # 🔥 NEW
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from supabase import create_client

# 🔥 NEW: language detection
from langdetect import detect

# 🔥 NEW: Text Intelligence
from backend.text_intelligence.pipeline import TextIntelligencePipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.abspath(os.path.join(BASE_DIR, "../../../../.env"))

load_dotenv(env_path)

# -------------------------
# SUPABASE CONFIG
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

from pydantic import BaseModel

from backend.storage.paths import ensure_generation_dirs, ensure_voice_dirs

from backend.services.tts_engine import EngineParams
from backend.services.tts_engine.generate import generate_chunks, GenerateConfig
from backend.services.postprocess.merge import merge_chunks, MergeConfig

from backend.workers.tasks import run_tts_job

router = APIRouter(prefix="/v1/tts", tags=["tts"])

# 🔥 NEW: initialize once
text_pipeline = TextIntelligencePipeline()

# ----------------------------
# 🔥 NEW: SMART CHUNKING HELPER
# ----------------------------
def _smart_chunk_text(text: str, max_len: int = 250):
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks


# ----------------------------
# Settings
# ----------------------------
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


# ----------------------------
# Upload Function (UNCHANGED)
# ----------------------------
def upload_to_supabase(file_path: Path, user_id: str, job_id: str):
    try:
        safe_user = (
            user_id.strip()
            .replace(" ", "_")
            .replace(".", "")
            .lower()
        )
        file_name = f"voice-clone/{safe_user}_{job_id}.wav"

        with open(file_path, "rb") as f:
            supabase.storage.from_("outputs").upload(file_name, f)

        public_url = supabase.storage.from_("outputs").get_public_url(file_name)

        print("Supabase URL:", public_url)

        return public_url

    except Exception as e:
        print("Supabase upload failed:", e)
        return None


# ----------------------------
# Schemas (UNCHANGED)
# ----------------------------
class TTSRequest(BaseModel):
    voice_id: str
    text: str
    user_id: str
    language: str = "en"

    emotion: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    gpt_cond_len: Optional[int] = None

    output_format: str = "wav"


class TTSResponse(BaseModel):
    job_id: str
    status: str


# ----------------------------
# Background job (UPDATED ONLY HERE)
# ----------------------------
def _run_tts_job(job_id: str, body: TTSRequest) -> None:
    try:
        _write_job(job_id, {"job_id": job_id, "status": "running", "updated_at": _utc_now_iso()})

        ref = _voice_ref_path(body.voice_id)

        base_params = EngineParams()

        p = base_params.__dict__.copy()

        if body.temperature is not None:
         p["temperature"] = body.temperature

        if body.top_k is not None:
          p["top_k"] = body.top_k

        if body.top_p is not None:
         p["top_p"] = body.top_p

        if body.repetition_penalty is not None:
         p["repetition_penalty"] = body.repetition_penalty

        # XTTS only
        if "gpt_cond_len" in p and body.gpt_cond_len is not None:
         p["gpt_cond_len"] = body.gpt_cond_len

        params = EngineParams(**p)

        # 🔥 language detection
        try:
            detected_lang = detect(body.text)
        except Exception:
            detected_lang = body.language or "en"

        language = "hi" if detected_lang == "hi" else "en"

        # 🔥 TEXT INTELLIGENCE
        analysis = text_pipeline.run(body.text)
        print("\n========== FULL ANALYSIS ==========")
        print(json.dumps(analysis, indent=2, default=str))
        print("==================================\n")

        # 🔥 Hindi protection
        if language == "hi":
            enhanced_text = body.text
        else:
            enhanced_text = analysis.get("enhanced_text", body.text)
            print("Enhanced Text Used:", enhanced_text)

        print("\n========== TEXT INTELLIGENCE ==========")
        print("Original:", body.text)
        print("Enhanced:", enhanced_text)
        print("Emotion:", analysis.get("emotion"))
        print("Detected Language:", language)
        print("=======================================\n")

        # 🔥 SMART CHUNKING
        text_chunks = _smart_chunk_text(enhanced_text)
        all_chunk_paths = []

        chunks_dir = _job_dir(job_id) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(text_chunks):
            print(f"Processing chunk {idx+1}/{len(text_chunks)}")

            chunk_paths = generate_chunks(
                job_id=job_id,
                text=chunk,
                language=language,
                speaker_wav_path=str(ref),
                params=params,
                cfg=GenerateConfig(),
            )

            for p in chunk_paths:
                p = Path(p)
                new_path = chunks_dir / f"{idx}_{p.name}"

                try:
                   p.rename(new_path)
                except Exception:
                   import shutil
                   shutil.move(str(p), str(new_path))

                all_chunk_paths.append(str(new_path))

        # merge
        out_ext = ".mp3" if body.output_format.lower() == "mp3" else ".wav"
        final_path = _job_dir(job_id) / f"final{out_ext}"

        merge_chunks(
            chunk_wavs=all_chunk_paths,
            out_path=str(final_path),
            cfg=MergeConfig(crossfade_ms=35, loudnorm=True),
        )

        # upload
        audio_url = upload_to_supabase(final_path, body.user_id, job_id)

        _write_job(
            job_id,
            {
                "job_id": job_id,
                "status": "done",
                "updated_at": _utc_now_iso(),
                "audio_url": audio_url,
                "text_analysis": analysis,
                "language": language,
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
# Endpoints (UNCHANGED)
# ----------------------------
@router.post("", response_model=TTSResponse)
def create_tts(body: TTSRequest, background: BackgroundTasks) -> TTSResponse:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    ensure_generation_dirs(job_id)

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
    return _read_job(job_id)


@router.get("/{job_id}/download")
def download_tts(job_id: str):
    job = _read_job(job_id)
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job not ready. status={job.get('status')}")
    audio_url = job.get("audio_url")
    if not audio_url:
        raise HTTPException(status_code=404, detail="Audio not found")

    return {"download_url": audio_url}
