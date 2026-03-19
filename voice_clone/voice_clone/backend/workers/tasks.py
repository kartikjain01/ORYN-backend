from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from backend.services.tts_engine.generate import generate_chunks, GenerateConfig
from backend.services.postprocess.merge import merge_chunks, MergeConfig
from backend.services.tts_engine.xtts_v2 import XTTSParams
from backend.storage.paths import ensure_voice_dirs, ensure_generation_dirs

import json
from datetime import datetime, timezone


def _utc() -> str:
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


def _voice_ref_path(voice_id: str) -> Path:
    vp = ensure_voice_dirs(voice_id)
    ref = Path(vp.clean_dir) / "reference_clean.wav"
    if not ref.exists():
        raise FileNotFoundError("reference_clean.wav missing; run voice build first.")
    return ref


def run_tts_job(job_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ worker task: long running synthesis.
    """
    try:
        _write_job(job_id, {"job_id": job_id, "status": "running", "updated_at": _utc()})

        voice_id = body["voice_id"]
        text = body["text"]
        language = body.get("language", "en")
        out_fmt = body.get("output_format", "wav").lower()

        ref = _voice_ref_path(voice_id)

        params = XTTSParams(
            temperature=float(body.get("temperature", 0.65)),
            top_k=int(body.get("top_k", 50)),
            top_p=float(body.get("top_p", 0.90)),
            repetition_penalty=float(body.get("repetition_penalty", 2.0)),
            gpt_cond_len=int(body.get("gpt_cond_len", 20)),
        )

        chunk_paths = generate_chunks(
            job_id=job_id,
            text=text,
            language=language,
            speaker_wav_path=str(ref),
            params=params,
            cfg=GenerateConfig(),
        )

        out_ext = ".mp3" if out_fmt == "mp3" else ".wav"
        final_path = _job_dir(job_id) / f"final{out_ext}"

        merge_chunks(
            chunk_wavs=chunk_paths,
            out_path=str(final_path),
            cfg=MergeConfig(crossfade_ms=80, loudnorm=True),
        )

        result = {
            "job_id": job_id,
            "status": "done",
            "updated_at": _utc(),
            "final_path": str(final_path),
        }
        _write_job(job_id, result)
        return result

    except Exception as e:
        err = {"job_id": job_id, "status": "failed", "updated_at": _utc(), "error": str(e)}
        _write_job(job_id, err)
        return err