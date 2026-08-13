import os
import uuid
import tempfile
import shutil
from typing import Literal
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from supabase import create_client

from app.services.pipeline import process_audio_mode
from app.services.silence_trimmer import process_silence_trim
from app.services.echo_remover import process_echo_removal
from app.services.smart_compressor import process_smart_compression
from app.services.intelligent_eq import process_intelligent_eq
from app.services.deesser_breath_control import process_uploaded_file
from app.services.final_audio_polishing import process_uploaded_file as youtube_polish_process
from app.services.ffmpeg_service import convert_to_wav
from dotenv import load_dotenv
from pathlib import Path

# 🔥 ALWAYS LOAD FROM BACKEND ROOT
BASE_DIR = Path(__file__).resolve().parents[3]  # go up to backend/
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

router = APIRouter()

# =========================
# ✅ SUPABASE CONFIG
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_supabase(file_path: str, user_id: str):
    try:
        safe_user = (
            user_id.strip()
            .replace(" ", "_")
            .replace(".", "")
            .lower()
        )

        # ✅ KEEP ORIGINAL FILE NAME
        original_name = os.path.basename(file_path)

        # ✅ ADD USER PREFIX (without changing actual name logic)
        file_name = f"editor/{safe_user}_{original_name}"

        with open(file_path, "rb") as f:
            supabase.storage.from_("outputs").upload(
                path=file_name,
                file=f,
                file_options={"upsert": "true"}
            )

        public_url = supabase.storage.from_("outputs").get_public_url(file_name)

        print("✅ Uploaded:", file_name)
        return public_url

    except Exception as e:
        print("❌ Upload failed:", e)
        return None


# =========================
# 1. Basic noise removal
# =========================
@router.post("/upload-audio/basic")
async def upload_basic(file: UploadFile = File(...)):
    return await process_audio_mode(file, "basic")


# =========================
# 2. Advanced noise removal
# =========================
@router.post("/upload-audio/advanced")
async def upload_advanced(file: UploadFile = File(...)):
    return await process_audio_mode(file, "advanced")


# =========================
# 3. DeepFilter
# =========================
@router.post("/upload-audio/deepfilter")
async def upload_deepfilter(file: UploadFile = File(...)):
    return await process_audio_mode(file, "deepfilter")


# =========================
# 4. Silence trim
# =========================
@router.post("/upload-audio/silence-trim")
async def upload_silence_trim(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    temp_output = input_path.replace(".wav", "_trimmed.wav")

    result = process_silence_trim(input_path, temp_output)

    DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    final_name = os.path.basename(temp_output)
    final_path = os.path.join(DOWNLOAD_DIR, final_name)

    if os.path.exists(temp_output):
        shutil.move(temp_output, final_path)

    result["output_path"] = final_path

    return {
        "filename": file.filename,
        "download_file": final_name,
        "download_url": f"/api/download/{final_name}",
        "result": result
    }


# =========================
# 5. Echo removal
# =========================
@router.post("/upload-audio/echo-remove")
async def upload_echo_remove(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", "_echo_clean.wav")

    result = process_echo_removal(input_path, output_path)

    return {
        "filename": file.filename,
        "download_file": os.path.basename(output_path),
        "result": result
    }


# =========================
# 6. Compression
# =========================
@router.post("/upload-audio/compress")
async def upload_compress(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", "_compressed.wav")

    result = process_smart_compression(input_path, output_path)

    return {
        "filename": file.filename,
        "download_file": os.path.basename(output_path),
        "result": result
    }


# =========================
# 7. Intelligent EQ
# =========================
@router.post("/upload-audio/eq")
async def upload_intelligent_eq(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    output_path = input_path.replace(".wav", "_eq.wav")

    result = process_intelligent_eq(input_path, output_path)

    return {
        "filename": file.filename,
        "download_file": os.path.basename(output_path),
        "result": result,
        "status": "processed"
    }


# =========================
# 8. Full Enhance
# =========================
@router.post("/upload-audio/full-enhance")
async def upload_full_enhance(
    file: UploadFile = File(...),
    mode: Literal["basic", "advanced", "deepfilter"] = Form("advanced"),
    youtube_polish: bool = Form(True)
):
    original_filename = file.filename.lower()

    # Directory for browser recordings
    RECORDINGS_DIR = BASE_DIR / "recordings" / "editor"
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert browser recordings (.webm) to wav
    if original_filename.endswith(".webm"):
        file_id = str(uuid.uuid4())

        webm_path = RECORDINGS_DIR / f"{file_id}.webm"
        wav_path = RECORDINGS_DIR / f"{file_id}.wav"

        with open(webm_path, "wb") as f:
            f.write(await file.read())

        convert_to_wav(str(webm_path), str(wav_path))

        file = UploadFile(
            filename=f"{file_id}.wav",
            file=open(wav_path, "rb")
        )

        print(f"🎙 WEBM Saved: {webm_path}")
        print(f"🎵 WAV Saved: {wav_path}")

    # Run enhancement pipeline
    result = await process_audio_mode(
        file=file,
        mode=mode,
        youtube_polish=youtube_polish
    )

    processed_path = os.path.join("outputs", result["download_file"])

    if not os.path.exists(processed_path):
        raise HTTPException(
            status_code=500,
            detail="Processed audio file not found."
        )

    # Final polish
    final_path = youtube_polish_process(processed_path, "outputs")

    # Delete intermediate file
    if os.path.exists(processed_path):
        os.remove(processed_path)

    final_file = os.path.basename(final_path)

    # Upload to Supabase
    supabase_url = upload_to_supabase(final_path, final_file)

    return {
        "filename": final_file,
        "download_file": final_file,
        "download_url": f"/api/download/{final_file}",
        "supabase_url": supabase_url,
        "status": "processed"
    }
