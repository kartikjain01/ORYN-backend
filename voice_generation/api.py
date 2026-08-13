from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid

from dotenv import load_dotenv
from supabase import create_client

from main import main
from voices import VOICES

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Voice Generation API"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://orynengine.com",
        "https://www.orynengine.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Voice Generation API Running 🚀"
    }

# ==========================================
# GET AVAILABLE VOICES
# ==========================================

@app.get("/voices")
def get_voices():

    return [
        {
            "id": key,
            "label": value["label"]
        }
        for key, value in VOICES.items()
    ]

# ==========================================
# SUPABASE UPLOAD
# ==========================================

def upload_to_supabase(
    file_path: str,
    user_id: str,
    job_id: str
):

    try:

        safe_user = (
            user_id
            .strip()
            .replace(" ", "_")
            .lower()
        )

        file_name = (
            f"tts/{safe_user}_{job_id}.wav"
        )

        print("Uploading:", file_name)

        with open(file_path, "rb") as f:

            supabase.storage.from_(
                "outputs"
            ).upload(
                file_name,
                f
            )

        public_url = (
            supabase.storage
            .from_("outputs")
            .get_public_url(file_name)
        )

        print("Public URL:", public_url)

        return public_url

    except Exception as e:

        print("❌ Upload error:", e)

        return None

# ==========================================
# GENERATE TTS
# ==========================================

@app.post("/generate")
async def handle_tts(
    payload: dict = Body(...)
):

    try:

        # ------------------------------
        # INPUTS
        # ------------------------------

        user_text = payload.get(
            "text",
            ""
        )

        user_id = payload.get(
            "user_id",
            "unknown_user"
        )

        selected_voice = payload.get(
            "voice",
            ""
        )

        selected_language = payload.get(
            "language",
            "en"
        )
        instruct = payload.get(
    "instruct",
    ""
)
        # ------------------------------
        # VALIDATION
        # ------------------------------

        if not user_text.strip():

            raise HTTPException(
                status_code=400,
                detail="No text provided"
            )

        # --------------------------------
# DEFAULT VOICES
# --------------------------------

        if not selected_voice:

           if selected_language == "hi":
              selected_voice = "omega"

           else:
              selected_voice = "michael"

        if selected_voice not in VOICES:

            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice: {selected_voice}"
            )

        # ------------------------------
        # JOB ID
        # ------------------------------

        job_id = uuid.uuid4().hex

        file_name = f"{job_id}.wav"

        print("\n==========================")
        print("🎤 GENERATING AUDIO")
        print("==========================")
        print("VOICE:", selected_voice)
        print("JOB ID:", job_id)

        # ------------------------------
        # GENERATE AUDIO
        # ------------------------------

        result_file = main(
    custom_text=user_text,
    custom_output=file_name,
    selected_voice=selected_voice,
    instruct=instruct
)

        print("RESULT FILE:", result_file)

        # ------------------------------
        # CHECK FILE
        # ------------------------------

        if not result_file:

            raise HTTPException(
                status_code=500,
                detail="Audio generation failed"
            )

        if not os.path.exists(result_file):

            raise HTTPException(
                status_code=500,
                detail=f"Generated file missing: {result_file}"
            )

        # ------------------------------
        # UPLOAD TO SUPABASE
        # ------------------------------

        audio_url = upload_to_supabase(
            result_file,
            user_id,
            job_id
        )

        if not audio_url:

            raise HTTPException(
                status_code=500,
                detail="Supabase upload failed"
            )

        # ------------------------------
        # OPTIONAL DELETE
        # ------------------------------

        # try:
        #     os.remove(result_file)
        # except Exception as e:
        #     print("Delete error:", e)

        # ------------------------------
        # SUCCESS RESPONSE
        # ------------------------------

        return {
            "success": True,
            "voice": selected_voice,
            "audio_url": audio_url
        }

    except HTTPException:
        raise

    except Exception as e:

        print("❌ API ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
