from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid

# Import your generation function
from main import main

app = FastAPI(title="Voice Generation API")

# ✅ Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route (avoid 404 confusion)
@app.get("/")
def root():
    return {"message": "Voice Generation API Running 🚀"}


# 🎤 Generate TTS
@app.post("/generate")
async def handle_tts(payload: dict = Body(...)):
    try:
        user_text = payload.get("text", "")

        if not user_text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        # ✅ Generate unique file name (VERY IMPORTANT)
        file_name = f"output_{uuid.uuid4().hex}.wav"

        # 🔊 Generate audio
        result_file = main(custom_text=user_text, custom_output=file_name)

        # ✅ Check file exists
        if result_file and os.path.exists(result_file):

            return FileResponse(
                path=result_file,
                media_type="audio/wav",
                filename="output.wav",  # forces correct handling
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )

        raise HTTPException(status_code=500, detail="Audio file not created")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
