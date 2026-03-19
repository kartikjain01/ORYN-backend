from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from main import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def handle_tts(payload: dict = Body(...)):
    try:
        user_text = payload.get("text", "")

        if not user_text:
            raise HTTPException(status_code=400, detail="No text provided")

        result_file = main(custom_text=user_text, custom_output="ui_output.wav")

        if result_file and os.path.exists(result_file):
            return FileResponse(result_file, media_type="audio/wav")

        raise HTTPException(status_code=500, detail="Audio file not created")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))