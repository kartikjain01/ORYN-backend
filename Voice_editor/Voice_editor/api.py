import os
import uuid
import sys
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice Editor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Voice Editor API running"}


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):

    try:
        # -----------------------------
        # Save uploaded file
        # -----------------------------
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        # -----------------------------
        # Run clean.py pipeline
        # -----------------------------
        clean_script = os.path.join(BASE_DIR, "clean.py")

        command = [
            sys.executable,
            clean_script,
            "--input",
            input_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=BASE_DIR
        )

        print("\n----- PIPELINE STDOUT -----")
        print(result.stdout)

        print("\n----- PIPELINE STDERR -----")
        print(result.stderr)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline failed:\n{result.stderr}"
            )

        # -----------------------------
        # Locate output file
        # -----------------------------
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_MASTERED.wav")

        if not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail="Pipeline finished but output file not found"
            )

        # -----------------------------
        # Return processed audio
        # -----------------------------
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="processed.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))