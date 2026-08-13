from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

import os
import uuid

from app.services.ffmpeg_service import convert_to_wav

router = APIRouter()

RECORDINGS_DIR = "recordings"

os.makedirs(RECORDINGS_DIR, exist_ok=True)

@router.websocket("/ws/record")
async def websocket_record(
    websocket: WebSocket
):

    await websocket.accept()

    file_id = str(uuid.uuid4())

    webm_path = os.path.join(
        RECORDINGS_DIR,
        f"{file_id}.webm"
    )

    wav_path = os.path.join(
        RECORDINGS_DIR,
        f"{file_id}.wav"
    )

    print(f"Recording Started: {file_id}")

    try:

        with open(webm_path, "ab") as audio_file:

            while True:

                data = await websocket.receive_bytes()

                audio_file.write(data)

    except WebSocketDisconnect:

        print("Client disconnected")

    except Exception as e:

        print("Error:", e)

    finally:

        print("Converting to WAV...")

        convert_to_wav(
            webm_path,
            wav_path
        )

        print(f"Saved WAV: {wav_path}")
