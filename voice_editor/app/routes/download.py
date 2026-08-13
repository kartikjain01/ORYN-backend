from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile

router = APIRouter()

@router.get("/download/{filename}")
async def download_file(filename: str):
    BASE_DIR = os.getcwd()
    DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")

    search_paths = [
        os.path.join(DOWNLOAD_DIR, filename),
        os.path.join(BASE_DIR, "outputs", filename),
        os.path.join(BASE_DIR, "uploads", filename),
        os.path.join(tempfile.gettempdir(), filename),
        os.path.join(BASE_DIR, filename),
    ]

    print("BASE_DIR:", BASE_DIR)
    print("DOWNLOAD_DIR:", DOWNLOAD_DIR)
    print("REQUESTED FILE:", filename)

    for file_path in search_paths:
        print("CHECKING:", file_path, "=>", os.path.isfile(file_path))
        if os.path.isfile(file_path):
            print("FOUND:", file_path)
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type="audio/wav"
            )

    raise HTTPException(status_code=404, detail="File not found")
