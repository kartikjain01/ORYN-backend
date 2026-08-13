from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.recorder import (
    router as recorder_router
)

app = FastAPI(
    title="voice_clone_tts",
    #root_path="/clone"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health route
@app.get("/health")
def health():
    return {"status": "ok"}

# Include recorder websocket
app.include_router(recorder_router)


# Lazy router loading
def _include_routes(app: FastAPI) -> None:
    from backend.api.routes.voices import router as voices_router
    from backend.api.routes.tts import router as tts_router

    app.include_router(voices_router)
    app.include_router(tts_router)

_include_routes(app)
