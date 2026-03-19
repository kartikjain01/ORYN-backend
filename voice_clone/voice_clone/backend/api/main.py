from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI

app = FastAPI(title="voice_clone_tts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _include_routes(app: FastAPI) -> None:
    # Import routes lazily to reduce circular import + heavy import issues
    from backend.api.routes.voices import router as voices_router
    from backend.api.routes.tts import router as tts_router  # if you have it

    app.include_router(voices_router)
    app.include_router(tts_router)
    @app.get("/health")
    def health():
      return {"status": "ok"}


_include_routes(app)
