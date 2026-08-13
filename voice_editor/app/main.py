from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.recorder import router as recorder_router
from app.routes.upload import router as upload_router
from app.routes import download

app = FastAPI(
    title="AI Audio Editing System",
    root_path="/editor"
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

# Routes
app.include_router(upload_router, prefix="/api")
app.include_router(recorder_router)
app.include_router(download.router, prefix="/api", tags=["Download"])
