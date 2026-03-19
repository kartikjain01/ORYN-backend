# backend/storage/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ----------------------------
# Base folders
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # backend/.. -> project root
DATA_DIR = PROJECT_ROOT / "data"

VOICES_DIR = DATA_DIR / "voices"
GENERATIONS_DIR = DATA_DIR / "generations"


@dataclass(frozen=True)
class VoicePaths:
    """All paths for a specific voice_id."""
    voice_id: str

    @property
    def root(self) -> Path:
        return VOICES_DIR / self.voice_id

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.root / "clean"


@dataclass(frozen=True)
class GenerationPaths:
    """All paths for a specific job_id."""
    job_id: str

    @property
    def root(self) -> Path:
        return GENERATIONS_DIR / self.job_id

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"


def ensure_dirs(*dirs: Path) -> None:
    """
    Create directories if missing.
    Safe to call repeatedly.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Convenience helpers
# ----------------------------
def get_voice_paths(voice_id: str) -> VoicePaths:
    return VoicePaths(voice_id=voice_id)


def get_generation_paths(job_id: str) -> GenerationPaths:
    return GenerationPaths(job_id=job_id)


def ensure_voice_dirs(voice_id: str) -> VoicePaths:
    """
    ✅ Output of phase:
    creates data/voices/<voice_id>/raw and clean reliably.
    """
    vp = get_voice_paths(voice_id)
    ensure_dirs(VOICES_DIR, vp.root, vp.raw_dir, vp.clean_dir)
    return vp


def ensure_generation_dirs(job_id: str) -> GenerationPaths:
    """
    Creates data/generations/<job_id>/chunks reliably.
    """
    gp = get_generation_paths(job_id)
    ensure_dirs(GENERATIONS_DIR, gp.root, gp.chunks_dir)
    return gp