import soundfile as sf
import librosa
from app.config import settings


def load_audio(path: str):
    audio, sr = librosa.load(path, sr=settings.TARGET_SR, mono=settings.MONO)
    return audio, sr


def save_audio(path: str, audio, sr: int = None):
    sample_rate = sr or settings.TARGET_SR
    sf.write(path, audio, sample_rate)
    return path