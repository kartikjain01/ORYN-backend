import librosa
import numpy as np


def analyze_audio(path: str):
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y**2)))
    silence_ratio = float(np.mean(np.abs(y) < 0.01))
    return {
        'sample_rate': sr,
        'duration_sec': round(duration, 2),
        'volume_rms': round(rms, 4),
        'silence_ratio': round(silence_ratio, 4),
        'voice_detected': rms > 0.01
    }