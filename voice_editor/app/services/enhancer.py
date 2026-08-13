# app/services/enhancer.py

import numpy as np
import librosa
import scipy.signal as signal
import noisereduce as nr
from app.utils.audio_io import load_audio, save_audio


def enhance_audio(input_path: str, output_path: str):
    audio, sr = load_audio(input_path)

    # Remove low rumble
    b, a = signal.butter(4, 80 / (0.5 * sr), btype="highpass")
    audio = signal.filtfilt(b, a, audio)

    # Basic noise reduction
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.9
    )

    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    save_audio(output_path, audio, sr)

    return {
        "output_path": output_path,
        "sample_rate": sr,
        "status": "enhanced",
        "mode": "basic"
    }
