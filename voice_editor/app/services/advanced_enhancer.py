# app/services/advanced_enhancer.py

import numpy as np
import librosa
import scipy.signal as signal
import noisereduce as nr
from app.utils.audio_io import load_audio, save_audio


def enhance_audio_advanced(input_path: str, output_path: str):
    audio, sr = load_audio(input_path)

    # High-pass filter (wind / rumble)
    b, a = signal.butter(4, 80 / (0.5 * sr), btype="highpass")
    audio = signal.filtfilt(b, a, audio)

    # Hum removal (50Hz / 60Hz)
    for freq in [50, 60]:
        bn, an = signal.iirnotch(freq, 30, sr)
        audio = signal.filtfilt(bn, an, audio)

    # Stronger noise reduction
    audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.95,
        n_std_thresh_stationary=1.2
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
        "status": "advanced_enhanced",
        "mode": "advanced"
    }
