# app/services/intelligent_eq.py

import numpy as np
import scipy.signal as signal
from app.utils.audio_io import load_audio, save_audio


def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


def bandpass(audio, sr, low_hz, high_hz, order=2):
    nyq = sr * 0.5
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)

    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, audio)


def lowpass(audio, sr, cutoff_hz, order=2):
    nyq = sr * 0.5
    b, a = signal.butter(order, cutoff_hz / nyq, btype="low")
    return signal.filtfilt(b, a, audio)


def highpass(audio, sr, cutoff_hz, order=2):
    nyq = sr * 0.5
    b, a = signal.butter(order, cutoff_hz / nyq, btype="high")
    return signal.filtfilt(b, a, audio)


def analyze_tone(audio):
    """
    Detect rough tonal balance.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    dynamic = peak / (rms + 1e-8)

    return {
        "rms": float(rms),
        "dynamic": float(dynamic)
    }


def smart_eq(audio, sr):
    """
    Bass / Mid / Treble auto enhancement for speech.
    """

    bass = lowpass(audio, sr, 180)
    mids = bandpass(audio, sr, 180, 4000)
    treble = highpass(audio, sr, 4000)

    tone = analyze_tone(audio)

    # Adaptive gains
    if tone["dynamic"] > 5:
        bass_gain = 0.90
        mid_gain = 1.20
        treble_gain = 1.05
    else:
        bass_gain = 1.00
        mid_gain = 1.15
        treble_gain = 1.10

    enhanced = (
        bass * bass_gain +
        mids * mid_gain +
        treble * treble_gain
    )

    # Mix with original for naturalness
    final = audio * 0.35 + enhanced * 0.65

    return final, {
        "bass_gain": bass_gain,
        "mid_gain": mid_gain,
        "treble_gain": treble_gain
    }


def process_intelligent_eq(input_path, output_path):
    audio, sr = load_audio(input_path)

    eq_audio, settings = smart_eq(audio, sr)
    eq_audio = normalize(eq_audio)

    save_audio(output_path, eq_audio, sr)

    return {
        "output_path": output_path,
        "sample_rate": sr,
        "status": "intelligent_eq_applied",
        "settings": settings,
        "features": [
            "bass_adjusted",
            "mids_enhanced",
            "treble_balanced",
            "voice_clarity_improved",
            "richer_voice",
            "youtube_ready",
            "podcast_ready"
        ]
    }
