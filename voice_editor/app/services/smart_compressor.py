# app/services/smart_compressor.py

import numpy as np
from app.utils.audio_io import load_audio, save_audio


def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x):
    return 20 * np.log10(np.maximum(x, 1e-8))


def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


def envelope_detector(audio, sr, attack_ms=8, release_ms=120):
    """
    Smooth loudness tracking.
    """
    attack = np.exp(-1.0 / (sr * attack_ms / 1000))
    release = np.exp(-1.0 / (sr * release_ms / 1000))

    env = np.zeros(len(audio))
    prev = 0.0

    for i, sample in enumerate(np.abs(audio)):
        if sample > prev:
            prev = attack * prev + (1 - attack) * sample
        else:
            prev = release * prev + (1 - release) * sample
        env[i] = prev

    return env


def smart_auto_settings(audio):
    """
    Detect audio dynamics and choose settings automatically.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    crest = peak / (rms + 1e-8)

    # More dynamic audio gets stronger compression
    if crest > 6:
        ratio = 3.5
        threshold_db = -24
    elif crest > 3:
        ratio = 2.5
        threshold_db = -20
    else:
        ratio = 1.8
        threshold_db = -16

    makeup_db = 4.0
    return threshold_db, ratio, makeup_db


def compress(audio, sr, threshold_db, ratio, makeup_db):
    env = envelope_detector(audio, sr)
    env_db = linear_to_db(env)

    gain_reduction_db = np.zeros(len(audio))

    over = env_db > threshold_db
    gain_reduction_db[over] = (
        (threshold_db + (env_db[over] - threshold_db) / ratio) - env_db[over]
    )

    gain = db_to_linear(gain_reduction_db + makeup_db)

    return audio * gain


def process_smart_compression(input_path, output_path):
    audio, sr = load_audio(input_path)

    threshold_db, ratio, makeup_db = smart_auto_settings(audio)

    compressed = compress(
        audio,
        sr,
        threshold_db,
        ratio,
        makeup_db
    )

    compressed = normalize(compressed)

    save_audio(output_path, compressed, sr)

    return {
        "output_path": output_path,
        "sample_rate": sr,
        "status": "smart_compressed",
        "settings": {
            "threshold_db": threshold_db,
            "ratio": ratio,
            "makeup_gain_db": makeup_db
        },
        "features": [
            "balanced_loud_soft_levels",
            "consistent_volume",
            "auto_dynamic_control",
            "youtube_ready_audio"
        ]
    }
