import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
import pyloudnorm as pyln
from scipy.signal import lfilter
from modules.intelligent_eq import apply_intelligent_eq

# ==========================================
# SAFETY LIMITS
# ==========================================

MAX_GAIN_DB = 10
MAX_COMP_RATIO = 4.0
MAX_NOISE_STRENGTH = 0.8
MAX_TILT = 0.5

TARGET_SR = 16000


# ==========================================
# GAIN MODULE
# ==========================================

def apply_gain(audio, gain_db):
    gain_db = np.clip(gain_db, -MAX_GAIN_DB, MAX_GAIN_DB)
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


# ==========================================
# NOISE REDUCTION CONTROL (DeepFilterNet)
# ==========================================

def apply_noise_reduction(audio, strength):
    strength = np.clip(strength, 0, MAX_NOISE_STRENGTH)

    # Placeholder logic — connect to your DeepFilterNet inference
    # Here we simulate scaling noise suppression intensity

    processed = audio * (1 - (strength * 0.15))
    return processed


# ==========================================
# ADAPTIVE COMPRESSOR
# ==========================================

def apply_compression(audio, ratio):

    ratio = min(ratio, MAX_COMP_RATIO)

    threshold = 0.1
    compressed = np.copy(audio)

    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            compressed[i] = np.sign(audio[i]) * (
                threshold + (abs(audio[i]) - threshold) / ratio
            )

    return compressed


# ==========================================
# EQ TILT FILTER
# ==========================================

def apply_tilt_eq(audio, tilt_amount):

    tilt_amount = np.clip(tilt_amount, -MAX_TILT, MAX_TILT)

    # Simple first-order tilt using high-pass blend
    b = [1, -tilt_amount]
    a = [1]

    return lfilter(b, a, audio)


# ==========================================
# DECISION CONTROLLER
# ==========================================

def decision_controller(audio, blueprint):

    print("\n--- DECISION CONTROLLER ---")

    # 1️⃣ Gain
    if abs(blueprint["gain_db"]) > 0.5:
        print("Applying Gain")
        audio = apply_gain(audio, blueprint["gain_db"])
    else:
        print("Skipping Gain")

    # 2️⃣ Noise Reduction
    if blueprint["noise_reduction_strength"] > 0.1:
        print("Applying Noise Reduction")
        audio = apply_noise_reduction(audio,
                                      blueprint["noise_reduction_strength"])
    else:
        print("Skipping Noise Reduction")

    # 3️⃣ Compression
    if blueprint["compression_ratio"] > 1.6:
        print("Applying Compression")
        audio = apply_compression(audio,
                                  blueprint["compression_ratio"])
    else:
        print("Skipping Compression")

    # 4️⃣ Tilt EQ
    if abs(blueprint["eq_tilt_correction"]) > 0.05:
        print("Applying Tilt EQ")
        audio = apply_tilt_eq(audio,
                              blueprint["eq_tilt_correction"])
    else:
        print("Skipping Tilt EQ")

    return audio


# ==========================================
# MAIN PROCESSOR
# ==========================================

def adaptive_process(audio_path, blueprint, output_path):

    audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    processed_audio = decision_controller(audio, blueprint)

    sf.write(output_path, processed_audio, TARGET_SR)

    print("\nAdaptive processing complete.")
    print("Output saved to:", output_path)

audio_tensor = torch.from_numpy(audio).float()

audio_tensor = apply_intelligent_eq(
    audio_tensor,
    sr=16000,
    blueprint_tilt=blueprint.get("eq_tilt_correction", 0.0)
)

audio = audio_tensor.detach().cpu().numpy()    

from .compressor import rms_compressor
from .multiband import multiband_compress
from .eq import apply_intelligent_eq

def process_with_delta(audio, sr, delta):
    """
    Route audio through adaptive processing
    based on delta map
    """

    # Gain adjust (RMS match)
    gain_db = delta["rms_diff"]
    gain = 10 ** (gain_db / 20)
    audio = audio * gain

    # EQ adjust (brightness)
    if delta["centroid_diff"] > 300:
        audio = apply_intelligent_eq(audio, sr)

    # Compression if peak too high
    if delta["peak_diff"] < -2:
        audio = multiband_compress(audio, sr)

    return audio
