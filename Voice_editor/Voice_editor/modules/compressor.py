import torch
import torchaudio.functional as F

# =========================================================
# Utility Functions
# =========================================================

def db_to_linear(db):
    return 10 ** (db / 20.0)


def linear_to_db(x, eps=1e-8):
    return 20 * torch.log10(torch.clamp(x, min=eps))


def calculate_rms(audio):
    return torch.sqrt(torch.mean(audio ** 2) + 1e-8)


def calculate_rms_db(audio):
    return linear_to_db(calculate_rms(audio))


def calculate_peak_db(audio):
    peak = torch.max(torch.abs(audio))
    return linear_to_db(peak)


# =========================================================
# Envelope Follower
# =========================================================

def envelope_follower(signal, attack_coeff, release_coeff):
    """
    Smooth envelope follower for compressor gain.
    """

    envelope = torch.zeros_like(signal)
    prev = torch.tensor(0.0, device=signal.device)

    for i in range(signal.shape[-1]):
        current = signal[..., i]

        if current > prev:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        prev = coeff * prev + (1 - coeff) * current
        envelope[..., i] = prev

    return envelope


# =========================================================
# Adaptive Voice Compressor
# =========================================================

def apply_adaptive_compression(
    audio,
    sr=16000,
    ratio=3.0,
    attack_ms=10.0,
    release_ms=100.0,
    knee_db=6.0,
    max_ratio=4.0
):
    """
    Studio adaptive compressor.

    Features:
    - Automatic threshold detection
    - Soft knee compression
    - Attack/Release smoothing
    - Auto makeup gain
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    device = audio.device

    ratio = min(ratio, max_ratio)

    # ==========================================
    # ANALYSIS
    # ==========================================

    rms_db = calculate_rms_db(audio)
    peak_db = calculate_peak_db(audio)

    dynamic_range = peak_db - rms_db

    # Adaptive threshold
    threshold_db = rms_db + dynamic_range * 0.25

    abs_audio = torch.abs(audio)
    audio_db = linear_to_db(abs_audio)

    over_threshold = audio_db - threshold_db

    # ==========================================
    # SOFT KNEE COMPRESSION
    # ==========================================

    knee_half = knee_db / 2

    gain_reduction_db = torch.zeros_like(audio_db)

    for i in range(audio_db.shape[-1]):

        x = over_threshold[0, i]

        if x < -knee_half:
            gain_reduction_db[0, i] = 0

        elif abs(x) <= knee_half:
            gain_reduction_db[0, i] = (
                (1 - 1/ratio) * ((x + knee_half) ** 2) / (2 * knee_db)
            )

        else:
            gain_reduction_db[0, i] = x * (1 - 1/ratio)

    # ==========================================
    # ATTACK / RELEASE
    # ==========================================

    attack_coeff = torch.exp(
        torch.tensor(-1.0 / (sr * attack_ms / 1000.0), device=device)
    )

    release_coeff = torch.exp(
        torch.tensor(-1.0 / (sr * release_ms / 1000.0), device=device)
    )

    smoothed_gain = envelope_follower(
        gain_reduction_db,
        attack_coeff,
        release_coeff
    )

    gain_linear = db_to_linear(-smoothed_gain)

    compressed = audio * gain_linear

    # ==========================================
    # AUTO MAKEUP GAIN
    # ==========================================

    before_rms = calculate_rms_db(audio)
    after_rms = calculate_rms_db(compressed)

    makeup_gain_db = before_rms - after_rms

    compressed = compressed * db_to_linear(makeup_gain_db)

    return compressed.squeeze(0)


# =========================================================
# RMS Compressor (Secondary Stage)
# =========================================================

def rms_compressor(
    audio,
    threshold_db=-20.0,
    ratio=2.0,
    sr=16000
):
    """
    Light RMS compressor for leveling voice.
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    eps = 1e-8

    rms = calculate_rms(audio)
    rms_db = 20 * torch.log10(rms + eps)

    if rms_db > threshold_db:
        gain_reduction = (rms_db - threshold_db) * (1 - 1 / ratio)
    else:
        gain_reduction = 0

    gain = db_to_linear(-gain_reduction)

    compressed = audio * gain

    return compressed.squeeze(0)


# =========================================================
# Transient Detection
# =========================================================

def transient_preserve(audio, threshold=0.2):
    """
    Detect fast transient peaks.
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    diff = torch.abs(audio[:, 1:] - audio[:, :-1])

    transient_mask = diff > threshold

    return transient_mask


# =========================================================
# Parallel Compression
# =========================================================

def apply_parallel_compression(
    audio,
    sr=16000,
    blend=0.15,
    threshold_db=-30.0,
    ratio=6.0
):
    """
    Parallel compression to add density.

    Blend recommended:
    10% – 20%
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    heavy = audio.clone()

    eps = 1e-8

    rms = calculate_rms(heavy)
    rms_db = 20 * torch.log10(rms + eps)

    if rms_db > threshold_db:
        gain_reduction_db = (rms_db - threshold_db) * (1 - 1 / ratio)
        gain = db_to_linear(-gain_reduction_db)
        heavy = heavy * gain

    blended = audio * (1 - blend) + heavy * blend

    return blended.squeeze(0)


# =========================================================
# FULL VOICE COMPRESSION CHAIN
# =========================================================

def apply_voice_compression_chain(audio, sr=16000):
    """
    Full professional compression pipeline.
    """

    # Stage 1 – Adaptive compressor
    audio = apply_adaptive_compression(audio, sr)

    # Stage 2 – RMS leveling
    audio = rms_compressor(audio)

    # Stage 3 – Parallel glue compression
    audio = apply_parallel_compression(audio, sr)

    return audio