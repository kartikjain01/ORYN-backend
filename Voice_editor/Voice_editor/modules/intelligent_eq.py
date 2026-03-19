import torch
import torchaudio.functional as F


# =========================================================
# Utility: Frequency Band Energy
# =========================================================

def band_energy(audio, sr, low_freq, high_freq):

    if audio.dim() == 2:
        audio = audio.squeeze(0)

    n_fft = 2048
    window = torch.hann_window(n_fft, device=audio.device)

    spec = torch.stft(
        audio,
        n_fft=n_fft,
        window=window,
        return_complex=True
    )

    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(audio.device)
    magnitude = torch.abs(spec)

    band_mask = (freqs >= low_freq) & (freqs <= high_freq)

    return magnitude[band_mask, :].mean()


# =========================================================
# FINAL Intelligent + Broadcast EQ
# =========================================================

def apply_intelligent_eq(
    audio,
    sr,
    blueprint_tilt=None,
    max_boost_db=6.0
):
    """
    Intelligent Broadcast EQ Engine:

    ✔ Mud reduction (150–400 Hz)
    ✔ Adaptive presence boost (3–5 kHz)
    ✔ Harshness control (6–10 kHz)
    ✔ Adaptive air boost (10–12 kHz)
    ✔ Low-cut (70–80 Hz)
    ✔ Blueprint tilt integration
    ✔ Safety clamp protection
    """

    # Ensure mono
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # -----------------------------------
    # ANALYZE SPECTRUM
    # -----------------------------------

    mud_energy = band_energy(audio, sr, 150, 400)
    presence_energy = band_energy(audio, sr, 3000, 5000)
    harsh_energy = band_energy(audio, sr, 6000, 10000)
    air_energy = band_energy(audio, sr, 10000, 14000)

    total_energy = mud_energy + presence_energy + harsh_energy + air_energy + 1e-6

    mud_ratio = mud_energy / total_energy
    presence_ratio = presence_energy / total_energy
    harsh_ratio = harsh_energy / total_energy
    air_ratio = air_energy / total_energy

    # -----------------------------------
    # DYNAMIC LOGIC
    # -----------------------------------

    # Mud Cut
    if mud_ratio > 0.45:
        mud_cut_db = 4.0
    elif mud_ratio > 0.35:
        mud_cut_db = 3.0
    else:
        mud_cut_db = 2.0

    # Presence Boost
    if presence_ratio < 0.25:
        presence_boost_db = 4.0
    elif presence_ratio < 0.35:
        presence_boost_db = 3.0
    else:
        presence_boost_db = 2.5

    # Harshness Protection
    if harsh_ratio > 0.30:
        presence_boost_db -= 1.0

    # Adaptive Air Boost
    if air_ratio < 0.15:
        air_boost_db = 2.0
    elif air_ratio < 0.25:
        air_boost_db = 1.5
    else:
        air_boost_db = 1.0

    # -----------------------------------
    # Blueprint Tilt Integration
    # -----------------------------------

    if blueprint_tilt is not None:
        tilt_db = float(blueprint_tilt) * 6.0
        tilt_db = max(-max_boost_db, min(max_boost_db, tilt_db))
        presence_boost_db += tilt_db

    # -----------------------------------
    # SAFETY CLAMPS
    # -----------------------------------

    mud_cut_db = min(mud_cut_db, max_boost_db)
    presence_boost_db = max(-max_boost_db,
                            min(max_boost_db, presence_boost_db))
    air_boost_db = max(-max_boost_db,
                       min(max_boost_db, air_boost_db))

    # -----------------------------------
    # APPLY BROADCAST EQ
    # -----------------------------------

    # 1. Low Cut (Broadcast Clean)
    audio = F.highpass_biquad(audio, sr, cutoff_freq=80)

    # 2. Mud Reduction
    audio = F.equalizer_biquad(
        audio,
        sr,
        center_freq=250.0,
        gain=-mud_cut_db,
        Q=1.0
    )

    # 3. Presence Boost
    audio = F.equalizer_biquad(
        audio,
        sr,
        center_freq=4500.0,
        gain=presence_boost_db,
        Q=1.2
    )

    # 4. Air Boost
    audio = F.equalizer_biquad(
        audio,
        sr,
        center_freq=11000.0,
        gain=air_boost_db,
        Q=0.7
    )

    return audio