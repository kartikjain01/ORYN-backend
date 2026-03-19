import torch
import numpy as np
import pyloudnorm as pyln

def analyze_audio_profile(audio, sr):
    """
    Extract full audio profile for matching
    """

    eps = 1e-8

    # Mono
    if audio.dim() == 2 and audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # RMS
    rms = torch.sqrt(torch.mean(audio ** 2) + eps)
    rms_db = 20 * torch.log10(rms + eps)

    # Peak
    peak = torch.max(torch.abs(audio))
    peak_db = 20 * torch.log10(peak + eps)

    # Spectral centroid
    spec = torch.stft(audio.squeeze(0), n_fft=2048, return_complex=True)
    magnitude = torch.abs(spec)
    freqs = torch.fft.rfftfreq(2048, 1.0 / sr)

    centroid = (freqs.unsqueeze(1) * magnitude).sum() / magnitude.sum()

    # LUFS
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio.squeeze().cpu().numpy())

    return {
        "rms_db": rms_db.item(),
        "peak_db": peak_db.item(),
        "centroid": centroid.item(),
        "lufs": loudness
    }
