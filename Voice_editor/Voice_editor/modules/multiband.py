import torch
import torchaudio.functional as F
from .compressor import rms_compressor

def multiband_compress(audio, sr):
    """
    Split into 3 bands and compress independently
    """

    low = F.lowpass_biquad(audio, sr, cutoff_freq=200)
    mid = F.bandpass_biquad(audio, sr, 1000, Q=0.7)
    high = F.highpass_biquad(audio, sr, cutoff_freq=5000)

    low = rms_compressor(low, threshold_db=-20, ratio=2.5)
    mid = rms_compressor(mid, threshold_db=-18, ratio=3.0)
    high = rms_compressor(high, threshold_db=-16, ratio=2.0)

    return low + mid + high
