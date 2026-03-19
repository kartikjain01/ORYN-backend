import torch
import torch.nn.functional as F


def compute_envelope(audio, window=512):
    """
    RMS envelope for stable dynamics processing
    """
    pad = window // 2
    audio_sq = audio ** 2

    env = F.avg_pool1d(
        audio_sq,
        kernel_size=window,
        stride=1,
        padding=pad
    )

    return torch.sqrt(env + 1e-8)


def smooth_gain(gain, attack=0.1, release=0.01):
    smoothed = torch.zeros_like(gain)
    prev = gain[..., 0]

    for i in range(gain.shape[-1]):
        g = gain[..., i]

        if g < prev:
            coeff = attack
        else:
            coeff = release

        prev = coeff * prev + (1 - coeff) * g
        smoothed[..., i] = prev

    return smoothed


def apply_downward_expander(audio, threshold=0.02, ratio=2.0):

    envelope = compute_envelope(audio)

    gain = torch.ones_like(audio)

    mask = envelope < threshold

    gain[mask] = (envelope[mask] / threshold) ** ratio

    gain = smooth_gain(gain)

    return audio * gain