import torch


# =========================================================
# Harmonic Saturation (Analog Style)
# =========================================================

def apply_harmonic_saturation(
    audio,
    drive=0.08,          # 0.05–0.15 recommended
    mix=0.12             # 10–20% blend
):
    """
    Subtle analog-style harmonic saturation.

    drive → intensity of harmonic generation
    mix   → dry/wet blend
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Soft clipping curve (tanh style)
    saturated = torch.tanh(audio * (1.0 + drive))

    # Blend dry and saturated
    output = audio * (1 - mix) + saturated * mix

    # Safety clamp
    output = torch.clamp(output, -1.0, 1.0)

    return output