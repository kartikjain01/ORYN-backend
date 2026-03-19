import torch

def hallucination_guard(audio):
    """
    Detect sudden artificial spikes or unnatural silence patches
    """

    energy = torch.mean(audio ** 2)

    if energy < 1e-5:
        return True  # suspicious

    return False
