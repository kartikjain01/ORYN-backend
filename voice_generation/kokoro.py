from __future__ import annotations

import os
import threading

from kokoro_onnx import Kokoro

# ---------------------------------------------------
# MODEL FILES
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "kokoro-v1.0.int8.onnx",
)

VOICE_FILE = os.path.join(
    BASE_DIR,
    "voices-v1.0.bin",
)

# ---------------------------------------------------
# SINGLETON
# ---------------------------------------------------

_kokoro = None
_lock = threading.Lock()


def get_kokoro() -> Kokoro:
    global _kokoro

    if _kokoro is not None:
        return _kokoro

    with _lock:

        if _kokoro is None:

            print("\nLoading Kokoro...\n")

            _kokoro = Kokoro(
                MODEL_FILE,
                VOICE_FILE,
            )

            print("\nKokoro loaded successfully.\n")

    return _kokoro


# ---------------------------------------------------
# CREATE
# ---------------------------------------------------

def create(
    text: str,
    speaker: str,
    language: str,
    speed: float = 1.0,
):
    """
    Returns:
        samples (np.ndarray), sample_rate (int)
    """

    model = get_kokoro()

    samples, sample_rate = model.create(
        text=text,
        voice=speaker,
        speed=speed,
        lang=language,
    )

    return samples, sample_rate
