import os
import numpy as np
from pydub import AudioSegment


def standardize_audio(input_path, output_path):
    """
    Converts audio to:
    - WAV
    - 16000 Hz
    - Mono
    - 16-bit PCM
    """
    try:
        print(f"Reading: {os.path.basename(input_path)}...")

        audio = AudioSegment.from_file(input_path)

        # Convert to Mono
        audio = audio.set_channels(1)

        # Set Sample Rate
        audio = audio.set_frame_rate(16000)

        # Export standardized WAV
        audio.export(output_path, format="wav")

        print(f"✅ Standardized: {output_path}")

    except Exception as e:
        print(f"❌ Failed to convert {input_path}: {e}")


# ------------------------------------------------------
# NEW: LOAD AUDIO FOR ANALYSIS / AI PROCESSING
# ------------------------------------------------------

def load_audio(input_path):
    """
    Loads WAV file and returns:
    - mono numpy array (float32, -1.0 to 1.0)
    - sample rate
    """

    try:
        audio = AudioSegment.from_file(input_path)

        # Ensure mono
        audio = audio.set_channels(1)

        sample_rate = audio.frame_rate

        # Convert raw samples to numpy
        samples = np.array(audio.get_array_of_samples())

        # Normalize from int16 to float32 (-1 to 1)
        samples = samples.astype(np.float32) / 32768.0

        # SAFETY FIX (Very Important)
        samples = np.nan_to_num(samples)

        return samples, sample_rate

    except Exception as e:
        print(f"❌ Failed to load audio for analysis: {e}")
        raise


# ------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------

if __name__ == "__main__":
    input_file = "test_beep.wav"
    output_file = "standardized_voice.wav"

    standardize_audio(input_file, output_file)

    audio_np, sr = load_audio(output_file)
    print("Loaded audio shape:", audio_np.shape)
    print("Sample rate:", sr)
