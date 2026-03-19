import os
import numpy as np
from pydub import AudioSegment, silence

# --------------------------------------------------
# Adaptive Silence Trimmer
# --------------------------------------------------

def adaptive_trim(
    input_path,
    output_path,
    noise_floor_db=-50,
    min_silence_len=400,
    keep_silence=200,
    fade_ms=30
):
    """
    Intelligent silence trimming based on measured noise floor.
    """

    print(f"\n✂️ Adaptive Trimming: {os.path.basename(input_path)}")

    audio = AudioSegment.from_file(input_path)

    # --------------------------------------------------
    # Dynamic Silence Threshold
    # --------------------------------------------------
    # Silence threshold = slightly above noise floor
    silence_thresh = noise_floor_db + 5

    print(f"🔎 Noise Floor: {noise_floor_db:.2f} dB")
    print(f"🎚 Silence Threshold: {silence_thresh:.2f} dB")

    # --------------------------------------------------
    # Detect Non-Silent Segments
    # --------------------------------------------------
    chunks = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    if not chunks:
        print("⚠️ No speech detected. Skipping trim.")
        return input_path

    print(f"🧩 Detected {len(chunks)} speech segments")

    # --------------------------------------------------
    # Expand segments slightly (Protect word edges)
    # --------------------------------------------------
    protected_chunks = []
    padding = 120  # ms extra to prevent cutting words

    for start, end in chunks:
        new_start = max(0, start - padding)
        new_end = min(len(audio), end + padding)
        protected_chunks.append((new_start, new_end))

    # --------------------------------------------------
    # Reconstruct Audio
    # --------------------------------------------------
    output_audio = AudioSegment.empty()

    for start, end in protected_chunks:
        segment = audio[start:end]

        # Add micro fade
        segment = segment.fade_in(fade_ms).fade_out(fade_ms)

        output_audio += segment
        output_audio += AudioSegment.silent(duration=keep_silence)

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    output_audio.export(output_path, format="wav")
    print(f"✅ Trimmed Output: {output_path}")

    return output_path


# --------------------------------------------------
# Manual Test
# --------------------------------------------------
if __name__ == "__main__":
    adaptive_trim(
        "inputs/raw_vocal_01.wav",
        "outputs/trimmed_test.wav",
        noise_floor_db=-48
    )
