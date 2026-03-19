import os
import numpy as np
import pyloudnorm as pyln
import soundfile as sf


# --------------------------------------------------
# True Peak Limiter (Safe Ceiling)
# --------------------------------------------------

def apply_true_peak_limiter(audio, ceiling_db=-1.0):
    """
    Apply simple true peak limiting with safety ceiling.
    """

    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio

    peak_db = 20 * np.log10(peak)

    if peak_db > ceiling_db:
        reduction_db = peak_db - ceiling_db
        reduction_linear = 10 ** (-reduction_db / 20)
        audio = audio * reduction_linear
        print(f"🛡 True Peak Limited by {reduction_db:.2f} dB")

    return audio


# --------------------------------------------------
# Advanced Intelligent Mastering
# --------------------------------------------------

def master_audio(
    input_path,
    output_path,
    target_lufs=-16.0,
    true_peak_limit=-1.0
):
    """
    Advanced Mastering:
    - Integrated LUFS normalization
    - True peak ceiling protection
    - Final loudness verification
    """

    print(f"\n🎚 Mastering: {os.path.basename(input_path)}")

    # --------------------------------------------------
    # Load Audio
    # --------------------------------------------------
    audio, sr = sf.read(input_path)

    # Convert to mono safely
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    meter = pyln.Meter(sr)

    # --------------------------------------------------
    # Measure Original Loudness
    # --------------------------------------------------
    original_loudness = meter.integrated_loudness(audio)
    print(f"📊 Original LUFS: {original_loudness:.2f} LUFS")

    # --------------------------------------------------
    # LUFS Normalization
    # --------------------------------------------------
    normalized_audio = pyln.normalize.loudness(
        audio,
        original_loudness,
        target_lufs
    )

    # --------------------------------------------------
    # Apply True Peak Limiter
    # --------------------------------------------------
    normalized_audio = apply_true_peak_limiter(
        normalized_audio,
        ceiling_db=true_peak_limit
    )

    # --------------------------------------------------
    # Final Loudness Measurement
    # --------------------------------------------------
    final_loudness = meter.integrated_loudness(normalized_audio)
    final_peak = np.max(np.abs(normalized_audio))
    final_peak_db = 20 * np.log10(final_peak) if final_peak > 0 else -100

    print(f"🎯 Final LUFS: {final_loudness:.2f} LUFS")
    print(f"🔊 Final Peak: {final_peak_db:.2f} dBTP")
    print(f"🛡 Ceiling Set To: {true_peak_limit} dBTP")

    # --------------------------------------------------
    # Save Output
    # --------------------------------------------------
    sf.write(output_path, normalized_audio, sr)
    print(f"✅ Mastered Output Saved: {output_path}")

    return output_path


# --------------------------------------------------
# Manual Test
# --------------------------------------------------
if __name__ == "__main__":
    master_audio(
        "temp/trimmed_test.wav",
        "outputs/mastered_test.wav",
        target_lufs=-16.0,
        true_peak_limit=-1.0
    )
