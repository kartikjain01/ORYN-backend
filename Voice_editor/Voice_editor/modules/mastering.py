import numpy as np
import pyloudnorm as pyln

def master_audio(audio, sr, target_lufs=-16.0, true_peak_ceiling=-1.0):
    """
    Professional mastering stage
    """

    # --------------------------------
    # Loudness meter
    # --------------------------------
    meter = pyln.Meter(sr)

    loudness = meter.integrated_loudness(audio)

    print(f"\n🎚 Mastering Stage")
    print(f"📊 Original LUFS: {loudness:.2f}")

    # --------------------------------
    # Normalize loudness
    # --------------------------------
    normalized_audio = pyln.normalize.loudness(
        audio,
        loudness,
        target_lufs
    )

    # --------------------------------
    # True peak limiting
    # --------------------------------
    peak = np.max(np.abs(normalized_audio))

    peak_db = 20 * np.log10(peak + 1e-8)

    if peak_db > true_peak_ceiling:

        reduction_db = peak_db - true_peak_ceiling
        gain = 10 ** (-reduction_db / 20)

        normalized_audio = normalized_audio * gain

        print(f"🛡 True Peak Limited by {reduction_db:.2f} dB")

    # --------------------------------
    # Final measurement
    # --------------------------------
    final_loudness = meter.integrated_loudness(normalized_audio)
    final_peak = 20 * np.log10(np.max(np.abs(normalized_audio)) + 1e-8)

    print(f"🎯 Final LUFS: {final_loudness:.2f}")
    print(f"🔊 Final Peak: {final_peak:.2f} dBTP")

    return normalized_audio