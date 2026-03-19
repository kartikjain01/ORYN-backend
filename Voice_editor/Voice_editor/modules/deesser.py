import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter


def bandpass_filter(data, sr, lowcut=5000, highcut=10000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def smooth_envelope(env, attack=0.2, release=0.01):
    """
    Smooth gain envelope to prevent clicks
    """
    smoothed = np.zeros_like(env)
    prev = env[0]

    for i, val in enumerate(env):
        if val < prev:
            coeff = attack
        else:
            coeff = release

        prev = coeff * prev + (1 - coeff) * val
        smoothed[i] = prev

    return smoothed


def apply_deesser(input_path, output_path,
                  threshold_db=-25,
                  reduction_db=6,
                  lowcut=5000,
                  highcut=10000):

    y, sr = librosa.load(input_path, sr=None)

    # Detect sibilance band
    sibilance_band = bandpass_filter(y, sr, lowcut, highcut)

    frame_length = int(0.02 * sr)  # 20ms
    hop_length = int(0.01 * sr)

    rms = librosa.feature.rms(
        y=sibilance_band,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Calculate gain reduction
    reduction_linear = 10 ** (-reduction_db / 20)

    gain_reduction = np.where(
        rms_db > threshold_db,
        reduction_linear,
        1.0
    )

    # Smooth gain envelope
    gain_reduction = smooth_envelope(gain_reduction)

    # Interpolate envelope to audio length
    frame_positions = np.arange(len(gain_reduction)) * hop_length
    sample_positions = np.arange(len(y))

    gain_env = np.interp(sample_positions, frame_positions, gain_reduction)

    processed = y * gain_env

    sf.write(output_path, processed, sr)

    return output_path