import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter, medfilt


class AdvancedDeEsserBreathControl:
    def __init__(self, sr=44100):
        self.sr = sr

    def bandpass(self, y, low=3500, high=11000, order=4):
        nyq = 0.5 * self.sr
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return lfilter(b, a, y)

    def smooth_envelope(self, x, kernel=101):
        env = np.abs(x)
        return medfilt(env, kernel_size=kernel)

    def adaptive_de_ess(self, y, base_threshold=0.08, max_reduction=0.45):
        sib = self.bandpass(y)
        env = self.smooth_envelope(sib)

        dynamic_threshold = np.mean(env) + np.std(env)
        threshold = max(base_threshold, dynamic_threshold)

        gain = np.ones_like(y)
        over = env > threshold
        strength = np.clip((env - threshold) / (threshold + 1e-8), 0, 1)
        gain[over] = 1.0 - (strength[over] * max_reduction)

        return y * gain

    def spectral_gate(self, y, n_fft=2048, hop=512):
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag = np.abs(stft)
        phase = np.angle(stft)

        noise_profile = np.median(mag, axis=1, keepdims=True)
        cleaned = np.maximum(mag - noise_profile * 0.35, 0)

        out = cleaned * np.exp(1j * phase)
        return librosa.istft(out, hop_length=hop)

    def suppress_breaths(self, y, frame=2048, hop=256, thresh_db=-38):
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame,
            hop_length=hop
        )[0]

        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=self.sr,
            hop_length=hop
        )[0]

        db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)

        breath_mask = (db < thresh_db) & (centroid > 1500)

        gains = np.where(breath_mask, 0.45, 1.0)
        gains = np.repeat(gains, hop)[:len(y)]

        return y * gains

    def remove_clicks(self, y, threshold=0.6):
        diff = np.abs(np.diff(y, prepend=y[0]))
        clicks = diff > threshold

        y[clicks] = 0.5 * (
            np.roll(y, 1)[clicks] +
            np.roll(y, -1)[clicks]
        )

        return y

    def limiter(self, y, ceiling=0.95):
        peak = np.max(np.abs(y)) + 1e-8

        if peak > ceiling:
            y = y * (ceiling / peak)

        return y

    def process(self, input_path, output_path):
        y, sr = librosa.load(input_path, sr=self.sr, mono=True)

        y = self.remove_clicks(y)
        y = self.adaptive_de_ess(y)
        y = self.spectral_gate(y)
        y = self.suppress_breaths(y)
        y = self.limiter(y)

        sf.write(output_path, y, sr)
        return output_path


# IMPORTANT: This is what your router imports
def process_uploaded_file(upload_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(
        os.path.basename(upload_path)
    )[0]

    output_path = os.path.join(
        output_dir,
        f"{base_name}_advanced_clean.wav"
    )

    engine = AdvancedDeEsserBreathControl()
    return engine.process(upload_path, output_path)


if __name__ == "__main__":
    result = process_uploaded_file("input.wav")
    print(result)
