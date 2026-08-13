import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

class SmartFinalAudioPolishing:
    def __init__(self, sr=44100):
        self.sr = sr

    def highpass(self, y, cutoff=80, order=2):
        nyq = 0.5 * self.sr
        b, a = butter(order, cutoff / nyq, btype='high')
        return lfilter(b, a, y)

    def band(self, y, low, high, order=2):
        nyq = 0.5 * self.sr
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return lfilter(b, a, y)

    def adaptive_presence(self, y):
        energy = np.mean(np.abs(y))
        amount = 0.08 if energy > 0.08 else 0.18
        return y + amount * self.band(y, 2500, 5500)

    def adaptive_air(self, y):
        energy = np.mean(np.abs(y))
        amount = 0.04 if energy > 0.08 else 0.10
        return y + amount * self.band(y, 9000, 14000)

    def smart_compress(self, y, threshold=0.25, ratio=3.0):
        mag = np.abs(y)
        over = mag > threshold
        y2 = y.copy()
        y2[over] = np.sign(y[over]) * (threshold + (mag[over] - threshold) / ratio)
        return y2

    def loudness_target(self, y, target_rms=0.18):
        rms = np.sqrt(np.mean(y ** 2)) + 1e-8
        return y * (target_rms / rms)

    def limiter(self, y, ceiling=0.95):
        peak = np.max(np.abs(y)) + 1e-8
        if peak > ceiling:
            y = y * (ceiling / peak)
        return y

    def soft_clip(self, y):
        return np.tanh(y)

    def process(self, input_path, output_path):
        y, sr = librosa.load(input_path, sr=self.sr, mono=True)
        y = self.highpass(y)
        y = self.adaptive_presence(y)
        y = self.adaptive_air(y)
        y = self.smart_compress(y)
        y = self.loudness_target(y)
        y = self.soft_clip(y)
        y = self.limiter(y)
        sf.write(output_path, y, sr)
        return output_path


def process_uploaded_file(upload_path, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(upload_path))[0]
    output_path = os.path.join(output_dir, f'{base}_youtube_polished.wav')
    engine = SmartFinalAudioPolishing()
    return engine.process(upload_path, output_path)

