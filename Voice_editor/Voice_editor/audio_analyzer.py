import numpy as np


class AudioAnalyzer:
    def __init__(self, audio: np.ndarray, sample_rate: int):
        """
        audio: mono float32 numpy array (-1.0 to 1.0)
        sample_rate: sampling rate
        """
        self.audio = audio
        self.sample_rate = sample_rate

    # --------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------

    def get_peak(self):
        peak = np.max(np.abs(self.audio))
        peak_db = 20 * np.log10(peak + 1e-12)
        return peak, peak_db

    def get_rms(self):
        rms = np.sqrt(np.mean(self.audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-12)
        return rms, rms_db

    def get_duration(self):
        return len(self.audio) / self.sample_rate

    # --------------------------------------------------
    # DYNAMIC RANGE
    # --------------------------------------------------

    def get_dynamic_range(self):
        _, peak_db = self.get_peak()
        _, rms_db = self.get_rms()
        crest_factor = peak_db - rms_db
        return crest_factor

    # --------------------------------------------------
    # NOISE FLOOR ESTIMATION
    # --------------------------------------------------

    def estimate_noise_floor(self, percentile=10):
        """
        Estimate noise floor using low-energy percentile.
        Default = bottom 10% of signal energy.
        """
        abs_signal = np.abs(self.audio)
        noise_level = np.percentile(abs_signal, percentile)
        noise_db = 20 * np.log10(noise_level + 1e-12)
        return noise_level, noise_db

    # --------------------------------------------------
    # SNR CALCULATION
    # --------------------------------------------------

    def calculate_snr(self):
        """
        SNR = RMS_signal - Noise_floor (in dB)
        """
        _, rms_db = self.get_rms()
        _, noise_db = self.estimate_noise_floor()
        snr = rms_db - noise_db
        return snr

    # --------------------------------------------------
    # IMPROVED SILENCE RATIO
    # --------------------------------------------------

    def get_silence_ratio(self):
        """
        Silence threshold = noise_floor + 6 dB
        This adapts based on recording environment.
        """
        _, noise_db = self.estimate_noise_floor()
        adaptive_threshold_db = noise_db + 6
        threshold_linear = 10 ** (adaptive_threshold_db / 20)

        silent_samples = np.sum(np.abs(self.audio) < threshold_linear)
        total_samples = len(self.audio)

        ratio = silent_samples / total_samples
        return ratio

    # --------------------------------------------------
    # CLIPPING RATIO
    # --------------------------------------------------

    def get_clipping_ratio(self, threshold=0.999):
        clipped_samples = np.sum(np.abs(self.audio) >= threshold)
        total_samples = len(self.audio)
        return clipped_samples / total_samples

    # --------------------------------------------------
    # MASTER ANALYZE
    # --------------------------------------------------

    def analyze(self):
        peak, peak_db = self.get_peak()
        rms, rms_db = self.get_rms()
        duration = self.get_duration()
        dynamic_range = self.get_dynamic_range()
        noise_level, noise_db = self.estimate_noise_floor()
        snr = self.calculate_snr()
        silence_ratio = self.get_silence_ratio()
        clipping_ratio = self.get_clipping_ratio()


        return {
            "technical": {
                "duration_sec": round(duration, 3),
                "sample_rate": self.sample_rate
            },
            "loudness": {
                "peak_dbfs": round(float(peak_db), 2),
                "rms_dbfs": round(float(rms_db), 2),
            },
            "dynamics": {
                "crest_factor_db": round(float(dynamic_range), 2)
            },
            "noise_analysis": {
                "noise_floor_dbfs": round(float(noise_db), 2),
                "snr_db": round(float(snr), 2)
            },
            "silence": {
                "silence_ratio": round(float(silence_ratio), 4)
            },
            "clipping": {
                "clipping_ratio": round(float(clipping_ratio), 6),
                "is_clipping": clipping_ratio > 0
            }
 
        }
