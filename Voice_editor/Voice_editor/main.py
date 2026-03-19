import librosa
import numpy as np
import soundfile as sf
import os

# 1. Create a 1-second 'A' note (440Hz)
sr = 22050  # Sample rate
duration = 1.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
audio_data = 0.5 * np.sin(2 * np.pi * 440.0 * t)

# 2. Save the file using soundfile
output_filename = "test_beep.wav"
sf.write(output_filename, audio_data, sr)
print(f"✅ Created file: {output_filename}")

# 3. Load it back using librosa (this tests FFmpeg integration)
y, s = librosa.load(output_filename, sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=s)

print(f"✅ Librosa successfully loaded the file.")
print(f"📊 Audio Duration: {librosa.get_duration(y=y, sr=s):.2f} seconds")