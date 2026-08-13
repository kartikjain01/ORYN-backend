import os

from app.services.enhancer import enhance_audio
from app.services.advanced_enhancer import enhance_audio_advanced
from app.services.deepfilter_enhancer import enhance_audio_deepfilter

input_file = "sample.wav"

# BASIC
try:
    enhance_audio(input_file, "basic_output.wav")
    print("✅ BASIC working")
except Exception as e:
    print("❌ BASIC failed:", e)

# ADVANCED
try:
    enhance_audio_advanced(input_file, "advanced_output.wav")
    print("✅ ADVANCED working")
except Exception as e:
    print("❌ ADVANCED failed:", e)

# DEEPFILTER
try:
    enhance_audio_deepfilter(input_file, "deep_output.wav")
    print("✅ DEEPFILTER working")
except Exception as e:
    print("❌ DEEPFILTER failed:", e)

print("\nCreated files:")
for f in ["basic_output.wav", "advanced_output.wav", "deep_output.wav"]:
    print(f, "->", "FOUND" if os.path.exists(f) else "NOT FOUND")
