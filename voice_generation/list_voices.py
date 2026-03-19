from kokoro_onnx import Kokoro

# Load the model metadata
kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

# Print all available voice IDs
print("--- 🎤 AVAILABLE KOKORO VOICES ---")
voices = kokoro.get_voices()
for voice in voices:
    print(f"ID: {voice}")
print(f"----------------------------------")
print(f"Total Voices Found: {len(voices)}")