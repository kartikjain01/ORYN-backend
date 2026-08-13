from fish_speech.inference_engine import TTSInferenceEngine

print("\nCreating Fish Engine...\n")

engine = TTSInferenceEngine()

print("ENGINE TYPE:")
print(type(engine))

print("\nALL PUBLIC METHODS:\n")

for item in sorted(dir(engine)):
    if not item.startswith("_"):
        print(item)

print("\nMETHODS CONTAINING KEYWORDS:\n")

for keyword in ["infer", "tts", "generate", "audio", "speech"]:
    print(f"\n--- {keyword.upper()} ---")

    for item in sorted(dir(engine)):
        if keyword in item.lower():
            print(item)
