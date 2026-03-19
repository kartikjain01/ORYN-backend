import whisper
import json
import os

def generate_transcript(audio_path, output_dir):
    """
    Generates:
    1. Plain text transcript
    2. Word-level alignment JSON
    """

    os.makedirs(output_dir, exist_ok=True)

    model = whisper.load_model("base")

    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False
    )

    words = []

    for segment in result["segments"]:
        for word_info in segment["words"]:
            words.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Save text file
    text_path = os.path.join(output_dir, base_name + ".txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    # Save alignment JSON
    json_path = os.path.join(output_dir, base_name + "_alignment.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)

    return text_path, json_path