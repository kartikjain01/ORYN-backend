import whisper
import numpy as np
import soundfile as sf


def remove_repeated_sentences(audio_path, output_path, model_size="base"):
    """
    Detect repeated sentences using Whisper transcription
    and remove the duplicate audio segments automatically.
    """

    print("🔄 Loading Whisper model...")
    model = whisper.load_model(model_size)

    print("🎧 Loading audio...")
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print("🧠 Transcribing audio...")
    result = model.transcribe(audio_path)

    segments = result["segments"]

    kept_segments = []
    seen_text = set()

    removed_count = 0

    for seg in segments:

        text = seg["text"].strip().lower()

        if text in seen_text:
            removed_count += 1
            continue

        seen_text.add(text)
        kept_segments.append(seg)

    print(f"🧹 Removed {removed_count} repeated sentence(s)")

    # rebuild audio
    cleaned_audio = []

    for seg in kept_segments:

        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)

        cleaned_audio.append(audio[start:end])

    if cleaned_audio:
        cleaned_audio = np.concatenate(cleaned_audio)
    else:
        cleaned_audio = audio

    sf.write(output_path, cleaned_audio, sr)

    print("✅ Cleaned audio saved:", output_path)

    return output_path