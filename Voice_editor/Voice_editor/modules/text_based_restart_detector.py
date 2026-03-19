import sys
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from rapidfuzz import fuzz


print("Loading Whisper model (base)...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper loaded successfully!")


def remove_text_restarts(audio, sr,
                         similarity_threshold=90,
                         time_window=6.0):

    print("🧠 Running Text-Based Restart Detection...")

    segments, _ = model.transcribe(audio)

    segments = list(segments)

    if len(segments) < 2:
        return audio

    cleaned_audio = audio.copy()
    removed_regions = []

    for i in range(1, len(segments)):

        prev_text = segments[i - 1].text.strip().lower()
        curr_text = segments[i].text.strip().lower()

        similarity = fuzz.ratio(prev_text, curr_text)
        time_gap = segments[i].start - segments[i - 1].end

        if similarity > similarity_threshold and time_gap < time_window:

            print(f"⚠️ Duplicate detected:")
            print(f"   \"{curr_text}\"")
            print(f"   Similarity: {similarity}%")

            start_sample = int(segments[i].start * sr)
            end_sample = int(segments[i].end * sr)

            removed_regions.append((start_sample, end_sample))

    for start, end in reversed(removed_regions):
        cleaned_audio = np.concatenate(
            (cleaned_audio[:start], cleaned_audio[end:])
        )

    print(f"🧹 Removed {len(removed_regions)} duplicate sentence(s)")
    return cleaned_audio


# ---------------------------------------------------
# CLI Execution Mode (IMPORTANT FOR OPTION B)
# ---------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python text_based_restart_detector.py input.wav")
        sys.exit(1)

    input_path = sys.argv[1]

    audio, sr = sf.read(input_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    cleaned_audio = remove_text_restarts(audio, sr)

    output_path = input_path.replace(".wav", "_text_clean.wav")
    sf.write(output_path, cleaned_audio, sr)

    print(f"✅ Saved cleaned file: {output_path}")
    
def detect_repetitions(word_list, threshold=3):
    """
    Detect consecutive repeated words.
    Returns list of (start_time, end_time) ranges to remove.
    """

    repeats = []
    i = 0

    while i < len(word_list) - threshold:
        current_word = word_list[i]["word"].strip().lower()
        count = 1

        j = i + 1
        while j < len(word_list) and \
              word_list[j]["word"].strip().lower() == current_word:
            count += 1
            j += 1

        if count >= threshold:
            start_time = word_list[i]["start"]
            end_time = word_list[j - 1]["end"]
            repeats.append((start_time, end_time))

        i = j

    return repeats    