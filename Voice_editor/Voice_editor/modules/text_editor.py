import os
import json
import whisper


# ---------------------------------------------------
# LOAD WHISPER MODEL (ONLY ONCE)
# ---------------------------------------------------

print("🔄 Loading Whisper Model for Transcript Generation...")

model = whisper.load_model("medium")

print("✅ Whisper Model Loaded Successfully!")


# ---------------------------------------------------
# TRANSCRIPT GENERATION FUNCTION
# ---------------------------------------------------

def generate_transcript(audio_path, output_dir):

    """
    Generates human readable transcript and alignment file.

    Output:
        transcript.txt  → user editable text
        alignment.json  → audio timing data
    """

    print("🎙 Running Speech Recognition...")

    result = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        word_timestamps=True
    )

    segments = result["segments"]

    transcript_lines = []
    alignment_data = []

    for i, seg in enumerate(segments):
        segments_id = i + 1

        start_time = seg["start"]
        end_time = seg["end"]

        text = seg["text"].strip()

        # ----------------------------
        # Clean Sentence
        # ----------------------------

        if len(text) == 0:
            continue

        text = text[0].upper() + text[1:]

        if not text.endswith((".", "?", "!")):
            text = text + "."

        # ----------------------------
        # Transcript Line
        # ----------------------------

        line = f"[{segments_id}] {text}"

        transcript_lines.append(line)

        # ----------------------------
        # Alignment Data
        # ----------------------------

        alignment_data.append({
            "id": segments_id,
            "start": float(start_time),
            "end": float(end_time),
            "text": text
        })

    # ---------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ---------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    transcript_file = os.path.join(output_dir, "transcript.txt")
    alignment_file = os.path.join(output_dir, "alignment.json")

    # ---------------------------------------------------
    # SAVE TRANSCRIPT
    # ---------------------------------------------------

    with open(transcript_file, "w", encoding="utf-8") as f:

        for line in transcript_lines:
            f.write(line + "\n")

    # ---------------------------------------------------
    # SAVE ALIGNMENT JSON
    # ---------------------------------------------------

    with open(alignment_file, "w", encoding="utf-8") as f:

        json.dump(alignment_data, f, indent=4)

    print("📄 Transcript Created:", transcript_file)
    print("📊 Alignment File Created:", alignment_file)

    return transcript_file, alignment_file


# ---------------------------------------------------
# EDIT TRANSCRIPT FUNCTION (USED BY clean.py)
# ---------------------------------------------------

def edit_transcript(audio_path, output_dir):

    """
    Wrapper function used by clean.py.

    Generates transcript and alignment
    and returns their paths.
    """

    print("📝 Generating editable transcript...")

    text_path, json_path = generate_transcript(
        audio_path,
        output_dir
    )

    return text_path, json_path