import os
import json
import re
from pydub import AudioSegment


# ---------------------------------------------------
# LOAD TRANSCRIPT (CLEAN TIMESTAMPS)
# ---------------------------------------------------

def load_transcript(path):

    print("📄 Loading transcript...")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if line == "":
            continue

        # Remove timestamps like [0.00 - 2.04]
        line = re.sub(r"\[.*?\]", "", line)

        # Normalize text
        line = line.lower()
        line = re.sub(r"[^\w\s]", "", line)
        line = line.strip()

        cleaned_lines.append(line)

    print(f"📝 Transcript lines detected: {len(cleaned_lines)}")

    return cleaned_lines


# ---------------------------------------------------
# LOAD ALIGNMENT
# ---------------------------------------------------

def load_alignment(path):

    print("📊 Loading alignment...")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        segments = data.get("segments", [])
    else:
        segments = data

    print(f"📊 Alignment segments detected: {len(segments)}")

    return segments


# ---------------------------------------------------
# CLEAN TEXT
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    return text


# ---------------------------------------------------
# SYNC TRANSCRIPT WITH ALIGNMENT
# ---------------------------------------------------

def sync_transcript_alignment(transcript_lines, segments):

    print("🔍 Matching transcript with alignment...")

    new_segments = []
    removed_count = 0

    for seg in segments:

        seg_text = clean_text(seg["text"])

        if seg_text in transcript_lines:
            new_segments.append(seg)
        else:
            removed_count += 1

    print(f"🧹 Removed {removed_count} segment(s)")

    return new_segments


# ---------------------------------------------------
# REBUILD AUDIO
# ---------------------------------------------------

def rebuild_audio(audio_path, segments, output_path):

    print("🎧 Loading audio...")

    audio = AudioSegment.from_file(audio_path)

    final_audio = AudioSegment.empty()

    print(f"✂️ Rebuilding audio from {len(segments)} segments...")

    for i, seg in enumerate(segments):

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        clip = audio[start_ms:end_ms]

        final_audio += clip

        print(f"  + Segment {i+1} added")

    print("💾 Exporting updated audio...")

    final_audio.export(output_path, format="wav")

    print(f"\n✅ Updated audio created: {output_path}")
    print(f"📊 Final Duration: {len(final_audio)/1000:.2f} seconds")


# ---------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------

def reconstruct_audio_from_text(audio_path, alignment_json, edited_text_path, output_path):

    transcript_lines = load_transcript(edited_text_path)

    segments = load_alignment(alignment_json)

    updated_segments = sync_transcript_alignment(
        transcript_lines,
        segments
    )

    rebuild_audio(
        audio_path,
        updated_segments,
        output_path
    )