import json
import re
import argparse


def sync_transcript_with_alignment(transcript_path, alignment_path):

    print("📄 Loading transcript...")

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ----------------------------------------
    # Extract IDs from transcript
    # ----------------------------------------

    transcript_ids = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        match = re.search(r"\[(\d+)\]", line)

        if match:
            segment_id = int(match.group(1))
            transcript_ids.append(segment_id)

    print(f"🧠 Transcript segments found: {len(transcript_ids)}")

    if len(transcript_ids) == 0:
        print("⚠ No segments detected in transcript")
        return

    # ----------------------------------------
    # Load alignment
    # ----------------------------------------

    print("📊 Loading alignment...")

    with open(alignment_path, "r", encoding="utf-8") as f:
        alignment = json.load(f)

    print(f"🧠 Alignment segments: {len(alignment)}")

    new_alignment = []
    removed = 0

    for segment in alignment:

        if segment["id"] in transcript_ids:
            new_alignment.append(segment)
        else:
            removed += 1
            print(f"❌ Removing: {segment['text']}")

    # ----------------------------------------
    # Save updated alignment
    # ----------------------------------------

    print(f"🧹 Removed {removed} segment(s)")

    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(new_alignment, f, indent=4)

    print("✅ alignment.json updated successfully")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--transcript", required=True)
    parser.add_argument("--alignment", required=True)

    args = parser.parse_args()

    sync_transcript_with_alignment(
        args.transcript,
        args.alignment
    )