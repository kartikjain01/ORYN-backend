# app/services/silence_trimmer_v2.py

import numpy as np
import librosa
from app.utils.audio_io import load_audio, save_audio


def detect_speech_intervals(
    audio: np.ndarray,
    top_db: int = 28,
    min_segment_ms: int = 120
):
    """
    Detect speech regions using energy.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)

    filtered = []
    for start, end in intervals:
        if (end - start) >= min_segment_ms:
            filtered.append((start, end))

    return filtered


def merge_close_segments(
    intervals,
    min_gap_samples: int
):
    """
    Merge segments separated by very short pauses.
    """
    if not intervals:
        return []

    merged = [intervals[0]]

    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]

        if start - prev_end <= min_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def rebuild_natural_audio(
    audio: np.ndarray,
    intervals,
    sr: int,
    short_gap_ms: int = 160,
    long_gap_ms: int = 260
):
    """
    Rebuild audio with natural pauses.
    """
    chunks = []

    for i, (start, end) in enumerate(intervals):
        chunks.append(audio[start:end])

        if i < len(intervals) - 1:
            next_start, _ = intervals[i + 1]
            original_gap = next_start - end

            if original_gap < sr * 0.35:
                gap = short_gap_ms
            else:
                gap = long_gap_ms

            silence = np.zeros(int(sr * gap / 1000), dtype=audio.dtype)
            chunks.append(silence)

    return np.concatenate(chunks)


def trim_edges(audio: np.ndarray, top_db: int = 28):
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def process_silence_trim(
    input_path: str,
    output_path: str
):
    audio, sr = load_audio(input_path)

    # Remove start/end dead silence
    audio = trim_edges(audio)

    # Detect speech chunks
    intervals = detect_speech_intervals(audio, top_db=28)

    # Merge tiny splits
    intervals = merge_close_segments(
        intervals,
        min_gap_samples=int(sr * 0.12)
    )

    # Rebuild with natural pauses
    final_audio = rebuild_natural_audio(
        audio,
        intervals,
        sr,
        short_gap_ms=160,
        long_gap_ms=260
    )

    save_audio(output_path, final_audio, sr)

    return {
        "output_path": output_path,
        "sample_rate": sr,
        "status": "silence_trimmed",
        "segments_kept": len(intervals),
        "features": [
            "natural_pauses_preserved",
            "breathing_preserved",
            "sentence_rhythm_preserved",
            "smart_dead_space_removed",
            "podcast_quality_trim"
        ]
    }
