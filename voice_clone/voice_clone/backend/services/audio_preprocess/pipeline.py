# backend/services/audio_preprocess/pipeline.py
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from backend.services.audio_preprocess.denoise import denoise_deepfilternet, DenoiseConfig
from backend.services.audio_preprocess.dereverb import simple_reduce_tail, DereverbConfig

import numpy as np

try:
    import soundfile as sf  # pip install soundfile
except Exception:
    sf = None

try:
    from scipy.io import wavfile  # pip install scipy
except Exception:
    wavfile = None


@dataclass(frozen=True)
class PreprocessConfig:
    target_sr: int = 24000
    target_rms: float = 0.10          # linear RMS target (0..1)
    peak_limit: float = 0.99          # prevent clipping after normalize
    ffmpeg_path: str = "ffmpeg"


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stderr}")


def decode_to_wav_mono_resampled(
    input_path: str | os.PathLike,
    output_wav_path: str | os.PathLike,
    cfg: PreprocessConfig,
) -> None:
    """
    Stage A: Decode ANY input -> WAV, mono, target_sr using ffmpeg.
    """
    in_p = str(Path(input_path))
    out_p = str(Path(output_wav_path))
    cmd = [
        cfg.ffmpeg_path, "-y",
        "-i", in_p,
        "-ac", "1",                    # mono
        "-ar", str(cfg.target_sr),     # resample
        "-vn",
        "-f", "wav",
        out_p,
    ]
    _run(cmd)


def _load_wav(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    p = str(Path(path))
    if sf is not None:
        audio, sr = sf.read(p, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32), int(sr)

    if wavfile is None:
        raise RuntimeError("Need either 'soundfile' or 'scipy' installed to read WAVs.")

    sr, x = wavfile.read(p)
    # Convert to float32 [-1, 1]
    if x.dtype == np.int16:
        audio = (x.astype(np.float32) / 32768.0)
    elif x.dtype == np.int32:
        audio = (x.astype(np.float32) / 2147483648.0)
    elif x.dtype == np.float32 or x.dtype == np.float64:
        audio = x.astype(np.float32)
    else:
        audio = x.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


def _save_wav(path: str | os.PathLike, audio: np.ndarray, sr: int) -> None:
    p = str(Path(path))
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    if sf is not None:
        sf.write(p, audio, sr, subtype="PCM_16")
        return

    if wavfile is None:
        raise RuntimeError("Need either 'soundfile' or 'scipy' installed to write WAVs.")

    wavfile.write(p, sr, (audio * 32767.0).astype(np.int16))


def rms_normalize(audio: np.ndarray, target_rms: float, peak_limit: float) -> np.ndarray:
    """
    Stage B: Basic RMS normalization (LUFS later).
    """
    x = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < 1e-8:
        return x

    gain = target_rms / rms
    y = x * gain

    peak = float(np.max(np.abs(y)) + 1e-12)
    if peak > peak_limit:
        y = y * (peak_limit / peak)
    return y.astype(np.float32)


def preprocess_stage1(
    input_audio_path: str | os.PathLike,
    output_clean_wav_path: str | os.PathLike,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> Path:
    """
    Pipeline stub (no denoise yet):
      1) decode -> wav mono @ target sr
      2) rms normalize
      3) write clean_stage1.wav
    """
    out_p = Path(output_clean_wav_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        decoded = Path(td) / "decoded.wav"
        decode_to_wav_mono_resampled(input_audio_path, decoded, cfg)

        audio, sr = _load_wav(decoded)
        if sr != cfg.target_sr:
            raise RuntimeError(f"Decode sr mismatch: got {sr}, expected {cfg.target_sr}")

        clean = rms_normalize(audio, cfg.target_rms, cfg.peak_limit)
        _save_wav(out_p, clean, sr)

    return out_p

def preprocess_stage1_5(
    input_stage1_wav: str | os.PathLike,
    output_stage1_5_wav: str | os.PathLike,
    cfg: PreprocessConfig = PreprocessConfig(),
    denoise_cfg: DenoiseConfig = DenoiseConfig(enabled=True),
    dereverb_cfg: DereverbConfig = DereverbConfig(enabled=True),
) -> Path:
    """
    Stage 1.5:
      clean_stage1.wav -> denoise (DeepFilterNet) -> simple dereverb tail reduce -> write clean_stage1_5.wav
    """
    in_p = Path(input_stage1_wav)
    out_p = Path(output_stage1_5_wav)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = _load_wav(in_p)
    if sr != cfg.target_sr:
        raise RuntimeError(f"Expected sr={cfg.target_sr}, got sr={sr}.")

    # 1) Denoise (DeepFilterNet)
    audio = denoise_deepfilternet(audio, sr, denoise_cfg)

    # 2) Dereverb (simple baseline)
    audio = simple_reduce_tail(audio, sr, dereverb_cfg)

    # 3) Safety normalize again (optional but recommended)
    audio = rms_normalize(audio, target_rms=cfg.target_rms, peak_limit=cfg.peak_limit)

    _save_wav(out_p, audio, sr)
    return out_p


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input audio file (mp3/wav/m4a/etc.)")
    ap.add_argument("--output", required=True, help="Output WAV path (e.g. clean_stage1.wav)")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--target-rms", type=float, default=0.10)
    ap.add_argument("--peak-limit", type=float, default=0.99)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()

    cfg = PreprocessConfig(
        target_sr=args.sr,
        target_rms=args.target_rms,
        peak_limit=args.peak_limit,
        ffmpeg_path=args.ffmpeg,
    )
    out = preprocess_stage1(args.input, args.output, cfg)
    print(str(out))


if __name__ == "__main__":
    _cli()


from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from backend.services.text_intel.normalize_text import normalize_text, NormalizeConfig
from backend.services.text_intel.ssml import ssml_to_text, SSMLParseResult
from backend.services.text_intel.grammar import grammar_correct, GrammarConfig
from backend.services.text_intel.emotion_tags import apply_emotion_style
from backend.services.text_intel.nlp import run_nlp_light
from backend.services.text_intel.phonemes import text_to_phonemes, PhonemeConfig


@dataclass(frozen=True)
class TextIntelConfig:
    normalize: NormalizeConfig = NormalizeConfig()
    grammar: GrammarConfig = GrammarConfig()
    phonemes: PhonemeConfig = PhonemeConfig()
    # style examples: "neutral", "calm", "energetic", "storytelling", "sad"
    default_style: str = "neutral"


@dataclass(frozen=True)
class TextIntelResult:
    text: str
    language: str
    sentences: List[str]
    pauses_ms: List[int]
    xtts_param_overrides: Dict[str, Any]


def process_text(
    raw_text_or_ssml: str,
    style: Optional[str] = None,
    cfg: TextIntelConfig = TextIntelConfig(),
) -> TextIntelResult:
    # 1) SSML -> text + pause tokens
    ssml_res: SSMLParseResult = ssml_to_text(raw_text_or_ssml)

    # 2) Normalize (whitespace/abbr/numbers)
    t = normalize_text(ssml_res.text, cfg.normalize)

    # 3) Grammar light cleanup
    t = grammar_correct(t, cfg.grammar)

    # 4) NLP light (language + sentence split)
    nlp = run_nlp_light(t)

    # 5) Emotion style (param overrides only)
    emo = apply_emotion_style(t, style or cfg.default_style)

    # 6) Optional phonemes (OFF by default)
    final_text = text_to_phonemes(emo.text, nlp.language, cfg.phonemes)

    # Re-split after phonemes conversion (usually unchanged if phonemes disabled)
    nlp2 = run_nlp_light(final_text)

    return TextIntelResult(
        text=final_text,
        language=nlp2.language,
        sentences=nlp2.sentences,
        pauses_ms=ssml_res.pauses_ms,
        xtts_param_overrides=emo.xtts_param_overrides,
    )
from backend.services.tts_engine.generate import generate_chunks
from backend.services.postprocess.merge import merge_chunks, MergeConfig

job_id = "job_001"
chunks = generate_chunks(
    job_id=job_id,
    text="Long text here ...",
    language="en",
    speaker_wav_path="data/voices/test_voice/clean/reference_clean.wav",
)

final = merge_chunks(
    chunk_wavs=chunks,
    out_path=f"data/generations/{job_id}/final.wav",
    cfg=MergeConfig(crossfade_ms=80, loudnorm=True),
)
print("Final:", final)