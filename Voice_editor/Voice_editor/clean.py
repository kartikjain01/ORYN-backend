import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import shutil
import argparse
import logging

import torch
import torchaudio
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from audio_analyzer import AudioAnalyzer
from ingest_audio import load_audio
from adaptive_trim import adaptive_trim
from mastering import master_audio

from modules.compressor import (
    apply_adaptive_compression,
    apply_parallel_compression
)

from modules.intelligent_eq import apply_intelligent_eq
from modules.text_based_restart_detector import remove_text_restarts
from modules.echo_remover import remove_echo
from modules.deesser import apply_deesser
from modules.saturation import apply_harmonic_saturation
from modules.downward_expander import apply_downward_expander
from modules.text_editor import edit_transcript
from modules.repetition_remover import remove_repeated_sentences

try:
    from ai_denoiser import process_ai_denoise
except ImportError as e:
    print(f"❌ Custom Script Missing: {e}")
    sys.exit(1)


from huggingface_hub import hf_hub_download
import zipfile

MODEL_DIR = "DeepFilterNet"

def download_model():
    if not os.path.exists(MODEL_DIR):
        print("⬇️ Downloading DeepFilterNet from Hugging Face...")

        zip_path = hf_hub_download(
            repo_id="Kartikjain12345/deepfilternet",
            filename="DeepFilterNet.zip"
        )

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("✅ Model ready!")
    else:
        print("✅ Model already exists")


#remove
def save_debug(audio_tensor, sr, name):
    import soundfile as sf, os, torch

    os.makedirs("debug_outputs", exist_ok=True)

    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.squeeze().cpu().numpy()

    path = f"debug_outputs/{name}.wav"

    sf.write(path, audio_tensor, sr)

    print(f"Saved debug audio: {path}")
    #remove

# ---------------------------------------------------
# DECISION ENGINE
# ---------------------------------------------------

class DecisionEngine:

    def __init__(self, snr, noise_floor, clipping):

        self.snr = snr
        self.noise_floor = noise_floor
        self.clipping = clipping

    def run_denoise(self):

        if self.snr < 35:
            return True

        return False

    def run_echo_removal(self):

        if self.snr < 40:
            return True

        return False

    def run_deesser(self):

        if self.snr > 20:
            return True

        return False

    def run_expander(self):

        if self.noise_floor > -60:
            return True

        return False

    def run_compressor(self):

        if self.clipping:
            return True

        if self.snr > 15:
            return True

        return False


# ---------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/voice_editor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VoiceEditor")


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

def main():

    download_model()
    parser = argparse.ArgumentParser(description="Voice Editor Production Pipeline")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--profile", type=str, default="default")
    parser.add_argument("--genre", type=str, default="podcast")
    parser.add_argument("--output", type=str, default="outputs/output.wav")
    parser.add_argument("--skip-echo", action="store_true")
    parser.add_argument("--text-restart-detect", action="store_true")

    args = parser.parse_args()

    temp_dir = "temp"
    output_dir = "outputs"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    raw_path = args.input

    if not os.path.exists(raw_path):
        logger.error(f"❌ FILE NOT FOUND: {raw_path}")
        return

    base_name = os.path.splitext(os.path.basename(raw_path))[0]

    logger.info("====================================")
    logger.info(f"🚀 Starting Pipeline: {base_name}")
    logger.info("====================================")

    # ---------------------------------------------------
    # STEP 1 — ANALYSIS
    # ---------------------------------------------------

    try:

        logger.info("🔍 Running Audio Analysis...")

        raw_np, raw_sr = load_audio(raw_path)
#remove
        save_debug(torch.tensor(raw_np), raw_sr, "00_input")
#remove
        analyzer = AudioAnalyzer(raw_np, raw_sr)

        report = analyzer.analyze()

        noise_analysis = report.get("noise_analysis", {})
        clipping_data = report.get("clipping", {})

        snr_value = noise_analysis.get("snr_db", 20.0)
        noise_floor = noise_analysis.get("noise_floor_db", -50.0)
        clipping_flag = clipping_data.get("is_clipping", False)

        logger.info(f"SNR: {snr_value:.2f} dB")
        logger.info(f"Noise Floor: {noise_floor:.2f} dB")

        if clipping_flag:
            logger.warning("⚠ Input contains clipping")

        decision = DecisionEngine(
            snr_value,
            noise_floor,
            clipping_flag
        )

    except Exception as e:

        logger.error(f"Analysis failed: {e}")
        return


    # ---------------------------------------------------
    # STEP 2 — AI DENOISE
    # ---------------------------------------------------

    try:

        denoised_file = os.path.join(temp_dir, f"{base_name}_denoised.wav")

        if decision.run_denoise():

            logger.info("Running AI Denoise...")

            denoised_path = process_ai_denoise(
                raw_path,
                denoised_file,
                snr=snr_value
            )
#remove
            audio_np, sr = load_audio(denoised_path)
            save_debug(torch.tensor(audio_np), sr, "01_denoised")
#remove

        else:

            logger.info("Skipping Denoise (clean recording)")
            denoised_path = raw_path

        if not os.path.exists(denoised_path):
            denoised_path = raw_path

    except Exception as e:

        logger.error(f"Denoise error: {e}")
        denoised_path = raw_path


    # ---------------------------------------------------
    # STEP 3 — ADAPTIVE TRIM
    # ---------------------------------------------------

    try:

        logger.info("Running Adaptive Trim...")

        trimmed_file = os.path.join(temp_dir, f"{base_name}_trimmed.wav")

        trimmed_path = adaptive_trim(
            denoised_path,
            trimmed_file,
            noise_floor_db=noise_floor
        )
#remove
        audio_np, sr = sf.read(trimmed_path)
        save_debug(torch.tensor(audio_np), sr, "02_trimmed")
#remove
        if not os.path.exists(trimmed_path):
            logger.error("Adaptive trim failed.")
            return

        # ---------------------------------------------------
        # APPLY MICRO FADES (VERY IMPORTANT)
        # ---------------------------------------------------

        logger.info("Applying micro fades to prevent clicks...")

        audio_np, sr = sf.read(trimmed_path)

        audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

        # Convert to [channels, samples]
        if audio_tensor.ndim == 1:
           audio_tensor = audio_tensor.unsqueeze(0)
        else:
           audio_tensor = audio_tensor.T

        fade_samples = int(sr * 0.01)

        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        fade_out = torch.linspace(1.0, 0.0, fade_samples)

        audio_tensor[:, :fade_samples] *= fade_in
        audio_tensor[:, -fade_samples:] *= fade_out

        sf.write(trimmed_path, audio_tensor.T.numpy(), sr)

        logger.info("Micro fades applied successfully.")

    except Exception as e:

        logger.error(f"Trim error: {e}")
        return


    # ---------------------------------------------------
    # STEP 4 — ECHO REMOVAL
    # ---------------------------------------------------

    dry_file = os.path.join(temp_dir, f"{base_name}_dry.wav")

    try:

        if args.skip_echo:
            shutil.copy(trimmed_path, dry_file)

        else:

            if decision.run_echo_removal():

                logger.info("Running Echo Removal...")
                remove_echo(trimmed_path, dry_file)

            else:

                logger.info("Skipping Echo Removal")
                shutil.copy(trimmed_path, dry_file)

        if not os.path.exists(dry_file):
            dry_file = trimmed_path
#remove
            audio_np, sr = sf.read(dry_file)
            save_debug(torch.tensor(audio_np), sr, "03_echo_removed")
#remove
    except Exception as e:

        logger.error(f"Echo stage failed: {e}")
        dry_file = trimmed_path


    # ---------------------------------------------------
    # STEP 5 — REPEAT REMOVAL
    # ---------------------------------------------------

    try:

        logger.info("Running Repetition Removal...")

        repeat_removed_path = os.path.join(
            temp_dir,
            f"{base_name}_no_repetition.wav"
        )

        remove_repeated_sentences(
            dry_file,
            repeat_removed_path
        )
#remove
        audio_np, sr = sf.read(repeat_removed_path)
        save_debug(torch.tensor(audio_np), sr, "04_no_repetition")
#remove
        if os.path.exists(repeat_removed_path):
            dry_file = repeat_removed_path

    except Exception as e:

        logger.warning(f"Repetition removal failed: {e}")


    # ---------------------------------------------------
    # STEP 6 — RESTART DETECTION + DE-ESSER
    # ---------------------------------------------------

    try:

        logger.info("Running Restart Detection...")

        audio_np, sr = sf.read(dry_file)

        if args.text_restart_detect:
            cleaned_audio = remove_text_restarts(audio_np, sr)
        else:
            cleaned_audio = audio_np

        restart_file = os.path.join(
            temp_dir,
            f"{base_name}_restart_clean.wav"
        )

        sf.write(restart_file, cleaned_audio, sr)
#remove
        save_debug(torch.tensor(cleaned_audio), sr, "05_restart_clean")
#remove
        deessed_path = os.path.join(
            temp_dir,
            f"{base_name}_deessed.wav"
        )

        if decision.run_deesser():

            logger.info("Running De-Esser...")
            apply_deesser(restart_file, deessed_path)
#remove
            audio_np, sr = sf.read(deessed_path)
            save_debug(torch.tensor(audio_np), sr, "06_deessed")
#remove
            dry_file = deessed_path

        else:

            logger.info("Skipping De-Esser")
            dry_file = restart_file

    except Exception as e:

        logger.warning(f"Restart stage failed: {e}")


    # ---------------------------------------------------
    # STEP 7 — DOWNWARD EXPANDER
    # ---------------------------------------------------

    try:

        logger.info("Running Downward Expander...")

        audio_np, sr = load_audio(dry_file)

        audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if decision.run_expander():

            audio_tensor = apply_downward_expander(
                audio_tensor,
                threshold=0.02,
                ratio=2.0
            )

        else:

            logger.info("Skipping Expander")

        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
#remove
        save_debug(audio_tensor, sr, "07_expander")
#remove
    except Exception as e:

        logger.warning(f"Noise gate failed: {e}")


    # ---------------------------------------------------
    # STEP 8 — COMPRESSION
    # ---------------------------------------------------

    try:

        if decision.run_compressor():

            logger.info("Applying 2-Stage Compression...")

            stage1 = apply_adaptive_compression(
                audio_tensor,
                sr=sr,
                ratio=2.0,
                attack_ms=20,
                release_ms=200
            )

            compressed_audio = apply_adaptive_compression(
                stage1,
                sr=sr,
                ratio=2.8,
                attack_ms=10,
                release_ms=80
            )

            if compressed_audio.ndim == 1:
                compressed_audio = compressed_audio.unsqueeze(0)

            compressed_audio = apply_parallel_compression(
                compressed_audio,
                sr,
                blend=0.06
            )

            compressed_audio = apply_harmonic_saturation(
                compressed_audio,
                drive=0.03,
                mix=0.08
            )

        else:

            logger.info("Skipping Compression")
            compressed_audio = audio_tensor

    except Exception as e:

        logger.warning(f"Compression failed: {e}")
        compressed_audio = audio_tensor
           #remove
    save_debug(compressed_audio, sr, "08_compressed")
    #remove
    # ---------------------------------------------------
    # STEP 9 — INTELLIGENT EQ
    # ---------------------------------------------------

    try:

        logger.info("Running Intelligent EQ...")

        eq_audio = apply_intelligent_eq(compressed_audio, sr)

        if eq_audio.ndim == 1:
            eq_audio = eq_audio.unsqueeze(0)

        eq_audio = torch.clamp(eq_audio * 0.995, -0.98, 0.98)

        eq_file = os.path.join(temp_dir, f"{base_name}_eq.wav")

        torchaudio.save(eq_file, eq_audio, sr)
#remove
        save_debug(eq_audio, sr, "09_eq")
#remove
    except Exception as e:

        logger.error(f"EQ stage failed: {e}")
        return


    # ---------------------------------------------------
    # STEP 10 — MASTERING
    # ---------------------------------------------------

    try:

        logger.info("Running Mastering...")

        final_output = os.path.join(
            output_dir,
            f"{base_name}_MASTERED.wav"
        )

        master_audio(
            eq_file,
            final_output,
            target_lufs=-17.0
        )
        #remove
        audio_np, sr = sf.read(final_output)
        save_debug(torch.tensor(audio_np), sr, "10_mastered")
#remove
    except Exception as e:

        logger.error(f"Mastering failed: {e}")
        return


    # ---------------------------------------------------
    # STEP 11 — TRANSCRIPT
    # ---------------------------------------------------

    try:

        logger.info("Generating Transcript...")

        text_path, json_path = edit_transcript(
            final_output,
            output_dir
        )

        logger.info(f"Transcript saved: {text_path}")
        logger.info(f"Alignment saved: {json_path}")

    except Exception as e:

        logger.warning(f"Transcript generation failed: {e}")


    logger.info("====================================")
    logger.info(f"✅ SUCCESS — Output: {final_output}")
    logger.info("====================================")


if __name__ == "__main__":
    main()
