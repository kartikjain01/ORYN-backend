import os
import numpy as np
import torch
import zipfile

from huggingface_hub import hf_hub_download
from df.enhance import enhance, init_df, load_audio, save_audio


# --------------------------------------------------
# Utility
# --------------------------------------------------

def tensor_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


# --------------------------------------------------
# DOWNLOAD + EXTRACT MODEL
# --------------------------------------------------

def get_model():
    model_base_dir = os.path.join("models")

    # Expected final path
    model_dir = os.path.join(model_base_dir, "DeepFilterNet3")

    # ✅ If already exists → skip download
    if os.path.exists(os.path.join(model_dir, "config.ini")):
        print("✅ Model already available locally")
        return model_dir

    print("⬇️ Downloading model from HuggingFace...")

    zip_path = hf_hub_download(
        repo_id="Kartikjain12345/deepfilternet",
        filename="DeepFilterNet.zip"
    )

    print("📦 Extracting model...")

    os.makedirs(model_base_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_base_dir)

    # 🔍 Handle nested folder case
    # (DeepFilterNet/DeepFilterNet3/)
    possible_nested = os.path.join(model_base_dir, "DeepFilterNet", "DeepFilterNet3")

    if os.path.exists(possible_nested):
        print("📁 Found nested model folder, fixing path...")
        return possible_nested

    return model_dir


# --------------------------------------------------
# MAIN DENOISE FUNCTION
# --------------------------------------------------

def process_ai_denoise(input_path, output_path, snr=None):

    print(f"\n🧠 AI Denoising: {os.path.basename(input_path)}")

    try:
        # --------------------------------------------------
        # Load Model
        # --------------------------------------------------
        local_model_path = get_model()

        print(f"📍 MODEL PATH: {local_model_path}")

        if not os.path.exists(local_model_path):
            print(f"❌ ERROR: Model not found at {local_model_path}")
            return input_path

        print("📦 Loading DeepFilterNet Model...")
        model, df_state, _ = init_df(model_base_dir=local_model_path)

        # --------------------------------------------------
        # Load Audio
        # --------------------------------------------------
        print("🎧 Loading Audio...")
        audio, _ = load_audio(input_path, sr=df_state.sr())

        audio_np = tensor_to_numpy(audio)

        print("\n🔎 BEFORE DENOISE STATS")
        print("RMS:", np.mean(audio_np**2))
        print("Peak:", np.max(np.abs(audio_np)))

        # --------------------------------------------------
        # Apply Denoising
        # --------------------------------------------------
        print("🚀 Applying DeepFilterNet Enhancement...")
        enhanced = enhance(model, df_state, audio)

        enhanced_np = tensor_to_numpy(enhanced)

        print("\n🔎 AFTER DENOISE STATS")
        print("RMS:", np.mean(enhanced_np**2))
        print("Peak:", np.max(np.abs(enhanced_np)))

        print("\n📊 DIFFERENCE")
        print("Mean Difference:", np.mean(audio_np - enhanced_np))
        print("Max Difference :", np.max(np.abs(audio_np - enhanced_np)))

        # --------------------------------------------------
        # Save Output
        # --------------------------------------------------
        save_audio(output_path, enhanced, df_state.sr())

        print(f"\n✅ Noise Removed → {output_path}")

        return output_path

    except Exception as e:
        print(f"❌ AI Denoiser failed: {e}")
        return input_path
