import os
import numpy as np
import torch
from df.enhance import enhance, init_df, load_audio, save_audio

# --------------------------------------------------
# AI Denoiser - TEST MODE (Force Full Denoise)
# --------------------------------------------------

def tensor_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def process_ai_denoise(input_path, output_path, snr=None):
    print(f"\n🧠 AI Denoising (FORCED TEST MODE): {os.path.basename(input_path)}")

    # --------------------------------------------------
    # Load Local Model
    # --------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = "DeepFilterNet/DeepFilterNet3"

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

    # Debug BEFORE stats
    print("\n🔎 BEFORE DENOISE STATS")
    print("RMS:", np.mean(audio_np**2))
    print("Peak:", np.max(np.abs(audio_np)))

    # --------------------------------------------------
    # Apply Full Denoising
    # --------------------------------------------------
    print("🚀 Applying FULL DeepFilterNet Enhancement...")
    enhanced = enhance(model, df_state, audio)

    enhanced_np = tensor_to_numpy(enhanced)

    # Debug AFTER stats
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
