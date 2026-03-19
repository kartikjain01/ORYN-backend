import os
import torch
import torchaudio
import shutil


def remove_echo(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"⚠️ Skipping Echo Removal: {input_path} not found.")
        return

    print(f"\n🏛️ Running Lightweight Dereverb: {os.path.basename(input_path)}")

    try:
        waveform, sr = torchaudio.load(input_path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Spectral gating style dereverb
        spec = torch.stft(
            waveform.squeeze(),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        )

        magnitude = torch.abs(spec)
        phase = torch.angle(spec)

        # Estimate reverb tail suppression
        noise_floor = torch.mean(magnitude, dim=1, keepdim=True)
        clean_mag = torch.clamp(magnitude - 0.2 * noise_floor, min=0.0)

        enhanced_spec = clean_mag * torch.exp(1j * phase)

        enhanced = torch.istft(
            enhanced_spec,
            n_fft=1024,
            hop_length=256
        )

        enhanced = enhanced.unsqueeze(0)

        torchaudio.save(output_path, enhanced, sr)
        print(f"✅ Dereverb Applied: {output_path}")

    except Exception as e:
        print(f"❌ Dereverb Failed: {e}")
        shutil.copy(input_path, output_path)