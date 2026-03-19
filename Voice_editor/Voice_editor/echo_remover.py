import torch
import torchaudio
import soundfile as sf

def remove_echo(input_path, output_path):
    print(f"\n🏛️ Running Lightweight Dereverb: {input_path}")

    # Load audio
    waveform, sr = torchaudio.load(input_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.squeeze()

    # STFT parameters
    n_fft = 1024
    hop_length = 256
    win_length = 1024

    # ✅ Proper Hann window (FIX)
    window = torch.hann_window(win_length)

    # STFT
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(spec)
    phase = torch.angle(spec)

    # Simple dereverb mask
    mag_mean = torch.mean(magnitude, dim=-1, keepdim=True)
    mask = magnitude / (mag_mean + 1e-8)

    mask = torch.clamp(mask, 0.0, 1.0)

    enhanced_mag = magnitude * mask

    enhanced_spec = enhanced_mag * torch.exp(1j * phase)

    # ISTFT
    enhanced = torch.istft(
        enhanced_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )

    enhanced = enhanced.unsqueeze(0)

    sf.write(output_path, enhanced.numpy().T, sr)

    print(f"✅ Dereverb Applied: {output_path}")