import os
import subprocess
import sys
from pathlib import Path


def enhance_audio_deepfilter(input_path: str, output_path: str):
    """
    AI Noise Removal using DeepFilterNet
    """

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Locate deepFilter inside the current Python virtual environment
    deepfilter_executable = Path(sys.executable).parent / "deepFilter"

    if not deepfilter_executable.exists():
        raise RuntimeError(
            f"deepFilter executable not found.\n"
            f"Expected location: {deepfilter_executable}"
        )

    cmd = [
        str(deepfilter_executable),
        input_path,
        "--output-dir",
        output_dir,
    ]

    print("Running:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"DeepFilterNet failed.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    generated_file = os.path.join(
        output_dir,
        os.path.basename(input_path).replace(
            ".wav",
            "_DeepFilterNet3.wav"
        )
    )

    if not os.path.exists(generated_file):
        raise RuntimeError(
            f"DeepFilter output not found:\n{generated_file}"
        )

    os.replace(generated_file, output_path)

    return {
        "output_path": output_path,
        "status": "enhanced",
        "model": "DeepFilterNet",
        "features_applied": [
            "traffic_noise_removed",
            "fan_noise_removed",
            "wind_noise_removed",
            "crowd_noise_removed",
            "keyboard_noise_removed",
            "hum_removed",
            "static_removed",
        ],
    }
