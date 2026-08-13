import subprocess

def convert_to_wav(
    input_path,
    output_path
):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        output_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
