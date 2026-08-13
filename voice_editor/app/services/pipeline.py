# app/services/pipeline.py

import os
import tempfile

from app.services.analyzer import analyze_audio
from app.services.enhancer import enhance_audio
from app.services.advanced_enhancer import enhance_audio_advanced
from app.services.deepfilter_enhancer import enhance_audio_deepfilter
from app.services.silence_trimmer import process_silence_trim
from app.services.echo_remover import process_echo_removal
from app.services.smart_compressor import process_smart_compression
from app.services.intelligent_eq import process_intelligent_eq
from app.services.deesser_breath_control import process_uploaded_file as deesser_process
from app.services.final_audio_polishing import process_uploaded_file as youtube_polish_process


async def process_audio_mode(
    file,
    mode: str,
    youtube_polish: bool = False
):
    """
    Full Professional Pipeline

    Default:
    Analyze
    -> Noise Removal
    -> Silence Trim
    -> Echo Removal
    -> Compression
    -> Intelligent EQ
    -> De-Esser & Breath Control

    Optional:
    -> Final YouTube Polishing (if button ON)
    """

    # ==================================
    # Save Uploaded File
    # ==================================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        input_path = tmp.name

    # ==================================
    # Analyze Original Audio
    # ==================================
    analysis = analyze_audio(input_path)

    # ==================================
    # Output Paths
    # ==================================
    cleaned_path = input_path.replace(".wav", f"_{mode}_cleaned.wav")
    trimmed_path = input_path.replace(".wav", f"_{mode}_trimmed.wav")
    echo_path = input_path.replace(".wav", f"_{mode}_echo.wav")
    compressed_path = input_path.replace(".wav", f"_{mode}_compressed.wav")
    eq_path = input_path.replace(".wav", f"_{mode}_eq.wav")

    # ==================================
    # Step 1: Noise Removal
    # ==================================
    if mode == "basic":
        noise_result = enhance_audio(input_path, cleaned_path)

    elif mode == "advanced":
        noise_result = enhance_audio_advanced(input_path, cleaned_path)

    elif mode == "deepfilter":
        noise_result = enhance_audio_deepfilter(input_path, cleaned_path)

    else:
        noise_result = enhance_audio(input_path, cleaned_path)

    # ==================================
    # Step 2: Silence Trimming
    # ==================================
    trim_result = process_silence_trim(
        cleaned_path,
        trimmed_path
    )

    # ==================================
    # Step 3: Echo Removal
    # ==================================
    echo_result = process_echo_removal(
        trimmed_path,
        echo_path
    )

    # ==================================
    # Step 4: Smart Compression
    # ==================================
    compression_result = process_smart_compression(
        echo_path,
        compressed_path
    )

    # ==================================
    # Step 5: Intelligent EQ
    # ==================================
    eq_result = process_intelligent_eq(
        compressed_path,
        eq_path
    )

    # ==================================
    # Step 6: De-Esser & Breath Control
    # ==================================
    deesser_output = deesser_process(eq_path, "outputs")
    final_output = deesser_output

    # ==================================
    # Step 7: Optional YouTube Polishing
    # ==================================
    polish_result = "OFF"

    if youtube_polish:
        polished_output = youtube_polish_process(
            deesser_output,
            "outputs"
        )
        final_output = polished_output
        polish_result = "ON"

    # ==================================
    # Final Response
    # ==================================
    return {
        "filename": file.filename,
        "selected_mode": mode,
        "youtube_polish": polish_result,
        "analysis": analysis,
        "step_1_noise_removal": noise_result,
        "step_2_silence_trim": trim_result,
        "step_3_echo_removal": echo_result,
        "step_4_smart_compression": compression_result,
        "step_5_intelligent_eq": eq_result,
        "step_6_deesser_breath_control": os.path.basename(deesser_output),
        "step_7_final_polishing": (
            os.path.basename(final_output)
            if youtube_polish else "Skipped"
        ),
        "download_file": os.path.basename(final_output),
        "output_path": final_output,
        "status": "processed"
    }
