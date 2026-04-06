import os
import re
import time
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langdetect import detect, DetectorFactory

# Force consistent results from the detector
DetectorFactory.seed = 0

# --- CONFIGURATION ---
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICE_FILE = "voices-v1.0.bin"
OUTPUT_FILE = "final_commercial_audio.wav"
SAMPLE_RATE = 24000
MAX_THREADS = 4

# 🔥 NEW SAFE LIMITS
MAX_CHARS = 180   # prevents phoneme overflow

# --- MASTER DEFAULT VOICES ---
DEFAULT_VOICES = {
    "hi": "hm_omega",
    "zh": "zm_yunxi",
    "ja": "jm_kumo",
    "es": "em_alex",
    "pt": "pm_santa",
    "it": "im_nicola",
    "fr": "ff_siwis",
    "en-gb": "bm_daniel",
    "en-us": "am_michael",
    "fallback": "am_michael"
}

# Initialize Model
if not os.path.exists(MODEL_FILE):
    print("❌ Model files not found!")
    exit()

kokoro = Kokoro(MODEL_FILE, VOICE_FILE)


# ✅ UPDATED SMART CHUNKING (IMPORTANT)
def split_text_into_sentences(text, max_chars=MAX_CHARS):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        # If single sentence too long → split further
        if len(sentence) > max_chars:
            parts = [sentence[i:i+max_chars] for i in range(0, len(sentence), max_chars)]
            for part in parts:
                chunks.append(part.strip())
            continue

        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "

    if current:
        chunks.append(current.strip())

    return chunks


def generate_chunk(sentence_data):
    index, text = sentence_data

    text = text.replace("\n", " ").strip()

    # 🔥 HARD SAFETY LIMIT (EXTRA PROTECTION)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    if not text:
        return index, None

    try:
        raw_lang = detect(text)

        if raw_lang == "hi":
            current_lang, current_voice = "hi", DEFAULT_VOICES["hi"]
        elif raw_lang.startswith("zh"):
            current_lang, current_voice = "zh", DEFAULT_VOICES["zh"]
        elif raw_lang == "ja":
            current_lang, current_voice = "ja", DEFAULT_VOICES["ja"]
        elif raw_lang == "es":
            current_lang, current_voice = "es", DEFAULT_VOICES["es"]
        elif raw_lang == "pt":
            current_lang, current_voice = "pt", DEFAULT_VOICES["pt"]
        elif raw_lang == "it":
            current_lang, current_voice = "it", DEFAULT_VOICES["it"]
        elif raw_lang == "fr":
            current_lang, current_voice = "fr", DEFAULT_VOICES["fr"]
        elif raw_lang == "uk":
            current_lang, current_voice = "en-gb", DEFAULT_VOICES["en-gb"]
        elif raw_lang == "en":
            current_lang, current_voice = "en-us", DEFAULT_VOICES["en-us"]
        else:
            current_lang, current_voice = "en-us", DEFAULT_VOICES["fallback"]

    except:
        current_lang, current_voice = "en-us", DEFAULT_VOICES["fallback"]

    try:
        samples, sr = kokoro.create(
            text,
            voice=current_voice,
            speed=1.0,
            lang=current_lang
        )

        # Trim end noise
        trim_samples = int(sr * 0.08)
        if len(samples) > trim_samples:
            samples = samples[:-trim_samples]

        # Add silence
        silence_padding = np.zeros(int(sr * 0.25))
        clean_samples = np.concatenate([samples, silence_padding])

        return index, clean_samples

    except Exception as e:
        print(f"Error in chunk {index}: {e}")
        return index, None


def main(custom_text=None, custom_output=None):

    if custom_text:
        input_text = custom_text
        active_output = custom_output
    else:
        try:
            with open("script.txt", "r", encoding="utf-8") as f:
                input_text = f.read()
            active_output = OUTPUT_FILE
        except FileNotFoundError:
            print("❌ script.txt not found!")
            return None

    # ✅ USE UPDATED CHUNKING
    sentences = split_text_into_sentences(input_text)

    indexed_sentences = list(enumerate(sentences))
    total_chunks = len(sentences)

    print(f"🚀 Processing {total_chunks} chunks with SAFE chunking...")
    start_total = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        list_of_tasks = list(
            tqdm(
                executor.map(generate_chunk, indexed_sentences),
                total=total_chunks,
                unit="sent",
                desc="Progress"
            )
        )
        results = list_of_tasks

    results.sort(key=lambda x: x[0])
    combined_audio = [res[1] for res in results if res[1] is not None]

    if combined_audio:

        print("\n🔗 Finalizing with High-Gain Optimization...")

        final_array = np.concatenate(combined_audio)

        peak = np.max(np.abs(final_array))
        if peak > 0:
            final_array = (final_array / peak)

        gain_factor = 1.8
        final_array = final_array * gain_factor

        final_array = np.clip(final_array, -0.98, 0.98)

        sf.write(active_output, final_array, SAMPLE_RATE)

        end_total = time.time()

        print("-" * 35)
        print(f"✅ COMPLETE!")
        print(f"⚡ Time Taken: {round(end_total - start_total, 2)}s")
        print(f"📁 Output: {active_output}")
        print("-" * 35)

        return active_output

    else:
        print("❌ Generation failed.")
        return None
