import os
import sys
import re
import time
import librosa
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langdetect import detect, DetectorFactory

from text_intelligence.pipeline import TextIntelligencePipeline
from text_intelligence.speech_planner import build_speech_plan
from text_intelligence.emotion import EmotionAnalyzer
from voices import VOICES, DEFAULT_VOICE

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

DetectorFactory.seed = 0

# --- CONFIG ---
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICE_FILE = "voices-v1.0.bin"
OUTPUT_FILE = "final_commercial_audio.wav"
SAMPLE_RATE = 24000
MAX_THREADS = 4
MAX_CHARS = 180

# --- VOICES ---
DEFAULT_VOICES = {
    "hi": "hm_omega",
    "en-us": "am_michael",
    "fallback": "am_michael"
}
#EMOTION_PROFILES
DEFAULT_VOICES = {
    "hi": "hm_omega",
    "en-us": "am_michael",
    "fallback": "am_michael"
}

EMOTION_PROFILES = {

    "happy": {
        "speed": 1.08,
        #"pitch": 1.10,
        "energy": 1.15,
        "pause": 0.12
    },

    "sad": {
        "speed": 1.0,
        #"pitch": 0.92,
        "energy": 0.96,
        "pause": 0.22
    },

    "angry": {
        "speed": 1.12,
        #"pitch": 1.18,
        "energy": 1.35,
        "pause": 0.08
    },

    "fear": {
        "speed": 1.02,
        #"pitch": 1.08,
        "energy": 0.95,
        "pause": 0.18
    },

    "love": {
        "speed": 0.95,
        #"pitch": 1.04,
        "energy": 0.92,
        "pause": 0.22
    },

    "neutral": {
        "speed": 1.0,
        #"pitch": 1.0,
        "energy": 1.0,
        "pause": 0.15
    }
}

# --- INIT ---
if not os.path.exists(MODEL_FILE):
    print("❌ Model files not found!")
    exit()

kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
pipeline = TextIntelligencePipeline()
emotion_analyzer = EmotionAnalyzer()

# --- CLEAN TEXT ---
def clean_text(text):
    text = text.replace("\ufeff", "")
    text = re.sub(r'\s+', ' ', text)

    # normalize punctuation spacing
    text = re.sub(r'\s*([.,!?।])\s*', r'\1 ', text)

    # remove repeated punctuation
    text = re.sub(r'[.,]{2,}', '.', text)
    text = re.sub(r'[!?]{2,}', '!', text)

    return text.strip()


# --- REMOVE PUNCTUATION FOR TTS ---
def tts_safe_text(text):

    # remove only problematic symbols
    text = re.sub(r'[#$%^&*+=<>{}[\]|~]', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# --- WORD SAFE SPLIT ---
def split_long_sentence(sentence, max_chars):
    words = sentence.split()
    chunks = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += word + " "
        else:
            chunks.append(current.strip())
            current = word + " "

    if current:
        chunks.append(current.strip())

    return chunks


# --- SENTENCE SPLIT ---
def split_text_into_sentences(text, max_chars=MAX_CHARS):
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) > max_chars:
            chunks.extend(split_long_sentence(sentence, max_chars))
            continue

        if len(current) + len(sentence) <= max_chars:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "

    if current:
        chunks.append(current.strip())

    return chunks


# --- LANGUAGE DETECTION ---
def detect_language_and_voice(text, selected_voice=None):
    # 🎯 If user selected a voice → use it directly
    if selected_voice and selected_voice in VOICES:
        config = VOICES[selected_voice]
        return config["voice"], config["lang"]

    # 🤖 Auto detection fallback (your old logic)
    if re.search(r'[\u0900-\u097F]', text):
        return DEFAULT_VOICES["hi"], "hi"

    return DEFAULT_VOICES["en-us"], "en-us"


# --- CROSSFADE ---
def crossfade(a, b, fade_samples=1200):
    if len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b])

    fade_out = np.linspace(1, 0, fade_samples)
    fade_in = np.linspace(0, 1, fade_samples)

    a_end = a[-fade_samples:] * fade_out
    b_start = b[:fade_samples] * fade_in

    return np.concatenate([
        a[:-fade_samples],
        a_end + b_start,
        b[fade_samples:]
    ])
def apply_pitch(samples, sr, pitch_factor):

    if pitch_factor == 1.0:
        return samples

    n_steps = (pitch_factor - 1.0) * 6

    return librosa.effects.pitch_shift(
        samples,
        sr=sr,
        n_steps=n_steps
    )
def apply_energy(samples, energy):

    #samples = samples * energy
    samples = samples * (0.7 + (energy * 0.3))

    return np.clip(samples, -1.0, 1.0)
def generate_pause(text, emotion, sr):

    profile = EMOTION_PROFILES.get(
        emotion,
        EMOTION_PROFILES["neutral"]
    )

    duration = profile["pause"]

    if "..." in text:
        duration += 0.4

    elif "!" in text:
        duration -= 0.05

    elif "?" in text:
        duration += 0.05

    duration = max(0.05, duration)

    return np.zeros(int(sr * duration))

# --- BREATH SOUND ---
def add_breath(sr):

    breath = np.random.normal(
        0,
        0.0012,
        int(sr * 0.06)
    )

    fade_in = np.linspace(
        0,
        1,
        len(breath)
    )

    fade_out = np.linspace(
        1,
        0,
        len(breath)
    )

    fade = fade_in * fade_out

    return breath * fade

# --- GENERATE AUDIO ---
def generate_chunk(task):
    index, text, speech_plan = task
    emotion_data = emotion_analyzer.analyze(text)
    if emotion_data["confidence"] < 0.65:
        emotion = "neutral"
    else:
        emotion = emotion_data["emotion"]

    profile = EMOTION_PROFILES.get(
        emotion,
        EMOTION_PROFILES["neutral"]
    )

    text = clean_text(text)

    if not text:
        return index, None

    print("\nDEBUG TEXT:", repr(text))

    voice, lang_code = detect_language_and_voice(text)

    print("DEBUG VOICE:", voice, "LANG:", lang_code)

    # 🔥 REMOVE punctuation before TTS
    text = tts_safe_text(text)

    base_speed = speech_plan.get("speed", 1.0)

    speed = base_speed * profile["speed"]
    speed = max(0.95, min(speed, 1.08))

    #pitch = profile["pitch"]

    energy = profile["energy"]

    try:
        samples, sr = kokoro.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang_code
        )

        samples = np.clip(samples, -1.0, 1.0)
        # emotion pitch
        #samples = apply_pitch(
         #   samples,
          #  sr,
           # pitch
        #)

        # emotion energy
        samples = apply_energy(
            samples,
            energy
    )
        # add emotional breath
        if emotion in ["sad", "fear", "love"]:

            breath = add_breath(sr)

            samples = np.concatenate([
        samples,
        breath
    ])

        # minimal pause only for small chunks
        pause = generate_pause(
            text,
            emotion,
            sr
        )

        samples = np.concatenate([
            samples,
            pause
        ])

        return index, samples

    except Exception as e:
        print(f"Error in chunk {index}: {e}")
        return index, None


# --- MAIN ---
def main(custom_text=None, custom_output=None, voice=None):

    if custom_text:
        input_text = clean_text(custom_text)
        output_file = custom_output
    else:
        try:
            with open("script.txt", "r", encoding="utf-8") as f:
                input_text = clean_text(f.read())
            output_file = OUTPUT_FILE
        except FileNotFoundError:
            print("❌ script.txt not found!")
            return

    base_analysis = pipeline.run(input_text)
    cleaned = [s["original"] for s in base_analysis["preprocessing"]]
    processed_text = clean_text(" ".join(cleaned))

    sentences = split_text_into_sentences(processed_text)

    print(f"\n🧠 Processing {len(sentences)} sentences...\n")

    tasks = []

    for i, sentence in enumerate(sentences):
        nlp = pipeline.run(sentence)
        nlp["text"] = sentence

        speech_plan = build_speech_plan(nlp)
        speech_plan["voice"] = voice
        tasks.append((i, sentence, speech_plan))

    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        results = list(tqdm(executor.map(generate_chunk, tasks), total=len(tasks)))

    results.sort(key=lambda x: x[0])
    audio_chunks = [r[1] for r in results if r[1] is not None]

    if not audio_chunks:
        print("❌ Generation failed")
        return

    print("\n🔗 Finalizing audio...")

    final = audio_chunks[0]

    for chunk in audio_chunks[1:]:
        final = crossfade(final, chunk)

    peak = np.max(np.abs(final))
    if peak > 0:
        final = final / peak

    final *= 1.2
    final = np.clip(final, -0.95, 0.95)

    sf.write(output_file, final, SAMPLE_RATE)

    print("✅ DONE")
    print(f"📁 Output: {output_file}")
    print(f"⏱ Time: {round(time.time() - start, 2)}s")


if __name__ == "__main__":
    main()
