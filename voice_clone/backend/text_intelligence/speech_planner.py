import re


# --- EMPHASIS DETECTION (kept for future use) ---
def detect_emphasis_words(text):
    important_words = [
        "not", "never", "always", "really", "very",
        "did", "do", "does", "must", "should",
        "important", "critical", "need", "have"
    ]

    words = re.findall(r'\b\w+\b', text.lower())

    return [w for w in words if w in important_words]


# --- MAIN SPEECH PLANNER ---
def build_speech_plan(nlp):
    emotion = nlp["emotion"]["emotion"]
    intent = nlp["intent"]["intent"]
    prosody = nlp["prosody"]
    text = nlp.get("text", "")

    # --- BASE PLAN ---
    plan = {
        "speed": prosody.get("speed", 1.0),
        "pause_after": 0.2,
        "pitch": 1.0,           # 🔥 NEW
        "energy_scale": 1.0,    # 🔥 NEW
        "intonation": "neutral",
    }

    # --- EMOTION RULES ---
    if emotion == "happy":
        plan["speed"] *= 1.08
        plan["pause_after"] = 0.15
        plan["pitch"] = 1.05
        plan["energy_scale"] = 1.1

    elif emotion == "sad":
        plan["speed"] *= 0.88
        plan["pause_after"] = 0.35
        plan["pitch"] = 0.95
        plan["energy_scale"] = 0.85

    elif emotion == "angry":
        plan["speed"] *= 1.12
        plan["pause_after"] = 0.12
        plan["pitch"] = 1.08
        plan["energy_scale"] = 1.2

    elif emotion == "neutral":
        plan["pitch"] = 1.0
        plan["energy_scale"] = 1.0

    # --- INTENT RULES ---
    if intent == "question":
        plan["pause_after"] = 0.3
        plan["intonation"] = "rising"
        plan["pitch"] *= 1.02  # slight lift

    elif intent == "exclamation":
        plan["speed"] *= 1.1
        plan["energy_scale"] *= 1.1

    elif intent == "statement":
        plan["intonation"] = "falling"

    # --- PUNCTUATION (final override) ---
    if text.endswith("?"):
        plan["pause_after"] = 0.35
        plan["pitch"] *= 1.03

    elif text.endswith("!"):
        plan["pause_after"] = 0.2
        plan["energy_scale"] *= 1.1

    elif text.endswith("."):
        plan["pause_after"] = 0.25

    return plan
