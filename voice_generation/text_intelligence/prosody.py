class ProsodyAnalyzer:
    def __init__(self):
        pass

    def analyze(self, text: str):
        text = text.strip()
        words = text.split()

        # --- DEFAULT VALUES ---
        speed = 1.0
        pause = 0.25
        pitch = 1.0
        energy_scale = 1.0

        # --- SENTENCE TYPE ---
        if text.endswith("?"):
            speed *= 0.95
            pitch *= 1.05
            pause = 0.3

        elif text.endswith("!"):
            speed *= 1.1
            energy_scale *= 1.15
            pitch *= 1.03
            pause = 0.2

        # --- LENGTH BASED CONTROL ---
        if len(words) > 20:
            speed *= 0.9
            pause += 0.05

        elif len(words) < 6:
            speed *= 1.05
            pause -= 0.05

        # --- KEYWORD-BASED EMPHASIS (LIGHT TOUCH) ---
        text_lower = text.lower()

        if any(word in text_lower for word in ["important", "must", "never", "critical"]):
            energy_scale *= 1.1
            pitch *= 1.02

        if any(word in text_lower for word in ["sad", "unfortunately", "lost"]):
            speed *= 0.9
            pitch *= 0.95
            energy_scale *= 0.9

        if any(word in text_lower for word in ["amazing", "great", "excited"]):
            speed *= 1.05
            pitch *= 1.05
            energy_scale *= 1.1

        # --- CLAMP VALUES (VERY IMPORTANT) ---
        speed = max(0.75, min(speed, 1.25))
        pitch = max(0.9, min(pitch, 1.1))
        energy_scale = max(0.8, min(energy_scale, 1.3))
        pause = max(0.15, min(pause, 0.5))

        return {
            "speed": speed,
            "pause": pause,
            "pitch": pitch,
            "energy_scale": energy_scale
        }
