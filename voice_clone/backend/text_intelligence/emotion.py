import re
from collections import defaultdict


class EmotionAnalyzer:
    def __init__(self):

        self.emotion_keywords = {
            "happy": [
                "happy", "great", "amazing", "awesome", "wonderful",
                "excited", "fantastic", "good", "excellent", "yay",
                "beautiful", "best", "nice", "love this"
            ],

            "sad": [
                "sad", "unhappy", "depressed", "cry", "crying",
                "bad", "hurt", "pain", "broken", "upset",
                "lonely", "miss you", "heartbroken"
            ],

            "angry": [
                "angry", "furious", "annoyed", "frustrated",
                "hate", "mad", "irritated", "stupid",
                "idiot", "nonsense", "damn"
            ],

            "fear": [
                "fear", "scared", "afraid", "terrified",
                "nervous", "worried", "panic", "anxious"
            ],

            "love": [
                "love", "care", "affection", "romantic",
                "darling", "sweetheart", "baby", "dear"
            ],

            "surprise": [
                "wow", "unbelievable", "shocking",
                "unexpected", "omg", "what", "seriously"
            ]
        }

        # Hinglish support
        self.hinglish_keywords = {
            "happy": ["khush", "mast", "badiya", "maza"],
            "sad": ["dukhi", "udaas", "rona"],
            "angry": ["gussa", "pagal", "chidh"],
            "fear": ["darr", "dar", "ghabra"],
            "love": ["pyaar", "mohabbat", "jaan"]
        }

    def preprocess(self, text):
        return re.sub(r"\s+", " ", text.strip().lower())

    def analyze(self, text: str):

        original_text = text
        text = self.preprocess(text)

        scores = defaultdict(float)

        # --------------------------
        # Keyword scoring
        # --------------------------

        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[emotion] += 1.0

        # Hinglish scoring
        for emotion, keywords in self.hinglish_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[emotion] += 1.0

        # --------------------------
        # Intensity boosts
        # --------------------------

        # Exclamation boost
        exclamations = original_text.count("!")
        if exclamations > 0:
            for emotion in scores:
                scores[emotion] += exclamations * 0.3

        # CAPS boost
        caps_words = re.findall(r"\b[A-Z]{3,}\b", original_text)
        if caps_words:
            for emotion in scores:
                scores[emotion] += 0.5

        # Repeated letters boost
        if re.search(r"(.)\1{2,}", text):
            for emotion in scores:
                scores[emotion] += 0.4

        # --------------------------
        # Detect dominant emotion
        # --------------------------

        if not scores:
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "scores": {}
            }

        dominant_emotion = max(scores, key=scores.get)

        total = sum(scores.values())
        confidence = round(scores[dominant_emotion] / total, 2)

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "scores": dict(scores)
        }


# --------------------------
# Example Usage
# --------------------------

if __name__ == "__main__":

    analyzer = EmotionAnalyzer()

    tests = [
        "I am SO HAPPY today!!!",
        "Mujhe bahut darr lag raha hai",
        "This is amazing wow!!",
        "I hate this stupid thing",
        "I miss you so much...",
        "Yeh bahut badiya hai!!!"
    ]

    for t in tests:
        print("\nText:", t)
        print(analyzer.analyze(t))
