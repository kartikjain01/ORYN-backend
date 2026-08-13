class IntentAnalyzer:
    def __init__(self):
        pass

    def analyze(self, text: str):
        text = text.strip()

        # Simple rule-based intent detection
        if text.endswith("?"):
            intent = "question"
        elif text.endswith("!"):
            intent = "exclamation"
        elif any(word in text.lower() for word in ["should", "try", "consider", "recommend"]):
            intent = "suggestion"
        elif any(word in text.lower() for word in ["please", "kindly"]):
            intent = "request"
        else:
            intent = "statement"

        return {
            "intent": intent
        }
