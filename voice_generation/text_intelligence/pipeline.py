from text_intelligence.preprocessing import TextPreprocessor
from text_intelligence.syntax import SyntaxAnalyzer
from text_intelligence.semantics import SemanticAnalyzer
from text_intelligence.intent import IntentAnalyzer
from text_intelligence.emotion import EmotionAnalyzer
from text_intelligence.prosody import ProsodyAnalyzer


class TextIntelligencePipeline:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.syntax = SyntaxAnalyzer()
        self.semantics = SemanticAnalyzer()
        self.intent = IntentAnalyzer()
        self.emotion = EmotionAnalyzer()
        self.prosody = ProsodyAnalyzer()

    def run(self, text: str):

        # Step 1: Preprocessing
        preprocessed = self.preprocessor.process(text)

        # Step 2: Syntax
        syntax = self.syntax.analyze(text)

        # Step 3: Semantics
        semantics = self.semantics.analyze(text)

        # Step 4: Intent
        intent = self.intent.analyze(text)

        # Step 5: Emotion
        emotion = self.emotion.analyze(text)

        # Step 6: Prosody
        prosody = self.prosody.analyze(text)

        return {
            "preprocessing": preprocessed,
            "syntax": syntax,
            "semantics": semantics,
            "intent": intent,
            "emotion": emotion,
            "prosody": prosody
        }
