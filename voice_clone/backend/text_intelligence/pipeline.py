from backend.text_intelligence.preprocessing import TextPreprocessor
from backend.text_intelligence.syntax import SyntaxAnalyzer
from backend.text_intelligence.semantics import SemanticAnalyzer
from backend.text_intelligence.intent import IntentAnalyzer
from backend.text_intelligence.emotion import EmotionAnalyzer
from backend.text_intelligence.prosody import ProsodyAnalyzer


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

        enhanced_text = " ".join(
          item["original"]
          for item in preprocessed
        )

        return {
           "enhanced_text": enhanced_text,
           "syntax": syntax,
           "semantics": semantics,
           "intent": intent,
           "emotion": emotion,
           "prosody": prosody
        }
