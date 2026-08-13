import re
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize


class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        # --- Normalize spaces ---
        text = re.sub(r'\s+', ' ', text)

        # --- Remove unwanted characters BUT keep Unicode (Hindi etc.) ---
        # Keep:
        # - all letters (Unicode)
        # - numbers
        # - punctuation
        text = re.sub(r'[^\w\s.,!?\'\u0900-\u097F]', '', text, flags=re.UNICODE)

        return text.strip()

    def sentence_split(self, text: str):
        try:
            # NLTK sometimes fails on Hindi → fallback to regex
            sentences = sent_tokenize(text)

            if len(sentences) == 0:
                raise ValueError("NLTK failed")

            return sentences

        except:
            # Fallback for Hindi / multilingual
            return re.split(r'(?<=[.!?।])\s+', text)

    def tokenize(self, sentence: str):
        try:
            return word_tokenize(sentence)
        except:
            # fallback: simple split (better for Hindi)
            return sentence.split()

    def normalize(self, tokens):
        return [token.lower() for token in tokens]

    def process(self, text: str):
        cleaned = self.clean_text(text)

        sentences = self.sentence_split(cleaned)

        processed_output = []

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            normalized = self.normalize(tokens)

            processed_output.append({
                "original": sentence,
                "tokens": tokens,
                "normalized": normalized
            })

        return processed_output
