import spacy


class SyntaxAnalyzer:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, text: str):
        doc = self.nlp(text)

        result = []

        for token in doc:
            result.append({
                "word": token.text,
                "lemma": token.lemma_,     # base form
                "pos": token.pos_,         # Part of Speech (NOUN, VERB, etc.)
                "tag": token.tag_,         # detailed POS
                "dep": token.dep_,         # dependency relation
                "head": token.head.text    # parent word
            })

        return result
