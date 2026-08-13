import spacy

nlp = spacy.load("en_core_web_sm")

class SemanticAnalyzer:
    def __init__(self):
        self.nlp = nlp

    def analyze(self, text: str):
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        return {"entities": entities, "keywords": keywords}
