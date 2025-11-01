import spacy
import re

def get_spacy_pipeline():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return nlp

def clean_text(text: str, nlp: spacy.Language) -> str:
    # Guardia per stringhe none
    if not isinstance(text, str):
        return ""

    # Rimuovi gli URL ma mantieni intatto il resto del contenuto
    text = re.sub(r'https?://\S+', '', text)
    text = text.lower().strip()

    # delega a spacy per tokenizzazione, rimozione stopword e lemmatizzazione 
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)
