# preprocessing/text_processing.py

import spacy

# load spacy english model
nlp = spacy.load("en_core_web_sm")

def preprocess_texts(records, config):
    """
    Cleans and annotates text records in-place.
    - Adds 'preprocessed_text'
    - Adds 'tokens'
    - Optionally adds 'spacy_entities'
    """
    use_ner = config.get("preprocessing", {}).get("enable_ner", True)

    for rec in records:
        text = rec["text"]

        # lowercase, strip whitespace, could add emoji/remove punctuation etc.
        clean_text = text.lower().strip()
        rec["preprocessed_text"] = clean_text

        doc = nlp(clean_text)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        rec["tokens"] = tokens

        if use_ner:
            rec["spacy_entities"] = [(ent.text, ent.label_) for ent in doc.ents]

    return records
