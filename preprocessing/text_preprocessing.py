import spacy

# Load spaCy model once (could be configurable)
nlp = spacy.load("en_core_web_sm")

def preprocess_texts(records, config):
    """
    Given a list of records, adds tokenized texts and entities extracted.
    Updates each record dict with:
    - 'tokens': list of token strings
    - 'preprocessed_text': cleaned/lowercased text (optional)
    - 'sentiment_labels': list or aggregated sentiment labels for training

    For now, just tokenizes text and assigns sentiment labels simplified.

    Returns updated list of records.
    """
    for record in records:
        doc = nlp(record['text'])
        record['tokens'] = [token.text for token in doc]
        record['preprocessed_text'] = record['text'].lower()

        # Simple placeholder: map sentiment dict to a single label for training
        # Example: pick highest scoring sentiment key
        if record['sentiments']:
            max_sentiment = max(record['sentiments'], key=record['sentiments'].get)
            record['sentiment_labels'] = max_sentiment
        else:
            record['sentiment_labels'] = "neutral"

    return records
