import numpy as np

def prepare_lstm_inputs(records, glove_vectorizer, timesteps):
    """
    Prepares fixed-length sequences of glove embeddings for LSTM input.
    Pads or truncates tokens per record to `timesteps`
    
    Returns numpy array shape (num_records, timesteps, embedding_dim)
    """
    embedding_dim = glove_vectorizer.embedding_dim
    inputs = []

    for rec in records:
        tokens = rec.get("tokens", [])
        vectors = [glove_vectorizer.embeddings_index.get(t.lower()) for t in tokens]
        vectors = [v for v in vectors if v is not None]

        # pad or truncate
        if len(vectors) > timesteps:
            vectors = vectors[:timesteps]
        else:
            while len(vectors) < timesteps:
                vectors.append(np.zeros(embedding_dim, dtype=np.float32))

        inputs.append(np.vstack(vectors))

    return np.array(inputs)
