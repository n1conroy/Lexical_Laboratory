# embeddings/vectorizer_factory.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os 

#USAGE IN YAML CONFIG

# #vectorizer:
#  type: glove
#  glove_path: data/glove.6B.100d.txt
#  embedding_dim: 100

class GloveVectorizer:
    def __init__(self, glove_path, embedding_dim):
        self.glove_path = glove_path
        self.embedding_dim = embedding_dim
        self.embeddings_index = {}
        self._load_embeddings()

    def _load_embeddings(self):
        print(f"🔍 Loading GloVe embeddings from {self.glove_path}")
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = np.asarray(parts[1:], dtype='float32')
                self.embeddings_index[word] = vector
        print(f"✅ Loaded {len(self.embeddings_index)} word vectors.")

    def fit_transform(self, records):
        """
        Converts tokenized records to flat fixed-length vectors.
        For LSTM input, use `prepare_lstm_inputs()` instead.
        This method is mainly for Dense NN baseline.
        """
        vectors = []
        for rec in records:
            tokens = rec.get("tokens", [])
            token_vectors = [self.embeddings_index.get(t.lower(), np.zeros(self.embedding_dim)) for t in tokens]
            if not token_vectors:
                token_vectors = [np.zeros(self.embedding_dim)]
            mean_vec = np.mean(token_vectors, axis=0)
            vectors.append(mean_vec)
        return np.vstack(vectors)


def get_vectorizer(config):
    vectorizer_type = config.get("type", "tfidf").lower()

    if vectorizer_type == "tfidf":
        return TfidfVectorizer(
            max_features=config.get("max_features", 5000),
            ngram_range=tuple(config.get("ngram_range", [1, 1]))
        )

    elif vectorizer_type == "glove":
        glove_path = config["glove_path"]
        embedding_dim = config["embedding_dim"]
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe file not found at {glove_path}")
        return GloveVectorizer(glove_path, embedding_dim)

    else:
        raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
