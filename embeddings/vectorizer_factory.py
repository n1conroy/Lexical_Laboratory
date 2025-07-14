import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GloveVectorizer:
    def __init__(self, glove_path: str, embedding_dim: int = 100):
        self.glove_path = glove_path
        self.embedding_dim = embedding_dim
        self.embeddings_index = {}
        self._load_glove_embeddings()

    def _load_glove_embeddings(self):
        if not os.path.exists(self.glove_path):
            raise FileNotFoundError(f"GloVe file not found at {self.glove_path}")
        logger.info(f"Loading GloVe embeddings from {self.glove_path}...")
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                self.embeddings_index[word] = vector
        logger.info(f"Loaded {len(self.embeddings_index)} word vectors from GloVe.")

    def transform(self, records: List[dict]) -> np.ndarray:
        """
        Given a list of records with 'tokens' field,
        produce a dense matrix of shape (num_records, embedding_dim)
        by averaging embeddings of tokens present in GloVe.
        """
        features = []
        for rec in records:
            tokens = rec.get("tokens", [])
            vectors = [self.embeddings_index.get(t.lower()) for t in tokens if t.lower() in self.embeddings_index]
            if vectors:
                avg_vec = np.mean(vectors, axis=0)
            else:
                # If no tokens found in glove, fallback to zero vector
                avg_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            features.append(avg_vec)
        return np.vstack(features)

    def fit_transform(self, records: List[dict]) -> np.ndarray:
        # No fitting needed for GloVe embeddings, just transform
        return self.transform(records)

def get_vectorizer(config):
    vec_type = config.get("type", "tfidf").lower()

    if vec_type == "tfidf":
        max_features = config.get("max_features", 10000)
        ngram_range = tuple(config.get("ngram_range", (1,2)))
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    elif vec_type == "glove":
        glove_path = config.get("glove_path")
        embedding_dim = config.get("embedding_dim", 100)
        if not glove_path:
            raise ValueError("For glove vectorizer, 'glove_path' must be provided in config.")
        return GloveVectorizer(glove_path, embedding_dim)

    else:
        raise ValueError(f"Vectorizer type {vec_type} not supported.")
