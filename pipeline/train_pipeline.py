import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_loader import load_entity_sentiment_data
from embeddings.vectorizer_factory import get_vectorizer
from models.model_factory import get_model
from preprocessing.text_processing import preprocess_texts


class TrainPipeline:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.get("output_dir", "reports")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        self.data = load_entity_sentiment_data(config["data"]["input_file"])

        # Preprocess texts (tokenization, NER, etc)
        self.data = preprocess_texts(self.data, config)

        # Setup vectorizer
        self.vectorizer = get_vectorizer(config["vectorizer"])

        # Setup model
        self.model = get_model(config["model"])

    def run(self):
        # Prepare features and labels depending on vectorizer type
        if self.config["vectorizer"]["type"].lower() == "glove":
            # GloVe vectorizer returns dense numpy matrix directly
            X = self.vectorizer.fit_transform(self.data)
        else:
            # TFIDF returns sparse matrix
            texts = [rec["preprocessed_text"] for rec in self.data]
            X = self.vectorizer.fit_transform(texts)

        y = [rec["sentiment_labels"] for rec in self.data]

        # encode y to integers for TF models or sklearn (assumes labels are strings)
        classes = sorted(set(y))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_encoded = np.array([class_to_idx[label] for label in y])

        # split train/test
        test_frac = self.config.get("test_fraction", 0.2)
        split_idx = int(len(X) * (1 - test_frac))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

        if hasattr(self.model, "fit") and "tensorflow" not in str(type(self.model)).lower():
            # sklearn models
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self._eval_and_save(y_test, y_pred)
        else:
            # TF keras models expect dense numpy arrays
            X_train = self._prepare_tf_input(X_train)
            X_test = self._prepare_tf_input(X_test)

            self.model.fit(
                X_train, y_train,
                epochs=self.config["model"].get("epochs", 10),
                batch_size=self.config["model"].get("batch_size", 32),
                validation_split=0.1,
                verbose=2
            )
            y_probs = self.model.predict(X_test)
            y_pred = np.argmax(y_probs, axis=1)
            self._eval_and_save(y_test, y_pred)

        print("Training complete.")

    def _prepare_tf_input(self, X):
        """
        Converts sparse matrix or dense matrix into proper shape
        for TF models. For LSTM expects 3D: (samples, timesteps, embedding_dim)
        For dense: (samples, input_dim)
        """

        if "lstm" in self.config["model"]["type"].lower():
            # reshape to 3D
            # Assuming input was flattened so reshape to (samples, timesteps, embedding_dim)
            timesteps = self.config["model"]["timesteps"]
            embedding_dim = self.config["model"]["embedding_dim"]

            X_dense = X if isinstance(X, np.ndarray) else X.toarray()
            samples = X_dense.shape[0]

            # Check if X size matches timesteps*embedding_dim
            if X_dense.shape[1] != timesteps * embedding_dim:
                raise ValueError(
                    f"Input feature dimension ({X_dense.shape[1]}) does not match timesteps*embedding_dim ({timesteps*embedding_dim})"
                )

            X_reshaped = X_dense.reshape((samples, timesteps, embedding_dim))
            return X_reshaped

        else:
            # dense or tfidf vectorizer output for dense net
            return X if isinstance(X, np.ndarray) else X.toarray()

    def _eval_and_save(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        report_path = os.path.join(self.output_dir, "classification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
