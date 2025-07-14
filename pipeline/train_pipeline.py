import os
import json
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
        # Prepare features and labels
        X = self.vectorizer.fi
