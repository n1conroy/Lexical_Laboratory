from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

def get_model(config):
    model_type = config.get("type", "logreg").lower()

    if model_type == "logreg":
        max_iter = config.get("max_iter", 1000)
        return LogisticRegression(max_iter=max_iter)

    elif model_type == "svm":
        kernel = config.get("kernel", "linear")
        C = config.get("C", 1.0)
        return SVC(kernel=kernel, C=C, probability=True)

    elif model_type == "dense":
        input_dim = config.get("input_dim")
        if not input_dim:
            raise ValueError("input_dim must be specified for dense model.")
        units = config.get("units", 64)
        dropout_rate = config.get("dropout_rate", 0.5)
        learning_rate = config.get("learning_rate", 0.001)

        model = models.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(units, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(units//2, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(config.get("num_classes", 2), activation="softmax")
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    elif model_type == "lstm":
        input_dim = config.get("input_dim")
        if not input_dim:
            raise ValueError("input_dim must be specified for lstm model.")
        timesteps = config.get("timesteps", 100)
        embedding_dim = config.get("embedding_dim", 100)
        units = config.get("units", 64)
        dropout_rate = config.get("dropout_rate", 0.5)
        learning_rate = config.get("learning_rate", 0.001)

        model = models.Sequential([
            layers.InputLayer(input_shap_
