# models/model_factory.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

def get_sklearn_model(model_type="logreg"):
    if model_type == "logreg":
        return LogisticRegression(max_iter=500)
    elif model_type == "svm":
        return SVC(probability=True)
    else:
        raise ValueError(f"Unknown sklearn model type: {model_type}")

def build_dense_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(timesteps, embedding_dim, num_classes, units=64, dropout_rate=0.4, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, input_shape=(timesteps, embedding_dim), return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model(config, input_shape):
    model_type = config["type"]

    if model_type in ["logreg", "svm"]:
        return get_sklearn_model(model_type)

    elif model_type == "dense":
        input_dim = config["input_dim"]
        return build_dense_model(input_dim, config["num_classes"])

    elif model_type == "lstm":
        return build_lstm_model(
            timesteps=config["timesteps"],
            embedding_dim=config["embedding_dim"],
            num_classes=config["num_classes"],
            units=config["units"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"]
        )

    else:
        raise ValueError(f"Unknown model type in config: {model_type}")
