# Lexical Lab

**Entity-Level Sentiment & Linking Evaluation Suite**

NLP experimentation framework for benchmarking entity-specific sentiment analysis and entity linking using a variety of modeling pipelines. Rapidly prototype, compare, and validate classic and neural NLP architectures using scikit-learn and TensorFlow.

---

## Project Goals

- Evaluate NLP models on complex structured tasks like **entity-level sentiment** and **entity linking**
- Support comparison across:
  - TF-IDF + Linear Models (LogisticRegression, SVM)
  - GloVe Embeddings + Dense Classifiers
  - Hand-crafted TensorFlow LSTM networks
- Enable **repeatable**, **transparent**, and **modular** experimentation
- Support **CI/CD** benchmarking via GitHub Actions or other pipelines

---

## What is Lexical Lab?

Given a CSV of documents and labeled entities with sentiment, Lexical Lab:

1. Performs **entity recognition** and **segmentation** using spaCy
2. Links entities to canonical concepts (via local dictionaries or external APIs)
3. Vectorizes segments using TF-IDF or GloVe embeddings
4. Runs selected models (linear, dense, or LSTM)
5. Outputs per-entity sentiment predictions
6. Logs evaluation results for comparison

---

## Features

- Modular training pipeline (`run_experiment.py`)
- Pre-built linear and neural model runners
- Customizable `config/config.yaml` for easy swaps
- Preprocessing & embedding utilities
- Evaluation and error reporting tools
- GitHub Actions CI support (for automated benchmarking)
- Export to CSV, JSON, or visualization via `matplotlib`

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/n1conroy/lexical-lab.git
cd lexical-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings (optional)
bash scripts/download_glove.sh

# Run an experiment
python run_experiment.py --config config/config.yaml
