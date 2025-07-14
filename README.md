# Lexical Lab

**Entity-Level Sentiment & Linking Evaluation Suite**

NLP experimentation framework for benchmarking entity-specific sentiment analysis and entity linking using a variety of modeling pipelines. Rapidly prototype, compare, and validate classic and neural NLP architectures using scikit-learn and TensorFlow.

---

## Project Goals

- Evaluate NLP models on complex structured tasks like **entity-level sentiment** and **entity linking**
- Support comparison across:
  - TF-IDF + Linear Models (LogisticRegression, SVM)
  - GloVe Embeddings + Dense Classifiers
  - TensorFlow networks
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

python run_experiment.py --config config/config.yaml
Example Dataset Format
Each row contains a document, a recognized entity within it, and the associated sentiment label.

```
```
doc_id,text,entity,sentiment
1,"I am concerned about my next trip overseas- are these airplanes safe?.",["Air Travel, TSA"], {"wary":7, "certainty":4}
1,"Basically breathless about my last trip to Tahiti. Make sure to check out Laguna Lodge.",["Laguna Lodge", "Tahiti"],{"adventurous":6, "energetic":7, "certainty":4}
2,"Amazon faced backlash over working conditions.",["Amazon"],{"dissent":7, "certainty":1}
```
```
lexical-lab/
├── config/                 # Experiment configuration files
├── ci/                     # GitHub Actions or other CI integrations
├── data/                   # Input datasets and intermediate files
├── embeddings/             # GloVe loader and vectorization tools
├── models/                 # Model implementations (linear, LSTM, etc.)
├── pipeline/               # Training and evaluation pipeline
├── preprocessing/          # Text processing and entity extraction
├── reports/                # Generated outputs and metrics
├── scripts/                # GloVe download and other shell tools
├── run_experiment.py       # Main experiment entrypoint
└── requirements.txt
```
## Configuration
Model and experiment settings are defined in config/config.yaml. This includes:
Model type (logreg, svm, dense, lstm)
Vectorizer type (tfidf, glove)
Tokenizer and segmentation settings
Training parameters (epochs, batch size, etc.)

```
Requirements
Python 3.9+

spaCy

scikit-learn

TensorFlow 2.x

pandas

numpy

matplotlib

seaborn
```
## Entity Linking
This framework includes simple local linking via dictionary-based mappings and can be extended to use APIs such as Wikidata. Entity linking is modular and pluggable via preprocessing hooks.

## Output
All experiments generate structured output under reports/ including:

Accuracy, precision, recall, F1

Per-entity performance breakdown

Confusion matrix visualizations

CSVs and JSON logs for CI compatibility

License
MIT License
