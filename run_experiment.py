# run_experiment.py

import argparse
import yaml
from pipeline.train_pipeline import TrainPipeline

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def override_config(config, args):
    if args.model:
        config["model"]["type"] = args.model
    if args.vectorizer:
        config["vectorizer"]["type"] = args.vectorizer
    if args.data:
        config["data"]["input_file"] = args.data
    if args.epochs:
        config["model"]["epochs"] = int(args.epochs)
    if args.batch_size:
        config["model"]["batch_size"] = int(args.batch_size)
    return config

def main():
    parser = argparse.ArgumentParser(description="Lexical Lab Experiment Runner")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Override model type (logreg, dense, lstm)")
    parser.add_argument("--vectorizer", type=str, help="Override vectorizer (tfidf, glove)")
    parser.add_argument("--data", type=str, help="Override input CSV path")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")

    args = parser.parse_args()
    config = load_config(args.config)
    config = override_config(config, args)

    pipeline = TrainPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
