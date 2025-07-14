import argparse
import yaml
import os
from pipeline.train_pipeline import TrainPipeline

def main(config_path: str):
    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(config.get("output_dir", "reports"), exist_ok=True)

    pipeline = TrainPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lexical Lab experiments")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
