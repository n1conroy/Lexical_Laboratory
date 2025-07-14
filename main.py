# main.py

import sys
import yaml
import traceback

from pipeline.train_pipeline import TrainPipeline

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        config = load_config(config_path)
        pipeline = TrainPipeline(config)
        pipeline.run()
    except Exception as e:
        print("🔥 Something went wrong during training:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
