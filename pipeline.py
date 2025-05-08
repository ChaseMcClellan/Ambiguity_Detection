# pipeline.py — full data pipeline

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.collect_and_label import run_collection_pipeline
from scripts.prepare_data import prepare_training_text
from scripts.fine_tune import run_fine_tuning


def run_full_pipeline():
    print("Starting full clarification pipeline...")

    # Step 1: Collect new labeled data
    run_collection_pipeline(max_pages=3)

    # Step 2: Prepare training data
    prepare_training_text("data/refined_requirements.json", "data/clarification_train.txt")

    # Step 3: Fine-tune GPT-2 model
    run_fine_tuning("data/clarification_train.txt")

    print("Pipeline finished. Model updated.")


if __name__ == "__main__":
    run_full_pipeline()