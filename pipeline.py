import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.collect_and_label import run_collection_pipeline
from scripts.prepare_data import prepare_training_text
#from scripts.fine_tune import run_fine_tuning


def run_full_pipeline():
    print("Starting full clarification pipeline...")

    #collect labeled data
    run_collection_pipeline(max_pages=3)

    #prep training data
    prepare_training_text("data/refined_requirements.json", "data/clarification_train.txt")
    #run_fine_tuning("data/clarification_train.txt")

    print("Pipeline finished. Model data updated.")


if __name__ == "__main__":
    run_full_pipeline()