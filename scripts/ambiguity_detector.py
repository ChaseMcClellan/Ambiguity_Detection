import sys
import re
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import os
import logging
from ollama_prompting import detect_ambiguity_with_llm
from tqdm import tqdm

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "ambiguity.log")
logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# File paths
INPUT_FILE = "../data/requirements.json"
OUTPUT_FILE = "output/ambiguity_report.json"

def load_input_requirements(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
        return [{"text": item["body"]} for item in data if "body" in item]

def generate_ambiguity_report(requirements):
    results = []
    for req in tqdm(requirements, desc="Detecting Ambiguity"):
        try:
            req_text = req if isinstance(req, str) else req.get("text", "")
            raw_response = detect_ambiguity_with_llm(req_text)

            # Extract quoted terms
            ambiguous_terms = extract_terms_from_response(raw_response)

            results.append({
                "original":  req_text,
                "ambiguous_terms": ambiguous_terms
            })

        except Exception as e:
            logging.error(f"Failed to process requirement: '{req}' | Error: {str(e)}")
            continue

    return results

def extract_terms_from_response(response: str) -> list:
        return re.findall(r'"(.*?)"', response)

def save_ambiguity_report(output_data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

def main():
    data = load_input_requirements(INPUT_FILE)
    results = generate_ambiguity_report(data)
    save_ambiguity_report(results, OUTPUT_FILE)

if __name__ == "__main__":
    main()



