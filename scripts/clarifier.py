'''
#clarification script
Then Chase, you can build the clarification script.
It will read from that ambiguity report, and for any requirement marked as vague,
it will call the LLM to generate a better version, saving it to output/refined_requirements.json.

 STEP #1
 Reads output/ambiguity_report.json (from ambiguity detector)

 STEP #2
 Processes each vague requirement (where ambiguous terms were detected)

 Step #3
 Calls clarify_requirement() from ollama_prompting.py to:
       -Ask 2 clarifying questions.
       -Rewrite the requirement.

 '''


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os
import logging
from ollama_prompting import clarify_requirement
from tqdm import tqdm

log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "clarifier.log")
logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

#TODO Change where input/output should be
INPUT_FILE = "output/ambiguity_report.json"
OUTPUT_FILE = "output/refined_requirements.json"

#load the report:

def load_ambiguity_report(filepath):
 if not os.path.exists(filepath):
  raise FileNotFoundError(f"input file not found: {filepath}")
 with open(filepath, "r", encoding="utf-8") as file:
  try:
   return json.load(file)
  except json.JSONDecodeError as e:
   raise ValueError(f"Cannot parse json: {e}")

def parse_llm_response(text):
    import re

    questions = []
    rewritten = ""

    lines = text.strip().split("\n")

    for i, line in enumerate(lines):
        line_clean = line.strip()
        lower = line_clean.lower()

        if lower.startswith("clarified requirement:") or lower.startswith("clarified:"):
            # Try to extract directly
            rewritten = line_clean.split(":", 1)[-1].strip()
        elif re.match(r"^(q[1-2]|[1-2]\.|- q[1-2])", lower):
            questions.append(line_clean)

    # Fallback to last non-question line if rewritten still empty
    if not rewritten:
        non_q_lines = [
            l.strip() for l in lines
            if l.strip() and not re.match(r"^(q[1-2]|[1-2]\.|questions|clarified requirement|clarified)", l.strip().lower())
        ]
        if non_q_lines:
            rewritten = non_q_lines[-1]

    if not questions or not rewritten:
        with open("logs/clarifier_debug.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- MALFORMED LLM RESPONSE ---\n{text.strip()}\n")
        raise ValueError("Malformed LLM response")

    return questions[:2], rewritten


def process_requirements(data):
    refined = []

    for entry in tqdm(data, desc="Clarifying Requirements"):
        original = entry.get("original")
        terms = entry.get("ambiguous_terms", [])

        #skip if there's no ambiguity
        if not terms or terms == ["None"]:
            continue

        try:

            llm_output = clarify_requirement(original, terms)
            print("🧠 LLM OUTPUT:", llm_output)
            with open("logs/clarifier_raw_output.txt", "a", encoding="utf-8") as f:
                f.write(f"\n---\nORIGINAL:\n{llm_output.strip()}\n---\n")

            questions, rewritten = parse_llm_response(llm_output)

            refined.append({
                "original": original,
                "ambiguous_terms": terms,
                "questions": questions,
                "rewritten": rewritten
            })

        except Exception as e:
            #skip and move on
            logging.error(f"Failed to process requirement: '{original}' | Error: {str(e)}")
            continue

    return refined


def save_refined_requirements(output_data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

def main():
    data = load_ambiguity_report(INPUT_FILE)
    results = process_requirements(data)
    save_refined_requirements(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()