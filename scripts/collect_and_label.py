# scripts/collect_and_label.py
import os
import json
from github_scraper import get_requirements, save_to_file as save_scraped
from ambiguity_detector import generate_ambiguity_report, extract_terms_from_response
from clarifier import parse_llm_response
from ollama_prompting import detect_ambiguity_with_llm, clarify_requirement
from tqdm import tqdm

RAW_OUTPUT = "data/requirements.json"
AMBIGUITY_OUTPUT = "output/ambiguity_report.json"
REFINED_OUTPUT = "data/refined_requirements.json"


def run_collection_pipeline(max_pages=2):
    print("🔎 Scraping GitHub issues tagged as requirements...")
    scraped = get_requirements(max_pages=max_pages)
    save_scraped(scraped, RAW_OUTPUT)

    print("🤖 Detecting ambiguous terms using Ollama...")
    requirements = [{"text": item["body"]} for item in scraped if item.get("body")]
    ambiguity_results = []
    for req in tqdm(requirements, desc="Detecting Ambiguity"):
        req_text = req["text"]
        raw_response = detect_ambiguity_with_llm(req_text)
        ambiguous_terms = extract_terms_from_response(raw_response)
        ambiguity_results.append({
            "original": req_text,
            "ambiguous_terms": ambiguous_terms
        })

    with open(AMBIGUITY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(ambiguity_results, f, indent=2)

    print("🪄 Clarifying ambiguous requirements using fine-tuned GPT-2...")
    refined = []
    for entry in tqdm(ambiguity_results, desc="Clarifying Requirements"):
        original = entry.get("original")
        terms = entry.get("ambiguous_terms", [])

        if not terms or terms == ["None"]:
            continue

        try:
            llm_output = clarify_requirement(original, terms)
            questions, rewritten = parse_llm_response(llm_output)

            refined.append({
                "original": original,
                "ambiguous_terms": terms,
                "questions": questions,
                "rewritten": rewritten
            })
        except Exception as e:
            print(f"Failed to clarify: {original}\n  Error: {e}")

    os.makedirs(os.path.dirname(REFINED_OUTPUT), exist_ok=True)
    with open(REFINED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)

    print(f"Pipeline complete. {len(refined)} refined requirements saved to {REFINED_OUTPUT}")


if __name__ == "__main__":
    run_collection_pipeline(max_pages=5)