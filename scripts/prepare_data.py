import json
import os

def prepare_training_text(input_json_path: str, output_txt_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        original = item.get("original", "").strip()
        terms = ", ".join(item.get("ambiguous_terms", []))
        questions = "\n".join(item.get("questions", []))
        rewritten = item.get("rewritten", "").strip()

        example = (
            f"Clarify this requirement: {original}\n"
            f"Ambiguous terms: {terms}\n"
            f"Questions:\n{questions}\n"
            f"Clarified requirement: {rewritten}\n"
        )
        examples.append(example)

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as out:
        out.write("\n\n".join(examples))

    print(f"Saved {len(examples)} clarification training examples to {output_txt_path}")

if __name__ == "__main__":
    prepare_training_text("data/refined_requirements.json", "data/clarification_train.txt")