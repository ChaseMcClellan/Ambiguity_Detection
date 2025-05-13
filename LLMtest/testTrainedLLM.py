from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "../scripts/fine_tuned_model"

#load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def clarify(requirement: str, ambiguous_terms: list):
    terms = ", ".join(ambiguous_terms)
    prompt = f"""
You are a strict formatting bot trained to clarify ambiguous software requirements.

INSTRUCTIONS:
- Output exactly in this format.
- No commentary or extra text.

FORMAT:
Questions:
1. [question 1]
2. [question 2]

Clarified requirement: [clarified statement]

Requirement: "{requirement}"
Ambiguous terms: {terms}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    examples = [
        {
            "requirement": "The system must be fast and user-friendly.",
            "ambiguous_terms": ["fast", "user-friendly"]
        },
        {
            "requirement": "Accept empty server and port in Create connection dialog with filling default values",
            "ambiguous_terms": ["empty", "default"]
        }
    ]

    for ex in examples:
        print("=== INPUT ===")
        print(f"Requirement: {ex['requirement']}")
        print(f"Ambiguous Terms: {ex['ambiguous_terms']}")
        print("=== OUTPUT ===")
        print(clarify(ex['requirement'], ex['ambiguous_terms']))
        print()
