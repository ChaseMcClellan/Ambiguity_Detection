# scripts/evaluate.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_PATH = "models/clarified_model"

def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def clarify_with_model(prompt: str, max_tokens=128):
    model, tokenizer = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(model_path=MODEL_PATH):
    print("\nClarified Output Test\n")
    prompt = (
        "Clarify this requirement: Accept empty server and port in Create connection dialog with filling default values\n"
        "Ambiguous terms: empty, default\n"
        "Questions:\n"
        "Q1: What exactly does \"empty\" mean?\n"
        "Q2: What are the specific default values used?\n"
        "Clarified requirement:"
    )
    response = clarify_with_model(prompt)
    print("Prompt:\n" + prompt)
    print("\nModel Output:\n" + response)


if __name__ == "__main__":
    evaluate_model()