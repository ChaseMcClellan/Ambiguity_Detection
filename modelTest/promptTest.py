#test to see if gpt2 will respond to different prompts
from loadGPT2 import load_model
import torch

def generate_response(prompt: str):
    model, tokenizer, device = load_model()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    test_prompt = (
        "Ambiguous requirement: The interface must be intuitive and efficient.\n"
        "Ambiguous terms: intuitive, efficient\n"
        "Ambiguous requirement: The system must be fast and user-friendly.\n"
        "Ambiguous terms:"
    )
    result = generate_response(test_prompt)
    print("=== GPT-2 Output ===")
    print(result)
