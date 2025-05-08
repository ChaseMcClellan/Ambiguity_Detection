from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_with_openai(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        with open("logs/openai_errors.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- ERROR ---\n{str(e)}\n")
        return ""


def clarify_requirement(requirement, ambiguous_terms):
    terms = ", ".join(ambiguous_terms)
    prompt = f"""
    You are a clarification engine. Do NOT explain anything.

    Your task is to:
    1. Ask two clarification questions about the vague requirement.
    2. Rewrite the requirement with greater clarity.

    Use **exactly this format**:

    Questions:
    1. ...
    2. ...

    Clarified requirement: ...

    DO NOT include any extra text, explanations, or introductions.
    DO NOT skip any of the sections.
    DO NOT number the final requirement or add labels.

    INPUT:
    Requirement: "{requirement}"
    Ambiguous terms: {terms}
    """

    return generate_with_openai(prompt)

def detect_ambiguity_with_llm(requirement: str):
    prompt = f"""
You are a software requirements reviewer.

Task:
- Read the requirement carefully.
- Identify and list any ambiguous, vague, or subjective words.
- Respond with only a JSON array of the ambiguous terms.

Examples:
Input: "The app should be fast and user-friendly."
Output: ["fast", "user-friendly"]

If no ambiguous terms are found, respond with:
["None"]

Requirement:
"{requirement}"
"""
    return generate_with_openai(prompt)
