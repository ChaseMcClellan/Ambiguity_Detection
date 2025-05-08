import requests

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_with_ollama(prompt, model=MODEL_NAME):
    """Send a prompt to Ollama and return the generated response."""
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_API, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}, {response.text}"


def clarify_requirement(requirement, ambiguous_terms):
    """
    Given a vague requirement and list of ambiguous terms, ask the model
    to generate clarifying questions and rewrite it clearly.
    """
    terms = ", ".join(ambiguous_terms)
    prompt = f"""

    You are a software requirements analyst.

    Given the requirement below, identify any ambiguous or vague terms. Then:
    1. Ask exactly **two clarifying questions** to help refine the requirement.
    2. Rewrite the requirement to be **clearer and more specific**, using neutral, concise language.
    
Original Requirement:
"{requirement}"

Ambiguous terms detected: {terms}

Please:
1. Ask 2 clarifying questions to improve the requirement.
2. Rewrite the requirement with specific details only making it more specific based on the ambiguous terms. Do not add extra functionality or assumptions.

Respond in this format:
Questions:
- Q1
- Q2

Rewritten Requirement:
<write your improved version here on this line>
"""
    return generate_with_ollama(prompt)


def detect_ambiguity_with_llm(requirement: str):
    prompt = f"""
    You are a software requirements reviewer.

    Task:
    - Analyze the requirement carefully.
    - List any words or phrases that are ambiguous, unclear, or subjective.
    - Return the list in a simple JSON array format. Example:
        ["fast", "secure", "easy to use"]

    If no ambiguous terms are found, return:
        ["None"]

    Requirement:
    \"\"\"{requirement}\"\"\"
    """
    return generate_with_ollama(prompt)
