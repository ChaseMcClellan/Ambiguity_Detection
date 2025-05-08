from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_with_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def clarify_requirement(requirement, ambiguous_terms):
    terms = ", ".join(ambiguous_terms)
    prompt = f"""
You are a strict formatting bot trained to clarify ambiguous software requirements.

INSTRUCTIONS:
- You will output in **EXACTLY** the following format — no changes, no explanations.
- If the input is unclear or vague, do your best.

FORMAT:
Questions:
1. [question 1]
2. [question 2]

Clarified requirement: [your clarified requirement on this line]

EXAMPLE INPUT:
Requirement: "The UI must be clean and user-friendly."
Ambiguous terms: clean, user-friendly

EXAMPLE OUTPUT:
Questions:
1. What does "clean" mean in terms of UI layout or elements?
2. What specific actions should be considered user-friendly?

Clarified requirement: The UI must display a minimal layout with no more than 3 colors, and each element must include tooltips and accessible labels.

----

Now process this:

Requirement: "{requirement}"
Ambiguous terms: {terms}

Respond ONLY using the format shown above.
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
""{requirement}""
"""
    return generate_with_openai(prompt)
