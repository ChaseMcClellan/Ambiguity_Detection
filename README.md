# Ambiguity Detection

This repository provides scripts for collecting software requirements from GitHub, detecting ambiguous terms using an LLM, and generating clarified requirements. It relies on the OpenAI API for language model interactions and Hugging Face Transformers for optional fine‑tuning.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**
   Create a `.env` file in the project root containing:
   ```
   OPENAI_API_KEY=your_openai_key
   GITHUB_TOKEN=your_github_token
   ```
   These keys are required for calling the OpenAI API and accessing the GitHub API.

## Running the Pipeline

The main pipeline scrapes issues from GitHub, detects ambiguous terms, and produces a dataset of clarified requirements.

```bash
python pipeline.py
```

Intermediate files are written to the `data/` and `output/` directories:
- `data/requirements.json` – raw issues scraped from GitHub
- `output/ambiguity_report.json` – ambiguous terms detected by the LLM
- `data/refined_requirements.json` – clarified requirements with follow‑up questions

## Operating the LLM

Language model calls are handled in `scripts/ollama_prompting.py`. Running the pipeline or individual scripts will automatically invoke the OpenAI model specified in that file.

To test the model directly, run the clarification script on an existing ambiguity report:
```bash
python scripts/clarifier.py
```
This reads `output/ambiguity_report.json`, calls the model to ask clarifying questions, and writes the results to `output/refined_requirements.json`.

For ambiguity detection only:
```bash
python scripts/ambiguity_detector.py
```

## Fine‑tuning (Optional)

If you have prepared training data (`data/clarification_train.txt`), you can fine‑tune a GPT‑2 model:
```bash
python scripts/fine_tune.py
```
The fine‑tuned model is saved under `scripts/fine_tuned_model/`.

## Testing the Trained Model

Example scripts inside `LLMtest/` demonstrate how to load and query the base or fine‑tuned GPT‑2 model.

---

This READ.ME was created by OpenAI's Codex, all other files are original. 
