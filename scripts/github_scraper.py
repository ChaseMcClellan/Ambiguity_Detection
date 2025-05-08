import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# GitHub authentication
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# Repo to scrape from
REPO = "eslint/eslint"

# Labels that likely indicate a requirement
TARGET_LABELS = {"enhancement", "feature", "idea", "proposal", "suggestion"}

# API endpoint
API_URL = f"https://api.github.com/repos/{REPO}/issues"

def get_requirements(max_pages=5):
    requirements = []
    for page in range(1, max_pages + 1):
        response = requests.get(API_URL, headers=HEADERS, params={
            "state": "open",
            "per_page": 100,
            "page": page
        })

        if response.status_code != 200:
            print(f"Failed to fetch issues: {response.status_code}")
            break

        issues = response.json()
        if not issues:
            break

        for issue in issues:
            # Skip pull requests
            if "pull_request" in issue:
                continue

            labels = {label["name"].lower() for label in issue.get("labels", [])}
            if labels & TARGET_LABELS:
                requirements.append({
                    "id": issue["number"],
                    "title": issue["title"],
                    "body": issue["body"] or "",
                    "labels": list(labels),
                    "url": issue["html_url"]
                })

    return requirements

def save_to_file(data, filename='../data/requirements.json'):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} requirements to {filename}")

if __name__ == "__main__":
    requirements = get_requirements()
    save_to_file(requirements)
