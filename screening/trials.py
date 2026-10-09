"""
Search ClinicalTrials.gov (API v2) for recruiting trials and save them.

Trials open, close, and change over time, so evaluations run on a saved
("frozen") copy in evals/trials/ instead of live search results.

Usage:
    python trials.py --save septic_icu "septic shock" "ARDS" "acute kidney injury"
"""

import argparse
import json
from pathlib import Path

import requests

from config import TRIALS_DIR, TRIALS_PAGE_SIZE, TRIALS_STATUS

API_URL = "https://clinicaltrials.gov/api/v2/studies"


def search_trials(conditions, status=TRIALS_STATUS, page_size=TRIALS_PAGE_SIZE):
    params = {
        "query.cond": " OR ".join(conditions),
        "filter.overallStatus": status,
        "pageSize": page_size,
        "format": "json",
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return [parse_study(s) for s in response.json().get("studies", [])]


def parse_study(study):
    protocol = study.get("protocolSection", {})
    ident = protocol.get("identificationModule", {})
    elig = protocol.get("eligibilityModule", {})
    nct_id = ident.get("nctId")
    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle"),
        "status": protocol.get("statusModule", {}).get("overallStatus"),
        "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
        "eligibility_text": elig.get("eligibilityCriteria", ""),
        "min_age": elig.get("minimumAge"),
        "max_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
    }


def save_trials(trials, name):
    path = Path(TRIALS_DIR) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trials, indent=2))
    return path


def load_trials(name):
    return json.loads((Path(TRIALS_DIR) / f"{name}.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search and save recruiting trials.")
    parser.add_argument("conditions", nargs="+", help='conditions, e.g. "septic shock" "ARDS"')
    parser.add_argument("--save", required=True, help="name for the saved trial set")
    args = parser.parse_args()

    found = search_trials(args.conditions)
    for t in found:
        print(f"{t['nct_id']}  {t['title']}")
    print(f"\n{len(found)} trials saved to {save_trials(found, args.save)}")