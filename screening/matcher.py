"""
Pre-screen a patient against clinical trial eligibility criteria.

For each criterion, the decision is TRUE, FALSE, or UNKNOWN for this patient:
- inclusion criterion TRUE  -> requirement met
- exclusion criterion TRUE  -> patient is excluded (a blocker)
- UNKNOWN                   -> not documented: check the chart

Each criterion goes one of three ways:
- numeric (labs, vitals, scores): the model only reads values; code compares
  them with hard rules (see rules.py)
- fact (diagnosis, medication, pregnancy...): the model judges, with a quote
- not checkable (consent, investigator judgment): left to the nurse

Safety rules:
- Criteria are split in code, and each one is judged on its own.
- A TRUE or FALSE decision needs an exact quote from the note. If the quote
  can't be found in the note, the decision is downgraded to UNKNOWN.
- The tool never says "eligible". It ranks possible matches and shows the
  evidence so a nurse can verify each decision quickly.

Usage:
    python -m screening.matcher --trials septic_icu --limit 5
    python -m screening.matcher --trials septic_icu --limit 5 --make-key
"""

import argparse
import json
import re
from pathlib import Path

from config import DATA_DIR, DEFAULT_FILENAME, OUTPUT_DIR
from screening.snapshot import (Evidence, ask_json, extract_patient, load_pages,
                                note_with_page_markers, verify_evidence)
from screening.rules import check_numeric, parse_criterion
from screening.trials import load_trials


HEADER_RE = re.compile(r"^\s*(?:key\s+)?(inclusion|exclusion)\s+criteria\b[^:]*:?\s*$", re.I)
BULLET_RE = re.compile(r"^\s*(?:[*\-•]|\d+[.)])\s+")


def split_criteria(text):
    """Turn ClinicalTrials.gov eligibility text into
    [{"type": "inclusion" or "exclusion", "text": "..."}]."""
    criteria, section = [], None
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        header = HEADER_RE.match(line)
        if header:
            section = header.group(1).lower()
            continue
        if section is None:
            continue
        if BULLET_RE.match(line):
            criteria.append({"type": section, "text": BULLET_RE.sub("", line).strip()})
        elif criteria and criteria[-1]["type"] == section:
            criteria[-1]["text"] += " " + line
        else:
            criteria.append({"type": section, "text": line})
    return criteria


def parse_age_years(value):
    """'18 Years' -> 18, '6 Months' -> 0.5, None or unparseable -> None."""
    if not value:
        return None
    match = re.match(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", value.strip(), re.I)
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2).lower()
    return number / {"year": 1, "month": 12, "week": 52, "day": 365}[unit]


def check_age(age, min_age, max_age):
    if age is None:
        return "UNKNOWN", "Patient age not documented."
    low, high = parse_age_years(min_age), parse_age_years(max_age)
    if low is not None and age < low:
        return "FALSE", f"Age {age} is below the minimum ({min_age})."
    if high is not None and age > high:
        return "FALSE", f"Age {age} is above the maximum ({max_age})."
    return "TRUE", f"Age {age} is within the trial's range ({min_age or 'none'} to {max_age or 'none'})."


def check_sex(sex, trial_sex):
    if not trial_sex or trial_sex.upper() == "ALL":
        return "TRUE", "Trial accepts all sexes."
    if not sex:
        return "UNKNOWN", "Patient sex not documented."
    if sex.strip().upper().startswith(trial_sex.upper()[0]):
        return "TRUE", f"Trial requires {trial_sex.lower()}; patient is {sex.lower()}."
    return "FALSE", f"Trial requires {trial_sex.lower()}; patient is {sex.lower()}."


CHECK_PROMPT = """You are helping a clinical research nurse pre-screen a patient for a
clinical trial. Decide whether the criterion statement below is TRUE, FALSE, or
UNKNOWN for this patient, using ONLY the clinical note.

- TRUE: the note clearly shows the statement is true for this patient.
- FALSE: the note clearly shows the statement is false for this patient.
- UNKNOWN: the note does not document enough to decide. When in doubt, choose
  UNKNOWN. Never guess, and never assume what is "likely" for this kind of patient.

For TRUE or FALSE, give the page number and an EXACT quote copied word for word
from the note (3 to 25 words) that proves it.

Criterion ({ctype}): {criterion}

Return JSON: {{"decision": "TRUE" or "FALSE" or "UNKNOWN", "page": <number or null>,
"quote": "<exact words>" or null, "reason": "<one short sentence>"}}"""


def check_criterion(criterion, note_text, pages):
    parsed = parse_criterion(criterion["text"])
    if parsed and parsed.kind == "not_checkable":
        return {**criterion, "decision": "UNKNOWN", "page": None, "quote": None, "verified": False,
                "reason": "Needs the nurse or investigator (consent, judgment, or not in a note).",
                "checked_by": "nurse"}
    numeric = check_numeric(parsed, note_text, pages)
    if numeric:
        return {**criterion, **numeric, "checked_by": "code"}
    return {**check_fact(criterion, note_text, pages), "checked_by": "model"}


def check_fact(criterion, note_text, pages):
    raw = ask_json(CHECK_PROMPT.format(ctype=criterion["type"], criterion=criterion["text"]),
                   note_text)
    decision = str(raw.get("decision", "UNKNOWN")).upper()
    if decision not in {"TRUE", "FALSE", "UNKNOWN"}:
        decision = "UNKNOWN"
    result = {**criterion, "decision": decision, "reason": raw.get("reason", ""),
              "page": raw.get("page"), "quote": raw.get("quote"), "verified": False}

    if decision in {"TRUE", "FALSE"}:
        try:
            evidence = verify_evidence(Evidence(page=int(raw.get("page") or 0),
                                                quote=raw.get("quote") or ""), pages)
        except (TypeError, ValueError):
            evidence = None
        if evidence and evidence.verified:
            result.update(page=evidence.page, verified=True)
        else:
            result["decision"] = "UNKNOWN"
            result["reason"] += " (Downgraded: the evidence quote could not be found in the note.)"
    return result


def is_blocker(c):
    return (c["type"] == "inclusion" and c["decision"] == "FALSE") or \
           (c["type"] == "exclusion" and c["decision"] == "TRUE")


def is_cleared(c):
    return (c["type"] == "inclusion" and c["decision"] == "TRUE") or \
           (c["type"] == "exclusion" and c["decision"] == "FALSE")


def screen_trial(trial, patient, note_text, pages):
    age_decision, age_reason = check_age(patient.age, trial.get("min_age"), trial.get("max_age"))
    sex_decision, sex_reason = check_sex(patient.sex, trial.get("sex"))
    results = [
        {"type": "inclusion", "text": "Age within the trial's age range", "decision": age_decision,
         "reason": age_reason, "page": None, "quote": None, "verified": age_decision != "UNKNOWN",
         "checked_by": "code"},
        {"type": "inclusion", "text": "Sex accepted by the trial", "decision": sex_decision,
         "reason": sex_reason, "page": None, "quote": None, "verified": sex_decision != "UNKNOWN",
         "checked_by": "code"},
    ]
    for criterion in split_criteria(trial.get("eligibility_text", "")):
        results.append(check_criterion(criterion, note_text, pages))

    blockers = sum(is_blocker(c) for c in results)
    unknown = sum(c["decision"] == "UNKNOWN" for c in results)
    cleared = sum(is_cleared(c) for c in results)
    if blockers:
        summary = f"Not a match ({blockers} blocker{'s' if blockers > 1 else ''})"
    elif unknown:
        summary = f"Possible match: {unknown} criteria for you to check"
    else:
        summary = "All criteria met per the note: confirm before referral"
    return {
        "nct_id": trial["nct_id"], "title": trial["title"], "url": trial["url"],
        "summary": summary,
        "blockers": blockers, "unknown": unknown, "cleared": cleared,
        "criteria": results,
    }


def screen_patient(pdf_path, trials):
    pages = load_pages(pdf_path)
    note_text = note_with_page_markers(pages)
    patient = extract_patient(note_text, pages)
    screened = [screen_trial(t, patient, note_text, pages) for t in trials]
    screened.sort(key=lambda r: (r["blockers"] > 0, r["unknown"], -r["cleared"]))
    return {"source": str(pdf_path), "patient": patient.model_dump(), "trials": screened}


def print_report(report):
    p = report["patient"]
    print(f"\nPatient: age {p.get('age')}, sex {p.get('sex')}\n")
    for r in report["trials"]:
        print(f"{r['nct_id']}  {r['summary']}")
        print(f"  {r['title']}")
        print(f"  cleared {r['cleared']} | blockers {r['blockers']} | unknown {r['unknown']}  {r['url']}")
        for c in r["criteria"]:
            if is_blocker(c):
                tag = "BLOCKER"
            elif c["decision"] == "UNKNOWN" and not r["blockers"]:
                tag = "CHECK"
            else:
                continue
            print(f"    {tag} ({c['type']}, {c['checked_by']}): {c['text'][:100]}")
            print(f"      {c['reason']} (p.{c['page']})" if c['page'] else f"      {c['reason']}")
        print()


def make_key(trials, path):
    """Write every criterion with an empty label, for hand labeling."""
    rows = []
    for t in trials:
        for c in split_criteria(t.get("eligibility_text", "")):
            rows.append({"nct_id": t["nct_id"], "type": c["type"], "text": c["text"],
                         "label": "", "notes": ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    return path, len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-screen a patient against saved trials.")
    parser.add_argument("--trials", required=True, help="name of a saved trial set in evals/trials/")
    parser.add_argument("--pdf", default=str(Path(DATA_DIR) / DEFAULT_FILENAME))
    parser.add_argument("--limit", type=int, default=5, help="how many trials to screen")
    parser.add_argument("--make-key", action="store_true",
                        help="write a labeling file instead of screening")
    args = parser.parse_args()

    trial_set = load_trials(args.trials)[: args.limit]
    if args.make_key:
        out, n = make_key(trial_set, Path("evals") / f"criteria_key_{args.trials}.json")
        print(f"Wrote {n} criteria to {out}. Fill in each label with TRUE, FALSE, or UNKNOWN.")
    else:
        report = screen_patient(args.pdf, trial_set)
        print_report(report)
        out = Path(OUTPUT_DIR) / f"match_{args.trials}_{Path(args.pdf).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Saved to {out}")