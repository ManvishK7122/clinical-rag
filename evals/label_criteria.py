"""
Label trial criteria one at a time in the terminal, instead of editing JSON by hand.

Shows each unlabeled criterion for the trials you pick. Type:
  t = TRUE, f = FALSE, u = UNKNOWN, s = skip for now, q = save and quit
Progress is saved after every answer, so you can stop and come back.

Remember: for an EXCLUSION criterion, TRUE means the patient HAS the excluding condition.

Usage:
    python -m evals.label_criteria                       # show criteria counts per trial
    python -m evals.label_criteria NCT07383103 NCT06849570
"""

import json
import sys
from collections import Counter
from pathlib import Path

KEY_PATH = Path("evals/criteria_key_septic_icu.json")
CHOICES = {"t": "TRUE", "f": "FALSE", "u": "UNKNOWN"}


def main(trial_ids):
    rows = json.loads(KEY_PATH.read_text())
    if not trial_ids:
        counts = Counter(r["nct_id"] for r in rows)
        done = Counter(r["nct_id"] for r in rows if r.get("label"))
        for nct, n in counts.items():
            print(f"{nct}: {done[nct]}/{n} labeled")
        return

    todo = [r for r in rows if r["nct_id"] in trial_ids and not r.get("label")]
    print(f"{len(todo)} criteria to label. Keep the note open next to you.\n")
    for i, r in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {r['nct_id']}  {r['type'].upper()}")
        print(f"  {r['text']}")
        if r["type"] == "exclusion":
            print("  (TRUE = patient HAS this, so they are excluded)")
        while True:
            answer = input("  t / f / u / s(kip) / q(uit): ").strip().lower()
            if answer in CHOICES or answer in {"s", "q"}:
                break
        if answer == "q":
            break
        if answer in CHOICES:
            r["label"] = CHOICES[answer]
            note = input("  note (optional, Enter to skip): ").strip()
            if note:
                r["notes"] = note
            KEY_PATH.write_text(json.dumps(rows, indent=2))
        print()
    KEY_PATH.write_text(json.dumps(rows, indent=2))
    print("Saved.")


if __name__ == "__main__":
    main(sys.argv[1:])