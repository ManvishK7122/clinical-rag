"""
Score trial screening against hand-labeled criteria.

The key comes from:  python matcher.py --trials <set> --limit 5 --make-key
Fill in each "label" with TRUE, FALSE, or UNKNOWN by reading the note yourself.

- false clear (most important): the tool cleared a criterion that your label
  says it shouldn't have. Could put an ineligible patient on a shortlist.
- false block: the tool ruled the patient out when it shouldn't have.
- missed info: the tool said UNKNOWN, but the note did answer it.

Usage:
    python eval_matcher.py outputs/match_septic_icu_Sample_ICU_note.json evals/criteria_key_septic_icu.json
"""

import json
import sys
from collections import Counter
from screening.matcher import is_blocker, is_cleared


def main(report_path, key_path):
    report = json.load(open(report_path))
    key = {(k["nct_id"], k["text"]): k for k in json.load(open(key_path))
           if k.get("label") in {"TRUE", "FALSE", "UNKNOWN"}}

    total = correct = 0
    false_clears, false_blocks, missed_info = [], [], []
    confusion = Counter()

    for trial in report["trials"]:
        for c in trial["criteria"]:
            k = key.get((trial["nct_id"], c["text"]))
            if not k:
                continue
            truth = {**c, "decision": k["label"]}
            total += 1
            confusion[(k["label"], c["decision"])] += 1
            if c["decision"] == k["label"]:
                correct += 1
                continue
            label = f"{trial['nct_id']} ({c['type']}): {c['text'][:90]}"
            if is_cleared(c) and not is_cleared(truth):
                false_clears.append(label)
            elif is_blocker(c) and not is_blocker(truth):
                false_blocks.append(label)
            elif c["decision"] == "UNKNOWN":
                missed_info.append(label)

    if total == 0:
        print("No labeled criteria matched the report. Fill in labels in the key first.")
        return

    print(f"\nLabeled criteria scored: {total}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.0%}")
    print(f"False clears (most important): {len(false_clears)}")
    print(f"False blocks: {len(false_blocks)}")
    print(f"Missed info (UNKNOWN but documented): {len(missed_info)}")

    print("\nConfusion (rows = your label, columns = tool):")
    labels = ["TRUE", "FALSE", "UNKNOWN"]
    print("            " + "".join(f"{l:>9}" for l in labels))
    for truth in labels:
        print(f"  {truth:<9} " + "".join(f"{confusion[(truth, tool)]:>9}" for tool in labels))

    for title, items in [("FALSE CLEARS", false_clears), ("FALSE BLOCKS", false_blocks)]:
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval_matcher.py <match_report.json> <criteria_key.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])