"""
Score a saved patient snapshot against a hand-made answer key.

- recall: how many key items were found AND put in the right group
- wrong group: key items found but put in the wrong group
- extra: extracted items that aren't in the key (review these by hand)

Usage:
    python eval_snapshot.py outputs/snapshot_Sample_ICU_note.json evals/snapshot_key_sample_icu.json
"""

import json
import re
import sys

MED_GROUPS = {"active": "given", "given_once": "given", "held": "not_given",
              "discontinued": "not_given", "considered": "not_given", "planned": "not_given"}


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def matches(key_entry, extracted_name):
    """A key entry like 'cr|creatinine' matches if all words of any alias
    appear in the extracted name. Whole words only, so 'pt' won't match 'aptt'."""
    name_words = words(extracted_name)
    return any(words(alias) and words(alias) <= name_words for alias in key_entry.split("|"))


def score_section(extracted, key_groups, group_of, check_conflicts=False):
    """check_conflicts: if one key item is extracted into two different groups
    (e.g. a medication both 'given' and 'not_given'), count it as a conflict,
    not a hit. A nurse would see contradictory information."""
    report, found_any = {}, set()
    for group, key_entries in key_groups.items():
        found = wrong = conflicts = 0
        wrong_items, conflict_items = [], []
        for entry in key_entries:
            hits = [e for e in extracted if matches(entry, e["name"])]
            if hits:
                found_any.update(id(h) for h in hits)
            groups_hit = {group_of(h) for h in hits}
            if check_conflicts and len(groups_hit) > 1:
                conflicts += 1
                conflict_items.append(f"{entry} -> {' and '.join(sorted(groups_hit))}")
            elif group in groups_hit:
                found += 1
            elif hits:
                wrong += 1
                wrong_items.append(f"{entry} -> {group_of(hits[0])}")
        report[group] = {"key_items": len(key_entries), "found_in_right_group": found,
                         "recall": round(found / len(key_entries), 2) if key_entries else None,
                         "wrong_group": wrong, "wrong_group_items": wrong_items,
                         "conflicts": conflicts, "conflict_items": conflict_items}
    extra = [e["name"] for e in extracted if id(e) not in found_any]
    return report, extra


def main(snapshot_path, key_path):
    snap = json.load(open(snapshot_path))
    key = json.load(open(key_path))
    results = {}

    if "medications" in key:
        results["medications"] = score_section(
            snap["medications"], key["medications"], lambda m: MED_GROUPS[m["status"]],
            check_conflicts=True)
    if "labs" in key:
        results["labs"] = score_section(
            snap["labs"], key["labs"], lambda l: l["interpretation"])

    for section, (groups, extra) in results.items():
        print(f"\n{section.upper()}")
        for group, r in groups.items():
            print(f"  {group:<13} recall {r['recall']}  ({r['found_in_right_group']}/{r['key_items']})"
                  f"  wrong group: {r['wrong_group']}  conflicts: {r['conflicts']}")
            for w in r["wrong_group_items"]:
                print(f"      wrong: {w}")
            for c in r["conflict_items"]:
                print(f"      CONFLICT: {c}")
        if extra:
            print(f"  extra (not in key, review by hand): {', '.join(extra)}")

    q = snap.get("quality", {})
    print(f"\nUnverified quotes: {q.get('unverified_quotes')} of {q.get('items')} items")


if __name__ == "__main__":
    snapshot_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/snapshot_Sample_ICU_note.json"
    key_path = sys.argv[2] if len(sys.argv) > 2 else "evals/snapshot_key_sample_icu.json"
    main(snapshot_path, key_path)