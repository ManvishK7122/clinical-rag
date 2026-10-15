"""
Run the patient snapshot on every held-out note that has an answer key, score it,
and print one table across all notes.

Usage:
    python -m evals.run_heldout            # run the tool, then score
    python -m evals.run_heldout --no-run   # only re-score saved outputs (no API calls)

Notes live in evals/heldout/notes/<name>.pdf and keys in evals/heldout/keys/<name>.json.
"""

import argparse
import json
import time
from pathlib import Path

from evals.eval_snapshot import MED_GROUPS, score_section

HELDOUT = Path("evals/heldout")
OUT_DIR = Path("outputs/heldout")


def score(snap, key):
    """Return {(section, group): (found, total, wrong)} plus extras for one note."""
    rows, extras = {}, []
    if "medications" in key:
        groups, extra = score_section(snap["medications"], key["medications"],
                                      lambda m: MED_GROUPS[m["status"]], check_conflicts=True)
        rows.update({("medications", g): r for g, r in groups.items()})
        extras += extra
    if "labs" in key:
        groups, extra = score_section(snap["labs"], key["labs"], lambda l: l["interpretation"])
        rows.update({("labs", g): r for g, r in groups.items()})
        extras += extra
    return rows, extras


def main(run):
    keys = sorted(p for p in HELDOUT.glob("keys/*.json") if not p.name.startswith("_"))
    if not keys:
        print("No keys found in evals/heldout/keys/. Copy _template.json to <name>.json first.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    totals, total_cost, total_seconds = {}, 0.0, 0.0
    for key_path in keys:
        name = key_path.stem
        pdf = HELDOUT / "notes" / f"{name}.pdf"
        out = OUT_DIR / f"snapshot_{name}.json"
        if run:
            from screening.snapshot import USAGE, build_snapshot, run_info
            USAGE.clear()
            started = time.perf_counter()
            snap = build_snapshot(pdf)
            snap["run"] = run_info(started)
            out.write_text(json.dumps(snap, indent=2))
        elif out.exists():
            snap = json.loads(out.read_text())
        else:
            print(f"\n{name}: no saved output, run without --no-run first")
            continue

        rows, extras = score(snap, json.loads(key_path.read_text()))
        q, r = snap.get("quality", {}), snap.get("run", {})
        total_cost += r.get("cost_usd", 0)
        total_seconds += r.get("total_seconds", 0)

        print(f"\n{name}")
        for (section, group), s in rows.items():
            print(f"  {section[:4]} {group:<13} {s['found_in_right_group']}/{s['key_items']}"
                  f"  wrong group: {s['wrong_group']}  conflicts: {s['conflicts']}")
            for c in s["conflict_items"]:
                print(f"      CONFLICT: {c}")
            t = totals.setdefault((section, group), [0, 0, 0, 0])
            t[0] += s["found_in_right_group"]; t[1] += s["key_items"]; t[2] += s["wrong_group"]
            t[3] += s["conflicts"]
        if extras:
            print(f"  extra (review by hand): {', '.join(extras)}")
        print(f"  unverified quotes: {q.get('unverified_quotes')} of {q.get('items')}"
              + (f"  |  ${r['cost_usd']:.4f}, {r['total_seconds']}s" if r else ""))

    print(f"\nALL HELD-OUT NOTES ({len(keys)})")
    for (section, group), (found, total, wrong, conflicts) in totals.items():
        recall = f"{found / total:.2f}" if total else "n/a"
        print(f"  {section[:4]} {group:<13} recall {recall}  ({found}/{total})  wrong group: {wrong}"
              f"  conflicts: {conflicts}")
    if total_cost:
        print(f"  cost ${total_cost:.4f}, {total_seconds:.0f}s total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and score held-out notes.")
    parser.add_argument("--no-run", action="store_true", help="re-score saved outputs only")
    main(run=not parser.parse_args().no_run)
