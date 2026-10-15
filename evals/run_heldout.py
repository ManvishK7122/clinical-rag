"""
Run the patient snapshot on every held-out note that has an answer key, score it,
and print one table across all notes.

Usage:
    python -m evals.run_heldout                # run the tool once, then score
    python -m evals.run_heldout --repeats 3    # run 3 times, report the range and worst run
    python -m evals.run_heldout --no-run       # only re-score saved outputs (no API calls)

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
    """Return {(section, group): report} plus extras for one note."""
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


def run_once(keys, run, verbose):
    """Score every note once. Returns totals {(section, group): [found, total, wrong, conflicts]}."""
    totals, cost, seconds = {}, 0.0, 0.0
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
        cost += r.get("cost_usd", 0)
        seconds += r.get("total_seconds", 0)
        if verbose:
            print(f"\n{name}")
        for (section, group), s in rows.items():
            if verbose:
                print(f"  {section[:4]} {group:<13} {s['found_in_right_group']}/{s['key_items']}"
                      f"  wrong group: {s['wrong_group']}  conflicts: {s['conflicts']}")
                for c in s["conflict_items"]:
                    print(f"      CONFLICT: {c}")
                for w in s["wrong_group_items"]:
                    print(f"      wrong: {w}")
            t = totals.setdefault((section, group), [0, 0, 0, 0])
            t[0] += s["found_in_right_group"]; t[1] += s["key_items"]; t[2] += s["wrong_group"]
            t[3] += s["conflicts"]
        if verbose:
            if extras:
                print(f"  extra (review by hand): {', '.join(extras)}")
            changed = [f"{m['name']} ({m['status_changed_by_code']})" for m in snap["medications"]
                       if m.get("status_changed_by_code")]
            if changed:
                print(f"  status set by code: {', '.join(changed)}")
            flagged = [m["name"] for m in snap["medications"] if not m.get("status_supported", True)]
            if flagged:
                print(f"  status flagged for nurse (not confirmed by note): {', '.join(flagged)}")
            print(f"  unverified quotes: {q.get('unverified_quotes')} of {q.get('items')}"
                  + (f"  |  ${r['cost_usd']:.4f}, {r['total_seconds']}s" if r else ""))
    return totals, cost, seconds


def print_totals(title, totals):
    print(f"\n{title}")
    for (section, group), (found, total, wrong, conflicts) in totals.items():
        recall = f"{found / total:.2f}" if total else "n/a"
        print(f"  {section[:4]} {group:<13} recall {recall}  ({found}/{total})  wrong group: {wrong}"
              f"  conflicts: {conflicts}")


def main(run, repeats):
    keys = sorted(p for p in HELDOUT.glob("keys/*.json") if not p.name.startswith("_"))
    if not keys:
        print("No keys found in evals/heldout/keys/. Copy _template.json to <name>.json first.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repeats = repeats if run else 1

    all_runs, total_cost = [], 0.0
    for i in range(repeats):
        if repeats > 1:
            print(f"\n========== RUN {i + 1} of {repeats} ==========")
        totals, cost, seconds = run_once(keys, run, verbose=True)
        all_runs.append(totals)
        total_cost += cost
        print_totals(f"ALL HELD-OUT NOTES ({len(keys)})" + (f", run {i + 1}" if repeats > 1 else ""), totals)
        if cost:
            print(f"  cost ${cost:.4f}, {seconds:.0f}s")

    if repeats > 1:
        print(f"\nACROSS {repeats} RUNS (worst to best)")
        for group_key in all_runs[0]:
            found = [r[group_key][0] for r in all_runs]
            total = all_runs[0][group_key][1]
            conflicts = max(r[group_key][3] for r in all_runs)
            wrong = max(r[group_key][2] for r in all_runs)
            section, group = group_key
            print(f"  {section[:4]} {group:<13} {min(found)}-{max(found)} of {total}"
                  f"  worst wrong group: {wrong}  worst conflicts: {conflicts}")
        print(f"  total cost ${total_cost:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and score held-out notes.")
    parser.add_argument("--no-run", action="store_true", help="re-score saved outputs only")
    parser.add_argument("--repeats", type=int, default=1, help="run each note this many times")
    args = parser.parse_args()
    main(run=not args.no_run, repeats=args.repeats)
