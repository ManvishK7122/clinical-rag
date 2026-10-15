# CLAUDE.md: clinical-rag

## What this project is
A tool that turns an ICU clinical note (PDF) into a verified patient snapshot for nurses and pre-screens the patient against clinical trial eligibility criteria. Every fact must be cited to a page and an exact quote.

## Core principle: the LLM reads, code decides
- The LLM only extracts: facts, exact quotes, values as written, and structured rules parsed from criteria.
- Code makes every decision: quote verification, unit conversion, numeric comparison, combining rules (TRUE / FALSE / UNKNOWN).
- Never move a decision into a prompt. If a change would let the model decide, stop and flag it.

## Hard rules (never break)
1. **Synthetic data only.** Never send real patient data or MIMIC data to OpenAI or any outside API.
2. **Uncertain means UNKNOWN.** Missing values, unverified quotes, unit mismatches, conflicting values, and time windows all return UNKNOWN for the nurse.
3. **The tool never says "eligible."** Output is cleared / blocker / check.
4. **Zero false clears.** Any change that adds a false clear is a regression, even if accuracy goes up.
5. **Never commit** `outputs/`, `data/`, `.env`, or personal PDFs (see `.gitignore`).
6. Don't edit answer keys in `evals/` to make a result pass. If a label looks wrong, show it to the user and let them decide.

## Layout
- `screening/snapshot.py`: snapshot extraction (one call per section), second pass for missed meds, `verify_evidence`, lab labeling
- `screening/rules.py`: numeric rule engine (parse criterion, find values, verify, convert units, compare, three-valued combine)
- `screening/matcher.py`: code rules first (time windows, pregnancy and sex), then numeric or fact checks
- `screening/trials.py`: fetch and freeze trials from ClinicalTrials.gov
- `evals/`: answer keys, `label_criteria.py`, `eval_snapshot.py`, `eval_matcher.py`, saved results
- `query.py`, `eval.py`, `prompts.py`: earlier question-answering stage

## How to run (from the repo root, with the venv active)
```bash
python -m screening.snapshot
python -m evals.eval_snapshot
python -m screening.matcher --trials septic_icu --only NCT06849570 NCT07383103
python -m evals.eval_matcher outputs/match_septic_icu_Sample_ICU_note.json evals/criteria_key_septic_icu.json
```

## Definition of done for any change
1. Run both evals and compare with the current baseline:
   - Snapshot: active meds 8/8, labs 23/23, not-given meds 2/3, wrong group 0, unverified quotes about 2–3 of 56
   - Matcher (38 labeled criteria): accuracy 79%, **false clears 0**, false blocks 0, missed info 8
2. If a number moves unexpectedly, run 3 times to separate a real change from run-to-run noise.
3. Show a before/after table. Read the raw outputs for any case that changed.
4. Keep commits small with clear messages.

## How to work with me
- Plan first for anything bigger than a small fix, and wait for approval.
- Keep explanations short and in simple wording.
- When asked for code, give full files rather than fragments.
- Don't add dependencies without asking.
