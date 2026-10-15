# Held-out notes

Notes the tool was **not** built or tuned on. They show whether results generalize beyond `Sample_ICU_note.pdf`.

## Rules
1. **Write the answer key before running the tool.** Never look at tool output first.
2. **No code changes between running and scoring.** If something breaks, fix it, then test on a *new* note.
3. **Never edit a key to make the tool pass.** Fix a key only if the key itself was wrong, and say why in the commit message.
4. **Public or synthetic notes only.** No real patient data.

## Layout
- `notes/<name>.pdf`: the note
- `keys/<name>.json`: its answer key (copy `keys/_template.json`)
- `SOURCES.md`: where each note came from and its license

## How to label a key
- `medications.given`: medications the note says were actually given or are running now
- `medications.not_given`: held, discontinued, considered, or planned (not given)
- `labs.abnormal`: labs the note itself calls abnormal (low, high, elevated, "AKI", etc.)
- `labs.not_abnormal`: labs the note calls normal, normalized, or at target
- `labs.not_labeled`: labs with a value but no interpretation in the note
- Use aliases for names the note might use: `"cr|creatinine"`, `"plt|platelets"`

## Run
```bash
python -m evals.run_heldout            # run the tool on every note with a key, then score
python -m evals.run_heldout --no-run   # re-score saved outputs without calling the API
```

## Labeling decisions (apply to every key)
- Discharge medications count as **given** (active at discharge).
- Labs the note calls abnormal without giving a number stay in the key.
- A lab is `abnormal` only if the note's own words say so, even when the value is obviously out of range.
