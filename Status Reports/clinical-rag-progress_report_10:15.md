## Session Update — 2026-10-15

### Context coming into this session

State at the start of the day (last commit `11ca294`):
- Patient snapshot on the one synthetic ICU note: active meds 8/8, labs 23/23, not-given meds **2/3** (dexmedetomidine, "consider... not today", always missed).
- Trial pre-screening on 38 hand-labeled criteria: 79% accuracy, **0 false clears**.
- **Everything was measured on one note.** No unit tests, no CI, no cost or latency numbers, and the README still described the old question-answering stage.

Goal for the day: move from "works on one note" to "tested, measured, and shown to hold up on notes it wasn't built on."

---

### What we did (in order)

| Time | Commit | Change |
|---|---|---|
| 12:54 | `fd503d2` | README rewritten (snapshot + trial pre-screening, "LLM reads, code decides", results, limitations). Added `CLAUDE.md` with hard safety rules and a definition of done. |
| 12:57 | `c1a6d19` | **56 unit tests** for the rule engine: unit conversion, number parsing, boundaries, three-valued logic, and every UNKNOWN safety rule (time windows, missing or mismatched units, disagreeing values, numbers not inside the quote). The model is faked, so there are no API calls. |
| 13:52 | `bb43d62` | **CI**: GitHub Actions runs the tests on every push. Fixed a broken `requirements.txt` (`requestspytest`). |
| 13:56 | `67e7895` | **Cost, latency, and run ID** saved with every run (tokens, $ cost, total and slowest-call seconds, git commit, model). Retries (3) and a 60s timeout on API calls. |
| 14:00 | `e065f16` | **Held-out evaluation setup**: key template, labeling rules, sources and licenses file, `run_heldout.py`. |
| 14:20 | `0da44f2` | First held-out notes. The eval now **counts conflicts** (the same med in two groups). Code checks every not-given status against the note's words. |
| 14:28 | `e668382` | Prompts catch drug classes ("dual antiplatelet therapy") and labs without numbers ("troponin elevated"). Fixed one self-written key (ibuprofen, documented). |
| 15:00 | `cfd0aa7` | **Medication status decided in code** from the words around each med. `run_heldout --repeats N`. 4 held-out notes, stable over 3 runs. 21 more tests (77 total). |
| Evening | (this commit) | README run commands and structure, CLAUDE.md baselines, this report. |

---

### What held-out testing found, and what we changed

Held-out notes: 4 notes from 3 sources, each with an answer key written **before** running the tool.
- 1 Asclepius discharge summary (takotsubo, 75M)
- 2 MTSamples notes (ICU GI bleed + COPD, 68M; respiratory-failure discharge)
- 1 self-written ICU note (GI bleed, 61F, SI units and abbreviations)

| Problem found | Why it mattered | Fix |
|---|---|---|
| Heparin listed as **both** active and discontinued, but the old eval scored it correct | The eval counted any hit as a pass, so contradictions were invisible | Eval now counts **conflicts**; contradicted entries are removed in code |
| Fake "discontinued" statuses (metoprolol) | Telling a nurse a running med was stopped is a dangerous error | Code reads the note's words and sets the status |
| Discharge meds labeled "discontinued" | Wrong meaning of "discharged with" | "discharged with" → active (labeling rule written down) |
| "We will add guaifenesin" came back **active** in one run | Same words, different answer run to run | Status is decided by code word rules, so the same words always give the same status |
| "Received" / "was given" not handled; "once daily" read as a single dose | Edge cases in the word rules | `given_once` rules fixed; covered by tests |
| Home-med lists given a status | A home list doesn't say what's running now | Home-med lists are left undecided and flagged for the nurse |

**Design change:** "LLM reads, code decides" now covers medication status, not just numbers. The model finds each med and its quote; code decides discontinued / held / considered / planned / given once / active. When the words are unclear, the model's status is kept and marked `[STATUS NOT CONFIRMED BY NOTE]`.

---

### Results: start of day vs. end of day

**Original note**

| Group | Start | End |
|---|---|---|
| Active meds | 8/8 | 8/8 |
| Not-given meds | 2/3 | **3/3** (dexmedetomidine found) |
| Labs | 23/23 | 23/23 |
| Conflicts | not measured | **0** |
| Albuterol status | wrong | corrected by code |

**Held-out notes (4 notes, 3 runs, worst to best)**

| Group | Result | Conflicts |
|---|---|---|
| Given meds | 22/22 every run | 0 |
| Not-given meds | 13–14 of 15 | 0 |
| Labs | 48/50 every run | 0 |

Remaining misses (all conservative, meaning under-claiming rather than wrong claims):
- "Avoid NSAIDs" is not always listed.
- Levaquin is ambiguous in the note (active in 2 of 3 runs, flagged in the 3rd).
- Two lab interpretations written in the plan section, far from the value, come back "not labeled."

**Engineering**

| | Start | End |
|---|---|---|
| Unit tests | 0 | **77** (56 rules + 21 med status) |
| CI | none | GitHub Actions, green |
| Cost / latency | unknown | $0.003–0.012 and 25–45s per note (gpt-4o-mini), saved per run |
| API robustness | none | 3 retries, 60s timeout |
| Notes evaluated | 1 | 1 + 4 held-out |
| Run-to-run variance | 3 manual runs | `--repeats N`, worst-to-best table |

Size of the day's change: 23 files, +1,455 / −85 lines (`11ca294..cfd0aa7`).
API spend for the day: about **$0.15–0.20**.

---

### Honest caveats

- The 4 held-out notes **informed fixes**, so they are no longer fully held out. A clean final score needs 2–3 fresh notes never used for fixing.
- The trial matcher has **not yet** been tested on held-out notes.
- Everything is synthetic. No real-world or clinician validation yet.

---

### Next steps

1. **Baseline comparison:** "LLM decides everything" vs. "LLM reads, code decides" on the same notes (the key chart).
2. 2–3 **fresh** MTSamples notes for a clean held-out score.
3. **Held-out matcher test**: false clears must stay at 0.
4. Run extraction calls in parallel (about 40s down to about 10s).
5. FastAPI + a simple web page, then a timed test with a clinician.
6. Start PhysioNet credentialing (MIMIC notes with a local model only; never send them to OpenAI).
7. Optional fixes: "avoid NSAIDs"; linking lab interpretations written far from the value.
