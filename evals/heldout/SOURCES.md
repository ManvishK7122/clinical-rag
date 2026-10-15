# Sources and licenses

| Note | Source | License | Notes |
|---|---|---|---|
| `asclepius_takotsubo_75m` | [Asclepius Synthetic Clinical Notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes) | CC BY-NC-SA 4.0 | Used unchanged. Discharge summary; sparse labs, drug classes, implausible platelet values left as-is. |
| `selfwritten_gibleed_61f` | Self-written synthetic note | Project-owned | Answer key written first, note written to match. Designed to test SI units, abbreviations, held/stopped/considered meds, values that change across days. Not an independent test. |
| `mtsamples_melena_icu_68m` | [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), "Melena - ICU Followup" | CC0 (Kaggle mirror) | Real-style transcription, independent. Line breaks restored (the CSV removed them); words unchanged. |
| `mtsamples_respfailure_dc` | [MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), "Discharge Summary - Respiratory Failure" | CC0 (Kaggle mirror) | Real-style transcription, independent. Line breaks restored; words unchanged. |

- MTSamples (Kaggle mirror): listed as CC0 public domain.
- Asclepius: non-commercial use only, attribution required. Kweon et al., "Publicly Shareable Clinical Large Language Model Built on Synthetic Clinical Notes" (arXiv:2309.00237).
