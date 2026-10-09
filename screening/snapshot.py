"""
Patient snapshot: structured facts from a clinical note, each with a page
number and an exact quote that is checked against the PDF text.

Design choices:
- The whole note goes to the model. Notes are a few pages, so nothing depends
  on retrieval picking the right chunks.
- One extraction call per section, so the model gathers one kind of item at a
  time instead of building every list at once.
- Every item carries a quote. Code checks that the quote really appears in the
  note. Items whose quote can't be found are flagged as unverified.
- Lab labels and target-range checks are decided in code, not by the model.
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator
from pypdf import PdfReader

from config import DATA_DIR, DEFAULT_FILENAME, EXTRACT_MODEL, OUTPUT_DIR

load_dotenv()
client = OpenAI()


class Evidence(BaseModel):
    page: int
    quote: str
    verified: bool = False


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    status: Literal["active", "given_once", "held", "discontinued", "considered", "planned"]
    evidence: Evidence


class Lab(BaseModel):
    name: str
    value: str
    note_label: Optional[str] = None
    target_range: Optional[str] = None
    interpretation: Literal["abnormal", "not_abnormal", "not_labeled"] = "not_labeled"
    evidence: Evidence

    @field_validator("value", "note_label", "target_range", mode="before")
    @classmethod
    def to_text(cls, v):
        """The model sometimes returns numbers or lists; store everything as text."""
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return "-".join(str(x) for x in v)
        if isinstance(v, dict):
            return "-".join(str(x) for x in v.values())
        return str(v)

    @field_validator("interpretation", mode="before")
    @classmethod
    def any_interpretation(cls, v):
        """Code decides the interpretation, so never reject a lab over this field."""
        return v if v in ("abnormal", "not_abnormal", "not_labeled") else "not_labeled"


class Item(BaseModel):
    name: str
    evidence: Evidence


class PatientInfo(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    evidence: Optional[Evidence] = None


BASE_RULES = """You extract facts from ONE patient's clinical note for a nurse.
Use ONLY the note. Never use outside medical knowledge or assumptions.
Every item needs evidence: the page number and an EXACT quote copied word for
word from that page (3 to 25 words) that supports it.
If nothing is documented, return an empty list. Return JSON only."""

EVIDENCE_SHAPE = '"evidence": {"page": <number>, "quote": "<exact words from the note>"}'

SECTIONS = {
    "medications": (Medication, f"""List EVERY medication mentioned anywhere in the note. Medications often
appear outside a medications heading: hemodynamics or vasopressors, sedation,
nutrition, infection, hematology, checklists, and plans. Check every page.

Set status using only the note's words:
- active: currently being given (infusions, drips, scheduled medications,
  "on X", "continue X", "weaning X")
- given_once: a single dose was given (for example, a bolus)
- held: paused. Note: "held at [dose]" means continued at that dose, so use active.
- discontinued: stopped or replaced
- considered: discussed or considered, but not started
- planned: ordered or planned for later
The same medication can appear twice with different statuses (for example, an
infusion and a separate bolus).
Include medications that were stopped, switched, held, considered, or decided
against. A nurse needs these as much as the active ones.

Return {{"items": [{{"name": "...", "dose": "..." or null, "status": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "labs": (Lab, f"""List EVERY lab value in the note, including basic chemistry (sodium,
potassium, chloride, bicarbonate), blood counts, coagulation, liver tests, and
inflammatory markers. Check every page.

For each value, copy into note_label the note's OWN interpretive words for it,
exactly as written (for example "low", "elevated", "severe AKI", "anemia",
"normalized", "therapeutic", "improving"). If the note puts several values under
one label (for example "mild transaminitis"), copy that label for each of them.
If the note gives no words about the value, note_label is null. Never write your
own judgment (such as "abnormal" or "normal") unless those exact words are in the note.
If the note gives a target or goal range for a value (for example "target
140-180"), copy it into target_range, otherwise null.

Return {{"items": [{{"name": "...", "value": "...", "note_label": "..." or null, "target_range": "..." or null, {EVIDENCE_SHAPE}}}]}}"""),

    "medication_changes": (Medication, f"""List ONLY medications that are NOT currently being given: stopped, switched
or de-escalated from, held, considered or suggested but not started, or planned
for later. Look for words like "de-escalated from", "switched from",
"discontinued", "stopped", "held", "consider", "if needed", and "avoid".
If one sentence names several medications, list each one separately.
Status must be one of: discontinued, held, considered, planned.

Return {{"items": [{{"name": "...", "dose": "..." or null, "status": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "diagnoses": (Item, f"""List the patient's diagnoses and active problems as the note states them.
Use the clinical diagnosis or assessment fields, never billing or ICD codes.
Return {{"items": [{{"name": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "devices": (Item, f"""List the lines, tubes, drains, and devices currently in place (for example
endotracheal tube, central lines, arterial lines, dialysis catheters, urinary
catheters, feeding tubes, compression devices).
Return {{"items": [{{"name": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "plans": (Item, f"""List pending results and planned actions, including tests, consults,
procedures, medication changes, and family communication.
Return {{"items": [{{"name": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "code_status": (Item, f"""Give the patient's code status exactly as documented. The name must be the
status itself (for example "Full Code" or "DNR/DNI"), not the heading "Code Status".
Return an empty list if it is not documented.
Return {{"items": [{{"name": "...", {EVIDENCE_SHAPE}}}]}}"""),

    "allergies": (Item, f"""List the patient's documented allergies. If the note says no known
allergies, return one item named "No known allergies". If allergies are not
mentioned at all, return an empty list.
Return {{"items": [{{"name": "...", {EVIDENCE_SHAPE}}}]}}"""),
}

MISSED_MEDS_PROMPT = f"""These medications were already found in the note:
FOUND_LIST

Read the WHOLE note again, every page and every section (including glucose or
insulin management, sedation and delirium plans, nutrition, prophylaxis, and
checklists). List ONLY medications that are mentioned in the note but are
MISSING from the list above. Use the same status rules: active, given_once,
held, discontinued, considered, planned. If nothing is missing, return an empty list.

Return {{"items": [{{"name": "...", "dose": "..." or null, "status": "...", {EVIDENCE_SHAPE}}}]}}"""

PATIENT_PROMPT = f"""Give the patient's age in years and sex, as stated in the note.
Return {{"age": <number or null>, "sex": "..." or null, {EVIDENCE_SHAPE} or null}}"""


def load_pages(filepath):
    """Return {page_number: text} for every page that has text."""
    reader = PdfReader(filepath)
    pages = {}
    for num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages[num] = text
    if not pages:
        raise ValueError("No text found in the PDF. It may be a scanned image that needs OCR.")
    return pages


def note_with_page_markers(pages):
    return "\n\n".join(f"[Page {num}]\n{text}" for num, text in pages.items())


_DASHES = str.maketrans({"–": "-", "—": "-", "’": "'", "‘": "'",
                         "“": '"', "”": '"'})


def _squash(text):
    """Lowercase and remove all whitespace, so PDF spacing quirks don't matter."""
    return re.sub(r"\s+", "", text.translate(_DASHES)).lower()


def _letters(text):
    """Lowercase letters and digits only. Ignores punctuation the PDF drops,
    such as a colon after a table heading."""
    return re.sub(r"[^a-z0-9]", "", (text or "").translate(_DASHES).lower())


def _on_page(quote, page_text):
    """Exact match ignoring whitespace, or a letters-only match for longer quotes."""
    if _squash(quote) in _squash(page_text):
        return True
    letters = _letters(quote)
    return len(letters) >= 12 and letters in _letters(page_text)


def verify_evidence(evidence, pages):
    """Mark the quote verified if it appears on the cited page. If it appears
    on a different page instead, correct the page number."""
    quote = evidence.quote or ""
    if len(_squash(quote)) < 6:
        evidence.verified = False
        return evidence
    if _on_page(quote, pages.get(evidence.page, "")):
        evidence.verified = True
        return evidence
    for num, text in pages.items():
        if _on_page(quote, text):
            evidence.page = num
            evidence.verified = True
            return evidence
    evidence.verified = False
    return evidence


def ask_json(instructions, note_text):
    response = client.chat.completions.create(
        model=EXTRACT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": BASE_RULES},
            {"role": "user", "content": f"{instructions}\n\nNOTE:\n{note_text}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


ABNORMAL_WORDS = {"low", "high", "elevated", "severe", "severely", "critical", "abnormal",
                  "decreased", "increased", "aki", "transaminitis", "anemia", "thrombocytopenia",
                  "leukocytosis", "leukopenia", "acidosis", "alkalosis", "coagulopathy"}
ABNORMAL_SUFFIXES = ("emia", "penia", "itis", "osis", "uria")
NORMAL_WORDS = {"normal", "normalized", "normalizing", "therapeutic", "wnl", "unremarkable"}
NORMAL_PHRASES = ("within normal", "within target", "at target", "in range", "within range")


def classify_label(note_label):
    """Decide lab interpretation from the note's own words, in code.
    Trends like 'improving' or 'down from X' are not_labeled."""
    if not note_label:
        return "not_labeled"
    text = note_label.lower()
    words = set(re.findall(r"[a-z]+", text))
    if words & ABNORMAL_WORDS or any(w.endswith(ABNORMAL_SUFFIXES) for w in words):
        return "abnormal"
    if words & NORMAL_WORDS or any(p in text for p in NORMAL_PHRASES):
        return "not_abnormal"
    return "not_labeled"


def _numbers(text):
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text or "")]


def classify_lab(lab):
    """The note's words decide first. If there are none but the note gives a
    target range, code compares the value to that range."""
    result = classify_label(lab.note_label)
    if result != "not_labeled":
        return result
    value, target = _numbers(lab.value), _numbers(lab.target_range)
    if len(value) == 1 and len(target) == 2:
        low, high = sorted(target)
        return "not_abnormal" if low <= value[0] <= high else "abnormal"
    return "not_labeled"


def extract_section(name, note_text, pages):
    model, instructions = SECTIONS[name]
    raw = ask_json(instructions, note_text)
    items, skipped = [], 0
    for entry in raw.get("items", []):
        try:
            item = model.model_validate(entry)
        except ValidationError:
            skipped += 1
            continue
        verify_evidence(item.evidence, pages)
        if name == "labs":
            page_text = pages.get(item.evidence.page, "")
            if item.note_label and _letters(item.note_label) not in _letters(page_text):
                item.note_label = None
            item.interpretation = classify_lab(item)
        items.append(item)
    return items, skipped


def find_missed_meds(meds, note_text, pages):
    """Second pass: show the model what it already found and ask only for what's missing.
    Anything already listed under the same name is dropped."""
    found = "\n".join(f"- {m['name']} ({m['status']})" for m in meds) or "(none)"
    raw = ask_json(MISSED_MEDS_PROMPT.replace("FOUND_LIST", found), note_text)
    known = {m["name"].lower() for m in meds}
    added, skipped = [], 0
    for entry in raw.get("items", []):
        try:
            med = Medication.model_validate(entry)
        except ValidationError:
            skipped += 1
            continue
        if med.name.lower() in known:
            continue
        verify_evidence(med.evidence, pages)
        added.append(med.model_dump())
        known.add(med.name.lower())
    return added, skipped

def extract_patient(note_text, pages):
    raw = ask_json(PATIENT_PROMPT, note_text)
    try:
        patient = PatientInfo.model_validate(raw)
    except ValidationError:
        return PatientInfo()
    if patient.evidence:
        verify_evidence(patient.evidence, pages)
    return patient


def build_snapshot(filepath):
    print(f"Reading {filepath}...")
    pages = load_pages(filepath)
    note_text = note_with_page_markers(pages)
    print(f"  {len(pages)} pages loaded. Using {EXTRACT_MODEL}.")

    print("  Extracting patient info...", flush=True)
    snapshot = {"source": str(filepath), "model": EXTRACT_MODEL,
                "patient": extract_patient(note_text, pages).model_dump()}
    skipped_total = 0
    for name in SECTIONS:
        print(f"  Extracting {name}...", flush=True)
        items, skipped = extract_section(name, note_text, pages)
        snapshot[name] = [item.model_dump() for item in items]
        skipped_total += skipped
        print(f"    {len(items)} found")

    changes = snapshot.pop("medication_changes", [])
    have = {(m["name"].lower(), m["status"]) for m in snapshot["medications"]}
    snapshot["medications"] += [m for m in changes if (m["name"].lower(), m["status"]) not in have]

    print("  Second pass for missed medications...", flush=True)
    missed, skipped = find_missed_meds(snapshot["medications"], note_text, pages)
    snapshot["medications"] += missed
    skipped_total += skipped
    print(f"    {len(missed)} added")
    
    all_evidence = [i["evidence"] for name in SECTIONS if name in snapshot for i in snapshot[name]]
    snapshot["quality"] = {
        "items": len(all_evidence),
        "unverified_quotes": sum(not e["verified"] for e in all_evidence),
        "malformed_items_skipped": skipped_total,
    }
    return snapshot


def _cite(evidence):
    flag = "" if evidence["verified"] else "  [UNVERIFIED QUOTE]"
    return f"(p.{evidence['page']}){flag}"


def print_snapshot(s):
    p = s["patient"]
    print(f"\nPATIENT: age {p.get('age')}, sex {p.get('sex')}")
    print("\nMEDICATIONS")
    for status in ["active", "given_once", "held", "discontinued", "considered", "planned"]:
        meds = [m for m in s["medications"] if m["status"] == status]
        if meds:
            print(f"  {status.replace('_', ' ').title()}:")
            for m in meds:
                dose = f" {m['dose']}" if m.get("dose") else ""
                print(f"    - {m['name']}{dose} {_cite(m['evidence'])}")
    print("\nLABS")
    for interp in ["abnormal", "not_abnormal", "not_labeled"]:
        labs = [l for l in s["labs"] if l["interpretation"] == interp]
        if labs:
            print(f"  {interp.replace('_', ' ').title()}:")
            for l in labs:
                label = f" ({l['note_label']})" if l.get("note_label") else ""
                print(f"    - {l['name']} {l['value']}{label} {_cite(l['evidence'])}")
    for section in ["diagnoses", "devices", "plans", "code_status", "allergies"]:
        print(f"\n{section.replace('_', ' ').upper()}")
        if not s[section]:
            print("  - Not documented")
        for item in s[section]:
            print(f"  - {item['name']} {_cite(item['evidence'])}")
    q = s["quality"]
    print(f"\nQuality: {q['items']} items, {q['unverified_quotes']} unverified quotes, "
          f"{q['malformed_items_skipped']} malformed items skipped")


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DATA_DIR) / DEFAULT_FILENAME
    snap = build_snapshot(pdf)
    print_snapshot(snap)
    out = Path(OUTPUT_DIR) / f"snapshot_{pdf.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2))
    print(f"\nSaved to {out}")