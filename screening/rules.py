"""
Hard number checks for trial criteria.

The model only reads text. Code makes every numeric decision:
1. The model turns a criterion into rules, e.g. "Platelets < 50,000/mm3"
   -> {"variable": "platelets", "operator": "<", "value": 50000, "unit": "/mm3"}
2. The model copies every value of that variable from the note, exactly as
   written, with a quote.
3. Code checks the number is really in the quote, converts units, and compares.

Anything uncertain returns UNKNOWN so the nurse decides: missing or
unconvertible units, values that disagree, unverified quotes, or time windows.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

from screening.snapshot import Evidence, ask_json, verify_evidence, _squash


class Rule(BaseModel):
    variable: str
    operator: Literal[">", ">=", "<", "<=", "=="]
    value: float
    unit: Optional[str] = None
    time_window: Optional[str] = None


class ParsedCriterion(BaseModel):
    kind: Literal["numeric", "fact", "not_checkable"]
    combine: Literal["all", "any"] = "all"
    rules: list[Rule] = []


class FoundValue(BaseModel):
    value_text: str
    unit: Optional[str] = None
    page: int
    quote: str


UNIT_ALIASES = {
    "k/ul": "k/ul", "x10^3/ul": "k/ul", "10^3/ul": "k/ul", "x10*3/ul": "k/ul",
    "thou/ul": "k/ul", "x10^9/l": "k/ul", "10^9/l": "k/ul", "k/mm3": "k/ul",
    "/ul": "/ul", "/mm3": "/ul", "cells/ul": "/ul", "/mcl": "/ul",
    "mg/dl": "mg/dl", "umol/l": "umol/l", "mmol/l": "mmol/l", "meq/l": "meq/l",
    "g/dl": "g/dl", "g/l": "g/l", "mmhg": "mmhg", "%": "%",
    "ng/ml": "ng/ml", "mcg/l": "ng/ml", "ug/l": "ng/ml",
    "kg": "kg", "years": "years", "year": "years",
}

CONVERSIONS = {
    ("/ul", "k/ul"): 0.001, ("k/ul", "/ul"): 1000,
    ("g/l", "g/dl"): 0.1, ("g/dl", "g/l"): 10,
}

VARIABLE_CONVERSIONS = {
    "creatinine": {("umol/l", "mg/dl"): 1 / 88.4, ("mg/dl", "umol/l"): 88.4},
    "bilirubin": {("umol/l", "mg/dl"): 1 / 17.1, ("mg/dl", "umol/l"): 17.1},
}


def normalize_unit(unit):
    if not unit:
        return None
    u = unit.strip().lower().replace("µ", "u").replace("μ", "u").replace("³", "3")
    u = u.replace(" ", "").replace("mcl", "ul")
    return UNIT_ALIASES.get(u, u)


def convert(value, from_unit, to_unit, variable=""):
    """Return the value in to_unit, or None if it can't be converted safely."""
    a, b = normalize_unit(from_unit), normalize_unit(to_unit)
    if a is None or b is None:
        return None
    if a == b:
        return value
    if (a, b) in CONVERSIONS:
        return value * CONVERSIONS[(a, b)]
    for name, table in VARIABLE_CONVERSIONS.items():
        if name in variable.lower() and (a, b) in table:
            return value * table[(a, b)]
    return None


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_number(text):
    """'45,000' -> 45000.0, '2.1' -> 2.1. None if there isn't exactly one number."""
    found = NUMBER_RE.findall(text or "")
    if len(found) != 1:
        return None
    try:
        return float(found[0].replace(",", ""))
    except ValueError:
        return None


def compare(a, operator, b):
    return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b, "==": a == b}[operator]


PARSE_PROMPT = """Turn this clinical trial criterion into rules a program can check.
Do not decide anything about a patient. Only describe the criterion.

- "numeric": the criterion is a threshold on measured values (labs, vitals, age,
  weight, scores). Write one rule per threshold. Keep the exact operator: "at
  least" is >=, "greater than" is >, "less than" is <, "no more than" is <=.
  Copy the unit as written, or null if none is given. Put any time limit
  ("within 24 hours", "at screening") in time_window, otherwise null.
  combine is "all" if every rule must hold, "any" if one is enough.
- "fact": a yes/no fact a clinical note could document (a diagnosis, a
  medication, a device, pregnancy, a past event).
- "not_checkable": needs consent, willingness, investigator judgment, or
  anything a clinical note cannot answer.

Criterion: {criterion}

Return JSON: {{"kind": "numeric" or "fact" or "not_checkable", "combine": "all" or "any",
"rules": [{{"variable": "...", "operator": ">" or ">=" or "<" or "<=" or "==",
"value": <number>, "unit": "..." or null, "time_window": "..." or null}}]}}"""

FIND_PROMPT = """Find every value of "{variable}" recorded for this patient in the note.
Copy each value exactly as written (value_text), its unit as written (or null if
the note gives none), the page, and an exact quote of 3 to 25 words that
contains the value. Do not compare, judge, round, or convert anything.
If the note has no value for it, return an empty list.

Return JSON: {{"values": [{{"value_text": "...", "unit": "..." or null,
"page": <number>, "quote": "..."}}]}}"""


def parse_criterion(text):
    raw = ask_json(PARSE_PROMPT.format(criterion=text), "(no note needed for this step)")
    try:
        return ParsedCriterion.model_validate(raw)
    except ValidationError:
        return None


def find_values(variable, note_text, pages):
    """Return verified values only. Each must have a quote found in the note
    AND its number must appear inside that quote."""
    raw = ask_json(FIND_PROMPT.format(variable=variable), note_text)
    found = []
    for entry in raw.get("values", []):
        try:
            v = FoundValue.model_validate(entry)
        except ValidationError:
            continue
        evidence = verify_evidence(Evidence(page=v.page, quote=v.quote), pages)
        number = parse_number(v.value_text)
        if not evidence.verified or number is None:
            continue
        if _squash(v.value_text) not in _squash(v.quote):
            continue
        found.append({"number": number, "value_text": v.value_text, "unit": v.unit,
                      "page": evidence.page, "quote": v.quote})
    return found


def check_rule(rule, values):
    """Return (decision, reason, value_used) for one rule against the note's values."""
    label = f"{rule.variable} {rule.operator} {rule.value:g}{' ' + rule.unit if rule.unit else ''}"
    if rule.time_window:
        return "UNKNOWN", f"Rule '{label}' has a time limit ({rule.time_window}); check timing in the chart.", None
    if not values:
        return "UNKNOWN", f"No verified value for {rule.variable} in the note.", None

    results = []
    for v in values:
        if v["unit"] is None and "age" in rule.variable.lower():
            v = {**v, "unit": "years"}
        if rule.unit is None and v["unit"] is None:
            converted = v["number"]
        else:
            converted = convert(v["number"], v["unit"], rule.unit, rule.variable)
        if converted is None:
            return "UNKNOWN", (f"Can't safely compare {v['value_text']} {v['unit'] or '(no unit)'} "
                               f"with rule '{label}'. Check units."), v
        results.append((compare(converted, rule.operator, rule.value), v))

    outcomes = {r for r, _ in results}
    if len(outcomes) > 1:
        shown = ", ".join(f"{v['value_text']} (p.{v['page']})" for _, v in results)
        return "UNKNOWN", f"Values disagree on '{label}': {shown}. Pick the right one.", None
    outcome, v = results[0]
    return ("TRUE" if outcome else "FALSE"), f"{v['value_text']} {v['unit'] or ''} vs '{label}'".strip(), v


def combine(decisions, how):
    """Three-valued logic. 'all' = AND, 'any' = OR, with UNKNOWN in between."""
    if how == "all":
        if "FALSE" in decisions:
            return "FALSE"
        return "TRUE" if all(d == "TRUE" for d in decisions) else "UNKNOWN"
    if "TRUE" in decisions:
        return "TRUE"
    return "FALSE" if all(d == "FALSE" for d in decisions) else "UNKNOWN"


def check_numeric(parsed, note_text, pages):
    """Return a result dict for a numeric criterion, or None if it isn't numeric."""
    if parsed is None or parsed.kind != "numeric" or not parsed.rules:
        return None
    details, decisions = [], []
    for rule in parsed.rules:
        decision, reason, v = check_rule(rule, find_values(rule.variable, note_text, pages))
        decisions.append(decision)
        details.append({"rule": rule.model_dump(), "decision": decision, "reason": reason,
                        "page": v["page"] if v else None, "quote": v["quote"] if v else None})
    final = combine(decisions, parsed.combine)
    first = next((d for d in details if d["decision"] == final and d["quote"]), {})
    return {"decision": final, "reason": "; ".join(d["reason"] for d in details),
            "page": first.get("page"), "quote": first.get("quote"),
            "verified": final != "UNKNOWN", "rules": details}