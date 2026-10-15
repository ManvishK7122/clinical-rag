"""
Unit tests for the rule engine (screening/rules.py).

These test the code that makes decisions, so no API calls are made.
Anything that would call the model is replaced with a fake.

Run from the repo root:
    python -m pytest -q
"""

from types import SimpleNamespace

import pytest

from screening import rules
from screening.rules import (
    ParsedCriterion,
    Rule,
    check_numeric,
    check_rule,
    combine,
    compare,
    convert,
    normalize_unit,
    parse_number,
)


def value(number, unit, text=None, page=1, quote=None):
    """A verified value as find_values would return it."""
    text = text or f"{number:g}"
    return {"number": number, "value_text": text, "unit": unit,
            "page": page, "quote": quote or f"value {text} {unit or ''}".strip()}


# ---------- Units ----------

@pytest.mark.parametrize("raw, expected", [
    ("x10^3/µL", "k/ul"),
    ("K/uL", "k/ul"),
    ("/mm³", "/ul"),
    ("cells/uL", "/ul"),
    ("mg/dL", "mg/dl"),
    ("mEq/L", "meq/l"),
    (" mmHg ", "mmhg"),
    ("year", "years"),
])
def test_normalize_unit_aliases(raw, expected):
    assert normalize_unit(raw) == expected


def test_normalize_unit_missing_is_none():
    assert normalize_unit(None) is None
    assert normalize_unit("") is None


def test_convert_same_unit_is_unchanged():
    assert convert(2.1, "mg/dL", "mg/dl") == 2.1


def test_convert_cells_per_ul_to_thousands():
    assert convert(45000, "/uL", "x10^3/uL") == pytest.approx(45)
    assert convert(45, "K/uL", "/mm3") == pytest.approx(45000)


def test_convert_hemoglobin_g_per_l_to_g_per_dl():
    assert convert(120, "g/L", "g/dL") == pytest.approx(12)


def test_convert_creatinine_umol_to_mg():
    assert convert(176.8, "umol/L", "mg/dL", "creatinine") == pytest.approx(2.0)


def test_convert_bilirubin_mg_to_umol():
    assert convert(1.0, "mg/dL", "umol/L", "total bilirubin") == pytest.approx(17.1)


def test_lab_specific_conversion_not_used_for_other_labs():
    # The creatinine factor must never be applied to glucose.
    assert convert(5.0, "umol/L", "mg/dL", "glucose") is None


def test_convert_unknown_pair_is_none():
    assert convert(7.0, "mmHg", "mg/dL") is None


def test_convert_missing_unit_is_none():
    assert convert(45, None, "k/ul") is None
    assert convert(45, "k/ul", None) is None


# ---------- Numbers ----------

@pytest.mark.parametrize("text, expected", [
    ("45,000", 45000.0),
    ("2.1", 2.1),
    ("117", 117.0),
    ("Platelets 45", 45.0),
])
def test_parse_number_single_value(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["", None, "normal", "1.2-1.5", "120/80"])
def test_parse_number_rejects_zero_or_multiple_numbers(text):
    assert parse_number(text) is None


@pytest.mark.parametrize("a, op, b, expected", [
    (50, "<", 50, False),
    (50, "<=", 50, True),
    (50, ">", 50, False),
    (50, ">=", 50, True),
    (50, "==", 50, True),
    (49.9, "<", 50, True),
])
def test_compare_boundaries(a, op, b, expected):
    assert compare(a, op, b) is expected


# ---------- Three-valued logic ----------

@pytest.mark.parametrize("decisions, expected", [
    (["TRUE", "TRUE"], "TRUE"),
    (["TRUE", "UNKNOWN"], "UNKNOWN"),
    (["TRUE", "FALSE"], "FALSE"),
    (["UNKNOWN", "FALSE"], "FALSE"),   # one FALSE settles an AND
])
def test_combine_all(decisions, expected):
    assert combine(decisions, "all") == expected


@pytest.mark.parametrize("decisions, expected", [
    (["FALSE", "FALSE"], "FALSE"),
    (["FALSE", "UNKNOWN"], "UNKNOWN"),
    (["FALSE", "TRUE"], "TRUE"),
    (["UNKNOWN", "TRUE"], "TRUE"),     # one TRUE settles an OR
])
def test_combine_any(decisions, expected):
    assert combine(decisions, "any") == expected


# ---------- check_rule: deciding one rule ----------

PLATELETS_UNDER_50K = Rule(variable="platelets", operator="<", value=50000, unit="/mm3")


def test_rule_true_after_unit_conversion():
    decision, _, used = check_rule(PLATELETS_UNDER_50K, [value(45, "K/uL")])
    assert decision == "TRUE"
    assert used["number"] == 45


def test_rule_false_after_unit_conversion():
    decision, _, _ = check_rule(PLATELETS_UNDER_50K, [value(180, "K/uL")])
    assert decision == "FALSE"


def test_rule_with_time_window_is_unknown():
    rule = Rule(variable="lactate", operator=">", value=2, unit="mmol/L",
                time_window="within 24 hours")
    decision, reason, _ = check_rule(rule, [value(5.2, "mmol/L")])
    assert decision == "UNKNOWN"
    assert "time" in reason.lower()


def test_rule_without_values_is_unknown():
    decision, _, _ = check_rule(PLATELETS_UNDER_50K, [])
    assert decision == "UNKNOWN"


def test_rule_with_unconvertible_unit_is_unknown():
    decision, reason, _ = check_rule(PLATELETS_UNDER_50K, [value(45, "mg/dL")])
    assert decision == "UNKNOWN"
    assert "unit" in reason.lower()


def test_value_unit_missing_but_rule_has_unit_is_unknown():
    # Never assume what unit an unlabeled number is in.
    decision, _, _ = check_rule(PLATELETS_UNDER_50K, [value(45, None)])
    assert decision == "UNKNOWN"


def test_rule_unit_missing_but_value_has_unit_is_unknown():
    rule = Rule(variable="platelets", operator="<", value=50, unit=None)
    decision, _, _ = check_rule(rule, [value(45, "K/uL")])
    assert decision == "UNKNOWN"


def test_both_units_missing_compares_raw_numbers():
    rule = Rule(variable="SOFA score", operator=">=", value=2, unit=None)
    decision, _, _ = check_rule(rule, [value(9, None)])
    assert decision == "TRUE"


def test_disagreeing_values_are_unknown():
    # Two platelet counts on different pages that land on opposite sides.
    values = [value(45, "K/uL", page=1), value(160, "K/uL", page=3)]
    decision, reason, _ = check_rule(PLATELETS_UNDER_50K, values)
    assert decision == "UNKNOWN"
    assert "disagree" in reason.lower()


def test_agreeing_values_are_decided():
    values = [value(45, "K/uL", page=1), value(38, "K/uL", page=3)]
    decision, _, _ = check_rule(PLATELETS_UNDER_50K, values)
    assert decision == "TRUE"


# ---------- check_numeric: whole criterion, model calls faked ----------

def fake_find_values(table):
    """Replace find_values so no API call happens. table: variable -> values."""
    return lambda variable, note_text, pages: table.get(variable, [])


def test_check_numeric_ignores_non_numeric_criteria():
    assert check_numeric(None, "note", []) is None
    assert check_numeric(ParsedCriterion(kind="fact"), "note", []) is None
    assert check_numeric(ParsedCriterion(kind="numeric", rules=[]), "note", []) is None


def test_check_numeric_all_rules_true(monkeypatch):
    monkeypatch.setattr(rules, "find_values", fake_find_values({
        "platelets": [value(45, "K/uL")],
        "creatinine": [value(2.4, "mg/dL")],
    }))
    parsed = ParsedCriterion(kind="numeric", combine="all", rules=[
        PLATELETS_UNDER_50K,
        Rule(variable="creatinine", operator=">", value=2.0, unit="mg/dL"),
    ])
    result = check_numeric(parsed, "note", [])
    assert result["decision"] == "TRUE"
    assert result["verified"] is True


def test_check_numeric_one_missing_value_makes_all_unknown(monkeypatch):
    monkeypatch.setattr(rules, "find_values", fake_find_values({
        "platelets": [value(45, "K/uL")],
    }))
    parsed = ParsedCriterion(kind="numeric", combine="all", rules=[
        PLATELETS_UNDER_50K,
        Rule(variable="bilirubin", operator=">", value=2.0, unit="mg/dL"),
    ])
    result = check_numeric(parsed, "note", [])
    assert result["decision"] == "UNKNOWN"
    assert result["verified"] is False


def test_check_numeric_any_with_one_true(monkeypatch):
    monkeypatch.setattr(rules, "find_values", fake_find_values({
        "lactate": [value(5.2, "mmol/L")],
    }))
    parsed = ParsedCriterion(kind="numeric", combine="any", rules=[
        Rule(variable="lactate", operator=">", value=2, unit="mmol/L"),
        Rule(variable="MAP", operator="<", value=65, unit="mmHg"),
    ])
    assert check_numeric(parsed, "note", [])["decision"] == "TRUE"


# ---------- find_values: the "number must be in the quote" guard ----------

def test_find_values_drops_numbers_not_in_quote(monkeypatch):
    monkeypatch.setattr(rules, "ask_json", lambda prompt, note: {"values": [
        # Quote checks out and contains the value: keep.
        {"value_text": "45", "unit": "K/uL", "page": 2, "quote": "Platelets 45 K/uL this morning"},
        # Quote is real, but the value isn't in it: drop.
        {"value_text": "30", "unit": "K/uL", "page": 2, "quote": "Platelets 45 K/uL this morning"},
        # More than one number in the value: drop.
        {"value_text": "40-50", "unit": "K/uL", "page": 2, "quote": "platelets 40-50 K/uL"},
    ]})
    monkeypatch.setattr(rules, "verify_evidence",
                        lambda ev, pages: SimpleNamespace(verified=True, page=ev.page))
    found = rules.find_values("platelets", "note", [])
    assert [v["number"] for v in found] == [45.0]


def test_find_values_drops_unverified_quotes(monkeypatch):
    monkeypatch.setattr(rules, "ask_json", lambda prompt, note: {"values": [
        {"value_text": "45", "unit": "K/uL", "page": 2, "quote": "Platelets 45 K/uL"},
    ]})
    monkeypatch.setattr(rules, "verify_evidence",
                        lambda ev, pages: SimpleNamespace(verified=False, page=ev.page))
    assert rules.find_values("platelets", "note", []) == []