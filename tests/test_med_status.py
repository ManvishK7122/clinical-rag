"""
Tests for medication status decisions in screening/snapshot.py.
Code reads the words around each medication in its verified quote and decides the
status, so the same note gives the same status every run. No API calls.

Run from the repo root:
    python -m pytest -q
"""

import pytest

from screening.snapshot import check_med_statuses, same_medication, statuses_in_words


def med(name, status, quote, verified=True):
    return {"name": name, "status": status, "dose": None,
            "evidence": {"page": 1, "quote": quote, "verified": verified}}


def one(name, model_status, quote, verified=True):
    kept, _ = check_med_statuses([med(name, model_status, quote, verified)])
    return kept[0]


def test_same_medication_ignores_generic_words():
    assert same_medication("heparin", "heparin infusion")
    assert same_medication("Home metoprolol", "metoprolol tartrate")
    assert not same_medication("insulin", "heparin")


@pytest.mark.parametrize("drug, quote, expected", [
    ("vanc", "Vanc discontinued on day 2 after MRSA nares PCR negative", "discontinued"),
    ("dopamine", "was successfully removed from dopamine", "discontinued"),
    ("enoxaparin", "Enoxaparin DVT prophylaxis HELD due to active GI bleed", "held"),
    ("iron", "consider iron, vitamin C, folate and B12 supplementation", "considered"),
    ("NSAIDs", "Dr. X recommended to avoid NSAIDs in the future", "considered"),
    ("guaifenesin", "We will add guaifenesin and N-acetyl-cysteine", "planned"),
    ("fentanyl", "fentanyl bolus given with improvement", "given_once"),
    ("Omnicef", "We will continue him on his oral Omnicef", "active"),
    ("hydroxyurea", "the patient was discharged with hydroxyurea 1000 mg", "active"),
    ("Bumex", "Bumex 2 mg p.o. daily", "active"),
    ("norepinephrine", "norepinephrine held at 0.05 mcg/kg/min", "active"),
])
def test_code_sets_status_from_words(drug, quote, expected):
    # The model guesses a different status on purpose; code must correct it.
    m = one(drug, "active" if expected != "active" else "discontinued", quote)
    assert m["status"] == expected
    assert m["status_supported"] is True


def test_each_drug_in_a_shared_sentence_gets_its_own_status():
    quote = "De-escalation from vancomycin completed yesterday; now on ceftriaxone 1g IV Q24h."
    assert one("vancomycin", "active", quote)["status"] == "discontinued"
    assert one("ceftriaxone", "discontinued", quote)["status"] == "active"


def test_unverified_quote_never_decides():
    m = one("enoxaparin", "held", "enoxaparin held", verified=False)
    assert m["status"] == "held" and m["status_supported"] is False


def test_no_status_words_keeps_model_status_and_flags_not_given():
    m = one("Levaquin", "discontinued", "treated successfully with Zyvox and Levaquin")
    assert m["status"] == "discontinued" and m["status_supported"] is False


def test_mixed_words_keep_model_status_only_if_supported():
    quote = "NE gtt 0.08 mcg/kg/min, will start weaning"
    assert statuses_in_words("NE", quote) >= {"active"}
    assert one("NE", "active", quote)["status_supported"] is True


def test_contradiction_and_duplicates_resolved():
    # The Asclepius case: the model also called a running heparin drip "discontinued".
    meds = [med("heparin", "active", "was started on heparin infusion"),
            med("heparin infusion", "discontinued", "was started on heparin infusion.")]
    kept, dropped = check_med_statuses(meds)
    assert [(m["name"], m["status"]) for m in kept] == [("heparin", "active")]
    assert len(dropped) == 1


def test_real_double_status_is_kept():
    meds = [med("insulin", "active", "insulin drip 2 units/hr"),
            med("insulin", "discontinued", "sliding scale insulin stopped")]
    kept, dropped = check_med_statuses(meds)
    assert {m["status"] for m in kept} == {"active", "discontinued"} and dropped == []


def test_home_medication_list_is_not_decided_by_code():
    quote = "Home medications: Metoprolol 25 mg BID, ibuprofen 400 mg TID, omeprazole 20 mg daily."
    assert statuses_in_words("omeprazole", quote) == set()


def test_received_is_a_completed_dose():
    assert one("PRBC", "active", "the patient has received 2 units of packed red blood cells",
               )["status"] == "given_once"


def test_once_daily_is_active_not_a_single_dose():
    assert one("Protonix", "given_once", "Protonix 40 mg once daily")["status"] == "active"
