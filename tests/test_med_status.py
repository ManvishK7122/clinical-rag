"""
Tests for check_med_statuses in screening/snapshot.py: a "not given" status
(held, discontinued, considered, planned) must be backed by the note's own words.
No API calls.

Run from the repo root:
    python -m pytest -q
"""

from screening.snapshot import check_med_statuses, same_medication


def med(name, status, quote, verified=True):
    return {"name": name, "status": status, "dose": None,
            "evidence": {"page": 1, "quote": quote, "verified": verified}}


def names(meds):
    return [(m["name"], m["status"]) for m in meds]


def test_same_medication_ignores_generic_words():
    assert same_medication("heparin", "heparin infusion")
    assert same_medication("Home metoprolol", "metoprolol tartrate")
    assert not same_medication("insulin", "heparin")


def test_supported_discontinued_is_kept():
    kept, dropped = check_med_statuses([med("vanc", "discontinued", "Vanc discontinued on day 2")])
    assert names(kept) == [("vanc", "discontinued")]
    assert kept[0]["status_supported"] and dropped == []


def test_contradiction_without_stop_word_is_dropped():
    # The Asclepius case: heparin is running, but the model also called it discontinued.
    meds = [med("heparin", "active", "was started on heparin infusion"),
            med("heparin infusion", "discontinued", "was started on heparin infusion.")]
    kept, dropped = check_med_statuses(meds)
    assert names(kept) == [("heparin", "active")]
    assert names(dropped) == [("heparin infusion", "discontinued")]


def test_both_statuses_kept_when_both_supported():
    # Infusion running and a separate dose stopped: both are real.
    meds = [med("insulin", "active", "insulin drip 2 units/hr"),
            med("insulin", "discontinued", "sliding scale insulin stopped")]
    kept, dropped = check_med_statuses(meds)
    assert len(kept) == 2 and dropped == []


def test_unsupported_status_alone_is_flagged_not_dropped():
    # Nothing contradicts it, so keep it for the nurse but mark it.
    kept, dropped = check_med_statuses(
        [med("albuterol", "considered", "albuterol MDI 4 puffs Q4h via in-line adapter")])
    assert dropped == []
    assert kept[0]["status_supported"] is False


def test_unverified_quote_cannot_support_a_status():
    kept, _ = check_med_statuses([med("enoxaparin", "held", "enoxaparin held", verified=False)])
    assert kept[0]["status_supported"] is False


def test_each_status_needs_its_own_cue():
    assert check_med_statuses([med("x", "held", "x held for procedure")])[0][0]["status_supported"]
    assert check_med_statuses([med("x", "considered", "consider x if worse")])[0][0]["status_supported"]
    assert check_med_statuses([med("x", "planned", "will start x tomorrow")])[0][0]["status_supported"]
    # "held" needs a hold word, not just any stop word
    assert not check_med_statuses([med("x", "held", "x discontinued")])[0][0]["status_supported"]


def test_active_status_needs_no_cue():
    kept, _ = check_med_statuses([med("norepinephrine", "active", "NE gtt 0.08 mcg/kg/min")])
    assert kept[0]["status_supported"] is True
