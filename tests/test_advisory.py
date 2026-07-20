"""AdvisoryTracker: grading offline advisories against realized moves."""

from __future__ import annotations

from gungnir.learning.advisory import AdvisoryTracker


def test_grades_previous_advice_against_realized_move(tmp_path):
    t = AdvisoryTracker(path=str(tmp_path / "adv.json"))
    # First call for a symbol: nothing to grade yet.
    assert t.record("EURUSD", "LONG", 100.0, "t0") is None
    # Price rose and we'd advised LONG → a hit, positive shadow return.
    g = t.record("EURUSD", "LONG", 101.0, "t1")
    assert g["correct"] is True and g["shadow_return"] > 0
    # Price fell while still advising LONG → a miss, negative shadow return.
    g2 = t.record("EURUSD", "SHORT", 99.0, "t2")  # grades the prior LONG@101
    assert g2["correct"] is False and g2["shadow_return"] < 0

    snap = t.snapshot()
    assert snap["n_graded"] == 2
    assert 0.0 <= snap["hit_rate"] <= 1.0
    assert snap["by_action"]["LONG"] == 2


def test_flat_advice_is_not_counted_directionally(tmp_path):
    t = AdvisoryTracker(path=str(tmp_path / "adv.json"))
    t.record("X", "FLAT", 100.0, "t0")
    t.record("X", "LONG", 105.0, "t1")     # grades prior FLAT → non-directional
    assert t.snapshot()["n_graded"] == 0
    assert t.by_action["FLAT"] == 1


def test_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "adv.json")
    t = AdvisoryTracker(path=p)
    t.record("X", "LONG", 100.0, "t0")
    t.record("X", "LONG", 102.0, "t1")
    t.save()
    t2 = AdvisoryTracker(path=p)
    assert t2.load() is True
    assert t2.snapshot()["n_graded"] == t.snapshot()["n_graded"]
    assert abs(t2.cum_return - t.cum_return) < 1e-12
