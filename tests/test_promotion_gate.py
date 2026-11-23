from __future__ import annotations

import promotion_gate


def _base(**extra) -> dict:
    payload = {
        "n": 40,
        "n_l": 20,
        "n_b": 20,
        "successes_l": 16,
        "successes_b": 10,
        "n_contrast": 12,
        "n_distinct": 2,
        "constant_advice": False,
        "level": "confirmed",
    }
    payload.update(extra)
    return payload


def _open_case(**extra) -> dict:
    payload = {
        "n": 4000,
        "n_l": 2000,
        "n_b": 2000,
        "successes_l": 1900,
        "successes_b": 200,
        "n_contrast": 2000,
        "n_distinct": 2,
        "constant_advice": False,
        "level": "confirmed",
    }
    payload.update(extra)
    return payload


def _legacy_margin_only_open(report: dict) -> bool:
    n = int(report.get("n") or 0)
    n_l = int(report.get("n_l") or 0)
    n_b = int(report.get("n_b") or 0)
    if n < 12 or n_l < 1 or n_b < 1:
        return False
    if int(report.get("n_contrast") or 0) < 2:
        return False
    if int(report.get("n_distinct") or 0) < 2:
        return False
    if report.get("constant_advice"):
        return False
    if str(report.get("level") or "") != "confirmed":
        return False
    return (report["successes_l"] / n_l) - (report["successes_b"] / n_b) >= 0.05


def test_16_of_20_vs_10_of_20_stays_closed():
    noisy = _base()
    result = promotion_gate.evaluate(noisy)
    assert result["open"] is False
    assert "wilson_not_separated" in result["reason"]
    assert _legacy_margin_only_open(noisy) is True


def test_16_of_20_still_closed_even_with_contrast_floor():
    result = promotion_gate.evaluate(_base(n_contrast=1500))
    assert result["open"] is False
    assert "wilson_not_separated" in result["reason"]


def test_n_contrast_is_total_not_per_arm():
    assert promotion_gate.MIN_N_CONTRAST == 1500
    total = _open_case(n_contrast=1500, n_l=2000, n_b=2000)
    result = promotion_gate.evaluate(total)
    assert result["open"] is True
    assert result["n_contrast"] == 1500
    half = promotion_gate.evaluate(_open_case(n_contrast=749))
    assert half["open"] is False
    assert "n_contrast<1500" in half["reason"]


def test_wilson_5pp_at_1500_per_arm_separates_on_full_floats():
    l_low, _, _ = promotion_gate._wilson_interval(788, 1500)
    _, _, b_high = promotion_gate._wilson_interval(712, 1500)
    assert l_low is not None and b_high is not None
    assert round(l_low, 6) == 0.50003
    assert round(b_high, 6) == 0.49997
    assert l_low > b_high
    assert round(l_low, 4) == round(b_high, 4) == 0.5
    result = promotion_gate.evaluate(
        _open_case(
            n=3000,
            n_l=1500,
            n_b=1500,
            successes_l=788,
            successes_b=712,
            n_contrast=1500,
        )
    )
    assert result["open"] is True
    assert result["wilson_l_low"] == 0.50003
    assert result["wilson_b_high"] == 0.49997


def test_open_when_wilson_intervals_do_not_overlap():
    result = promotion_gate.evaluate(_open_case())
    assert result["open"] is True
    assert result["reason"] == "-"
    assert result["min_n"] == 12
    assert result["min_n_contrast"] == 1500
    assert result["conservative_margin"] > 0
    assert promotion_gate.MIN_N == 12
    assert promotion_gate.MIN_N_CONTRAST == 1500


def test_closed_when_n_below_min():
    result = promotion_gate.evaluate(_base(n=11, n_l=6, n_b=5, successes_l=5, successes_b=2))
    assert result["open"] is False
    assert "n<12" in result["reason"]


def test_closed_when_constant_advice():
    result = promotion_gate.evaluate(_open_case(n_distinct=1, constant_advice=True))
    assert result["open"] is False
    assert "constant_advice" in result["reason"]


def test_closed_when_margin_below_threshold():
    result = promotion_gate.evaluate(_open_case(successes_l=1000, successes_b=1000))
    assert result["open"] is False
    assert "margin_below_threshold" in result["reason"]


def test_closed_without_level_3():
    result = promotion_gate.evaluate(_open_case(level="technical"))
    assert result["open"] is False
    assert "level!=confirmed" in result["reason"]


def test_does_not_lower_min_n():
    assert promotion_gate.MIN_N >= 12
    assert promotion_gate.MIN_N_CONTRAST >= 1500


def test_rollback_closes_previously_open_gate():
    previous = promotion_gate.evaluate(_open_case())
    assert previous["open"] is True
    current = promotion_gate.rollback(previous, _base())
    assert current["open"] is False
    assert current["rollback"] is True
    assert current["reason"].startswith("rollback:")
