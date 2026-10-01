"""ftlib.metrics, checked against hand-computed values."""
import numpy as np
import pytest

from ftlib import metrics

AB = ["a", "b"]


def test_macro_f1_hand_computed():
    # class a: P 1/1, R 1/2 -> F1 2/3;  class b: P 2/3, R 2/2 -> F1 4/5;  mean = 11/15
    assert metrics.macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"], AB) == pytest.approx(11 / 15)


def test_macro_f1_unparseable_prediction_counts_as_wrong():
    # 'missing' is in no class: a gets P 1, R 1/2 (F1 2/3); b stays perfect (F1 1)
    assert metrics.macro_f1(["a", "a", "b", "b"], ["a", "missing", "b", "b"], AB) == pytest.approx(5 / 6)


def test_macro_f1_label_absent_from_gold_and_pred_scores_zero():
    # zero_division=0: an unused label drags the macro average down instead of being skipped
    assert metrics.macro_f1(["a", "b"], ["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)


def test_core_metrics():
    out = metrics.core_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], AB)
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["macro_f1"] == pytest.approx(11 / 15)
    np.testing.assert_array_equal(out["confusion"], [[1, 1], [0, 2]])   # rows = gold, columns = predicted
    assert "precision" in out["report"]


def test_bootstrap_ci_is_deterministic_and_brackets_the_score():
    rng = np.random.default_rng(1)
    gold = rng.choice(AB, 200)
    pred = np.where(rng.random(200) < 0.8, gold, rng.choice(AB, 200))
    lo, hi = metrics.bootstrap_ci(gold, pred, AB, n_resamples=300, seed=5)
    assert (lo, hi) == metrics.bootstrap_ci(gold, pred, AB, n_resamples=300, seed=5)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo <= metrics.macro_f1(gold, pred, AB) <= hi
    assert (lo, hi) != metrics.bootstrap_ci(gold, pred, AB, n_resamples=300, seed=6)


def test_bootstrap_ci_of_perfect_predictions_is_a_point():
    gold = ["a", "b"] * 20
    assert metrics.bootstrap_ci(gold, gold, AB, n_resamples=50) == (1.0, 1.0)
