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
