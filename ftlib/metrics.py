"""Classification evaluation helpers, reusable across tasks.

Every function takes plain lists/arrays of label strings, so it works the same for
a TF-IDF baseline, FinBERT, or an LLM.
"""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def macro_f1(y_true, y_pred, labels):
    # An invalid/unparseable prediction simply counts as wrong for every class.
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def core_metrics(y_true, y_pred, labels):
    """Eval #1: macro-F1 (headline), accuracy (secondary), per-class report, confusion matrix."""
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def bootstrap_ci(y_true, y_pred, labels, n_resamples=1000, seed=0, alpha=0.05):
    """Eval #4: 95% CI on macro-F1 by resampling the test set with replacement.

    Tells you how much the score would wobble on a different test set of the same size.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [macro_f1(y_true[idx], y_pred[idx], labels)
              for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def mcnemar(y_true, pred_a, pred_b):
    """Eval #4: are systems A and B different on the SAME test items? (exact McNemar test)

    Only items where exactly one system is right carry information:
      b = A right, B wrong      c = A wrong, B right
    If A and B were equally good, b and c would be a coin flip -> binomial test.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_right": b, "b_only_right": c, "p_value": float(p)}


def expected_calibration_error(y_true, probs, labels, n_bins=10):
    """Eval #6: ECE -- average gap between confidence and actual accuracy, per confidence bin.

    probs: array (n, len(labels)) of class probabilities in `labels` order.
    """
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(1)
    correct = np.asarray(labels)[probs.argmax(1)] == np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            rows.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return float(ece), rows


def bootstrap_ece(y_true, probs, labels, n_resamples=1000, seed=0, alpha=0.05):
    """95% CI on ECE by resampling the test set -- ECE on ~800 items is itself noisy."""
    y_true, probs = np.asarray(y_true), np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = [expected_calibration_error(y_true[idx], probs[idx], labels)[0]
            for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff_ci(correct_a, correct_b, n_resamples=2000, seed=0, alpha=0.05):
    """CI on accuracy(A) - accuracy(B) when both were scored on the SAME items.

    Resamples items (not systems), so the pairing is kept. If the interval is [-0.05, +0.04],
    the data rule out a drop larger than 5 points -- and nothing smaller. That's how to phrase
    "no evidence of forgetting" honestly.
    """
    a, b = np.asarray(correct_a, dtype=float), np.asarray(correct_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = [(a[idx] - b[idx]).mean() for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    return float((a - b).mean()), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def plot_reliability(curves, path, title="Reliability", n_bins=10):
    """Reliability diagram: confidence (x) vs actual accuracy (y) per bin, several systems.

    curves: {name: (y_true, probs, labels)}. On the diagonal = calibrated; below = overconfident.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfectly calibrated")
    for name, (y_true, probs, labels) in curves.items():
        ece, rows = expected_calibration_error(y_true, probs, labels, n_bins)
        xs = [r[3] for r in rows if r[2] >= 5]   # bins with >= 5 items; smaller bins are noise
        ys = [r[4] for r in rows if r[2] >= 5]
        ax.plot(xs, ys, marker="o", ms=4, label=f"{name} (ECE {ece:.3f})")
    ax.set_xlabel("confidence (top label probability)")
    ax.set_ylabel("accuracy in that bin")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
