"""FinBen FPB helpers that need only pandas (no torch), so tests and CI can import them.

  finben_parse   FinBen/PIXIU's answer parser
  attach_gold    re-attach gold labels to a shipped prediction file (ids only, no dataset text or labels)
  accuracy_by_agreement   accuracy per PhraseBank annotator-agreement level (the memorisation diagnostic)
"""
import pandas as pd

FPB_CHOICES = ["positive", "neutral", "negative"]   # FinBen FPB's choice order: the first match wins


def finben_parse(raw, choices):
    """FinBen/PIXIU rule: first choice (dataset order) contained in the lower-cased reply, else 'missing'."""
    low = raw.lower()
    return next((c.lower() for c in choices if c.lower() in low), "missing")


def attach_gold(pred, gold):
    """`pred` with a `gold` column taken from `gold` (a Series indexed by sentence id).

    The repo ships predictions without text or labels (FinBen FPB is gated), so every report joins the
    labels back from the local data. Files that already carry `gold` are returned unchanged.
    """
    if "gold" in pred:
        return pred
    missing = ~pred["id"].isin(gold.index)
    if missing.any():
        raise KeyError(f"{int(missing.sum())} prediction ids have no gold label, e.g. {pred['id'][missing].iloc[0]!r}")
    out = pred.copy()
    out.insert(out.columns.get_loc("id") + 1, "gold", gold.loc[out["id"]].to_numpy())
    return out


AGREEMENT_LEVELS = ["AllAgree", "75Agree", "66Agree", "50Agree"]   # PhraseBank: share of annotators who agreed


def accuracy_by_agreement(pred, gold, agreement, levels=AGREEMENT_LEVELS):
    """Accuracy per PhraseBank annotator-agreement level, {level: (accuracy, n)} (the b5_report section 4b check).
    On 50Agree sentences the annotators split, so a model that never saw the gold label can't do much better
    than the label's own ambiguity allows; one that memorised it can."""
    pred, gold, agreement = (pd.Series(x).to_numpy() for x in (pred, gold, agreement))
    out = {}
    for level in levels:
        m = agreement == level
        out[level] = (float((pred[m] == gold[m]).mean()) if m.any() else float("nan"), int(m.sum()))
    return out
