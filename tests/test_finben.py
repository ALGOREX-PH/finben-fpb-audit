"""ftlib.finben: FinBen/PIXIU's FPB answer parser and the gold-label join."""
import pandas as pd
import pytest

from ftlib.finben import FPB_CHOICES, attach_gold, finben_parse


@pytest.mark.parametrize("raw, expected", [
    ("positive", "positive"),
    ("neutral", "neutral"),
    ("negative", "negative"),
    ("Negative.", "negative"),                        # case and punctuation don't matter
    ("  NEUTRAL\n", "neutral"),
    ("The sentiment is positive.", "positive"),       # a word anywhere in the reply counts
])
def test_parse_plain_answers(raw, expected):
    assert finben_parse(raw, FPB_CHOICES) == expected


@pytest.mark.parametrize("raw, expected", [
    ("not positive", "positive"),                     # substring match, no negation handling
    ("negative, not neutral", "neutral"),             # first CHOICE in choice order wins, not first in the reply
    ("positive or negative", "positive"),
    ("nonneutral", "neutral"),                        # substring, not whole word
])
def test_parse_follows_choice_order_quirks(raw, expected):
    assert finben_parse(raw, FPB_CHOICES) == expected


@pytest.mark.parametrize("raw", ["", "mixed", "I cannot tell.", "pos", "neg"])
def test_parse_missing(raw):
    assert finben_parse(raw, FPB_CHOICES) == "missing"


def test_parse_respects_the_given_choice_order():
    assert finben_parse("negative, not positive", ["negative", "neutral", "positive"]) == "negative"
    assert finben_parse("negative, not positive", ["positive", "neutral", "negative"]) == "positive"


def test_parse_lowercases_choices():
    assert finben_parse("POSITIVE", ["Positive", "Neutral", "Negative"]) == "positive"


GOLD = pd.Series({"s1": "positive", "s2": "neutral", "s3": "negative"})


def test_attach_gold_joins_by_id_and_keeps_row_order():
    pred = pd.DataFrame({"id": ["s3", "s1"], "raw": ["negative", "neutral"], "pred": ["negative", "neutral"]})
    out = attach_gold(pred, GOLD)
    assert out["gold"].tolist() == ["negative", "positive"]
    assert out["id"].tolist() == ["s3", "s1"]


def test_attach_gold_puts_gold_right_after_id():
    pred = pd.DataFrame({"raw": ["x"], "id": ["s2"], "pred": ["neutral"]})
    assert attach_gold(pred, GOLD).columns.tolist() == ["raw", "id", "gold", "pred"]


def test_attach_gold_does_not_modify_its_input():
    pred = pd.DataFrame({"id": ["s1"], "pred": ["positive"]})
    attach_gold(pred, GOLD)
    assert "gold" not in pred


def test_attach_gold_passes_through_files_that_already_have_gold():
    pred = pd.DataFrame({"id": ["s1"], "gold": ["negative"], "pred": ["positive"]})
    out = attach_gold(pred, GOLD)
    assert out is pred
    assert out["gold"].tolist() == ["negative"]     # never overwritten by the lookup


def test_attach_gold_rejects_unknown_ids():
    pred = pd.DataFrame({"id": ["s1", "zz9"], "pred": ["positive", "neutral"]})
    with pytest.raises(KeyError, match="zz9"):
        attach_gold(pred, GOLD)
