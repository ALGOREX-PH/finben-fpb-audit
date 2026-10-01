"""ftlib.finben: FinBen/PIXIU's FPB answer parser and the gold-label join."""
import pytest

from ftlib.finben import FPB_CHOICES, finben_parse


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
