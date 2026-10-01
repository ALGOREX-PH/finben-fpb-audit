"""Part C, step 0 (for people reproducing this repo): rebuild the fresh 2026 test set from public URLs.

  uv run python scripts/c0_rebuild_fresh.py

The repo ships data/fresh/fresh_set.csv with each sentence's press-release URL, a SHA-1 fingerprint of its
normalised text, its normalised word count and its label -- but NOT the text (press releases are
copyrighted). This script fetches each URL, slides a window of `n_words` words over the page, and keeps
the span whose fingerprint matches; the original wording (case, punctuation) is cut back out of the page.
Output: data/fresh/candidates.csv + data/fresh/labels.csv, exactly the files c3_eval.py / c4_report.py read.
Sentences whose page has changed or disappeared are reported and left out.
"""
import hashlib
import html
import re
import time
import urllib.request

import config  # noqa: F401
import pandas as pd

from config import DATA_DIR

FRESH = DATA_DIR / "fresh"
WORD = re.compile(r"[A-Za-z0-9]+")          # the same tokens ftlib.contamination.normalize keeps


def fetch(url):
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; finben-fpb-audit)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except OSError:
            time.sleep(3 * (i + 1))
    return ""


def page_text(page):
    page = re.sub(r"(?is)<(script|style).*?</\1>", " ", page)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))


def find(text, sha1, n):
    """Original-text span whose normalised words hash to sha1, or None."""
    toks = list(WORD.finditer(text))
    for i in range(len(toks) - n + 1):
        words = " ".join(m.group().lower() for m in toks[i:i + n])
        if hashlib.sha1(words.encode()).hexdigest() == sha1:
            a, b = toks[i].start(), toks[i + n - 1].end()
            while a > 0 and not text[a - 1].isspace():          # leading quote / bracket / non-ASCII letters
                a -= 1
            while b < len(text) and not text[b].isspace():      # trailing punctuation
                b += 1
            return text[a:b].strip()
    return None


if __name__ == "__main__":
    fresh = pd.read_csv(FRESH / "fresh_set.csv")
    found, pages = {}, {}
    for k, row in enumerate(fresh.itertuples(), 1):
        if row.url not in pages:
            pages[row.url] = page_text(fetch(row.url))
            time.sleep(0.5)
        span = find(pages[row.url], row.text_sha1, int(row.n_words))
        if span:
            found[row.id] = span
        print(f"\r{k}/{len(fresh)} sentences, {len(found)} recovered", end="", flush=True)
    print()

    ok = fresh[fresh.id.isin(found)].copy()
    ok.insert(1, "text", ok.id.map(found))
    ok[["id", "text", "company", "date", "url", "selection"]].to_csv(FRESH / "candidates.csv", index=False, encoding="utf-8")
    pd.DataFrame({"id": ok.id, "human": ok.label, "final": "", "note": "", "labeled_by": ok.labeled_by}).to_csv(
        FRESH / "labels.csv", index=False, encoding="utf-8")
    missing = fresh[~fresh.id.isin(found)]
    print(f"recovered {len(ok)}/{len(fresh)} sentences -> {FRESH}")
    if len(missing):
        print(f"{len(missing)} not recoverable (page changed or removed): {missing.id.tolist()[:10]}{' ...' if len(missing) > 10 else ''}")
