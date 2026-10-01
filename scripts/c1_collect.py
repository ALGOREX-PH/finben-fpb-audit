"""Part C, step 1: collect FRESH financial-news sentences that FinMA cannot have seen. Network, no GPU.

  uv run python scripts/c1_collect.py            # ~300 sentences -> data/fresh/candidates.csv

Why: FinMA (2023, LLaMA-1 base) behaves on FinBen's test set like a model that trained on it (Part B,
finding 4). A test set written after FinMA existed settles it: if FinMA only memorised, it should
drop toward our level on new sentences while our model holds steady.

Source: Cision's public press-release feed (news.cision.com) -- Nordic-listed company announcements,
the same kind of source Financial PhraseBank was built from (Finnish company news). Only releases
dated 2025-01-01 or later are used.
Filters: English, 8-45 words, a complete sentence, no contact/disclaimer/legal boilerplate, max 3
sentences per release (so no single company dominates), and NO near-copy of any FinBen FPB sentence
(train, validation or test; ftlib.contamination, same detectors as Part A).
The texts are copyrighted press releases: kept locally in data/ (git-ignored) for evaluation only.
"""
import argparse
import html
import random
import re
import time
import urllib.request
from email.utils import parsedate_to_datetime

import config  # noqa: F401  (cache paths)
import pandas as pd

from config import DATA_DIR, NEAR_DUP_THRESHOLD, NGRAM_N, SPLIT_FILES
from ftlib.contamination import contamination_table

FEED = "https://news.cision.com/ListItems?format=rss&pageSize=100&pageIx={page}"
MIN_DATE = pd.Timestamp("2025-01-01", tz="UTC")
BOILERPLATE = re.compile(
    r"(for (further|more) information|contact|e-?mail|tel\.?|phone|\+\d|www\.|http|@|forward[- ]looking|disclaimer|"
    r"this (information|announcement|press release) (is|was)|inside information|pursuant to|in accordance with|"
    r"market abuse regulation|regulation \(eu\)|not for (release|distribution)|subscribe|click|cookie|"
    r"about [A-Z][a-z]+|the company'?s? (auditor|shares are listed)|certified adviser|/cnw/|\(the \"company\"\))",
    re.IGNORECASE)
ENGLISH = re.compile(r"\b(the|and|of|to|in|for|with|is|was|has|will)\b", re.IGNORECASE)
# page headers, event logistics and fragments: trivially "neutral" or unlabelable, would make the test too easy
HEADER = re.compile(r"((?i:press release|stock exchange release|investor news|company announcement|half-year report|please find attached)|Managers[’'] Transactions|\bName:|"
                    r"^(EEST|CEST|CET|EET|GMT)\b|^[A-Z0-9&.,]{2,}(\s+[A-Z0-9&.,]{2,}){1,}\s)")
LOGISTICS = re.compile(r"\b(webcast|conference call|teleconference|will (host|be held|be conducted|be presented|"
                       r"release|publish|present)|presentation will|capital markets day|general meeting|"
                       r"chairman of the meeting|interim report for|q&a|registration|livestream|dial[- ]in)\b",
                       re.IGNORECASE)


def fetch(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; finetuning-training-2026)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except OSError:
            time.sleep(3 * (i + 1))
    return ""


def items(xml):
    for it in re.findall(r"<item>(.*?)</item>", xml, re.S):
        g = lambda t: html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", (re.search(rf"<{t}>(.*?)</{t}>", it, re.S) or [None, ""])[1])).strip()
        yield {"title": g("title"), "link": g("link"), "date": g("pubDate"), "body": g("description")}


def sentences(body):
    text = html.unescape(re.sub(r"<[^>]+>", " ", body))
    text = re.sub(r"^.{0,120}?\d{4}\s*[-–—:]\s*", "", text.strip(), count=1)     # drop a leading dateline
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", re.sub(r"\s+", " ", text)):
        s = s.strip()
        words = s.split()
        if (8 <= len(words) <= 45 and s[-1] in ".!" and not BOILERPLATE.search(s)
                and not HEADER.search(s) and not LOGISTICS.search(s) and s[0].isupper()
                and len(ENGLISH.findall(s)) >= 2 and sum(c.isascii() for c in s) / len(s) > 0.97
                and not s.isupper() and s.count("(") == s.count(")")):
            yield s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="sentences to keep")
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--per-release", type=int, default=3)
    ap.add_argument("--enrich-negative", type=int, default=0,
                    help="ADD this many sentences containing negative-leaning words to the existing candidates.csv "
                         "(press releases are mostly good news; without this the negative class is too small to measure)")
    ap.add_argument("--min-date", default="2026-06-30", help="oldest release date kept (default: the shipped set's window)")
    ap.add_argument("--max-date", default="2026-09-30", help="newest release date kept, inclusive")
    args = ap.parse_args()
    min_date = pd.Timestamp(args.min_date, tz="UTC")
    max_date = pd.Timestamp(args.max_date, tz="UTC") + pd.Timedelta(days=1)     # inclusive of the whole last day

    rows, seen_links, oldest = [], set(), None
    for page in range(1, args.max_pages + 1):
        batch = list(items(fetch(FEED.format(page=page))))
        if not batch:
            break
        for it in batch:
            try:
                date = parsedate_to_datetime(it["date"])
            except (TypeError, ValueError):
                continue
            oldest = date
            if not min_date <= date < max_date or it["link"] in seen_links:
                continue
            seen_links.add(it["link"])
            company = (re.search(r"news\.cision\.com/([^/]+)/", it["link"]) or [None, "?"])[1]
            for s in list(sentences(it["body"]))[:args.per_release]:
                rows.append({"text": s, "company": company, "date": date.date().isoformat(), "url": it["link"]})
        print(f"\rpage {page}: {len(seen_links)} releases, {len(rows)} candidate sentences (oldest {oldest.date() if oldest else '-'})",
              end="", flush=True)
        if oldest is not None and oldest < MIN_DATE:
            break
        time.sleep(1)                                                # be polite to the server
    print()
    df = pd.DataFrame(rows).drop_duplicates("text")

    # no near-copy of ANY FinBen FPB sentence (the point is: sentences no model trained on)
    finben = pd.concat([pd.read_csv(DATA_DIR / f) for f in SPLIT_FILES.values()]).text
    tab = contamination_table(df.text, finben, NEAR_DUP_THRESHOLD, NGRAM_N)
    df = df[~tab["contaminated"].values]
    print(f"after removing near-copies of FinBen FPB: {len(df)} sentences from {df.url.nunique()} releases, "
          f"{df.company.nunique()} companies, {df.date.min()} .. {df.date.max()}")

    out = DATA_DIR / "fresh"
    out.mkdir(parents=True, exist_ok=True)
    if args.enrich_negative:
        # Keyword only decides which sentences get LABELLED; the label itself still comes from reading the sentence.
        existing = pd.read_csv(out / "candidates.csv")
        neg = re.compile(r"\b(decline[sd]?|decreas\w*|los[st]\w*|lower|weak\w*|down\w*|fell|fall\w*|drop\w*|negative\w*|"
                         r"impairment|write-?down|layoffs?|redundanc\w*|dismiss\w*|cut\w*|reduc\w*|delay\w*|behind|"
                         r"postpon\w*|terminat\w*|cancel\w*|warn\w*|challeng\w*|difficult\w*|pressure\w*|shortfall|"
                         r"miss\w*|lawsuit|fine[sd]?|penalt\w*|bankrupt\w*|restructur\w*|change negotiations)\b", re.IGNORECASE)
        pool = df[df.text.str.contains(neg) & ~df.text.isin(existing.text)]
        add = pool.sample(min(args.enrich_negative, len(pool)), random_state=13).reset_index(drop=True)
        add.insert(0, "id", [f"fresh{len(existing) + i:03d}" for i in range(len(add))])
        existing["selection"] = existing.get("selection", "random")
        add["selection"] = "negative-keyword enrichment"
        keep = pd.concat([existing, add], ignore_index=True)
        keep.to_csv(out / "candidates.csv", index=False, encoding="utf-8")
        print(f"added {len(add)} negative-keyword sentences (pool {len(pool)}) -> {len(keep)} candidates")
        raise SystemExit(0)
    random.seed(13)
    keep = df.sample(min(args.n, len(df)), random_state=13).reset_index(drop=True)
    keep.insert(0, "id", [f"fresh{i:03d}" for i in range(len(keep))])
    keep["selection"] = "random"
    keep.to_csv(out / "candidates.csv", index=False, encoding="utf-8")
    print(f"kept {len(keep)} -> {out / 'candidates.csv'}")
    print(keep.sample(8, random_state=1).text.str[:110].to_string())
