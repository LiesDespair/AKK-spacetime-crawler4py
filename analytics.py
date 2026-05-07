"""analytics.py — persistent crawl statistics for the ICS crawler report.

This module is imported by scraper.py and updated on every successfully
processed page.  All state is stored in a shelve file (analytics.shelve)
so that progress survives crawler restarts.

Four things are tracked to answer the report questions:
  Q1. Unique page count (URL, fragment-stripped — matches frontier logic)
  Q2. Longest page by word count
  Q3. 50 most common words across all pages (stop-words excluded)
  Q4. Subdomain → unique page count (for *.uci.edu)

Thread-safety note: A threading.Lock guards every shelve write.  The
crawler runs multiple worker threads, so all shared state is protected
by this lock to prevent data corruption.

Usage:
  from analytics import record_page, generate_report
  record_page(url, soup)   # call once per crawled page
  generate_report()        # call after crawling to print results
"""

import os
import re
import shelve
import threading
from collections import Counter
from urllib.parse import urlparse

# ── File paths ───────────────────────────────────────────────────────────────
# Keep analytics next to the frontier save file so both are found together.
ANALYTICS_SAVE = "analytics.shelve"
REPORT_FILE    = "analytics_report.txt"

# ── Module-level lock for thread safety ──────────────────────────────────────
_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# English stop words
# Source: https://www.ranks.nl/stopwords  (the list referenced by the assignment)
# Any word in this set is excluded from Q3 word-frequency counts.
#
# Implementation note on contractions:
#   The tokenizer uses re.findall(r"[a-zA-Z]+", ...) which strips apostrophes,
#   so "aren't" produces tokens ["aren", "t"].  "t" / "s" / "d" etc. are
#   already dropped by MIN_WORD_LEN = 2.  We store the meaningful root of
#   each contraction (e.g. "aren", "cant", "couldn") so those fragments are
#   also suppressed.
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    # ── ranks.nl list verbatim (apostrophes kept as reference) ──────────────
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by",
    "can", "can't", "cannot", "could", "couldn't", "will",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during",
    "each",
    "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself",
    "let's",
    "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # ── Alphabetic roots of contractions (what the tokenizer actually sees) ──
    # re.findall(r"[a-zA-Z]+") splits on apostrophes, so "aren't" → "aren"+"t".
    # Single-char fragments (t, s, d, m, ll, ve, re) are dropped by MIN_WORD_LEN.
    "aren", "cant", "cannot", "couldn", "didnt", "doesnt", "dont",
    "hadnt", "hasnt", "havent", "hed", "hell", "hes", "heres", "hows",
    "id", "ill", "im", "ive", "isnt", "its", "lets",
    "mustnt", "shant", "shed", "shell", "shes", "shouldnt",
    "thats", "theres", "theyd", "theyll", "theyre", "theyve",
    "wasnt", "wed", "well", "were", "werent", "whats", "whens",
    "wheres", "whos", "whys", "wont", "wouldnt",
    "youd", "youll", "youre", "youve",
    # ── URL / domain fragments ────────────────────────────────────────────────
    # These appear when pages display URLs or domain names as visible text.
    "edu", "uci", "ics", "www", "http", "https", "html", "php", "asp",
    # ── Months ───────────────────────────────────────────────────────────────
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

# Minimum token length to count — single-character tokens (e.g. "i", "a"
# already covered by stop-words, but stray punctuation-stripped chars)
# are almost never meaningful.
MIN_WORD_LEN = 2

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_shelf():
    """Open (or create) the analytics shelve.  Caller must hold _lock."""
    return shelve.open(ANALYTICS_SAVE, writeback=True)


def _get_or_default(shelf, key, default):
    """Return shelf[key] if it exists, otherwise set and return default."""
    if key not in shelf:
        shelf[key] = default
    return shelf[key]


def _is_uci_subdomain(hostname: str) -> bool:
    return hostname.endswith(".uci.edu")


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint-based exact and near-duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
#   1. Parse document into words (tokens).
#   2. Group tokens into contiguous n-grams (overlapping sequences of n words).
#   3. Hash each n-gram using FNV-1a (fast, no libraries needed).
#   4. Select a subset of hashes using modular selection (H mod k == 0).
#   5. Store the selected hash set as the document's "fingerprint".
#   6. Compare documents using Jaccard similarity (|A ∩ B| / |A ∪ B|).
#   7. If similarity ≥ threshold τ, documents are near-duplicates.
#
# FNV-1a 64-bit (public domain algorithm, no external libraries needed).
# ─────────────────────────────────────────────────────────────────────────────
_FNV_PRIME  = 0x00000100000001B3
_FNV_OFFSET = 0xcbf29ce484222325
_MASK64     = 0xFFFFFFFFFFFFFFFF

_NGRAM_SIZE          = 3     # 3-grams as shown in lecture example
_MOD_SELECT          = 4     # keep hashes where H mod 4 == 0 (lecture slide 34)
_SIMILARITY_THRESHOLD = 0.8  # Jaccard ≥ this → near-duplicate


def _fnv1a(text: str) -> int:
    """64-bit FNV-1a hash of a UTF-8 string."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & _MASK64
    return h


def _ngrams(tokens: list, n: int) -> list:
    """Generate overlapping word n-grams from a token list."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _fingerprint(tokens: list) -> frozenset:
    """Compute the fingerprint of a document (lecture method).

    Steps:
      1. Build 3-grams from the token list.
      2. Hash each 3-gram with FNV-1a.
      3. Select only hashes where H mod 4 == 0.
      4. Return the selected hashes as a frozenset.
    """
    grams = _ngrams(tokens, _NGRAM_SIZE)
    all_hashes = [_fnv1a(gram) for gram in grams]
    selected = frozenset(h for h in all_hashes if h % _MOD_SELECT == 0)
    return selected


def _jaccard(set_a, set_b) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Split text into lowercase alpha tokens, filtering stop-words.

    We keep only tokens that:
      • consist solely of ASCII letters (no digits, no punctuation)
      • are longer than MIN_WORD_LEN characters
      • are not in STOP_WORDS

    This intentionally excludes numbers and hyphenated compound tokens so
    that raw HTML artefacts (e.g. hex colour codes, version strings) don't
    pollute the word frequency counts.
    """
    raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [
        t for t in raw_tokens
        if len(t) > MIN_WORD_LEN and t not in STOP_WORDS
    ]


def is_duplicate(page_text: str) -> bool:
    """Return True if page_text is an exact or near-duplicate of a seen page.

    Exact duplicate : FNV-1a hash of the full text matches a stored hash.
    Near-duplicate  : Jaccard similarity of n-gram fingerprints ≥ threshold τ
                      against any previously stored fingerprint.

    Method:
      1. Tokenize the page text.
      2. Build 3-grams from the tokens.
      3. Hash each 3-gram and select a subset (H mod 4 == 0).
      4. Compare this fingerprint set against all stored fingerprints
         using Jaccard similarity (|A ∩ B| / |A ∪ B|).
      5. If Jaccard ≥ threshold, it's a near-duplicate.

    Both stores are persisted in the analytics shelve so detection works
    correctly across crawler restarts.
    """
    tokens = tokenize(page_text)
    if not tokens:
        return False

    exact_hash = _fnv1a(page_text)
    fp = _fingerprint(tokens)

    with _lock:
        db = _open_shelf()
        try:
            exact_hashes     = _get_or_default(db, "exact_hashes",     set())
            fingerprint_store = _get_or_default(db, "fingerprint_store", [])

            # ── Exact duplicate check ────────────────────────────────
            if exact_hash in exact_hashes:
                return True

            # ── Near-duplicate check (Jaccard similarity) ────────────
            # Only compare if the new document produced a non-empty
            # fingerprint; pages with too few tokens may produce an
            # empty set after mod-selection, which would give a
            # misleading Jaccard of 0.0.
            if fp:
                for stored_fp in fingerprint_store:
                    if _jaccard(fp, stored_fp) >= _SIMILARITY_THRESHOLD:
                        return True

            # ── Not a duplicate — store fingerprint for future checks ─
            exact_hashes.add(exact_hash)
            if fp:
                fingerprint_store.append(fp)
            db["exact_hashes"]     = exact_hashes
            db["fingerprint_store"] = fingerprint_store
            db.sync()
            return False
        finally:
            db.close()


def record_page(url: str, soup) -> None:
    """Record analytics for one successfully crawled page.

    Parameters
    ----------
    url  : defragmented URL of the page (matches frontier key convention)
    soup : BeautifulSoup object for the page (already parsed in scraper.py —
           we reuse it here to avoid a second parse)

    This function is intentionally tolerant of errors: if anything goes
    wrong updating analytics we print a warning and continue rather than
    crashing the crawler.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""

        # Extract visible text and tokenize once — used for Q2 and Q3.
        page_text = soup.get_text(separator=" ", strip=True)
        # Word count for Q2: split on whitespace, include ALL words (even
        # stop-words) because "words" here means the naive definition the
        # assignment uses — any whitespace-separated token.
        word_count = len(page_text.split())
        # Filtered token list for Q3 word frequency.
        tokens = tokenize(page_text)

        with _lock:
            db = _open_shelf()
            try:
                # ── Q1: unique pages ──────────────────────────────────
                # Use a set stored in the shelf.  The frontier also deduplicates,
                # but we keep our own set so Q1 is self-contained and can be
                # computed independently (e.g. after a partial crawl).
                unique_pages = _get_or_default(db, "unique_pages", set())
                unique_pages.add(url)
                db["unique_pages"] = unique_pages

                # ── Q2: longest page ──────────────────────────────────
                longest = _get_or_default(db, "longest_page", {"url": "", "word_count": 0})
                if word_count > longest["word_count"]:
                    longest = {"url": url, "word_count": word_count}
                    db["longest_page"] = longest

                # ── Q3: global word frequency ─────────────────────────
                # We accumulate a Counter in the shelf.  Adding a large Counter
                # on every page is O(unique_tokens_on_page), which is fine.
                freq = _get_or_default(db, "word_freq", Counter())
                freq.update(tokens)
                db["word_freq"] = freq

                # ── Q4: subdomain page counts ─────────────────────────
                # Track all *.uci.edu subdomains crawled.
                if _is_uci_subdomain(hostname):
                    subdomain_urls = _get_or_default(db, "subdomain_urls", {})
                    if hostname not in subdomain_urls:
                        subdomain_urls[hostname] = set()
                    subdomain_urls[hostname].add(url)
                    db["subdomain_urls"] = subdomain_urls

                db.sync()
            finally:
                db.close()

    except Exception as exc:
        # Never crash the crawler due to analytics.
        print(f"[analytics] Warning: failed to record {url}: {exc}")


def generate_report() -> None:
    """Read the analytics shelve and print / write the four report answers.

    Safe to call at any time — even while the crawl is running — because
    it only reads the shelf (no writes).  The report is printed to stdout
    and also written to analytics_report.txt.
    """
    with _lock:
        db = _open_shelf()
        try:
            unique_pages   = db.get("unique_pages",   set())
            longest_page   = db.get("longest_page",   {"url": "N/A", "word_count": 0})
            word_freq      = db.get("word_freq",      Counter())
            subdomain_urls = db.get("subdomain_urls", {})
        finally:
            db.close()

    lines = []

    # ── Q1: Unique pages ──────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("Q1. Unique pages found")
    lines.append("=" * 70)
    lines.append(f"  Total: {len(unique_pages):,}")
    lines.append("")

    # ── Q2: Longest page by word count ───────────────────────────────────────
    lines.append("=" * 70)
    lines.append("Q2. Longest page (by word count, HTML markup excluded)")
    lines.append("=" * 70)
    lines.append(f"  URL:        {longest_page['url']}")
    lines.append(f"  Word count: {longest_page['word_count']:,}")
    lines.append("")

    # ── Q3: 50 most common words ──────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("Q3. 50 most common words (stop-words and single-chars excluded)")
    lines.append("=" * 70)
    top50 = word_freq.most_common(50)
    for rank, (word, count) in enumerate(top50, start=1):
        lines.append(f"  {rank:>3}. {word:<30} {count:>10,}")
    lines.append("")

    # ── Q4: Subdomains of uci.edu ─────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("Q4. Subdomains found in uci.edu (alphabetical)")
    lines.append("=" * 70)
    lines.append(f"  Total subdomains: {len(subdomain_urls):,}")
    lines.append("")
    for subdomain in sorted(subdomain_urls):
        lines.append(f"  {subdomain}, {len(subdomain_urls[subdomain])}")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[analytics] Report written to {REPORT_FILE}")


def print_status() -> None:
    """Lightweight status check — prints counts without the full report.

    Useful for quick mid-crawl progress checks.
    """
    with _lock:
        db = _open_shelf()
        try:
            n_unique     = len(db.get("unique_pages", set()))
            longest      = db.get("longest_page", {"url": "N/A", "word_count": 0})
            n_words      = sum(db.get("word_freq", Counter()).values())
            n_subdomains = len(db.get("subdomain_urls", {}))
        finally:
            db.close()

    print(
        f"[analytics] "
        f"unique={n_unique:,}  "
        f"longest={longest['word_count']:,} words ({longest['url']})  "
        f"total_word_occurrences={n_words:,}  "
        f"subdomains={n_subdomains}"
    )
