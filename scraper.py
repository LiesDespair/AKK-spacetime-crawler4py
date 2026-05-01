import re
from collections import Counter
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup
from analytics import record_page

# ──────────────────────────────────────────────────────────────
# Allowed domains/paths — only URLs under these will be crawled.
# Derived from the seed URLs in config.ini.
# We allow any subdomain of these (e.g. "vision.ics.uci.edu").
# For "today.uci.edu" we restrict to the /department/information_computer_sciences path.
# ──────────────────────────────────────────────────────────────
ALLOWED_DOMAINS = [
    ".ics.uci.edu",
    ".cs.uci.edu",
    ".informatics.uci.edu",
    ".stat.uci.edu",
]

# ──────────────────────────────────────────────────────────────
# Low-information / binary file extensions that should never be crawled.
# This extends the original list with a few more traps we've observed.
# ──────────────────────────────────────────────────────────────
NON_PAGE_EXTENSIONS = re.compile(
    r".*\.(css|js|bmp|gif|jpe?g|ico"
    + r"|png|tiff?|mid|mp2|mp3|mp4"
    + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
    + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
    + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
    + r"|epub|dll|cnf|tgz|sha1"
    + r"|thmx|mso|arff|rtf|jar|csv"
    + r"|rm|smil|wmv|swf|wma|zip|rar|gz"
    # ── additional trap / binary extensions ──
    + r"|img|war|apk|sql|db|sqlite"
    + r"|json|xml|rss|atom|feed"
    + r"|py|java|c|cpp|h|class|o"
    + r"|svg|webp|bak|log|conf|cfg|ini"
    + r"|mpg|flv|m4a|aac|odt|ods|odp|ttf|woff2?|eot)$",
)


def scraper(url, resp):
    """Main entry point called by the crawler worker.

    Extracts links from the page and filters them through is_valid
    so only in-scope, crawlable URLs are returned to the frontier.
    """
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]


def extract_next_links(url, resp):
    """Parse the HTTP response and return a list of absolute, defragmented URLs.

    Design decisions
    ────────────────
    • We only process responses with status 200 and a non-empty raw_response.
      Any status ≥ 400 or in the 600-range (cache-server errors) is skipped
      because there is no useful HTML to parse.
    • We use BeautifulSoup with the fast 'lxml' parser to find all <a href>
      tags.  This is more robust than regex-based link extraction because it
      handles malformed HTML, relative URLs, encoded entities, etc.
    • We resolve every href to an absolute URL using urljoin against the
      *actual* response URL (resp.url) — this correctly handles redirects
      where the final URL differs from the requested one.
    • Fragments (#…) are stripped with urldefrag because the same document
      content is served regardless of fragment.
    • We skip empty hrefs, javascript: links, and mailto: links early to
      avoid polluting the frontier with garbage.
    """
    extracted_links = []

    # ── Guard: only process successful, non-empty responses ──────────
    # Status 200 means the cache server delivered the page successfully.
    # Any other status (4xx, 5xx, 6xx cache errors) means we have no
    # usable content to extract links from.
    if resp.status != 200:
        return extracted_links

    # raw_response can be None when the cache server couldn't unpickle
    # the response (see utils/response.py).  content can be empty on
    # legitimate-but-empty pages.
    if resp.raw_response is None:
        return extracted_links
    if not resp.raw_response.content:
        return extracted_links
    if len(resp.raw_response.content) > 5_000_000:  # skip files larger than 5 MB
        return extracted_links

    # ── Determine the base URL for resolving relative links ──────────
    # We prefer resp.raw_response.url (the final URL after any redirects)
    # because relative links on the page are relative to that location.
    # Fall back to resp.url (the URL we originally requested) if needed.
    base_url = resp.raw_response.url if resp.raw_response.url else resp.url

    try:
        # ── Parse HTML with BeautifulSoup + lxml for speed ───────────
        soup = BeautifulSoup(resp.raw_response.content, "lxml")
    except Exception:
        # If parsing fails entirely (e.g. binary content mislabeled as
        # text/html), just return nothing rather than crashing.
        return extracted_links

    # ── Low-content page detection ───────────────────────────────────
    # Pages with very little visible text are often soft-404s, login
    # walls, or error pages.  We still extract links from them because
    # they may link to real content, but this is a natural place to add
    # a threshold if we wanted to skip low-value pages entirely.
    page_text = soup.get_text(separator=" ", strip=True)
    # Skip pages that are essentially empty (< 50 visible characters).
    # These are usually server error pages or blank templates with no
    # real content worth indexing or following links from.
    if len(page_text) < 50:
        return extracted_links

    # ── Record analytics for this page ───────────────────────────────
    clean_url, _ = urldefrag(url)
    record_page(clean_url, soup)

    # ── Extract all <a> tags with an href attribute ──────────────────
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        # Skip trivially invalid hrefs before doing any URL work.
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue

        # Resolve relative URLs against the page's actual location.
        absolute_url = urljoin(base_url, href)

        # Remove the fragment portion — same content regardless of anchor.
        defragmented_url, _ = urldefrag(absolute_url)

        # Basic sanity: only keep http/https schemes.
        if urlparse(defragmented_url).scheme in ("http", "https"):
            extracted_links.append(defragmented_url)

    return extracted_links


def is_valid(url):
    """Decide whether the crawler should download this URL.

    Returns True for URLs we want to crawl, False otherwise.

    Design decisions
    ────────────────
    • Domain allow-list: We only crawl pages under the four UCI
      sub-domains (ics, cs, informatics, stat) plus the ICS news
      section on today.uci.edu.  Everything else is out-of-scope.
    • Trap avoidance:
        – Calendar pages (e.g. ?ical, /calendar/, /events/ with dates)
          create near-infinite crawl paths by generating a page for
          every day into the future/past.
        – Very long URLs or deeply nested paths are typically auto-
          generated content or directory traversals.
        – Query strings with session-like parameters or sort/filter
          controls create combinatorial explosions of duplicate pages.
        – Repeating path components (e.g. /foo/foo/foo/) are a classic
          crawler trap.
    • File extension filter: We reject any URL whose path ends with a
      known binary / non-HTML extension.
    """
    try:
        parsed = urlparse(url)

        # ── Scheme check ─────────────────────────────────────────────
        if parsed.scheme not in ("http", "https"):
            return False

        # ── Domain allow-list ────────────────────────────────────────
        # The hostname (lowered) must be one of the allowed domains or
        # a subdomain thereof.  E.g. "vision.ics.uci.edu" ends with
        # ".ics.uci.edu".  We also accept the bare form "ics.uci.edu".
        hostname = parsed.hostname
        if hostname is None:
            return False
        hostname = hostname.lower()

        # Known trap domains — block entirely.
        BLOCKED_HOSTS = {"grape.ics.uci.edu", "intranet.ics.uci.edu", "wiki.ics.uci.edu"}
        if hostname in BLOCKED_HOSTS:
            return False

        # Special case: today.uci.edu is only allowed under the ICS department path.
        if hostname == "today.uci.edu":
            if not parsed.path.lower().startswith("/department/information_computer_sciences"):
                return False
        else:
            # General domain check — hostname must match or be a subdomain
            # of one of our allowed domains.
            domain_ok = any(
                hostname == domain.lstrip(".")          # exact match (e.g. "ics.uci.edu")
                or hostname.endswith(domain)            # subdomain (e.g. "vision.ics.uci.edu")
                for domain in ALLOWED_DOMAINS
            )
            if not domain_ok:
                return False

        # ── File extension filter ────────────────────────────────────
        # Reject known binary / non-page file extensions.  We compile
        # the regex once at module load (NON_PAGE_EXTENSIONS) for speed.
        if NON_PAGE_EXTENSIONS.match(parsed.path.lower()):
            return False

        # ── Trap detection heuristics ────────────────────────────────

        # 1. Extremely long URLs are almost always auto-generated junk
        #    or infinite redirect chains.  800 chars is very generous.
        if len(url) > 800:
            return False

        # 2. Very deep paths (> 15 components) suggest crawler traps
        #    like infinitely nested directories.
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) > 15:
            return False

        # 3. Repeating path segments — e.g. /a/b/a/b/a/b — are a
        #    classic trap pattern.  If any single segment appears 4+
        #    times, it's almost certainly a loop.
        if path_parts:
            seg_counts = Counter(path_parts)
            if seg_counts.most_common(1)[0][1] >= 4:
                return False

        # 4. Calendar / event traps — these pages generate infinite
        #    paginated date ranges.  We detect them by path keywords
        #    combined with date-like patterns in the path or query.
        path_lower = parsed.path.lower()
        query_lower = parsed.query.lower()

        # Block URLs with calendar-related path segments that also
        # contain a date pattern — these are per-day views.
        calendar_keywords = ("/calendar", "/events", "/event", "/day", "/week", "/month")
        if any(kw in path_lower for kw in calendar_keywords):
            # If the URL also has a date-like pattern (YYYY-MM-DD or YYYY/MM/DD),
            # it's almost certainly an infinite calendar page.
            if re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", url):
                return False
            # Even without a full date, calendar paths with "?ical" or
            # date-related query params are traps.
            if "ical" in query_lower or "date" in query_lower:
                return False

        # 5. Query-string complexity — URLs with many query parameters
        #    are often filter/sort combinations that produce duplicate
        #    or low-value content.
        if parsed.query:
            # Count the number of parameters (separated by &).
            param_count = parsed.query.count("&") + 1
            if param_count > 5:
                return False

            # Known trap parameters that create infinite pagination or
            # session-dependent views.
            trap_params = ("sessionid", "sid", "phpsessid", "jsessionid",
                           "share", "replytocom", "action=login",
                           "action=download", "do=hierarchicalnamespaceurlrewrite",
                           "do=media", "tab_files", "tab_details")
            if any(tp in query_lower for tp in trap_params):
                return False

        # 6. Specific known trap patterns from ICS sites
        #    e.g. wiki diffs, revision histories, endless pagination
        if re.search(r"(difftype|diffold|rev=|oldid=|action=edit|action=diff)", query_lower):
            return False

        # 7. /wp-admin, /wp-login, /wp-json, and similar WordPress
        #    back-end paths never have useful crawlable content.
        if re.search(r"/(wp-admin|wp-login|wp-json|cgi-bin|api/)", path_lower):
            return False

        # ── If we passed all filters, the URL is valid ───────────────
        return True

    except TypeError:
        print("TypeError for ", parsed)
        raise
