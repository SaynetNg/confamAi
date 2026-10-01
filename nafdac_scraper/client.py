"""HTTP layer for the NAFDAC Greenbook site.

Three access patterns, all plain ``GET`` with no authentication:

* :meth:`GreenbookClient.fetch_page` - the DataTables server-side JSON endpoint
  that backs the home-page product table.
* :meth:`GreenbookClient.fetch_detail` - a product detail page, for the few
  fields the bulk JSON omits.
* :meth:`GreenbookClient.fetch_index` - a paginated reference index page
  (manufacturers, applicants, ingredients).

All three share one retry policy and one session, except the detail crawl which
uses a small thread pool with a per-thread session.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

import config

log = logging.getLogger(__name__)


class GreenbookError(RuntimeError):
    """Raised when the Greenbook backend cannot be used at all."""


# --------------------------------------------------------------------------
# HTML helpers (no BeautifulSoup/lxml dependency - the markup is simple and
# well-formed enough for targeted regexes)
# --------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_COMMENT = re.compile(r"<!--.*?-->", re.S)

# Detail pages render every extra field as:
#   <h1 class="p-1 bg-gray-200 text-left">Label</h1>
#   <p class="m-2 text-justify">value</p>
_DETAIL_FIELD = re.compile(
    r'<h1[^>]*class="[^"]*bg-gray-200[^"]*"[^>]*>(.*?)</h1>\s*<p[^>]*>(.*?)</p>',
    re.S,
)

# Index pages render one card per entry:
#   <a href=".../manufacturer/products/379"> <div class="card-body"> <h5>Name</h5>
#   <span> 3 Products, 1 Ingredients </span> </div> </a>
# Deliberately tolerant: the exact whitespace between tags varies per page and
# counts are sometimes wrapped in an HTML comment.
_INDEX_CARD = re.compile(
    r'href="[^"]*?/(?P<kind>manufacturer|applicant|ingredient)/products/(?P<id>\d+)"[^>]*>'
    r'.*?<h5>(?P<name>.*?)</h5>(?P<rest>.*?)</div>\s*</a>',
    re.S,
)

_PAGINATION_PAGE = re.compile(r"[?&]page=(\d+)")

# ATC codes on /atc-classification, e.g. "A01AA - Caries prophylactic agents".
# The separator is a mangled en-dash in the page, so any run of non-word
# characters or spaces is accepted.
_ATC_LABEL = re.compile(
    r"^(?P<code>[A-Z]\d{0,2}[A-Z]{0,2})\s*[^\w]{1,4}\s*(?P<desc>.+)$"
)
_ATC_ANCHOR = re.compile(r'class="pain-link"[^>]*>(.*?)</a>', re.S)

# ATC depth by code length: 1 char = anatomical group, 3 = therapeutic area,
# 4 = therapeutic subgroup, 5 = chemical subgroup.
ATC_LEVEL_BY_LENGTH = {1: 1, 3: 2, 4: 3, 5: 4}


def strip_tags(fragment: str) -> str:
    import html as _html

    text = _TAG.sub(" ", fragment or "")
    text = _html.unescape(text).replace("\xa0", " ")
    return _WS.sub(" ", text).strip()


def parse_detail_fields(html: str) -> Dict[str, str]:
    """Extract the label/value pairs from a product detail page."""
    out: Dict[str, str] = {}
    for label_html, value_html in _DETAIL_FIELD.findall(html):
        label = strip_tags(label_html)
        value = strip_tags(value_html)
        if label:
            out[label] = value
    return out


def parse_index_cards(html: str) -> List[Dict[str, Any]]:
    """Extract id/name/counts triples from a paginated index page."""
    cards: List[Dict[str, Any]] = []
    for match in _INDEX_CARD.finditer(html):
        counts = strip_tags(_COMMENT.sub(" ", match.group("rest")))
        products = re.search(r"([\d,]+)\s*Products", counts, re.I)
        ingredients = re.search(r"([\d,]+)\s*Ingredients", counts, re.I)
        cards.append({
            "kind": match.group("kind"),
            "id": int(match.group("id")),
            "name": strip_tags(match.group("name")),
            "products": int(products.group(1).replace(",", "")) if products else None,
            "ingredients": int(ingredients.group(1).replace(",", "")) if ingredients else None,
        })
    return cards


def parse_atc_classification(html: str) -> List[Dict[str, str]]:
    """Extract the ATC hierarchy as a flat list of code/level/description rows.

    Comments are stripped first: many anchors on this page sit inside HTML
    comments and would otherwise appear twice.
    """
    clean = _COMMENT.sub(" ", html)

    entries: List[Dict[str, str]] = []
    seen: set = set()
    for anchor in _ATC_ANCHOR.findall(clean):
        label = strip_tags(anchor)
        match = _ATC_LABEL.match(label)
        if not match:
            continue
        code = match.group("code")
        if code in seen:
            continue
        seen.add(code)
        entries.append({
            "ATC Code": code,
            "ATC Level": ATC_LEVEL_BY_LENGTH.get(len(code), ""),
            "Description": match.group("desc").strip(),
        })
    return entries


def parse_total_pages(html: str) -> int:
    pages = [int(n) for n in _PAGINATION_PAGE.findall(html)]
    return max(pages) if pages else 1


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class GreenbookClient:
    def __init__(
        self,
        base_url: str = config.BASE_URL,
        timeout: int = config.REQUEST_TIMEOUT,
        max_retries: int = config.MAX_RETRIES,
        page_size: int = config.PAGE_SIZE,
        delay: float = config.REQUEST_DELAY,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.page_size = page_size
        self.delay = delay
        self.draw = 0

        self.session = requests.Session()
        self.session.headers.update(config.AJAX_HEADERS)

    # -- low level ---------------------------------------------------------
    def _request(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ) -> requests.Response:
        """GET with retries. Raises on persistent failure."""
        active = session or self.session
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = active.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code in config.RETRY_STATUS_CODES:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                return response

            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                wait = min(
                    config.BACKOFF_CAP, config.BACKOFF_FACTOR ** (attempt - 1)
                ) + random.uniform(0, 0.75)
                log.warning(
                    "attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise GreenbookError(f"request failed after {self.max_retries} attempts: {last_error}")

    def _get_json(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request(self.base_url, params=params)
        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            raise GreenbookError(
                "expected JSON but got "
                f"'{content_type or 'unknown'}'. The AJAX header is most likely missing."
            )
        payload = response.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise GreenbookError(f"unexpected payload shape: {list(payload)[:10]}")
        return payload

    # -- bulk product table -------------------------------------------------
    def fetch_page(self, start: int = 0, length: Optional[int] = None) -> Dict[str, Any]:
        if self.delay and start:
            time.sleep(self.delay)

        self.draw += 1
        params: Dict[str, Any] = {
            "draw": self.draw,
            "start": start,
            "length": self.page_size if length is None else length,
            "search[value]": "",
            "search[regex]": "false",
        }
        return self._get_json(params)

    def count(self) -> Dict[str, int]:
        payload = self._get_json({"draw": self.draw + 1, "start": 0, "length": 1})
        self.draw += 1
        return {
            "total": int(payload.get("recordsTotal", 0)),
            "filtered": int(payload.get("recordsFiltered", 0)),
        }

    # -- detail pages -------------------------------------------------------
    def fetch_detail(self, product_id: int) -> Dict[str, str]:
        url = config.DETAIL_URL_TEMPLATE.format(product_id=product_id)
        response = self._request(url, headers=config.REQUEST_HEADERS)
        return parse_detail_fields(response.text)

    def fetch_details(
        self,
        product_ids: List[int],
        workers: int = config.DETAIL_WORKERS,
        delay: float = config.DETAIL_DELAY,
        cache_path: Optional[str] = None,
        progress_every: int = 250,
        settle_time: float = 5.0,
        dead_file: Optional[str] = None,
    ) -> Dict[int, Dict[str, str]]:
        """Fetch detail pages concurrently, with an on-disk resume cache.

        Returns ``{product_id: {label: value}}``. Products whose detail page
        cannot be retrieved are simply absent from the result, so one bad page
        never aborts the crawl. Two full passes are made: the site returns
        spurious HTTP 500s under concurrency, and a second attempt after a
        short settle recovers nearly all of them.

        Ids that still fail both passes are recorded in ``dead_file`` and skipped
        on later runs. A small number of products have a permanently broken
        detail page (HTTP 500 on every request); retrying them forever costs
        minutes and never succeeds.
        """
        cached = self._load_detail_cache(cache_path)
        results: Dict[int, Dict[str, str]] = dict(cached)

        dead = set(self._load_dead_ids(dead_file))
        if dead:
            log.info("  skipping %s ids previously recorded as permanently broken", len(dead))

        outstanding = [
            pid for pid in dict.fromkeys(product_ids)
            if pid not in results and pid not in dead
        ]
        if not outstanding:
            log.info("Detail crawl: all %s products cached or known-broken", len(results))
            return results

        log.info(
            "Detail crawl: %s to fetch (%s cached), %s workers",
            len(outstanding), len(results), workers,
        )

        for attempt in (1, 2):
            if not outstanding:
                break
            if attempt > 1:
                log.info(
                    "  second pass for %s products that failed; settling %.0fs",
                    len(outstanding), settle_time,
                )
                time.sleep(settle_time)

            outstanding = self._detail_pass(
                outstanding, results, workers, delay, cache_path,
                progress_every, label=f"pass {attempt}",
            )

        if outstanding:
            log.warning(
                "  %s products unreachable after 2 passes; recording them as "
                "permanently broken: %s",
                len(outstanding), outstanding[:10],
            )
            if dead_file:
                dead.update(outstanding)
                self._write_dead_ids(dead_file, dead)

        return results

    @staticmethod
    def _load_dead_ids(dead_file: Optional[str]) -> List[int]:
        if not dead_file:
            return []
        try:
            with open(dead_file, "r", encoding="utf-8") as handle:
                return [int(x) for x in json.load(handle)]
        except (FileNotFoundError, ValueError, TypeError):
            return []

    @staticmethod
    def _write_dead_ids(dead_file: str, dead: set) -> None:
        try:
            with open(dead_file, "w", encoding="utf-8") as handle:
                json.dump(sorted(dead), handle, indent=2)
        except OSError as exc:
            log.warning("Could not record permanently broken ids: %s", exc)

    def _detail_pass(
        self,
        todo: List[int],
        results: Dict[int, Dict[str, str]],
        workers: int,
        delay: float,
        cache_path: Optional[str],
        progress_every: int,
        label: str = "",
    ) -> List[int]:
        """One concurrent sweep over ``todo``; returns the ids that failed."""
        lock = threading.Lock()
        local = threading.local()
        completed = 0
        failed = 0
        started = time.time()

        def worker(product_id: int) -> Optional[Dict[str, str]]:
            nonlocal completed, failed
            if not hasattr(local, "session"):
                local.session = requests.Session()
                local.session.headers.update(config.REQUEST_HEADERS)
            try:
                fields = self.fetch_detail(product_id)
                out: Optional[Dict[str, str]] = fields
            except Exception as exc:
                log.debug("detail %s failed: %s", product_id, exc)
                out = None

            if delay:
                time.sleep(delay + random.uniform(0, delay))

            with lock:
                completed += 1
                if out is not None:
                    results[product_id] = out
                else:
                    failed += 1
                if completed % progress_every == 0 or completed == len(todo):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed else 0
                    eta = (len(todo) - completed) / rate if rate else 0
                    log.info(
                        "  %s details %d/%d (%.1f/s, ~%.0f min left, %s failed)",
                        label, completed, len(todo), rate, eta / 60, failed,
                    )
                    if cache_path:
                        self._write_detail_cache(cache_path, results)
            return out

        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(worker, todo))

        if cache_path:
            self._write_detail_cache(cache_path, results)

        return [pid for pid, fields in zip(todo, fetched) if fields is None]

    @staticmethod
    def _load_detail_cache(cache_path: Optional[str]) -> Dict[int, Dict[str, str]]:
        if not cache_path:
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = {}
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        cached[int(entry["product_id"])] = entry["fields"]
                    except (ValueError, KeyError, TypeError):
                        continue
                return cached
        except FileNotFoundError:
            return {}

    @staticmethod
    def _write_detail_cache(cache_path: str, results: Dict[int, Dict[str, str]]) -> None:
        """Rewrite the cache atomically so an interrupted run never corrupts it."""
        temporary = f"{cache_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            for product_id, fields in results.items():
                handle.write(
                    json.dumps({"product_id": product_id, "fields": fields}) + "\n"
                )
        os.replace(temporary, cache_path)

    # -- reference index pages ---------------------------------------------
    def fetch_index_page(self, url: str, page: int = 1) -> str:
        if page > 1:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}page={page}"
        response = self._request(url, headers=config.REQUEST_HEADERS)
        return response.text

    def fetch_index(self, url: str, label: str) -> List[Dict[str, Any]]:
        """Crawl every page of a reference index. Returns the merged card list."""
        first = self.fetch_index_page(url, 1)
        total_pages = parse_total_pages(first)
        log.info("  %s: %s page(s)", label, total_pages)

        seen: Dict[int, Dict[str, Any]] = {}
        for card in parse_index_cards(first):
            seen[card["id"]] = card

        for page in range(2, total_pages + 1):
            time.sleep(self.delay)
            html = self.fetch_index_page(url, page)
            found = parse_index_cards(html)
            if not found:
                break
            for card in found:
                seen[card["id"]] = card
            log.info("  %s: page %s/%s, %s entries", label, page, total_pages, len(seen))

        return list(seen.values())

    def close(self) -> None:
        self.session.close()
