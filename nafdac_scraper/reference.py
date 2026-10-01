"""Reference data from the site's index pages.

These are separate browsable pages, not endpoints, and each is paginated:

* ``/manufacturers``  - 29 pages, 50 cards per page (~1,450 manufacturers)
* ``/applicants``     - 34 pages
* ``/ingredients``    - 48 pages
* ``/productCategories`` - single page, 6 categories
* ``/atc-classification``  - single page, ~640 KB of the ATC hierarchy

Each becomes its own sheet in the workbook so the product data is not polluted
with unrelated rows.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import config
from client import GreenbookClient, parse_atc_classification, strip_tags

log = logging.getLogger(__name__)


def fetch_manufacturers(client: GreenbookClient) -> List[Dict[str, Any]]:
    cards = client.fetch_index(config.MANUFACTURERS_URL, "manufacturers")
    return [
        {
            "Manufacturer ID": card["id"],
            "Manufacturer Name": card["name"],
            "Products": card["products"],
            "Ingredients": card["ingredients"],
        }
        for card in cards
    ]


def fetch_applicants(client: GreenbookClient) -> List[Dict[str, Any]]:
    cards = client.fetch_index(config.APPLICANTS_URL, "applicants")
    return [
        {
            "Applicant ID": card["id"],
            "Applicant Name": card["name"],
            "Products": card["products"],
            "Ingredients": card["ingredients"],
        }
        for card in cards
    ]


def fetch_ingredients(client: GreenbookClient) -> List[Dict[str, Any]]:
    cards = client.fetch_index(config.INGREDIENTS_URL, "ingredients")
    return [
        {
            "Ingredient ID": card["id"],
            "Active Ingredients": card["name"],
        }
        for card in cards
    ]


def fetch_product_categories(client: GreenbookClient) -> List[Dict[str, Any]]:
    """The six product classes the site itself publishes.

    Note: the counts rendered on this page are unreliable. Disinfectants
    (id 12) advertises brands on the page but its product list is empty in the
    API, so the rendered numbers are decorative. The live counts come from the
    bulk crawl instead.
    """
    html = client.fetch_index_page(config.PRODUCT_CATEGORIES_URL, 1)
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r'href="[^"]*productCategory/products/(\d+)"[^>]*>.*?<h5>(.*?)</h5>'
        r'(?P<counts>.*?)</div>\s*</a>',
        re.S,
    )
    for category_id, name_html, counts_html in pattern.findall(html):
        rows.append({
            "Product Category ID": int(category_id),
            "Product Category": strip_tags(name_html),
            "Counts As Published": strip_tags(counts_html),
        })
    return rows


def fetch_atc_classification(client: GreenbookClient) -> List[Dict[str, str]]:
    """The ATC hierarchy as a flat list of codes with descriptions.

    The page renders a nested tree of headings; each ATC code is a bolded span
    followed by its description.
    """
    html = client.fetch_index_page(config.ATC_URL, 1)
    return parse_atc_classification(html)
