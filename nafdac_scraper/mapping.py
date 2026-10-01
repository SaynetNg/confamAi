"""JSON row -> dataset columns, plus merging of detail-page fields.

The bulk JSON already contains most of the dataset. Three fields exist only on
the product detail pages and are merged in afterwards:

    Manufacturer Name, Manufacturer Country, Marketing Category

De-duplication uses ``product_id`` (the true primary key from the backend) and
reports NRN collisions separately instead of silently dropping rows. An earlier
version keyed on NRN, which discarded a handful of genuinely different products
that share an NRN upstream.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, Optional

import config

_WHITESPACE = re.compile(r"\s+")


def _clean(value: Any) -> str:
    """Coerce a JSON value to a clean string; never invent a value."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if v not in (None, ""))
    text = html.unescape(str(value))
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _multiline(value: Any) -> str:
    """Like :func:`_clean` but keeps line breaks, for compositions."""
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _product_name(value: Any) -> str:
    name = _clean(value)
    if config.STRIP_PRODUCT_NAME_MARKERS:
        name = name.replace("#", "").replace("*", "")
        name = _WHITESPACE.sub(" ", name).strip()
    return name


def _id(value: Any) -> str:
    return "" if value is None else str(value).strip()


def extract_record(row: Dict[str, Any]) -> Dict[str, str]:
    """Map one bulk API row to every output column except the detail-only three."""
    ingredient = row.get("ingredient") or {}
    category = row.get("product_category") or {}
    form = row.get("form") or {}
    route = row.get("route") or {}
    applicant = row.get("applicant") or {}

    product_id = row.get("product_id")

    record = {
        # --- core ten ---
        "Product Name": _product_name(row.get("product_name")),
        "Active Ingredients": _clean(
            ingredient.get("ingredient_name") or row.get("ingredient_name")
        ),
        "Product Category": _clean(category.get("name") or row.get("category_name")),
        "NRN": _clean(row.get("NAFDAC")),
        "Form": _clean(form.get("name") or row.get("form_name")),
        "ROA": _clean(route.get("name") or row.get("route_name")),
        "Strengths": _clean(row.get("strength")),
        "Applicant Name": _clean(applicant.get("name") or row.get("applicant_name")),
        "Approval Date": _clean(row.get("approval_date")),
        "Status": _clean(row.get("status")),

        # --- identity ---
        "Product ID": _id(product_id),

        # --- classification ---
        "Product Category ID": _id(row.get("product_category_id")),
        "ATC Code": _clean(row.get("atc")),

        # --- composition ---
        "Ingredient ID": _id(row.get("ingredient_id")),
        "Ingredient Synonym": _clean(
            ingredient.get("synonym") or row.get("synonym")
        ),
        "Form ID": _id(row.get("form_id")),
        "Route ID": _id(row.get("route_id")),

        # --- presentation ---
        "Product Description": _clean(row.get("product_description")),
        "Composition": _multiline(row.get("composition")),
        "Pack Size": _clean(row.get("pack_size")),

        # --- parties ---
        "Applicant ID": _id(row.get("applicant_id")),
        "Applicant Address": _clean(applicant.get("address")),
        "Manufacturer ID": _id(row.get("manufacturer_id")),
        # detail-page fields, populated later by merge_details()
        "Manufacturer Name": "",
        "Manufacturer Country": "",
        "Country ID": _id(row.get("country_id")),

        # --- regulatory ---
        "Marketing Category ID": _id(row.get("marketing_category_id")),
        "Marketing Category": "",
        "Expiry Date": _clean(row.get("expiry_date")),
        "SMPC URL": _clean(row.get("smpc")),
        "Biosimilar": _clean(row.get("biosimilar")),

        # --- metadata ---
        "Created At": _clean(row.get("created_at")),
        "Updated At": _clean(row.get("updated_at")),
        "Source URL": (
            config.DETAIL_URL_TEMPLATE.format(product_id=product_id)
            if product_id not in (None, "")
            else ""
        ),
    }
    return record


def merge_details(
    records: list[Dict[str, str]],
    details: Dict[int, Dict[str, str]],
) -> int:
    """Fill the three detail-only fields. Returns the number merged."""
    merged = 0
    for record in records:
        product_id = record.get("Product ID", "")
        if not product_id:
            continue
        fields = details.get(int(product_id))
        if not fields:
            continue
        record["Manufacturer Name"] = _clean(fields.get("Manufacturer Name"))
        record["Manufacturer Country"] = _clean(fields.get("Manufacturer Country"))
        record["Marketing Category"] = _clean(fields.get("Marketing Category"))
        merged += 1
    return merged


# --------------------------------------------------------------------------
# De-duplication
# --------------------------------------------------------------------------
def record_identity(row: Dict[str, Any]) -> str:
    """Primary key: product_id. Falls back to NRN then name+applicant."""
    if row.get("product_id") not in (None, ""):
        return f"pid:{row['product_id']}"
    record = extract_record(row)
    nrn = record["NRN"].strip().upper()
    if nrn:
        return f"nrn:{nrn}"
    return "nam:{}|{}".format(
        record["Product Name"].upper(), record["Applicant Name"].upper()
    )


def find_nrn_collisions(records: list[Dict[str, str]]) -> list[str]:
    """NRNs used by more than one product_id.

    These are upstream data problems, not duplicates to remove. Reported so the
    user knows which NRNs are ambiguous.
    """
    by_nrn: Dict[str, set] = {}
    for record in records:
        nrn = record["NRN"].strip().upper()
        if not nrn:
            continue
        by_nrn.setdefault(nrn, set()).add(record["Product ID"])

    return sorted(nrn for nrn, ids in by_nrn.items() if len(ids) > 1)


# --------------------------------------------------------------------------
# Backfill for products whose detail page returns HTTP 500
# --------------------------------------------------------------------------
def build_manufacturer_lookup(
    manufacturers: list[Dict[str, Any]]
) -> Dict[str, str]:
    """``manufacturer_id`` -> ``Manufacturer Name``, from the /manufacturers index.

    The index page is the same data the detail page renders, keyed by the same
    id, so joining on it recovers the name without inventing anything.
    """
    lookup: Dict[str, str] = {}
    for row in manufacturers:
        identifier = str(row.get("Manufacturer ID", "")).strip()
        name = _clean(row.get("Manufacturer Name"))
        if identifier and name:
            lookup[identifier] = name
    return lookup


def build_marketing_category_lookup(
    records: list[Dict[str, str]], details: Dict[int, Dict[str, str]]
) -> Dict[str, str]:
    """``marketing_category_id`` -> ``Marketing Category``, learned from products
    whose detail page did load.

    The id space is tiny and one-to-one in practice (0 = unset, 1 = POM,
    2 = OTC, 5 = not applicable), so a single successful page per id is
    sufficient. Ids seen with conflicting names are dropped rather than guessed.
    """
    candidates: Dict[str, set] = {}
    for record in records:
        identifier = record.get("Marketing Category ID", "").strip()
        if not identifier:
            continue
        product_id = record.get("Product ID", "")
        if not product_id.isdigit():
            continue
        fields = details.get(int(product_id))
        if not fields:
            continue
        name = _clean(fields.get("Marketing Category"))
        if name:
            candidates.setdefault(identifier, set()).add(name)

    return {
        identifier: next(iter(names))
        for identifier, names in candidates.items()
        if len(names) == 1
    }


def backfill_detail_fields(
    records: list[Dict[str, str]],
    details: Dict[int, Dict[str, str]],
    manufacturers: list[Dict[str, Any]],
) -> Dict[str, int]:
    """Fill Manufacturer Name and Marketing Category for products whose detail
    page is permanently broken (HTTP 500).

    Only keys the site itself published are used: the manufacturer index and the
    marketing-category ids observed on the pages that did load. Nothing is
    guessed. ``Manufacturer Country`` is left empty because no other source on
    the site carries it.
    """
    stats = {"manufacturer_name": 0, "marketing_category": 0, "unresolved": 0}

    manufacturer_lookup = build_manufacturer_lookup(manufacturers)
    marketing_lookup = build_marketing_category_lookup(records, details)

    for record in records:
        product_id = record.get("Product ID", "")
        if product_id.isdigit() and int(product_id) in details:
            continue  # detail page loaded; nothing to do

        if not record.get("Manufacturer Name"):
            identifier = record.get("Manufacturer ID", "").strip()
            name = manufacturer_lookup.get(identifier)
            if name:
                record["Manufacturer Name"] = name
                stats["manufacturer_name"] += 1

        if not record.get("Marketing Category"):
            identifier = record.get("Marketing Category ID", "").strip()
            name = marketing_lookup.get(identifier)
            if name:
                record["Marketing Category"] = name
                stats["marketing_category"] += 1

        if not record.get("Manufacturer Name"):
            stats["unresolved"] += 1

    return stats
