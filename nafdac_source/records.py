"""The normalised product record returned by every source.

All sources return the same shape so a CSV row, an XLSX row and a PostgreSQL row
are directly comparable. The 34 greenbook columns are kept, because a NAFDAC
verification answer often needs the applicant, the manufacturer country or the
marketing category, and those are the fields a user actually asks about.

`Product ID` is the greenbook's own primary key and is always unique. NRN is
*not* unique: 257 of the 8935 rows share an NRN with another row, so a lookup
can legitimately return several records. Callers must handle `len(matches) > 1`
rather than assuming one row.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

STATUS_ACTIVE = "Active"
STATUS_INACTIVE = "Inactive"
STATUS_UNKNOWN = "Unknown"

# The fields a human verification answer is built from, in the order they are
# shown. Everything else in the row is available on the record but not printed.
CORE_FIELDS = (
    "nrn",
    "product_name",
    "active_ingredients",
    "strengths",
    "form",
    "route",
    "product_category",
    "status",
    "marketing_category",
    "applicant_name",
    "manufacturer_name",
    "manufacturer_country",
    "approval_date",
    "expiry_date",
    "atc_code",
)

# CSV / XLSX header -> record attribute. The greenbook uses spaced, capitalised
# headers; the record uses snake_case. This is the only place that mapping
# lives, so adding a source means adding rows here and nowhere else.
_COLUMN_MAP: Dict[str, str] = {
    "Product Name": "product_name",
    "Active Ingredients": "active_ingredients",
    "Product Category": "product_category",
    "NRN": "nrn",
    "Form": "form",
    "ROA": "route",
    "Strengths": "strengths",
    "Applicant Name": "applicant_name",
    "Approval Date": "approval_date",
    "Status": "status",
    "Product ID": "product_id",
    "Product Category ID": "product_category_id",
    "ATC Code": "atc_code",
    "Ingredient ID": "ingredient_id",
    "Ingredient Synonym": "ingredient_synonym",
    "Form ID": "form_id",
    "Route ID": "route_id",
    "Product Description": "product_description",
    "Composition": "composition",
    "Pack Size": "pack_size",
    "Applicant ID": "applicant_id",
    "Applicant Address": "applicant_address",
    "Manufacturer ID": "manufacturer_id",
    "Manufacturer Name": "manufacturer_name",
    "Manufacturer Country": "manufacturer_country",
    "Country ID": "country_id",
    "Marketing Category ID": "marketing_category_id",
    "Marketing Category": "marketing_category",
    "Expiry Date": "expiry_date",
    "SMPC URL": "smpc_url",
    "Biosimilar": "biosimilar",
    "Created At": "created_at",
    "Updated At": "updated_at",
    "Source URL": "source_url",
}

# Reverse mapping, for the PostgreSQL writer.
FIELD_COLUMNS: Dict[str, str] = {v: k for k, v in _COLUMN_MAP.items()}

# One pass over the row, built once. Both the greenbook header form
# (`Product Name`) and the record attribute form (`product_name`) are looked up
# in the same iteration, so a row never needs a second sweep.
_LOOKUP: Dict[str, str] = {}
for _column, _attribute in _COLUMN_MAP.items():
    _LOOKUP[_column] = _attribute
    _LOOKUP.setdefault(_attribute, _attribute)

# Fields that are stored as integers in PostgreSQL.
_INT_FIELDS = frozenset(
    {
        "product_id",
        "product_category_id",
        "ingredient_id",
        "form_id",
        "route_id",
        "applicant_id",
        "manufacturer_id",
        "country_id",
        "marketing_category_id",
    }
)


def _clean(value: Any) -> str:
    """Trim, and turn pandas/psycopg NA sentinels into a real empty string."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "n/a", "<na>"}:
        return ""
    return text


def classify_status(value: Any) -> str:
    """Map the greenbook `Status` column onto a small known set.

    The published data only ever contains `Active` or `Inactive`, but a
    superseded row loaded from PostgreSQL can carry anything, so unknown values
    are preserved rather than being flattened to Active.
    """
    text = _clean(value)
    lowered = text.lower()
    if lowered in {"active", "approved", "registered", "valid", "current"}:
        return STATUS_ACTIVE
    if lowered in {
        "inactive",
        "suspended",
        "withdrawn",
        "recalled",
        "cancelled",
        "canceled",
        "expired",
        "superseded",
        "revoked",
    }:
        return STATUS_INACTIVE
    return STATUS_UNKNOWN if not text else text


def is_authorised(status: str) -> Optional[bool]:
    """Whether the record counts as currently authorised.

    `None` when the status is not one this code recognises, so a caller never
    claims a product is authorised on the strength of an unknown status.
    """
    if status == STATUS_ACTIVE:
        return True
    if status == STATUS_INACTIVE:
        return False
    return None


@dataclass
class ProductRecord:
    """One greenbook product, normalised across all sources."""

    nrn: str = ""
    product_id: Optional[int] = None
    product_name: str = ""
    active_ingredients: str = ""
    strengths: str = ""
    form: str = ""
    route: str = ""
    product_category: str = ""
    status: str = STATUS_UNKNOWN
    marketing_category: str = ""
    applicant_name: str = ""
    applicant_address: str = ""
    manufacturer_name: str = ""
    manufacturer_country: str = ""
    approval_date: str = ""
    expiry_date: str = ""
    atc_code: str = ""
    product_category_id: Optional[int] = None
    ingredient_id: Optional[int] = None
    ingredient_synonym: str = ""
    form_id: Optional[int] = None
    route_id: Optional[int] = None
    product_description: str = ""
    composition: str = ""
    pack_size: str = ""
    applicant_id: Optional[int] = None
    manufacturer_id: Optional[int] = None
    country_id: Optional[int] = None
    marketing_category_id: Optional[int] = None
    smpc_url: str = ""
    biosimilar: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_url: str = ""

    # Provenance. `sources` is every source that confirmed this row, in the
    # order they were consulted.
    sources: List[str] = field(default_factory=list)

    @property
    def nrn_key(self) -> str:
        from .extract import normalize_nrn_key

        return normalize_nrn_key(self.nrn)

    @property
    def authorised(self) -> Optional[bool]:
        return is_authorised(self.status)

    @property
    def duplicate_marker(self) -> bool:
        """True for the 268 rows the scraper deliberately kept.

        The scraper retains rows whose name contains `(duplicate)`, so a user
        asking about one of these should be told it is a duplicate entry rather
        than shown as two separate products.
        """
        return "(duplicate" in self.product_name.lower()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["nrn_key"] = self.nrn_key
        data["authorised"] = self.authorised
        return data

    def field(self, name: str) -> str:
        """Read a field by record attribute name, safely."""
        return _clean(getattr(self, name, ""))


def _coerce_int(value: Any) -> Optional[int]:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def record_from_row(row: Dict[str, Any], source: str = "") -> ProductRecord:
    """Build a `ProductRecord` from a CSV/XLSX/psycopg row.

    Accepts either the greenbook's spaced headers (`Product Name`) or the
    record's snake_case attribute names, so the same function serves the file
    loaders and the database reader.
    """
    record = ProductRecord()
    for key, value in row.items():
        attribute = _LOOKUP.get(key)
        if attribute is None or value is None:
            continue
        if attribute in _INT_FIELDS:
            setattr(record, attribute, _coerce_int(value))
        else:
            setattr(record, attribute, _clean(value))

    if record.status:
        record.status = classify_status(record.status)
    if source:
        record.sources = [source]
    return record


def merge_records(base: ProductRecord, other: ProductRecord) -> ProductRecord:
    """Overlay `other` onto `base` without discarding anything already known.

    Earlier sources win: a field already set on `base` is kept, and only empty
    fields are filled from `other`. This is what makes the local CSV/XLSX files
    able to fill gaps in a thin PostgreSQL row without overwriting it.
    """
    merged = ProductRecord(**{
        key: value
        for key, value in asdict(base).items()
        if key != "sources"
    })
    for attribute in FIELD_COLUMNS:
        current = getattr(merged, attribute, None)
        incoming = getattr(other, attribute, None)
        if current in (None, ""):
            setattr(merged, attribute, incoming)

    seen: List[str] = []
    for name in list(base.sources) + list(other.sources):
        if name and name not in seen:
            seen.append(name)
    merged.sources = seen
    if merged.status:
        merged.status = classify_status(merged.status)
    return merged


def render_record(record: ProductRecord, indent: str = "  ") -> str:
    """A compact human-readable block for a terminal or a prompt.

    Only non-empty core fields are printed, so a sparse row does not produce a
    wall of blank labels. The output is plain text with no markdown, because it
    is fed straight into the model's context.
    """
    lines: List[str] = []
    title = record.product_name or "(name not recorded)"
    header = f"{record.nrn} - {title}" if record.nrn else title
    lines.append(indent + header)

    authorised = record.authorised
    if authorised is True:
        state = "AUTHORISED"
    elif authorised is False:
        state = "NOT CURRENTLY AUTHORISED"
    else:
        state = "STATUS UNKNOWN"
    lines.append(f"{indent}  Status: {state} ({record.status or 'not recorded'})")

    if record.active_ingredients:
        lines.append(f"{indent}  Active ingredients: {record.active_ingredients}")
    if record.strengths:
        lines.append(f"{indent}  Strength: {record.strengths}")
    if record.form or record.route:
        shape = " / ".join(part for part in (record.form, record.route) if part)
        lines.append(f"{indent}  Form: {shape}")
    if record.product_category:
        lines.append(f"{indent}  Category: {record.product_category}")
    if record.marketing_category:
        lines.append(f"{indent}  Marketing category: {record.marketing_category}")
    if record.applicant_name:
        lines.append(f"{indent}  Applicant: {record.applicant_name}")
    if record.manufacturer_name or record.manufacturer_country:
        who = " / ".join(
            part for part in (record.manufacturer_name, record.manufacturer_country) if part
        )
        lines.append(f"{indent}  Manufacturer: {who}")
    if record.approval_date:
        lines.append(f"{indent}  Approved: {record.approval_date}")
    if record.expiry_date:
        lines.append(f"{indent}  Expires: {record.expiry_date}")
    if record.atc_code:
        lines.append(f"{indent}  ATC code: {record.atc_code}")

    detail = record.duplicate_marker
    if detail:
        lines.append(f"{indent}  Note: the greenbook lists this as a duplicate entry.")
    if record.sources:
        lines.append(f"{indent}  Source: {', '.join(record.sources)}")
    if record.source_url:
        lines.append(f"{indent}  Greenbook page: {record.source_url}")

    return "\n".join(lines)
