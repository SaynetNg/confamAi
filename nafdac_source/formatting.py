"""Terminal and prompt formatting for NAFDAC results.

Kept apart from `verify.py` so the presentation of a result can change without
touching the lookup logic, and so the same result can be rendered three ways:

    format_verification()    a table for a terminal
    VerificationResult.render()   a plain block for the model's context
    format_search_results()  a ranked list for /search

All three are plain text with no markdown syntax, because the model's block is
fed straight into a prompt and a stray `**` would reach the user verbatim.
"""

from typing import Iterable, List, Optional

from .records import ProductRecord

# Terminal width for the boxes below. 78 keeps the table inside an 80-column
# window, which is still the common case in cmd.exe.
WIDTH = 78

NAFDAC_HELP = """NAFDAC verification
  /verify <number>   Check one NAFDAC number, e.g. /verify A4-100160
  /search <text>     Search products by name or ingredient, e.g. /search malaria
  /nafdac            Show which data sources are available

You can also just ask in your own words. "Is A4-100160 genuine?" is
verified against the greenbook before N-ATLaS answers, and the model is
given the verified values so it cannot invent them."""


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _label_value(label: str, value: str, width: int = 16) -> str:
    return "  %-*s %s" % (width, label + ":", value)


def _truncate(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _status_line(record: ProductRecord) -> str:
    """One line describing whether a product is currently authorised."""
    authorised = record.authorised
    if authorised is True:
        state = "AUTHORISED"
    elif authorised is False:
        state = "NOT AUTHORISED"
    else:
        state = "STATUS NOT RECORDED"
    return f"{record.status or 'Unknown'} - {state}"


def format_verification(result, show_data: bool = True) -> str:
    """Render a VerificationResult as a terminal table.

    `show_data=False` gives just the verdict, which is what `/nafdac` needs when
    it is summarising several lookups.
    """
    if not result.searched:
        return (
            "No NAFDAC number found. It looks like A4-100160 or 04-3877 - the "
            "number NAFDAC printed on the pack."
        )

    if not result.found:
        return format_not_found(result)

    lines: List[str] = []

    if result.ambiguous:
        lines.append(
            f"NAFDAC {result.nrn} - {len(result.matches)} different products "
            "share this number:"
        )
    else:
        verdict = (
            "AUTHORISED"
            if result.authorised is True
            else "NOT CURRENTLY AUTHORISED"
            if result.authorised is False
            else "STATUS NOT RECORDED"
        )
        lines.append(f"NAFDAC {result.nrn} - {verdict}")
    lines.append(_rule("="))

    for record in result.matches:
        lines.extend(_format_record(record, show_data))
        lines.append("")

    if result.greenbook_url:
        lines.append(f"  Greenbook: {result.greenbook_url}")
    lines.append(f"  Confidence: {result.confidence}")

    failed = _format_failures(result)
    if failed:
        lines.append(failed)

    for note in result.notes:
        lines.append(f"  Note: {note}")

    return "\n".join(lines).rstrip()


def _format_record(record: ProductRecord, show_data: bool) -> List[str]:
    """The per-product block inside a verification result."""
    lines: List[str] = []
    name = record.product_name or "(name not recorded)"
    if result_is_duplicate(record):
        name += "  [duplicate entry]"
    lines.append(f"  {name}")
    lines.append(_label_value("Number", record.nrn or "not recorded"))

    if not show_data:
        lines.append(_label_value("Status", _status_line(record)))
        return lines

    if record.active_ingredients:
        lines.append(_label_value("Ingredients", _truncate(record.active_ingredients, 60)))
    if record.strengths:
        lines.append(_label_value("Strength", _truncate(record.strengths, 60)))
    if record.form or record.route:
        shape = " / ".join(part for part in (record.form, record.route) if part)
        lines.append(_label_value("Form", shape))
    if record.product_category:
        lines.append(_label_value("Category", record.product_category))
    if record.marketing_category:
        lines.append(_label_value("Marketing", record.marketing_category))
    if record.applicant_name:
        lines.append(_label_value("Applicant", _truncate(record.applicant_name, 60)))
    if record.manufacturer_name or record.manufacturer_country:
        who = " / ".join(
            part for part in (record.manufacturer_name, record.manufacturer_country)
            if part
        )
        lines.append(_label_value("Manufacturer", _truncate(who, 60)))
    if record.pack_size:
        lines.append(_label_value("Pack size", _truncate(record.pack_size, 60)))
    if record.approval_date:
        lines.append(_label_value("Approved", record.approval_date))
    if record.expiry_date:
        lines.append(_label_value("Expires", record.expiry_date))
    if record.atc_code:
        lines.append(_label_value("ATC code", record.atc_code))
    lines.append(_label_value("Status", _status_line(record)))
    return lines


def result_is_duplicate(record: ProductRecord) -> bool:
    return bool(getattr(record, "duplicate_marker", False))


def format_not_found(result) -> str:
    """Render a miss without over-claiming.

    The wording is deliberate: not being in the greenbook is not the same as
    being fake, and a user who is told a real product is counterfeit because of
    a stale export will not trust the next answer either.
    """
    lines = [f"NAFDAC {result.nrn} - not found in the greenbook.", _rule("=")]
    lines.append("")
    lines.append("  This number is not in any data source N-ATLaS can reach.")
    if result.sources_checked:
        lines.append(f"  Checked: {', '.join(result.sources_checked)}")
    failed = _format_failures(result)
    if failed:
        lines.append(failed)
    lines.append("")
    lines.append("  That is not proof the product is fake. The export may be")
    lines.append("  out of date, or the number may be mistyped. Check the")
    lines.append("  number against the pack, and try the search command if the")
    lines.append("  product has a different number.")
    return "\n".join(lines)


def _format_failures(result) -> str:
    if not result.sources_failed:
        return ""
    parts = [f"{name} ({why})" for name, why in result.sources_failed.items()]
    return f"  Unavailable: {'; '.join(parts)}"


def format_search_results(
    records: Iterable[ProductRecord], query: str, limit: int = 20
) -> str:
    """Render a free-text search as a ranked list."""
    records = list(records)
    if not records:
        return (
            f"No product matched {query!r}. Try a shorter word, or an active "
            "ingredient such as 'paracetamol' or 'artemether'."
        )

    shown = records[: max(1, limit)]
    lines = [
        f"{len(records)} product(s) matched {query!r}"
        + (f" (showing {len(shown)})" if len(records) > len(shown) else "")
        + ":",
        _rule("="),
    ]
    for record in shown:
        name = _truncate(record.product_name or "(name not recorded)", 46)
        number = record.nrn or "no number"
        lines.append(f"  {name}  [{number}]")
        detail = " / ".join(
            part
            for part in (
                _truncate(record.active_ingredients, 40),
                _truncate(record.applicant_name, 34),
            )
            if part
        )
        if detail:
            lines.append(f"    {detail}")
    lines.append("")
    lines.append("  Verify one with: /verify <number>")
    return "\n".join(lines)


def format_status(sources: Iterable) -> str:
    """Render which sources are available, for /nafdac."""
    sources = list(sources)
    lines = ["NAFDAC data sources", _rule("=")]
    for source in sources:
        try:
            description = source.describe()
        except Exception as exc:
            description = f"error: {exc}"
        lines.append(f"  {source.name:<9} {description}")
    lines.append("")
    lines.append("  The PostgreSQL source carries superseded and withdrawn")
    lines.append("  products. Without it, a number that has since been")
    lines.append("  withdrawn will not be distinguishable from one that never")
    lines.append("  existed.")
    return "\n".join(lines)


def source_status(sources: Optional[Iterable]) -> str:
    return format_status(sources or [])
