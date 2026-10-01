"""Scraper for the NAFDAC Greenbook registered product database.

Collects every field the site publishes and exports it as a CSV plus a
multi-sheet workbook.

Sources
-------
1. Bulk product JSON (DataTables server-side) - all products, core fields.
2. Product detail pages - adds Manufacturer Name, Manufacturer Country and
   Marketing Category, which the bulk JSON does not carry.
3. Reference index pages - manufacturers, applicants, ingredients, product
   categories, ATC classification. Each becomes its own sheet.

Usage
-----
    python scraper.py --limit 20      # quick smoke test
    python scraper.py                 # full crawl, all sources
    python scraper.py --no-details    # skip the ~8.9k detail requests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

import config
import mapping
import reference
from client import GreenbookClient, GreenbookError

HERE = Path(__file__).resolve().parent
log = logging.getLogger("nafdac")


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------
# Phase 1: bulk product crawl
# --------------------------------------------------------------------------
def crawl_products(
    client: GreenbookClient, limit: int = 0, max_pages: int = 0
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Walk every page of the product table. Returns (records, raw rows)."""
    counters = client.count()
    total = counters["total"]
    log.info("Bulk API reports %s records", total)

    if limit and limit < total:
        log.info("Limit mode: fetching at most %s records", limit)

    records: List[Dict[str, str]] = []
    raw_rows: List[Dict[str, Any]] = []
    seen: set = set()
    duplicates = 0
    start = 0
    page = 0
    started = time.time()

    while True:
        page += 1

        if max_pages and page > max_pages:
            log.warning("Reached --max-pages=%s, stopping early", max_pages)
            break
        if limit and len(records) >= limit:
            log.info("Reached --limit=%s, stopping early", limit)
            break
        if total and start >= total:
            log.info("End of dataset after %s records", len(records))
            break

        page_length = client.page_size
        if limit:
            page_length = min(page_length, max(1, limit - len(records)))

        payload = client.fetch_page(start=start, length=page_length)
        rows = payload.get("data") or []

        if not rows:
            log.info("Page %s returned no rows; nothing left to fetch", page)
            break

        for row in rows:
            identity = mapping.record_identity(row)
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            raw_rows.append(row)
            records.append(mapping.extract_record(row))

        total = int(payload.get("recordsTotal", total) or total)
        start += len(rows)

        if limit and len(records) >= limit:
            records = records[:limit]
            raw_rows = raw_rows[:limit]

        elapsed = time.time() - started
        rate = len(records) / elapsed if elapsed else 0
        log.info(
            "  page %-4s fetched %-5s rows | collected %-6s / %s (%.1f rows/s)",
            page, len(rows), len(records), total, rate,
        )

    if duplicates:
        log.info("Skipped %s rows with a repeated product_id", duplicates)

    return records, {"raw_rows": len(raw_rows)}


# --------------------------------------------------------------------------
# Phase 2: detail pages
# --------------------------------------------------------------------------
def crawl_details(
    client: GreenbookClient,
    records: List[Dict[str, str]],
    workers: int,
    delay: float,
    cache: Path | None,
    dead_file: Path | None,
) -> Dict[str, int]:
    product_ids = [
        int(r["Product ID"]) for r in records if r.get("Product ID", "").isdigit()
    ]
    if not product_ids:
        log.warning("No product IDs available; skipping detail crawl")
        return {"detail_products": 0, "detail_merged": 0}

    details = client.fetch_details(
        product_ids,
        workers=workers,
        delay=delay,
        cache_path=str(cache) if cache else None,
        dead_file=str(dead_file) if dead_file else None,
    )
    merged = mapping.merge_details(records, details)

    for column in ("Manufacturer Name", "Manufacturer Country", "Marketing Category"):
        filled = sum(1 for r in records if r.get(column))
        log.info("  %-24s populated on %s / %s records", column, filled, len(records))

    return {
        "detail_products": len(details),
        "detail_merged": merged,
        "details": details,
    }


# --------------------------------------------------------------------------
# Phase 3: reference indexes
# --------------------------------------------------------------------------
def crawl_reference(client: GreenbookClient) -> Dict[str, List[Dict[str, Any]]]:
    log.info("Crawling reference indexes")
    data: Dict[str, List[Dict[str, Any]]] = {}
    for key, label, fetcher in [
        ("Manufacturers", "manufacturers", reference.fetch_manufacturers),
        ("Applicants", "applicants", reference.fetch_applicants),
        ("Ingredients", "ingredients", reference.fetch_ingredients),
        ("Product Categories", "categories", reference.fetch_product_categories),
        ("ATC Classification", "atc", reference.fetch_atc_classification),
    ]:
        try:
            rows = fetcher(client)
            data[key] = rows
            log.info("  %-20s %s entries", label, len(rows))
        except Exception as exc:
            log.error("  %-20s failed: %s", label, exc)
            data[key] = []
    return data


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def print_summary(
    records: List[Dict[str, str]], reference_data: Dict[str, List[Dict[str, Any]]]
) -> None:
    frame = pd.DataFrame(records, columns=config.OUTPUT_COLUMNS)
    empty = lambda col: int((frame[col].astype(str).str.strip() == "").sum())

    nrn = frame["NRN"].astype(str).str.strip().str.upper()
    nrn_values = nrn[nrn != ""]
    collisions = mapping.find_nrn_collisions(records)

    print()
    print("=" * 74)
    print("SCRAPE SUMMARY")
    print("=" * 74)
    print(f"Total records:             {len(frame)}")
    print(f"Unique NRNs:               {nrn_values.nunique()}")
    print(f"Unique Product IDs:        {frame['Product ID'].nunique()}")
    print(f"Records with missing Product Name:   {empty('Product Name')}")
    print(f"Records with missing NRN:            {empty('NRN')}")
    print(f"Records with missing Applicant Name: {empty('Applicant Name')}")
    print(f"Records with missing Approval Date:  {empty('Approval Date')}")
    print()
    print(f"NRNs shared by 2+ products (upstream collisions, kept): {len(collisions)}")
    if collisions:
        print(f"  {', '.join(collisions[:12])}{' ...' if len(collisions) > 12 else ''}")
    print()
    print("Completeness per column:")
    for column in config.OUTPUT_COLUMNS:
        missing = empty(column)
        flag = "" if missing == 0 else f"  <- {missing} empty"
        print(f"  {column:<24} {len(frame) - missing:>6}/{len(frame)}{flag}")
    print()
    print("Records per product category (live counts from the API):")
    for category, count in frame["Product Category"].value_counts().items():
        print(f"  {category:<32} {count}")
    print()
    print("Records per status:")
    for status, count in frame["Status"].value_counts().items():
        print(f"  {status:<32} {count}")
    print()
    for name, rows in reference_data.items():
        if rows:
            print(f"Reference sheet {name:<22} {len(rows)} rows")
    print()
    print(f"Sample of up to {config.SAMPLE_ROWS} records:")
    print("=" * 74)
    print(
        frame[
            ["Product Name", "Active Ingredients", "Product Category", "NRN",
             "Form", "ROA", "Strengths", "Applicant Name", "Approval Date", "Status",
             "Manufacturer Name", "Manufacturer Country", "Marketing Category"]
        ]
        .head(config.SAMPLE_ROWS)
        .to_string(index=False)
    )
    print("=" * 74)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def _autosize(writer: Any, sheet_name: str, frame: pd.DataFrame) -> None:
    sheet = writer.sheets[sheet_name]
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, column in enumerate(frame.columns, start=1):
        longest = frame[column].astype(str).str.len().max() if len(frame) else 12
        sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = (
            max(12, min(45, int(longest or 12) + 2))
        )


def save(
    records: List[Dict[str, str]],
    reference_data: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    skip_xlsx: bool = False,
    extra_sheet: List[Dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records, columns=config.OUTPUT_COLUMNS)

    csv_path = output_dir / config.CSV_FILENAME
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Wrote %s (%s rows, %s columns)", csv_path.name, len(frame), len(frame.columns))

    if skip_xlsx:
        log.info("Skipping XLSX export (--no-xlsx)")
        return

    xlsx_path = output_dir / config.XLSX_FILENAME
    sheets: List[Tuple[str, pd.DataFrame]] = [
        ("Products", frame),
        ("Manufacturers", pd.DataFrame(reference_data.get("Manufacturers", []))),
        ("Applicants", pd.DataFrame(reference_data.get("Applicants", []))),
        ("Ingredients", pd.DataFrame(reference_data.get("Ingredients", []))),
        ("Product Categories", pd.DataFrame(reference_data.get("Product Categories", []))),
        ("ATC Classification", pd.DataFrame(reference_data.get("ATC Classification", []))),
    ]
    if extra_sheet:
        sheets.append(("Uncategorised", pd.DataFrame(extra_sheet)))

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, sheet_frame in sheets:
                if sheet_frame.empty and name != "Products":
                    continue
                sheet_frame.to_excel(writer, index=False, sheet_name=name[:31])
                _autosize(writer, name[:31], sheet_frame)
        log.info("Wrote %s (%s sheets)", xlsx_path.name, len(sheets))
    except ImportError:
        log.error(
            "openpyxl is not installed, so the XLSX file was not written. "
            "Run: pip install openpyxl"
        )


def write_manifest(
    path: Path, payload: Dict[str, Any]
) -> None:
    """Write a machine-readable record of what was crawled."""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        log.info("Wrote %s", path.name)
    except OSError as exc:
        log.error("Could not write %s: %s", path.name, exc)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape every field the NAFDAC Greenbook site publishes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT,
                        help="stop after N products (0 = full crawl)")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="stop after N bulk pages (0 = no limit)")
    parser.add_argument("--page-size", type=int, default=config.PAGE_SIZE,
                        help="rows per bulk request")
    parser.add_argument("--delay", type=float, default=config.REQUEST_DELAY,
                        help="seconds between bulk page requests")
    parser.add_argument("--timeout", type=int, default=config.REQUEST_TIMEOUT,
                        help="per-request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=config.MAX_RETRIES,
                        help="attempts per request before giving up")
    parser.add_argument("--detail-workers", type=int, default=config.DETAIL_WORKERS,
                        help="concurrent workers for the detail crawl")
    parser.add_argument("--detail-delay", type=float, default=config.DETAIL_DELAY,
                        help="pause per detail worker between requests")
    parser.add_argument("--detail-cache", type=Path,
                        default=HERE / config.DETAIL_CACHE,
                        help="resume cache for the detail crawl")
    parser.add_argument("--detail-dead-file", type=Path,
                        default=HERE / config.DETAIL_DEAD_FILE,
                        help="ids of products with permanently broken detail pages")
    parser.add_argument("--no-details", action="store_true",
                        help="skip the detail crawl (faster, 3 columns stay empty)")
    parser.add_argument("--no-indexes", action="store_true",
                        help="skip the reference index crawl")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="directory for output files")
    parser.add_argument("--no-xlsx", action="store_true", help="skip the Excel export")
    parser.add_argument("--no-log-file", action="store_true", help="no log file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log_file = None if args.no_log_file else args.output_dir / config.LOG_FILENAME
    setup_logging(verbose=args.verbose, log_file=log_file)

    client = GreenbookClient(
        timeout=args.timeout,
        max_retries=args.max_retries,
        page_size=args.page_size,
        delay=args.delay,
    )

    started = time.time()
    stats: Dict[str, Any] = {}
    details: Dict[int, Dict[str, str]] = {}
    reference_data: Dict[str, List[Dict[str, Any]]] = {}

    try:
        log.info("Phase 1/3: bulk product crawl")
        records, bulk_stats = crawl_products(
            client, limit=args.limit, max_pages=args.max_pages
        )
        stats.update(bulk_stats)

        if args.no_details:
            log.info("Phase 2/3: detail crawl skipped (--no-details)")
            stats.update({"detail_products": 0, "detail_merged": 0})
        else:
            log.info("Phase 2/3: detail pages for Manufacturer/Marketing fields")
            detail_stats = crawl_details(
                client, records,
                workers=args.detail_workers,
                delay=args.detail_delay,
                cache=args.detail_cache,
                dead_file=args.detail_dead_file,
            )
            details = detail_stats.pop("details", {})
            stats.update(detail_stats)

        if args.no_indexes:
            log.info("Phase 3/3: reference indexes skipped (--no-indexes)")
        else:
            log.info("Phase 3/3: reference indexes")
            reference_data = crawl_reference(client)

    except GreenbookError as exc:
        log.error("Scrape aborted: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        return 130
    finally:
        client.close()

    if not records:
        log.error("No records retrieved; nothing to save")
        return 1

    # Products whose detail page is permanently broken can still recover
    # Manufacturer Name and Marketing Category from data the site published
    # elsewhere. Runs after the index crawl because it needs that data.
    if details:
        backfill = mapping.backfill_detail_fields(
            records, details, reference_data.get("Manufacturers", [])
        )
        stats.update(backfill)
        log.info(
            "Backfilled from site reference data: %s manufacturer name(s), "
            "%s marketing category value(s); %s still unresolved",
            backfill["manufacturer_name"],
            backfill["marketing_category"],
            backfill["unresolved"],
        )

    elapsed = time.time() - started
    print_summary(records, reference_data)
    save(records, reference_data, args.output_dir, skip_xlsx=args.no_xlsx)

    stats.update({
        "total_records": len(records),
        "elapsed_seconds": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "details_included": not args.no_details,
        "indexes_included": not args.no_indexes,
        "nrn_collisions": len(mapping.find_nrn_collisions(records)),
        "reference_counts": {k: len(v) for k, v in reference_data.items()},
        "detail_products_unreachable": sorted(
            record["Product ID"]
            for record in records
            if not record.get("Manufacturer Country")
        ) if details else [],
    })
    write_manifest(args.output_dir / "scrape_manifest.json", stats)

    log.info("Done in %.1f min", elapsed / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
