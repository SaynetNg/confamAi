#!/usr/bin/env python3
"""Weekly NAFDAC Greenbook refresh: scrape the live site, then load PostgreSQL.

This is the single command a weekly scheduler runs:

    python weekly_nafdac_refresh.py

It performs the two documented steps in order:

    1. nafdac_scraper/scraper.py    -> nafdac_scraper/nafdac_greenbook.csv
    2. nafdac_source.load_postgres  -> upsert into nafdac_products and archive
                                       rows that have left the export into
                                       nafdac_product_history

The database connection string is read from NAFDAC_DATABASE_URL (or DATABASE_URL)
by the loader; nothing here prints it, and a missing value fails loudly instead
of silently skipping the load.

Each step runs as a subprocess so a failure carries its real exit code, the two
never share interpreter state, and the output streams straight through to the
scheduler's log.

Exit codes:
    0  both steps succeeded
    1  the scrape produced no data, or the load failed
    2  the runner was misconfigured (scraper folder missing)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRAPER_DIR = ROOT / "nafdac_scraper"


def _step(label: str, command: list, cwd: Path) -> int:
    """Run one step, streaming its output. Returns the process exit code."""
    line = "=" * 74
    print(line, flush=True)
    print(f"[weekly-refresh] {label}", flush=True)
    print(f"[weekly-refresh] cwd: {cwd}", flush=True)
    print(f"[weekly-refresh] $ {' '.join(command)}", flush=True)
    print(line, flush=True)
    started = time.time()
    try:
        completed = subprocess.run(command, cwd=str(cwd))
    except FileNotFoundError as exc:
        print(f"[weekly-refresh] could not start: {exc}", file=sys.stderr, flush=True)
        return 127
    elapsed = time.time() - started
    print(
        f"[weekly-refresh] {label} finished in {elapsed / 60:.1f} min "
        f"(exit {completed.returncode})",
        flush=True,
    )
    return completed.returncode


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape the NAFDAC Greenbook and load it into PostgreSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="load the existing CSV without re-scraping",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="scrape only; do not touch the database",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="skip the ~8.9k detail pages (faster; 3 columns stay empty)",
    )
    parser.add_argument(
        "--with-xlsx",
        action="store_true",
        help="also write the multi-sheet Excel export (not needed for the database)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="scrape at most N products (0 = full crawl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="rows per database INSERT batch",
    )
    parser.add_argument(
        "--dry-run-load",
        action="store_true",
        help="read and validate the CSV without writing to the database",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not SCRAPER_DIR.is_dir():
        print(f"[weekly-refresh] scraper folder not found: {SCRAPER_DIR}", file=sys.stderr)
        return 2

    if args.skip_scrape:
        print("[weekly-refresh] scrape skipped (--skip-scrape)", flush=True)
    else:
        scrape_cmd = [
            sys.executable,
            str(SCRAPER_DIR / "scraper.py"),
            "--output-dir",
            str(SCRAPER_DIR),
        ]
        if args.no_details:
            scrape_cmd.append("--no-details")
        if not args.with_xlsx:
            # The database is the target; the workbook costs ~40 s and is not read
            # by the loader.
            scrape_cmd.append("--no-xlsx")
        if args.limit:
            scrape_cmd.extend(["--limit", str(args.limit)])

        code = _step("scrape", scrape_cmd, SCRAPER_DIR)
        if code != 0:
            print(
                f"[weekly-refresh] scrape failed (exit {code}); not loading.",
                file=sys.stderr,
                flush=True,
            )
            print(
                "[weekly-refresh] hint: if the log shows CERTIFICATE_VERIFY_FAILED, "
                "NAFDAC's certificate has expired; set NAFDAC_SCRAPER_INSECURE_TLS=1 "
                "for this run (see nafdac_scraper/README.md).",
                file=sys.stderr,
                flush=True,
            )
            return 1

    if args.skip_load:
        print("[weekly-refresh] load skipped (--skip-load)", flush=True)
        return 0

    load_cmd = [
        sys.executable,
        "-m",
        "nafdac_source.load_postgres",
        "--batch-size",
        str(max(1, args.batch_size)),
    ]
    if args.dry_run_load:
        load_cmd.append("--dry-run")

    code = _step("load", load_cmd, ROOT)
    if code != 0:
        print(f"[weekly-refresh] load failed (exit {code}).", file=sys.stderr, flush=True)
        return 1

    print(
        "[weekly-refresh] done: greenbook scraped and loaded into PostgreSQL.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
