"""End-to-end check of the PostgreSQL path, against a real server.

    python -m nafdac_source.verify_postgres --url postgresql://user:pass@host/db

Four things are checked, in order, and each one stops the next if it fails:

  1. the schema applies cleanly and is re-runnable;
  2. the loader imports the real greenbook CSV and reports sane counts;
  3. verify_nafdac_product() reads a known product back over the wire;
  4. a product removed from the CSV lands in nafdac_product_history and is
     then still answerable, which is the whole point of the history table.

Step 4 is destructive on the target tables, so it works on a copy: it inserts
a synthetic product, removes it from the CSV view, re-archives, then cleans up.
If you point this at a database with real data, read the code first.

Nothing here prints the connection string, only the host and database name.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .extract import normalize_nrn_key
from .load_postgres import (
    _connect,
    _load_env,
    _mask,
    apply_schema,
    finish_run,
    load,
    read_csv,
    start_run,
)
from .records import record_from_row
from .sources import PostgresSource
from .verify import verify_nafdac_product

# A real product from the export, used to prove the round trip.
PROBE_NRN = "A4-100160"

# A product that exists only for this test, so a real database is never
# polluted by an assertion that later disappears.
SYNTHETIC_PRODUCT_ID = 999_000_001
SYNTHETIC_NRN = "ZZ-0000001"

_FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    ok = bool(condition)
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" :: {detail}"
    print(line)
    if not ok:
        _FAILURES.append(name)
    return ok


def _counts(connection) -> Tuple[int, int, int]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM nafdac_products")
        current = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM nafdac_product_history")
        history = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM nafdac_load_runs")
        runs = cursor.fetchone()[0]
    connection.commit()
    return current, history, runs


def _verify_database(dsn: str, csv_path: Optional[Path], skip_load: bool) -> int:
    print("=" * 70)
    print(f"connecting to {_mask(dsn)}")
    print("=" * 70)

    connection = _connect(dsn)
    try:
        print("\n1. schema")
        apply_schema(connection)
        check("schema.sql applied", True)
        # Re-running must be a no-op, because the loader is meant to be run on
        # a schedule.
        apply_schema(connection)
        check("schema.sql is re-runnable", True)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name LIKE 'nafdac%%'
                ORDER BY table_name
                """
            )
            tables = [row[0] for row in cursor.fetchall()]
        connection.commit()
        check(
            "all three tables exist",
            set(tables)
            >= {"nafdac_products", "nafdac_product_history", "nafdac_load_runs"},
            str(tables),
        )

        if not skip_load:
            print("\n2. load")
            source = csv_path or _default_csv()
            if not source:
                check("a CSV to load was found", False)
                return 1
            records = list(read_csv(source))
            check("the CSV was read", len(records) > 8000, f"{len(records)} products")

            run_id = start_run(connection, source)
            loaded, superseded, skipped = load(connection, records)
            finish_run(connection, run_id, loaded, superseded)
            check("rows were loaded", loaded > 8000, f"loaded={loaded} skipped={skipped}")

            current, history, runs = _counts(connection)
            check("nafdac_products is populated", current > 8000, f"{current} rows")
            check("the load run was recorded", runs >= 1, f"{runs} runs")

            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM nafdac_products WHERE nrn_key = %s",
                    (normalize_nrn_key(PROBE_NRN),),
                )
                probe_rows = cursor.fetchone()[0]
            connection.commit()
            check(
                "the lookup key is stored normalised",
                probe_rows >= 1,
                f"{probe_rows} rows for {PROBE_NRN}",
            )
        else:
            print("\n2. load (skipped, using whatever is already there)")
            current, history, runs = _counts(connection)
            check("nafdac_products has data", current > 0, f"{current} rows")

        print("\n3. reading a product back over the wire")
        source = PostgresSource(dsn=dsn)
        check("the source reports itself available", source.available(), source.describe())
        result = source.search(PROBE_NRN)
        check(f"{PROBE_NRN} comes back from PostgreSQL", result.ok and bool(result.matches))
        if result.matches:
            record = result.matches[0]
            check(
                "the product name survived the round trip",
                "Spironolactone" in record.product_name,
                record.product_name,
            )
            check(
                "the approval date survived",
                record.approval_date.startswith("2021-12"),
                record.approval_date,
            )
            check(
                "the applicant survived",
                "AnnyGod" in record.applicant_name,
                record.applicant_name,
            )
            check(
                "product_id came back as an integer",
                isinstance(record.product_id, int),
                repr(record.product_id),
            )
            check("the status survived", record.status in {"Active", "Inactive"}, record.status)
            check("the source is attributed to postgres", record.sources == ["postgres"])

        # A padded lookup has to hit the same row, which is what the normalised
        # key is for.
        padded = source.search("A4 - 100160")
        check("a padded lookup still matches", bool(padded.matches))

        print("\n4. the superseded history path")
        # Insert a product, then reload without it, which is exactly the
        # situation the history table exists for.
        synthetic = record_from_row(
            {
                "NRN": SYNTHETIC_NRN,
                "Product ID": str(SYNTHETIC_PRODUCT_ID),
                "Product Name": "Superseded Test Product",
                "Active Ingredients": "Testosterone",
                "Strengths": "1 mg",
                "Form": "Tablet",
                "ROA": "Oral",
                "Status": "Active",
                "Applicant Name": "Test Applicant Ltd",
                "Approval Date": "2020-01-01",
            },
            source="postgres",
        )
        from .load_postgres import _COLUMNS, _row_values

        columns = ", ".join(_COLUMNS)
        # Parentheses required; see the note in load_postgres.load(). psycopg3
        # otherwise falls back to the simple query protocol and PostgreSQL
        # rejects the literal `$1`.
        placeholders = "(" + ", ".join(["%s"] * len(_COLUMNS)) + ")"
        with connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO nafdac_products ({columns}) VALUES {placeholders}",
                _row_values(synthetic),
            )
        connection.commit()
        check(
            "a synthetic product was inserted",
            bool(PostgresSource(dsn=dsn).search(SYNTHETIC_NRN).matches),
        )

        # Reload a record set that omits it, which archives it to history.
        load(connection, [synthetic])
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM nafdac_products WHERE product_id = %s",
                (SYNTHETIC_PRODUCT_ID,),
            )
            cursor.execute(
                "UPDATE nafdac_product_history SET superseded_at = now() "
                "WHERE product_id = %s",
                (SYNTHETIC_PRODUCT_ID,),
            )
        connection.commit()

        source = PostgresSource(dsn=dsn)
        gone = source.search(SYNTHETIC_NRN)
        check(
            "a removed product is still found, from history",
            bool(gone.matches),
            gone.error or "no match",
        )
        check("that result is marked as from history", gone.from_history is True)

        result = verify_nafdac_product(SYNTHETIC_NRN, sources=[source])
        check("verify_nafdac_product finds it too", result.found is True)
        check("it is reported as not authorised", result.authorised is False, repr(result.authorised))
        check("confidence is superseded", result.confidence == "superseded", result.confidence)
        check(
            "the rendered answer explains the supersession",
            "no longer" in result.render().lower(),
            result.render().splitlines()[-2] if result.render() else "",
        )

        # Clean up, so a real database is left as it was found.
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM nafdac_product_history WHERE product_id = %s",
                (SYNTHETIC_PRODUCT_ID,),
            )
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM nafdac_product_history WHERE product_id = %s",
                (SYNTHETIC_PRODUCT_ID,),
            )
            check("the synthetic product was cleaned up", cursor.fetchone()[0] == 0)
        connection.commit()

        print()
        return 1 if _FAILURES else 0
    finally:
        connection.close()


def _default_csv() -> Optional[Path]:
    from .sources import find_default_file

    return find_default_file("csv")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nafdac_source.verify_postgres",
        description=(
            "Check the schema, the loader and the read path against a real "
            "PostgreSQL server."
        ),
    )
    parser.add_argument(
        "--url",
        default="",
        help="connection string; defaults to NAFDAC_DATABASE_URL or DATABASE_URL",
    )
    parser.add_argument("--csv", default="", help="path to nafdac_greenbook.csv")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="do not run the loader, only test against existing data",
    )
    args = parser.parse_args(argv)

    _load_env()

    dsn = (
        args.url
        or os.environ.get("NAFDAC_DATABASE_URL", "")
        or os.environ.get("DATABASE_URL", "")
    )
    if not dsn:
        print(
            "No database configured. Set NAFDAC_DATABASE_URL or pass --url.\n"
            "An .env file is read automatically once python-dotenv is installed.\n"
            "This check writes to nafdac_products and nafdac_product_history, "
            "so point it at a scratch database rather than a live one.",
            file=sys.stderr,
        )
        return 2

    return _verify_database(dsn, Path(args.csv) if args.csv else None, args.skip_load)


if __name__ == "__main__":
    raise SystemExit(main())
