"""Load the NAFDAC Greenbook CSV into PostgreSQL.

This is the "superseded database" side of the setup. The client and the Kaggle
server both read from these tables, so this script is what makes a
superseding update visible to a user who asks about an old product.

Two things it does that a plain COPY does not:

  * computes `nrn_key`, the normalised lookup key, so a lookup for `A4 - 5180`
    finds the row stored as `A4-5180`;
  * moves rows that have left the CSV into `nafdac_product_history` rather than
    deleting them, so "this used to be registered" stays answerable.

Usage:
    python -m nafdac_source.load_postgres --dry-run
    python -m nafdac_source.load_postgres
    python -m nafdac_source.load_postgres --url postgresql://... --no-history

The connection string comes from --url, or NAFDAC_DATABASE_URL, or
DATABASE_URL, or the same `.env` files the laptop client reads. It is never
printed back: only the host and database name are shown in the log, because
these are hosted credentials.
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .extract import normalize_nrn_key
from .records import (
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    classify_status,
    record_from_row,
)
from .sources import find_default_file

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE / "schema.sql"

# Column order for every INSERT. Must match `_SELECT_COLUMNS` in sources.py,
# which is what the reader queries.
_COLUMNS: Tuple[str, ...] = (
    "nrn",
    "nrn_key",
    "product_id",
    "product_name",
    "active_ingredients",
    "strengths",
    "form",
    "route",
    "product_category",
    "status",
    "marketing_category",
    "applicant_name",
    "applicant_address",
    "manufacturer_name",
    "manufacturer_country",
    "approval_date",
    "expiry_date",
    "atc_code",
    "product_category_id",
    "ingredient_id",
    "ingredient_synonym",
    "form_id",
    "route_id",
    "product_description",
    "composition",
    "pack_size",
    "applicant_id",
    "manufacturer_id",
    "country_id",
    "marketing_category_id",
    "smpc_url",
    "biosimilar",
    "created_at",
    "updated_at",
    "source_url",
)

# Columns holding a date or timestamp. An unparseable value becomes NULL
# rather than aborting the load, because the scraped data does contain a few
# odd date strings and one bad row should not cost 8934 good ones.
_DATE_COLUMNS = frozenset({"approval_date", "expiry_date"})
_TIMESTAMP_COLUMNS = frozenset({"created_at", "updated_at"})
_INT_COLUMNS = frozenset(
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


def _safe_date(value: str) -> Optional[str]:
    """A date or timestamp PostgreSQL will accept, or None.

    The scraper emits a year-1 sentinel wherever a timestamp was missing, which
    serialises as '-000001-11-30T00:00:00.000000Z' -- datetime.min pushed
    through the proleptic Gregorian calendar. PostgreSQL rejects it, and because
    one poisoned row aborts a whole 500-row batch, 2,211 of the 8,930 greenbook
    rows were enough to stop the load dead and force a per-row retry that took
    fifteen minutes to write nothing.

    Anything empty, unparseable or out of range becomes NULL, which is what the
    note above _DATE_COLUMNS promises. Valid values are passed through
    untouched so PostgreSQL, not this function, decides what it accepts.
    """
    text = (value or "").strip()
    if not text or text.lower() in {"none", "nan", "null", "nat"}:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    # PostgreSQL's timestamp range starts at 4713 BC, but anything before year
    # 1000 in this data is a sentinel rather than a real approval date.
    if parsed.year < 1000:
        return None
    return text


def _row_values(record) -> List[Any]:
    """Turn a ProductRecord into the parameter tuple for INSERT."""
    values: List[Any] = []
    for column in _COLUMNS:
        if column == "nrn_key":
            values.append(record.nrn_key)
        elif column in _DATE_COLUMNS or column in _TIMESTAMP_COLUMNS:
            values.append(_safe_date(record.field(column)))
        elif column in _INT_COLUMNS:
            values.append(getattr(record, column, None))
        else:
            values.append(record.field(column))
    return values


def _connect(dsn: str):
    try:
        import psycopg  # noqa: WPS433
    except ImportError:
        try:
            import psycopg2 as psycopg  # type: ignore[no-redef]
        except ImportError:
            print(
                "No PostgreSQL driver found. Install one with:\n"
                "    pip install \"psycopg[binary]\"",
                file=sys.stderr,
            )
            raise SystemExit(2)
    return psycopg.connect(dsn, connect_timeout=30)


def _mask(dsn: str) -> str:
    """A loggable form of the connection string: host and database only."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(dsn)
        database = parsed.path.lstrip("/") or "?"
        host = parsed.hostname or "local socket"
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return f"postgresql://{host}/{database}"
    except Exception:
        return "postgresql://<configured>"


def _load_env() -> bool:
    """Read the `.env` files into the environment, the way the client does.

    The loader is started as a bare `python -m nafdac_source.load_postgres`, so
    unlike the client nothing has put the connection string in the environment
    first. Without this the same database the client reads is invisible to the
    tool that fills it, and the run stops at "No database configured" even
    though the URL is sitting in a `.env` file a directory away.

    The search order matches `n_atlas_client/config.py`: working directory
    first, then the project root, then this package's own folder, all with
    `override=False`. A real environment variable always wins and the first
    file to define a key keeps it, so a `.env` cannot rewrite the shell.

    Returns whether python-dotenv was importable. It is deliberately optional:
    without it the loader still runs, it just sees only `--url` and real
    environment variables.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(HERE.parent / ".env", override=False)
    load_dotenv(HERE / ".env", override=False)
    return True


def apply_schema(connection) -> None:
    """Run schema.sql. Every statement is IF NOT EXISTS, so this is re-runnable."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(sql)
    connection.commit()


def _try_trigram(connection) -> bool:
    """Enable pg_trgm and a trigram index, tolerating a managed-host refusal.

    Supabase, Neon and most hosted PostgreSQL already have pg_trgm available.
    Where it is not, the plain lower() indexes still serve exact-prefix lookups,
    so this is a best-effort upgrade rather than a requirement.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS nafdac_products_name_trgm_idx "
                "ON nafdac_products USING gin (product_name gin_trgm_ops)"
            )
        connection.commit()
        return True
    except Exception as exc:
        connection.rollback()
        print(f"note: trigram search unavailable ({type(exc).__name__}); "
              "falling back to prefix matching")
        return False


def read_csv(path: Path) -> Iterable[Any]:
    """Yield a ProductRecord per CSV row.

    The file is UTF-8 with a BOM and its Composition column contains embedded
    newlines, so `csv` with newline="" is required; a line-based reader sees
    14953 lines instead of 8935 rows.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            record = record_from_row(row, source="postgres")
            if not record.nrn_key:
                # A row with no usable NRN cannot be looked up by number, so
                # loading it would only bloat the table.
                continue
            yield record


def load(
    connection,
    records: List[Any],
    batch_size: int = 500,
    keep_history: bool = True,
) -> Tuple[int, int, int]:
    """Upsert every record. Returns (loaded, superseded, skipped).

    Loaded rows that were not in the table before are counted as new, and rows
    that have disappeared from the CSV are moved to history.
    """
    # The parentheses are load-bearing, not decoration. Without them psycopg3
    # cannot split the parameter list out of `VALUES %s, %s`, so it sends the
    # statement through the simple query protocol where $1 is literal text and
    # PostgreSQL answers `syntax error at or near "$1"`. Measured on this
    # project: `VALUES (%s, %s)` succeeds, `VALUES %s, %s` always fails.
    placeholders = "(" + ", ".join(["%s"] * len(_COLUMNS)) + ")"
    columns = ", ".join(_COLUMNS)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT product_id FROM nafdac_products WHERE product_id = ANY(%s)",
            ([record.product_id for record in records if record.product_id],),
        )
        existing = {row[0] for row in cursor.fetchall()}

    loaded = 0
    skipped = 0
    batch: List[tuple] = []
    product_ids: List[int] = []

    with connection.cursor() as cursor:
        for record in records:
            if not record.product_id:
                # The greenbook always has a Product ID, but a row without one
                # has no primary key to upsert on.
                skipped += 1
                continue
            batch.append(tuple(_row_values(record)))
            product_ids.append(record.product_id)
            if len(batch) >= batch_size:
                _flush(cursor, columns, placeholders, batch)
                loaded += len(batch)
                batch = []
        if batch:
            _flush(cursor, columns, placeholders, batch)
            loaded += len(batch)

    connection.commit()

    superseded = 0
    if keep_history:
        superseded = _archive_missing(connection, product_ids, existing)
        connection.commit()

    return loaded, superseded, skipped


def _flush(cursor, columns: str, placeholders: str, batch: List[tuple]) -> None:
    """Upsert one batch, retrying row by row if a single value is rejected."""
    statement = (
        f"INSERT INTO nafdac_products ({columns}) VALUES {placeholders} "
        "ON CONFLICT (product_id) DO UPDATE SET "
        + ", ".join(
            f"{column} = EXCLUDED.{column}"
            for column in _COLUMNS
            if column not in {"product_id"}
        )
        + ", loaded_at = now()"
    )
    try:
        cursor.executemany(statement, batch)
    except Exception:
        # One malformed date should not cost the whole batch. Roll back to a
        # clean state, then insert the good rows individually.
        cursor.connection.rollback()
        for values in batch:
            try:
                cursor.execute(statement, values)
            except Exception as exc:
                cursor.connection.rollback()
                print(f"  skipped a row: {type(exc).__name__}: {exc}")


def _archive_missing(
    connection, keep_ids: List[int], existing_ids: set
) -> int:
    """Move rows that left the CSV into the history table.

    A row already in history is not re-archived, so this is safe to re-run: the
    history table accumulates genuine supersessions, not loader runs.
    """
    if not existing_ids:
        return 0

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO nafdac_product_history (
                nrn, nrn_key, product_id, product_name, active_ingredients,
                strengths, form, route, product_category, status,
                marketing_category, applicant_name, applicant_address,
                manufacturer_name, manufacturer_country, approval_date,
                expiry_date, atc_code, product_category_id, ingredient_id,
                ingredient_synonym, form_id, route_id, product_description,
                composition, pack_size, applicant_id, manufacturer_id,
                country_id, marketing_category_id, smpc_url, biosimilar,
                created_at, updated_at, source_url, superseded_reason
            )
            SELECT
                nrn, nrn_key, product_id, product_name, active_ingredients,
                strengths, form, route, product_category,
                CASE WHEN status = 'Active' THEN 'Superseded' ELSE status END,
                marketing_category, applicant_name, applicant_address,
                manufacturer_name, manufacturer_country, approval_date,
                expiry_date, atc_code, product_category_id, ingredient_id,
                ingredient_synonym, form_id, route_id, product_description,
                composition, pack_size, applicant_id, manufacturer_id,
                country_id, marketing_category_id, smpc_url, biosimilar,
                created_at, updated_at, source_url,
                'no longer present in the current greenbook export'
            FROM nafdac_products AS old
            WHERE NOT (old.product_id = ANY(%s))
              AND NOT EXISTS (
                  SELECT 1 FROM nafdac_product_history AS h
                  WHERE h.product_id = old.product_id
              )
            """,
            (keep_ids,),
        )
        archived = cursor.rowcount
        if archived:
            cursor.execute(
                "DELETE FROM nafdac_products WHERE NOT (product_id = ANY(%s))",
                (keep_ids,),
            )
    return max(0, archived)


def start_run(connection, path: Path) -> int:
    stat = path.stat()
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO nafdac_load_runs (source_file, source_size, source_mtime)
            VALUES (%s, %s, to_timestamp(%s))
            RETURNING run_id
            """,
            (str(path), stat.st_size, int(stat.st_mtime)),
        )
        run_id = cursor.fetchone()[0]
    connection.commit()
    return run_id


def finish_run(
    connection,
    run_id: int,
    loaded: int,
    superseded: int,
    status: str = "complete",
    detail: str = "",
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE nafdac_load_runs
            SET rows_loaded = %s, rows_superseded = %s, status = %s,
                detail = %s, finished_at = now()
            WHERE run_id = %s
            """,
            (loaded, superseded, status, detail, run_id),
        )
    connection.commit()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nafdac_source.load_postgres",
        description="Load the NAFDAC Greenbook CSV into PostgreSQL.",
    )
    parser.add_argument(
        "--url",
        default="",
        help="connection string; defaults to NAFDAC_DATABASE_URL or DATABASE_URL",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="path to nafdac_greenbook.csv; auto-detected by default",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="rows per INSERT batch"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="do not archive rows that have left the CSV",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate the CSV without touching the database",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else find_default_file("csv")
    if not csv_path or not csv_path.is_file():
        print(
            "Could not find nafdac_greenbook.csv. Run the scraper first, or pass "
            "--csv PATH.",
            file=sys.stderr,
        )
        return 1

    records = list(read_csv(csv_path))
    active = sum(1 for record in records if record.status == STATUS_ACTIVE)
    inactive = sum(1 for record in records if record.status == STATUS_INACTIVE)
    print(f"read {len(records)} products from {csv_path}")
    print(f"  active   {active}")
    print(f"  inactive {inactive}")
    print(f"  distinct NRNs {len({record.nrn_key for record in records})}")

    if args.dry_run:
        print("\ndry run: nothing was written to the database.")
        return 0

    if not _load_env():
        print(
            "note: python-dotenv is not installed, so no .env file was read.\n"
            "      Install it with:  pip install python-dotenv\n"
            "      or set NAFDAC_DATABASE_URL in the shell, or pass --url.",
            file=sys.stderr,
        )

    dsn = (
        args.url
        or os.environ.get("NAFDAC_DATABASE_URL", "")
        or os.environ.get("DATABASE_URL", "")
    )
    if not dsn:
        print(
            "No database configured. Set NAFDAC_DATABASE_URL or pass --url.\n"
            "An .env file is read automatically once python-dotenv is installed.\n"
            "For a hosted database the string looks like:\n"
            "    postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require",
            file=sys.stderr,
        )
        return 1

    print(f"\nconnecting to {_mask(dsn)}")
    connection = _connect(dsn)
    try:
        apply_schema(connection)
        print("schema applied")
        _try_trigram(connection)

        run_id = start_run(connection, csv_path)
        loaded, superseded, skipped = load(
            connection,
            records,
            batch_size=max(1, args.batch_size),
            keep_history=not args.no_history,
        )
        finish_run(connection, run_id, loaded, superseded)

        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM nafdac_products")
            current = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM nafdac_product_history")
            history = cursor.fetchone()[0]
        connection.commit()

        print(f"\nloaded       {loaded} rows ({skipped} skipped, no product id)")
        print(f"superseded   {superseded} rows moved to nafdac_product_history")
        print(f"nafdac_products         {current}")
        print(f"nafdac_product_history  {history}")
        print(f"\nload run {run_id} recorded in nafdac_load_runs")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
