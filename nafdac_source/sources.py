"""The three lookup sources: remote PostgreSQL, local CSV, local XLSX.

Every source exposes the same three methods:

    available()          is this source usable right now?
    describe()           a short status string, for /nafdac status
    search(nrn)          -> SourceResult

`search` never raises for a missing or unreachable source. It returns a
`SourceResult` with `ok=False` and an `error` string, so one broken source
degrades the answer instead of failing the lookup.
"""

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extract import normalize_nrn_key
from .records import ProductRecord, record_from_row

# The greenbook files, relative to the project root. Located automatically so
# the client works from any working directory.
_DEFAULT_FILENAMES = {
    "csv": "nafdac_greenbook.csv",
    "xlsx": "nafdac_greenbook.xlsx",
}

_DEFAULT_DIRS = ("nafdac_scraper", ".", "data", "nafdac_source")


def find_default_file(kind: str, start: Optional[Path] = None) -> Optional[Path]:
    """Locate nafdac_greenbook.csv / .xlsx by walking up from `start`."""
    filename = _DEFAULT_FILENAMES[kind]
    here = Path(start or Path(__file__).resolve().parent)
    candidates = [here, *here.parents]
    for base in candidates:
        for sub in _DEFAULT_DIRS:
            path = base / sub / filename
            if path.is_file():
                return path
        direct = base / filename
        if direct.is_file():
            return direct
    return None


@dataclass
class SourceResult:
    """The outcome of querying one source."""

    source: str
    ok: bool = False
    error: str = ""
    matches: List[ProductRecord] = field(default_factory=list)
    total_records: int = 0
    detail: str = ""
    # True when every match came from a superseded/withdrawn history table
    # rather than the current one. This is an explicit flag rather than
    # something inferred from the detail string, because "this product used to
    # be registered" changes what the answer should say.
    from_history: bool = False

    def __bool__(self) -> bool:
        return self.ok


class NafdacSource:
    """Base class. Subclasses implement `search` and `describe`."""

    name = "base"

    def available(self) -> bool:
        return False

    def describe(self) -> str:
        return "not available"

    def search(self, nrn: str) -> SourceResult:  # pragma: no cover - interface
        return SourceResult(source=self.name, ok=False, error="not implemented")

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Local CSV
# ---------------------------------------------------------------------------
class CsvSource(NafdacSource):
    """nafdac_scraper/nafdac_greenbook.csv.

    Read with the `csv` module rather than pandas so this works on the Kaggle
    server and on the laptop with nothing installed. The file is UTF-8 with a
    BOM and its `Composition` column contains real newlines, so `csv` with
    `newline=""` is the only correct reader; a line-based reader silently
    produces 14953 rows instead of 8935.
    """

    name = "csv"

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else find_default_file("csv")
        self._index: Optional[Dict[str, List[ProductRecord]]] = None
        self._total = 0

    def available(self) -> bool:
        return bool(self.path) and Path(self.path).is_file()

    def describe(self) -> str:
        if not self.available():
            return "not found (nafdac_greenbook.csv)"
        if self._index is None:
            return f"{self.path} (not loaded yet)"
        return f"{self.path} ({self._total} products)"

    def load(self) -> Dict[str, List[ProductRecord]]:
        """Read and index the file once, then reuse it.

        The index maps a normalised NRN key to every record carrying it. A key
        can hold more than one record because 257 NRNs are shared.
        """
        if self._index is not None:
            return self._index

        index: Dict[str, List[ProductRecord]] = {}
        if not self.available():
            self._index = index
            return index

        with open(self.path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                record = record_from_row(row, source=self.name)
                key = record.nrn_key
                if not key:
                    continue
                index.setdefault(key, []).append(record)

        self._index = index
        self._total = sum(len(records) for records in index.values())
        return index

    def search(self, nrn: str) -> SourceResult:
        key = normalize_nrn_key(nrn)
        if not key:
            return SourceResult(source=self.name, ok=False, error="no valid NRN")
        if not self.available():
            return SourceResult(
                source=self.name, ok=False, error=f"file not found: {self.path}"
            )
        try:
            index = self.load()
        except OSError as exc:
            return SourceResult(source=self.name, ok=False, error=str(exc))

        matches = list(index.get(key, []))
        return SourceResult(
            source=self.name,
            ok=True,
            matches=matches,
            total_records=self._total,
            detail=str(self.path),
        )


# ---------------------------------------------------------------------------
# Local XLSX
# ---------------------------------------------------------------------------
class XlsxSource(NafdacSource):
    """nafdac_scraper/nafdac_greenbook.xlsx, the `Products` sheet.

    openpyxl is optional. It is present in the scraper venv and on Kaggle but
    not in the client's requirements, so its absence downgrades this source to
    unavailable rather than raising — the CSV covers the same rows.

    Reading the workbook takes about 40 seconds, which is far too slow to do on
    the first user message, so the parsed rows are cached next to the data as a
    small CSV keyed on the workbook's size and modification time. The cache is
    pure CSV rather than JSON or pickle on purpose: it reads in about 1.5
    seconds instead of 40, it is inspectable, and it cannot execute anything
    when loaded. The cache is a speed optimisation only, so a miss or an
    unreadable cache just means the workbook is parsed again.
    """

    name = "xlsx"
    CACHE_SUFFIX = ".nafdac_rows.csv"

    def __init__(
        self,
        path: Optional[Path] = None,
        sheet: str = "Products",
        use_cache: bool = True,
    ):
        self.path = Path(path) if path else find_default_file("xlsx")
        self.sheet = sheet
        self.use_cache = use_cache
        self._index: Optional[Dict[str, List[ProductRecord]]] = None
        self._total = 0
        self._import_error = ""
        self._cache_hit = False

    def _openpyxl(self):
        try:
            import openpyxl  # noqa: WPS433 - optional dependency
        except ImportError as exc:
            self._import_error = f"openpyxl not installed ({exc})"
            return None
        return openpyxl

    def cache_path(self) -> Optional[Path]:
        if not self.path:
            return None
        directory = self.path.parent
        if not os.access(str(directory), os.W_OK):
            # A read-only checkout is still perfectly usable, just slower.
            return None
        return self.path.with_name(self.path.name + self.CACHE_SUFFIX)

    def _cache_stamp(self) -> Dict[str, Any]:
        stat = self.path.stat()
        return {
            "source": self.path.name,
            "sheet": self.sheet,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "version": 1,
        }

    def available(self) -> bool:
        return bool(self.path) and Path(self.path).is_file() and self._openpyxl() is not None

    def describe(self) -> str:
        if not self.path or not Path(self.path).is_file():
            return "not found (nafdac_greenbook.xlsx)"
        if not self._openpyxl():
            return f"{self.path} (openpyxl not installed)"
        if self._index is None:
            return f"{self.path} (not loaded yet)"
        origin = "cached" if self._cache_hit else "parsed"
        return f"{self.path} ({self._total} products, {origin})"

    # -- cache ---------------------------------------------------------
    def _stamp_path(self) -> Optional[Path]:
        path = self.cache_path()
        return path.with_suffix(path.suffix + ".stamp") if path else None

    def _load_cache(self) -> Optional[Dict[str, List[ProductRecord]]]:
        path = self.cache_path()
        stamp_path = self._stamp_path()
        if path is None or stamp_path is None or not path.is_file():
            return None
        if not stamp_path.is_file():
            return None
        try:
            import json

            if json.loads(stamp_path.read_text(encoding="utf-8")) != self._cache_stamp():
                return None
        except Exception:
            return None

        index: Dict[str, List[ProductRecord]] = {}
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    record = record_from_row(row, source=self.name)
                    key = record.nrn_key
                    if key:
                        index.setdefault(key, []).append(record)
        except Exception:
            return None

        self._total = sum(len(records) for records in index.values())
        self._cache_hit = True
        return index

    def _save_cache(self, index: Dict[str, List[ProductRecord]]) -> None:
        path = self.cache_path()
        stamp_path = self._stamp_path()
        if path is None or stamp_path is None:
            return
        try:
            import json

            from .records import FIELD_COLUMNS

            # FIELD_COLUMNS maps record attribute -> greenbook column name, so
            # the attribute is the key and the column is the CSV header. Writing
            # it this way round means the cache reloads through
            # `record_from_row` unchanged.
            pairs = list(FIELD_COLUMNS.items())
            columns = [column for _attribute, column in pairs]
            temporary = path.with_suffix(path.suffix + ".tmp")
            with open(temporary, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for records in index.values():
                    for record in records:
                        writer.writerow(
                            {column: record.field(attribute) for attribute, column in pairs}
                        )
            os.replace(temporary, path)
            stamp_path.write_text(
                json.dumps(self._cache_stamp(), sort_keys=True), encoding="utf-8"
            )
        except Exception:
            # A cache that cannot be written is not an error; the index is
            # already in memory.
            pass

    def load(self) -> Dict[str, List[ProductRecord]]:
        if self._index is not None:
            return self._index

        if self.use_cache:
            cached = self._load_cache()
            if cached is not None:
                self._index = cached
                return cached

        openpyxl = self._openpyxl()
        index: Dict[str, List[ProductRecord]] = {}
        if openpyxl is None or not self.available():
            self._index = index
            return index

        workbook = openpyxl.load_workbook(
            self.path, read_only=True, data_only=True
        )
        try:
            if self.sheet not in workbook.sheetnames:
                self._import_error = (
                    f"sheet {self.sheet!r} not in {workbook.sheetnames}"
                )
                self._index = index
                return index
            sheet = workbook[self.sheet]
            rows = sheet.iter_rows(values_only=True)
            try:
                headers = next(rows)
            except StopIteration:
                self._index = index
                return index

            headers = [str(h).strip() if h is not None else "" for h in headers]
            for values in rows:
                if not any(values):
                    continue
                row = dict(zip(headers, values))
                record = record_from_row(row, source=self.name)
                key = record.nrn_key
                if not key:
                    continue
                index.setdefault(key, []).append(record)
        finally:
            workbook.close()

        self._index = index
        self._total = sum(len(records) for records in index.values())
        if self.use_cache:
            self._save_cache(index)
        return index

    def search(self, nrn: str) -> SourceResult:
        key = normalize_nrn_key(nrn)
        if not key:
            return SourceResult(source=self.name, ok=False, error="no valid NRN")
        if not self.available():
            return SourceResult(
                source=self.name, ok=False, error=self._import_error or f"file not found: {self.path}"
            )
        try:
            index = self.load()
        except Exception as exc:  # openpyxl raises a variety of types
            return SourceResult(source=self.name, ok=False, error=f"{type(exc).__name__}: {exc}")

        return SourceResult(
            source=self.name,
            ok=True,
            matches=list(index.get(key, [])),
            total_records=self._total,
            detail=str(self.path),
        )


# ---------------------------------------------------------------------------
# Remote PostgreSQL
# ---------------------------------------------------------------------------
# Column order and types must match schema.sql. Written out rather than built
# from the map so the query is readable in server logs and so a schema change
# is a deliberate edit in two places.
_SELECT_COLUMNS = """
    nrn, product_id, product_name, active_ingredients, strengths, form, route,
    product_category, status, marketing_category, applicant_name,
    applicant_address, manufacturer_name, manufacturer_country, approval_date,
    expiry_date, atc_code, product_category_id, ingredient_id,
    ingredient_synonym, form_id, route_id, product_description, composition,
    pack_size, applicant_id, manufacturer_id, country_id,
    marketing_category_id, smpc_url, biosimilar, created_at, updated_at,
    source_url
"""


class PostgresSource(NafdacSource):
    """The remote PostgreSQL database, read over the network.

    This is the only source that can be reached from the Kaggle server, because
    it is hosted rather than on the laptop. That makes it the source that
    carries the current, and superseded, state of a product.

    psycopg is imported lazily and is optional. The connection is opened once
    and reused, because the laptop client and the Kaggle server both run a
    lookup per message and reconnecting each time is wasteful.
    """

    name = "postgres"

    def __init__(
        self,
        dsn: str = "",
        table: str = "nafdac_products",
        history_table: str = "nafdac_product_history",
        connect_timeout: int = 10,
        sslmode: str = "",
    ):
        self.dsn = dsn or os.environ.get("NAFDAC_DATABASE_URL", "") or os.environ.get(
            "DATABASE_URL", ""
        )
        self.table = table
        self.history_table = history_table
        self.connect_timeout = connect_timeout
        self.sslmode = sslmode
        self._connection = None
        self._driver_error = ""
        self._last_error = ""
        self._includes_history = True
        # Cached on first use, so a missing history table costs one query per
        # process rather than one per lookup.
        self._history_exists: Optional[bool] = None

    # -- lifecycle ------------------------------------------------------
    def _driver(self):
        try:
            import psycopg  # noqa: WPS433 - optional dependency
        except ImportError:
            try:
                import psycopg2 as psycopg  # type: ignore[no-redef]
            except ImportError as exc:
                self._driver_error = (
                    f"no PostgreSQL driver installed ({exc}); "
                    "pip install 'psycopg[binary]'"
                )
                return None
        return psycopg

    def connect(self):
        """Open the connection, or return None and record why."""
        if self._connection is not None:
            return self._connection
        if not self.dsn:
            self._last_error = "NAFDAC_DATABASE_URL is not set"
            return None
        driver = self._driver()
        if driver is None:
            self._last_error = self._driver_error
            return None
        try:
            self._connection = driver.connect(
                self.dsn,
                connect_timeout=self.connect_timeout,
                sslmode=self.sslmode or None,
            )
            self._last_error = ""
            return self._connection
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._connection = None
            return None

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def available(self) -> bool:
        return bool(self.dsn) and self._driver() is not None

    def describe(self) -> str:
        if not self.dsn:
            return "not configured (set NAFDAC_DATABASE_URL)"
        if self._driver() is None:
            return f"configured but {self._driver_error}"
        if self._last_error:
            return f"configured but unreachable: {self._last_error}"
        return "configured"

    # -- queries --------------------------------------------------------
    @staticmethod
    def _table_exists(connection, table: str) -> bool:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = current_schema() AND table_name = %s
                """,
                (table,),
            )
            return cursor.fetchone() is not None

    def _history_table_present(self, connection) -> bool:
        """Whether the history table exists, answered once per connection.

        A database that has had the current table loaded but not the history
        table is a normal state, not an error, and re-asking on every lookup
        would add a round trip to each one.
        """
        if self._history_exists is None:
            try:
                self._history_exists = self._table_exists(connection, self.history_table)
            except Exception as exc:
                # The connection is in a failed transaction; clear it so the
                # lookup can still return what the current table had.
                try:
                    connection.rollback()
                except Exception:
                    pass
                raise exc
        return self._history_exists

    def _query(self, connection, table: str, key: str) -> List[ProductRecord]:
        sql = (
            f"SELECT {_SELECT_COLUMNS} "
            f"FROM {table} "
            f"WHERE nrn_key = %s "
            f"ORDER BY product_id NULLS LAST"
        )
        with connection.cursor() as cursor:
            cursor.execute(sql, (key,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        return [
            record_from_row(dict(zip(columns, row)), source=self.name)
            for row in rows
        ]

    def search(self, nrn: str) -> SourceResult:
        key = normalize_nrn_key(nrn)
        if not key:
            return SourceResult(source=self.name, ok=False, error="no valid NRN")
        if not self.available():
            return SourceResult(
                source=self.name,
                ok=False,
                error=self._driver_error or self._last_error or "not configured",
            )

        connection = self.connect()
        if connection is None:
            return SourceResult(
                source=self.name,
                ok=False,
                error=self._last_error or "connection failed",
            )

        try:
            matches = self._query(connection, self.table, key)
            detail = f"{self.table}"
            from_history = False

            # A product that is no longer in the current table may still have a
            # superseded row. Surfacing it lets the caller say "this used to be
            # registered" instead of a flat "not found".
            if not matches and self._includes_history:
                try:
                    if self._history_table_present(connection):
                        historical = self._query(connection, self.history_table, key)
                        if historical:
                            for record in historical:
                                record.status = record.status or "Superseded"
                            matches.extend(historical)
                            detail = f"{self.table} + {self.history_table}"
                            from_history = True
                except Exception as exc:
                    detail += f" (history check skipped: {type(exc).__name__})"

            return SourceResult(
                source=self.name,
                ok=True,
                matches=matches,
                detail=detail,
                from_history=from_history,
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            # A failed query may have poisoned the transaction.
            try:
                connection.rollback()
            except Exception:
                pass
            return SourceResult(source=self.name, ok=False, error=self._last_error)


# Process-wide source cache. `build_sources` is called on every verification,
# and each source loads its file once into an index. Without this, every user
# message would re-read the greenbook; with it, the cost is paid once per
# process. Keyed on the arguments so a caller passing explicit paths still gets
# its own sources rather than someone else's.
_SOURCE_CACHE: Dict[tuple, List[NafdacSource]] = {}


def build_sources(
    csv_path: Optional[Path] = None,
    xlsx_path: Optional[Path] = None,
    database_url: str = "",
    use_database: bool = True,
    use_csv: bool = True,
    use_xlsx: bool = True,
    use_cache: bool = True,
) -> List[NafdacSource]:
    """The default source list, in merge order.

    PostgreSQL first: it holds the current and superseded state, so its values
    win. The two local files follow and only fill fields that are still empty.

    The returned sources are shared and reused across calls, so their file
    indexes are loaded at most once per process. Pass `use_cache=False` for
    isolated sources, which the tests do.
    """
    key = (
        str(csv_path or ""),
        str(xlsx_path or ""),
        database_url,
        use_database,
        use_csv,
        use_xlsx,
    )
    if use_cache and key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]

    sources: List[NafdacSource] = []
    if use_database:
        sources.append(PostgresSource(dsn=database_url))
    if use_csv:
        sources.append(CsvSource(csv_path))
    if use_xlsx:
        sources.append(XlsxSource(xlsx_path))

    if use_cache:
        _SOURCE_CACHE[key] = sources
    return sources


def clear_source_cache() -> None:
    """Drop the shared sources, closing any open database connection."""
    for sources in _SOURCE_CACHE.values():
        for source in sources:
            source.close()
    _SOURCE_CACHE.clear()
