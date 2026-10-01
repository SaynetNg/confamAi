"""Exercise Cell 17 on its own, without a GPU or a loaded model.

The full notebook cannot run here: Cells 1-4 need torch, CUDA and the N-ATLaS
weights. Cell 17 needs none of that, so it is executed against a hand-built
namespace holding only the globals it actually reads.

This checks the three things that can break Cell 17 without a GPU:
  * it finds nafdac_source and imports it;
  * it degrades to "unavailable" instead of raising when it cannot;
  * a real number verifies and the verified block is produced.
"""

import logging
import os
import sys
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import notebook_cells  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default=0):
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def build_namespace():
    """The globals Cell 17 reads.

    NAFDAC_HOOKS is normally created by Cell 9, which owns the routes. Cell 17
    fills it, so an empty dict stands in for it here — that is the real
    relationship, not a stub to make the test pass.
    """
    return {
        "os": os,
        "re": __import__("re"),
        "sys": sys,
        "Path": Path,
        "logger": logging.getLogger("cell17"),
        "WORKDIR": HERE,
        "_env_flag": _env_flag,
        "_env_int": _env_int,
        "Any": typing.Any,
        "List": typing.List,
        "Dict": typing.Dict,
        "Optional": typing.Optional,
        "NAFDAC_HOOKS": {},
    }


def main() -> int:
    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" :: {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("=" * 70)
    print("CELL 17 — NAFDAC greenbook (offline)")
    print("=" * 70)

    ns = build_namespace()
    result = notebook_cells.exec_range(17, 17, ns)

    check(
        "Cell 17 executed without raising",
        "NAFDAC_AVAILABLE" in result,
        f"available={result.get('NAFDAC_AVAILABLE')}",
    )
    if not result.get("NAFDAC_AVAILABLE"):
        print(f"  nafdac_source could not be loaded: {result.get('NAFDAC_IMPORT_ERROR')}")
        print("  Nothing further can be checked on this machine.")
        return 1

    status = result["_nafdac_status"]()
    check("status reports enabled", status.get("enabled") is True)
    check(
        "at least one source is usable",
        status.get("any_source_usable") is True,
        str([(s["name"], s["usable"]) for s in status.get("sources", [])]),
    )
    check(
        "Cell 17 filled the verify hook",
        callable(result["NAFDAC_HOOKS"].get("verify")),
    )
    check(
        "Cell 17 filled the enrich hook",
        callable(result["NAFDAC_HOOKS"].get("enrich")),
    )

    # A real number from the greenbook, verified through the same hook the chat
    # endpoint uses.
    found = result["verify_nafdac_product"]("A4-100160", sources=result["NAFDAC_SOURCES"])
    check("A4-100160 verifies", found.found is True, f"nrn={found.nrn!r}")
    if found.found:
        record = found.primary
        check(
            "the record carries the real product name",
            "Spironolactone" in record.product_name,
            record.product_name,
        )
        check(
            "the record carries an approval date",
            bool(record.approval_date),
            record.approval_date,
        )
        check("sources are named", bool(record.sources), str(record.sources))

    inactive = result["verify_nafdac_product"]("C4-1142", sources=result["NAFDAC_SOURCES"])
    check(
        "an inactive product reports as not authorised",
        inactive.found and inactive.authorised is False,
        f"authorised={inactive.authorised}",
    )

    missing = result["verify_nafdac_product"]("A4-0000000", sources=result["NAFDAC_SOURCES"])
    check("an unknown number is not found", missing.found is False)
    check("an unknown number claims nothing", missing.authorised is None)

    # Extraction out of a sentence, which is the path a real user message takes.
    enriched = result["_nafdac_enrich"](
        [{"role": "user", "content": "Is A4-100160 a genuine product?"}]
    )
    check("a sentence is enriched", enriched is not None)
    if enriched:
        check(
            "the block carries the instruction",
            "NAFDAC" in enriched["content"],
            enriched["content"][:60],
        )
        check(
            "the block carries the product name",
            "Spironolactone" in enriched["content"],
        )

    not_a_lookup = result["_nafdac_enrich"](
        [{"role": "user", "content": "Báwo ni? How are you today?"}]
    )
    check(
        "ordinary conversation is not enriched",
        not_a_lookup is None,
        repr(not_a_lookup)[:60],
    )

    unverified = result["_nafdac_enrich"](
        [{"role": "user", "content": "Is A4-0000000 genuine?"}]
    )
    check(
        "a miss is not passed to the model as context",
        unverified is None,
    )

    search = result["search_products"]("paracetamol", sources=result["NAFDAC_SOURCES"], limit=5)
    check("free-text search returns results", len(search) > 0, f"count={len(search)}")

    print("=" * 70)
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        return 1
    print("All Cell 17 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
