"""Build the Kaggle dataset archive the notebook's Cell 18 needs.

Cell 18 searches these paths for the package, in order:

    /kaggle/input/n-atlas-source/nafdac_source/__init__.py
    /kaggle/input/nafdac-source/nafdac_source/__init__.py
    /kaggle/input/natlas-source/nafdac_source/__init__.py
    /kaggle/input/n-afdac-source/nafdac_source/__init__.py

so the archive must contain `nafdac_source/` at its root. Upload it as a Kaggle
dataset whose slug is one of those four and Cell 18 finds it with no
configuration; otherwise set NAFDAC_SOURCE_DIR to the mount path.

The greenbook CSV goes in `nafdac_scraper/` beside the package, because
nafdac_source.sources.find_default_file walks up from the package directory
looking for nafdac_greenbook.csv under nafdac_scraper/, ., data/ and
nafdac_source/ at each level. That is the same layout as the repository, so the
CSV source works on Kaggle without setting NAFDAC_CSV_PATH.

`.env` is excluded on purpose. One of them lives *inside* the package directory
and holds the live PostgreSQL password, and a Kaggle dataset is not a place for
a credential. Cell 1 already carries the DSN, and a Kaggle Secret named
NAFDAC_DATABASE_URL overrides it there.

Usage:
    python build_kaggle_dataset.py
    python build_kaggle_dataset.py --no-csv      # package only, smaller upload
"""

import argparse
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PACKAGE_DIR = ROOT / "nafdac_source"
CSV_PATH = ROOT / "nafdac_scraper" / "nafdac_greenbook.csv"
TARGET = ROOT / "n-atlas-source.zip"

# Kept deliberately small. tests.py is not imported by __init__.py, and the
# notebook never runs it; excluding it keeps the archive to what is loaded.
INCLUDE_SUFFIXES = (".py", ".sql", ".md", ".txt")

EXCLUDE_NAMES = {
    ".env",
    "__pycache__",
    "tests.py",
    ".pytest_cache",
    ".git",
}


def package_members():
    """(archive_name, source_path) for every file that belongs in the package."""
    if not PACKAGE_DIR.is_dir():
        raise SystemExit(f"missing {PACKAGE_DIR}")
    for path in sorted(PACKAGE_DIR.iterdir()):
        if path.name in EXCLUDE_NAMES or path.is_dir():
            continue
        if path.suffix not in INCLUDE_SUFFIXES:
            continue
        yield f"nafdac_source/{path.name}", path


def csv_member():
    if not CSV_PATH.is_file():
        return None
    return f"nafdac_scraper/{CSV_PATH.name}", CSV_PATH


def build(include_csv: bool = True) -> Path:
    members = list(package_members())
    if include_csv:
        found = csv_member()
        if found is None:
            print(f"note: {CSV_PATH} not found; building without the CSV source.")
        else:
            members.append(found)

    if not any(name.endswith("__init__.py") for name, _ in members):
        raise SystemExit("refusing to build: nafdac_source/__init__.py is missing")

    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in members:
            archive.write(path, name)

    return TARGET


def verify(archive_path: Path) -> int:
    """Prove the archive has the shape Cell 18 and find_default_file expect."""
    problems = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()

        if "nafdac_source/__init__.py" not in names:
            problems.append(
                "missing nafdac_source/__init__.py at the archive root. The "
                "package files must sit inside a nafdac_source/ directory: "
                "zipping the folder from a file browser flattens them to the "
                "archive root, where Python cannot import them as "
                "nafdac_source at all."
            )
        for name in names:
            if name.endswith(".env"):
                problems.append(f"credential file leaked into the archive: {name}")
            if "__pycache__" in name:
                problems.append(f"bytecode leaked into the archive: {name}")
            if "/" not in name:
                problems.append(
                    f"{name} sits at the archive root instead of inside a "
                    "directory; see the note about nafdac_source/__init__.py"
                )
        for required in (
            "nafdac_source/sources.py",
            "nafdac_source/verify.py",
            "nafdac_source/records.py",
            "nafdac_source/extract.py",
            "nafdac_source/formatting.py",
            "nafdac_source/schema.sql",
        ):
            if required not in names:
                problems.append(f"missing {required}")

        print(f"archive: {archive_path}")
        print(f"  {len(names)} entries, {archive_path.stat().st_size / 1024:.0f} KB")
        for name in names:
            print(f"  {name}")

    if problems:
        print("\nINVALID")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nVALID: nafdac_source/ is at the archive root, no credentials, "
          "CSV laid out for autodetect.")
    print("       Upload this as a Kaggle dataset with slug 'n-atlas-source'.")
    print("       Do not re-zip the folder by hand: that flattens the package "
          "and Cell 18 cannot import it.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="omit nafdac_greenbook.csv; PostgreSQL becomes the only source",
    )
    args = parser.parse_args(argv)
    return verify(build(include_csv=not args.no_csv))


if __name__ == "__main__":
    sys.exit(main())