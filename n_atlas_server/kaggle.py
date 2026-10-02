"""Run the N-ATLaS Kaggle notebook as a plain Python script.

Some people prefer a script over a .ipynb, and Kaggle also accepts a plain
Python file. This runs the exact same cells as natlas_notebook.py — there is no
second implementation.

    python kaggle.py                    # all 18 cells, then exit
    python kaggle.py --cells 2-8        # config + tokenizer + model + engine only
    python kaggle.py --cells 2-11       # also builds and starts the API server
    python kaggle.py --list             # show the cell map

Cells are numbered to match the notebook. The default range stops at Cell 18 on
purpose: Cell 19 logs in and opens a tunnel, which blocks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebook_cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the N-ATLaS Kaggle server cells as a script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cells",
        default="1-18",
        help="cell range to run, 1-based inclusive (e.g. 2-8, 11, 1-18)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the cell map and exit",
    )
    return parser


def print_cell_map() -> None:
    source = notebook_cells.read_source()
    notebook_cells.verify_numbering(source)
    titles = notebook_cells.cell_titles(source)
    print(f"{notebook_cells.NOTEBOOK_SOURCE}")
    print(f"{len(titles)} code cells")
    print("-" * 70)
    for index, title in enumerate(titles, start=1):
        print(f"{index:>2}. {title}")
    print("-" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print_cell_map()
        return 0

    total = notebook_cells.cell_count()
    try:
        start, end = notebook_cells.parse_cell_range(args.cells, total)
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"running cells {start}-{end} of {total}")
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_cells.NOTEBOOK_SOURCE),
    }
    notebook_cells.exec_range(start, end, namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
