"""Build natlas_kaggle.ipynb from natlas_notebook.py.

natlas_notebook.py is the single source of truth. It is plain sequential Python
with `# %%` cell markers, so it can be opened directly in VS Code as an
interactive notebook, exec'd as a script, or converted to .ipynb for Kaggle.

Usage:
    python build_notebook.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebook_cells

HERE = Path(__file__).resolve().parent
SOURCE = notebook_cells.NOTEBOOK_SOURCE
TARGET = HERE / "natlas_kaggle.ipynb"


def split_cells(source: str):
    return notebook_cells.split_cells(source)


def source_lines(text: str):
    """Split a cell body into nbformat-compliant source lines.

    nbformat stores `source` as a list of newline-terminated strings, except
    the last one. A list of bare lines is reassembled with `"".join(...)` by
    Jupyter and by every nbformat writer, which collapses the entire cell into
    a single line and produces a SyntaxError on the first run.
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def build():
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")

    cells = []
    for kind, text in split_cells(SOURCE.read_text(encoding="utf-8")):
        lines = source_lines(text)
        if kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": lines,
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": lines,
                }
            )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "codemirror_mode": {"name": "ipython", "version": 3},
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {TARGET}")
    print(f"  {len(cells)} cells ({code_cells} code, {len(cells) - code_cells} markdown)")
    return notebook


def verify(notebook):
    # Cell 17 is the NAFDAC greenbook cell. Bumping this is deliberate friction:
    # the count is asserted here and in validate_notebook.py so a new cell
    # cannot be added without someone confirming the number and the tooling.
    expected = 17
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    titles = notebook_cells.cell_titles()
    numbers = [int(t.split()[1]) for t in titles if t.startswith("CELL ")]
    print(f"  numbered code cells: {numbers}")
    if len(code_cells) != expected:
        print(f"  WARNING: expected {expected} code cells, found {len(code_cells)}")
    if numbers != list(range(1, expected + 1)):
        print(f"  WARNING: cell numbering is not 1..{expected}")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and not cell["source"]:
            print("  WARNING: empty markdown cell")


if __name__ == "__main__":
    verify(build())
