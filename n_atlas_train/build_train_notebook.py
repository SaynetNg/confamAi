"""Build confam_train_kaggle.ipynb from train_notebook.py.

train_notebook.py is the single source of truth. It is plain sequential Python
with `# %%` cell markers, so it can be opened directly in VS Code as an
interactive notebook, exec'd as a script, or converted to .ipynb for Kaggle.

This is notebook A of the two-notebook split: it trains the adapter and serves
nothing. The cell-splitting and line-encoding logic is imported from the server
folder rather than copied, because a drifting second copy of that logic is how
"it validates locally but breaks on Kaggle" happens.

Usage:
    python build_train_notebook.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "train_notebook.py"
TARGET = HERE / "confam_train_kaggle.ipynb"

sys.path.insert(0, str(HERE.parent / "n_atlas_server"))

import notebook_cells  # noqa: E402

EXPECTED_CODE_CELLS = 9


def source_lines(text: str):
    """Split a cell body into nbformat-compliant source lines.

    nbformat stores `source` as a list of newline-terminated strings, except the
    last one. A list of bare lines is reassembled with `"".join(...)` by Jupyter
    and by every nbformat writer, which collapses the entire cell into a single
    line and produces a SyntaxError on the first run.
    """
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + lines[-1:]


def build():
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")

    cells = []
    for kind, text in notebook_cells.split_cells(
        SOURCE.read_text(encoding="utf-8")
    ):
        lines = source_lines(text)
        if kind == "markdown":
            cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": lines}
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

    TARGET.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    code_cells = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {TARGET}")
    print(f"  {len(cells)} cells ({code_cells} code, {len(cells) - code_cells} markdown)")
    return notebook


def verify(notebook):
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    titles = notebook_cells.cell_titles(SOURCE.read_text(encoding="utf-8"))
    numbers = [int(t.split()[1]) for t in titles if t.startswith("CELL ")]
    print(f"  numbered code cells: {numbers}")
    if len(code_cells) != EXPECTED_CODE_CELLS:
        print(f"  WARNING: expected {EXPECTED_CODE_CELLS} code cells, "
              f"found {len(code_cells)}")
    if numbers != list(range(1, EXPECTED_CODE_CELLS + 1)):
        print("  WARNING: cell numbering is not 1..%d" % EXPECTED_CODE_CELLS)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and not cell["source"]:
            print("  WARNING: empty markdown cell")


if __name__ == "__main__":
    verify(build())
