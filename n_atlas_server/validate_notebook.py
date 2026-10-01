"""Validate the generated .ipynb: JSON shape, nbformat source encoding, per-cell Python syntax."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "natlas_kaggle.ipynb"
SOURCE = HERE / "natlas_notebook.py"

problems = []

nb = json.loads(TARGET.read_text(encoding="utf-8"))

if nb.get("nbformat") != 4:
    problems.append(f"unexpected nbformat {nb.get('nbformat')}")

if "accelerator" not in json.dumps(nb.get("metadata", {})):
    problems.append("GPU accelerator metadata missing")

EXPECTED_CODE_CELLS = 17  # Cells 1-16 plus Cell 17, the NAFDAC greenbook cell.

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
if len(code_cells) != EXPECTED_CODE_CELLS:
    problems.append(
        f"expected {EXPECTED_CODE_CELLS} code cells, found {len(code_cells)}"
    )

for index, cell in enumerate(nb["cells"], start=1):
    source = cell["source"]
    if isinstance(source, str):
        problems.append(f"cell {index} source must be a list of lines, not a string")
        continue
    if len(source) > 1:
        if any(not line.endswith("\n") for line in source[:-1]):
            problems.append(
                f"cell {index} source lines are missing their newline terminators; "
                'Jupyter rejoins them with "" and the whole cell collapses to one line'
            )
        if source[-1].endswith("\n"):
            problems.append(f"cell {index} last source line has a trailing newline")
    if "".join(source).count("# %%"):
        problems.append(f"cell {index} still contains a '# %%' cell marker")

for index, cell in enumerate(code_cells, start=1):
    text = "".join(cell["source"])
    try:
        compile(text, f"cell-{index}", "exec")
    except SyntaxError as exc:
        problems.append(f"code cell {index} does not compile: {exc}")
    if not any(line.startswith(f"# CELL {index}") for line in text.split("\n")):
        problems.append(f"code cell {index} is missing its '# CELL {index}' banner")

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        if "".join(cell["source"]).strip().startswith('"""'):
            problems.append("markdown cell still wrapped in a docstring")

try:
    compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")
except SyntaxError as exc:
    problems.append(f"{SOURCE.name} does not compile: {exc}")

source_lines_kept = [
    line
    for line in SOURCE.read_text(encoding="utf-8").split("\n")
    if line.strip() and not line.strip().startswith("# %%")
]
notebook_lines_kept = [
    line.rstrip("\n")
    for cell in nb["cells"]
    for line in cell["source"]
    if line.strip()
]
if source_lines_kept != notebook_lines_kept:
    problems.append("notebook cells do not reassemble into natlas_notebook.py")

if problems:
    print("INVALID")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(f"VALID: {EXPECTED_CODE_CELLS} code cells compile, nbformat source is "
      "newline-terminated,")
print("       markdown cell is unwrapped, and the cells reassemble exactly")
print("       into natlas_notebook.py.")
