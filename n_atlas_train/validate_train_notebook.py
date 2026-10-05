"""Validate the generated training .ipynb.

Same three things the server's validate_notebook.py checks, plus one that only
matters here: that every cell carries its `# CELL N` banner and that the cells
reassemble exactly into train_notebook.py, so the artefact on Kaggle is provably
the one in this folder.

Usage:
    python validate_train_notebook.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "confam_train_kaggle.ipynb"
SOURCE = HERE / "train_notebook.py"

# Cell 1 config, 2 deps, 3 HF login, 4 dataset checks, 5 model, 6 render,
# 7 train, 8 push, 9 behavioural check.
EXPECTED_CODE_CELLS = 9

problems = []

nb = json.loads(TARGET.read_text(encoding="utf-8"))

if nb.get("nbformat") != 4:
    problems.append(f"unexpected nbformat {nb.get('nbformat')}")

if "accelerator" not in json.dumps(nb.get("metadata", {})):
    problems.append("GPU accelerator metadata missing")

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
    if cell["cell_type"] == "code" and "!" in "".join(source).split("\n")[0][:1]:
        problems.append(f"cell {index} starts with a '!' shell escape")

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

# The dataset filename the notebook looks for has to be the one this repo ships,
# or the Kaggle upload and the notebook disagree about the input.
dataset_name = "confam_training_dataset.jsonl"
if dataset_name not in TARGET.read_text(encoding="utf-8"):
    problems.append(f"notebook does not reference {dataset_name}")

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
    problems.append("notebook cells do not reassemble into train_notebook.py")

if problems:
    print("INVALID")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(f"VALID: {EXPECTED_CODE_CELLS} code cells compile, nbformat source is "
      "newline-terminated,")
print("       markdown cell is unwrapped, no shell escapes, and the cells")
print("       reassemble exactly into train_notebook.py.")
