"""Shared loader for the `# %%`-delimited cells in natlas_notebook.py.

natlas_notebook.py is the single source of truth for the N-ATLaS server. This
module lets the standalone scripts (kaggleserver.py, apiserver.py, kaggle.py)
and the notebook builder all consume those exact cells instead of keeping
drifting copies of the logic.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
NOTEBOOK_SOURCE = HERE / "natlas_notebook.py"

CELL_MARKER_RE = re.compile(r"^#\s*%%\s*(?:\[[a-zA-Z]+\])?\s*$")
TITLE_RE = re.compile(r"^#\s*(CELL\s+\d+\s*[-—:].*)$")


def read_source(source_path: Optional[Path] = None) -> str:
    path = Path(source_path) if source_path else NOTEBOOK_SOURCE
    if not path.exists():
        raise FileNotFoundError(f"not found: {path}")
    return path.read_text(encoding="utf-8")


def split_cells(source: str) -> List[Tuple[str, str]]:
    """Split source into [("markdown"|"code", text), ...]."""
    cells: List[Tuple[str, str]] = []
    kind = "code"
    buffer: List[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip("\n")
        if text.strip():
            cells.append((kind, text.strip()))

    for line in source.split("\n"):
        if CELL_MARKER_RE.match(line.strip()):
            flush()
            buffer = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buffer.append(line)
    flush()
    return cells

def code_cells(source: Optional[str] = None) -> List[str]:
    return [text for kind, text in split_cells(source if source is not None else read_source()) if kind == "code"]


def cell_count(source: Optional[str] = None) -> int:
    return len(code_cells(source))


def cell_titles(source: Optional[str] = None) -> List[str]:
    titles = []
    for text in code_cells(source):
        title = None
        for line in text.split("\n"):
            match = TITLE_RE.match(line.strip())
            if match:
                title = match.group(1)
                break
        titles.append(title or "(untitled)")
    return titles


def verify_numbering(source: Optional[str] = None) -> None:
    """Every code cell must carry a matching `# CELL N` banner."""
    source = source if source is not None else read_source()
    titles = cell_titles(source)
    for index, title in enumerate(titles, start=1):
        match = re.match(r"CELL\s+(\d+)", title)
        if not match:
            raise ValueError(f"code cell {index} has no '# CELL N' banner: {title!r}")
        if int(match.group(1)) != index:
            raise ValueError(
                f"code cell {index} is banner-numbered '{title}'; numbering must be sequential"
            )


def exec_range(
    start: int,
    end: int,
    namespace: Dict[str, object],
    source_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute code cells `start..end` (1-based, inclusive) into `namespace`."""
    path = Path(source_path) if source_path else NOTEBOOK_SOURCE
    source = read_source(path)
    verify_numbering(source)
    cells = code_cells(source)
    total = len(cells)
    if not (1 <= start <= end <= total):
        raise ValueError(f"cell range {start}-{end} is outside 1-{total}")

    body = "\n\n".join(cells[start - 1 : end])
    namespace.setdefault("__file__", str(path))
    exec(compile(body, f"{path.name}:cells {start}-{end}", "exec"), namespace)
    return namespace


def parse_cell_range(spec: str, total: Optional[int] = None) -> Tuple[int, int]:
    total = total if total is not None else cell_count()
    spec = str(spec).strip()
    if not spec:
        raise ValueError("empty cell range")
    if "-" in spec:
        lo_text, _, hi_text = spec.partition("-")
        start, end = int(lo_text), int(hi_text)
    else:
        start = end = int(spec)
    if not (1 <= start <= end <= total):
        raise ValueError(f"cell range {start}-{end} is outside 1-{total}")
    return start, end
