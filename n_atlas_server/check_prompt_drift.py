"""Fail when the prompt copies and confam_prompt.txt disagree.

`confam_prompt.txt` at the project root is the single source of truth: the
client resolves it at runtime and the notebook serves it from the uploaded
dataset. The two Python constants are fallbacks for a machine where the text
file was not copied, and DEFAULT_SYSTEM_PROMPT is the notebook's last resort
before the built-in prompt. None of them is read when the file is present, so
nothing complains when one of them goes stale — this does.

Exit code 0 when every copy matches, 1 otherwise. No network, no imports from
the project: the constants are read with ast, so this runs before the client's
dependencies are installed.

Usage:
    python n_atlas_server/check_prompt_drift.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / "confam_prompt.txt"

COPIES = (
    (ROOT / "n_atlas_client" / "prompt.py", "CONFAM_AI_SYSTEM_PROMPT"),
    (ROOT / "n_atlas_client" / "n_atlas.py", "CONFAM_AI_SYSTEM_PROMPT"),
    (ROOT / "n_atlas_server" / "natlas_notebook.py", "DEFAULT_SYSTEM_PROMPT"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _constant(path: Path, name: str) -> str:
    """The literal value assigned to `name` at module level."""
    tree = ast.parse(_read(path), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", "") == name for target in node.targets):
            continue
        try:
            return str(ast.literal_eval(node.value)).strip()
        except ValueError:
            raise SystemExit(
                "%s: %s is not a plain string literal, so it cannot be compared"
                % (path.name, name)
            )
    raise SystemExit("%s: no module-level %s found" % (path.name, name))


def _first_difference(left: str, right: str):
    """(1-based line number, left line, right line) for the first difference."""
    left_lines = left.splitlines()
    right_lines = right.splitlines()
    for index in range(max(len(left_lines), len(right_lines))):
        a = left_lines[index] if index < len(left_lines) else "<no line>"
        b = right_lines[index] if index < len(right_lines) else "<no line>"
        if a != b:
            return index + 1, a, b
    return None


def main() -> int:
    if not PROMPT_FILE.is_file():
        print("INVALID: %s is missing." % PROMPT_FILE)
        print("        It is the single source of truth for the persona.")
        return 1

    truth = _read(PROMPT_FILE)
    if not truth:
        print("INVALID: %s is empty." % PROMPT_FILE.name)
        return 1

    print("confam_prompt.txt: %d chars, %d lines"
          % (len(truth), len(truth.splitlines())))

    problems = 0
    for path, name in COPIES:
        label = "%s: %s" % (path.parent.name + "/" + path.name, name)
        if not path.is_file():
            print("  MISSING  %s" % label)
            problems += 1
            continue
        try:
            value = _constant(path, name)
        except SystemExit as exc:
            print("  UNREADABLE %s (%s)" % (label, exc))
            problems += 1
            continue
        if value == truth:
            print("  ok       %s" % label)
            continue
        problems += 1
        print("  DRIFT    %s" % label)
        difference = _first_difference(truth, value)
        if difference:
            line, expected, found = difference
            print("           first difference at line %d" % line)
            print("           confam_prompt.txt: %s" % expected[:100])
            print("           constant:          %s" % found[:100])

    if problems:
        print("\nINVALID: %d copy/copies differ from %s."
              % (problems, PROMPT_FILE.name))
        print("Copy the file's text into the constants above, or delete a")
        print("constant if the file is the only copy you want to keep.")
        return 1

    print("\nVALID: every copy matches confam_prompt.txt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())