"""Lightweight notebook integrity checks.

This script deliberately does not claim that the historical 2022 notebooks are
fully reproducible. It checks that every notebook is valid JSON/notebook format
and that the modern quickstart has Python code that can be parsed after IPython
magics are neutralized.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
MODERN_NOTEBOOK = ROOT / "notebooks" / "00_quickstart_pythainlp.ipynb"


def sanitize_ipython(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith(("!", "%")):
            lines.append(f"{indent}pass  # IPython magic/shell command")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_notebook(path: Path, check_python_syntax: bool = False) -> None:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)

    if check_python_syntax:
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = sanitize_ipython(cell.source)
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise SyntaxError(
                    f"{path}: code cell {index} is not valid Python after "
                    f"sanitizing IPython magics: {exc}"
                ) from exc


def main() -> None:
    notebooks = sorted(ROOT.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found.")

    for path in notebooks:
        validate_notebook(
            path,
            check_python_syntax=(path.resolve() == MODERN_NOTEBOOK.resolve()),
        )
        print(f"OK: {path.relative_to(ROOT)}")

    print(f"Validated {len(notebooks)} notebook(s).")


if __name__ == "__main__":
    main()
