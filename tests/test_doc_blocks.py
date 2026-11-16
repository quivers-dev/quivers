"""Compile QVR documentation and run tutorials and marked Python examples.

QVR blocks
----------

Each fenced ```qvr block under ``docs/`` (and ``README.md``) is treated
as a test case. An uppercase ```R fence is treated the same way; the root
README uses GitHub's R lexer as the closest available syntax-highlighting
fallback for QVR. The block's *compile mode* is chosen by an HTML comment on
the line immediately preceding the opening fence:

* ``<!-- compile: false -->``      — not a test case; the block is
  illustrative prose (e.g. a bind step shown outside any program body).
* ``<!-- compile: cumulative -->`` — concatenate this block with every
  prior cumulative block in the same file before compiling. Use when a
  guide walks the reader through one model in incrementally-elaborated
  fragments.
* ``<!-- compile: qiec -->``       — parse and lower the block through the
  exact QVR to QIEC route beside the categorical Program compiler.
* ``<!-- compile: qiec-cumulative -->`` — concatenate successive blocks and
  lower each accumulated source through QIEC.
* (no marker)                      — ``standalone``: the block must
  compile on its own.

Python blocks
-------------

Each fenced ```python block under ``docs/tutorials/`` is executed in
order, with per-file shared state: every block sees the namespace
built by the previous blocks in the same file (matching the way a
reader copies the chapter into a REPL). Outside tutorials, an explicit
``<!-- python: run -->`` marker opts an example into the same execution.
The block's mode is chosen by
the same HTML-comment surface, scoped to a separate marker so QVR and
Python markers don't collide:

* ``<!-- python: skip -->`` — not a test case. Use for blocks that
  reference on-disk files (``open("foo.qvr")``), shell commands, or
  illustrative fragments that aren't meant to run.
* ``<!-- python: run -->`` — run in any documentation page.
* (no marker) — run in tutorials; elsewhere remain illustrative.

Run with the rest of the suite::

    pytest tests/test_doc_blocks.py

CI gates the doc surface on this file: a release-blocking failure here
means a published example doesn't compile or doesn't run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quivers.dsl import loads

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_ROOTS = (_REPO_ROOT / "docs", _REPO_ROOT / "README.md")
_TUTORIAL_ROOT = _REPO_ROOT / "docs" / "tutorials"

_QVR_FENCE_RE = re.compile(r"^([ \t]*)```(?:qvr|R)\n(.*?)^\1```", re.M | re.S)
_PY_FENCE_RE = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.M | re.S)


def _dedent_fence_body(indent: str, body: str) -> str:
    """Strip ``indent`` from the start of every line in ``body``.

    Material-for-MkDocs requires fenced blocks inside ``=== "..."`` tabs
    to be indented (typically by four spaces). The fence regex captures
    that indent so we can strip it before compiling.
    """
    if not indent:
        return body
    out_lines: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.startswith(indent):
            out_lines.append(line[len(indent) :])
        elif line.strip() == "":
            out_lines.append(line)
        else:
            # A line shorter than ``indent`` that isn't blank: leave it
            # alone so the compiler reports an accurate column.
            out_lines.append(line)
    return "".join(out_lines)


_QVR_MARKER_RE = re.compile(
    r"<!--\s*compile:\s*"
    r"(false|standalone|cumulative|qiec|qiec-cumulative)\s*-->"
)
_PY_MARKER_RE = re.compile(r"<!--\s*python:\s*(skip|run)\s*-->")


def _qvr_marker_before(text: str, fence_start: int) -> str:
    """Return the compile-mode marker on the line preceding ``fence_start``."""
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _QVR_MARKER_RE.search(prev_line)
    return m.group(1) if m else "standalone"


def _py_marker_before(text: str, fence_start: int, *, default: str = "run") -> str:
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _PY_MARKER_RE.search(prev_line)
    return m.group(1) if m else default


def _iter_md_files() -> list[Path]:
    files: list[Path] = []
    for root in _DOC_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
        elif root.is_file():
            files.append(root)
    return files


def _collect_qvr_blocks() -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for md in _iter_md_files():
        text = md.read_text()
        rel = str(md.relative_to(_REPO_ROOT))
        cumulative_prefix = ""
        for idx, m in enumerate(_QVR_FENCE_RE.finditer(text)):
            indent, body = m.group(1), m.group(2)
            body = _dedent_fence_body(indent, body)
            mode = _qvr_marker_before(text, m.start())
            if mode == "false":
                continue
            if mode in ("cumulative", "qiec-cumulative"):
                source = cumulative_prefix + body
                cumulative_prefix = source + "\n"
            else:
                source = body
            out.append((rel, idx, mode, source))
    return out


def _collect_py_blocks() -> list[tuple[str, int, str, str]]:
    """Collect tutorial blocks and explicitly runnable examples on other pages."""
    out: list[tuple[str, int, str, str]] = []
    for md in _iter_md_files():
        text = md.read_text()
        rel = str(md.relative_to(_REPO_ROOT))
        default = "run" if md.is_relative_to(_TUTORIAL_ROOT) else "skip"
        for idx, m in enumerate(_PY_FENCE_RE.finditer(text)):
            indent, body = m.group(1), m.group(2)
            body = _dedent_fence_body(indent, body)
            mode = _py_marker_before(text, m.start(), default=default)
            if mode == "skip":
                continue
            out.append((rel, idx, mode, body))
    return out


_QVR_BLOCKS = _collect_qvr_blocks()
_PY_BLOCKS = _collect_py_blocks()


@pytest.mark.parametrize(
    ("path", "index", "mode", "source"),
    _QVR_BLOCKS,
    ids=[f"{p}:blk{i}:{m}" for p, i, m, _ in _QVR_BLOCKS],
)
def test_qvr_doc_block(path: str, index: int, mode: str, source: str) -> None:
    del path  # carried only for readable test ids
    if mode in ("qiec", "qiec-cumulative"):
        from quivers.dsl.parser import parse
        from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

        lower_qvr_to_qiec(parse(source), module_name=f"docs.block.{index}")
        return
    loads(source)


# Per-file namespaces accumulated as python blocks execute in file order.
_PY_NAMESPACES: dict[str, dict] = {}


@pytest.mark.parametrize(
    ("path", "index", "mode", "source"),
    _PY_BLOCKS,
    ids=[f"{p}:pyblk{i}:{m}" for p, i, m, _ in _PY_BLOCKS],
)
def test_python_doc_block(path: str, index: int, mode: str, source: str) -> None:
    del mode  # every collected block runs
    ns = _PY_NAMESPACES.setdefault(path, {"__name__": "__doc_block__"})
    try:
        exec(compile(source, f"{path}:pyblk{index}", "exec"), ns)
    except Exception:
        # Make subsequent blocks in the same file see a clean slate so
        # one failure doesn't cascade.
        _PY_NAMESPACES[path] = {"__name__": "__doc_block__"}
        raise
