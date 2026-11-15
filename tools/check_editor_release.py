"""Fail when the checked-in VSIX differs from its authoritative sources."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "editors" / "vscode-qvr"
PARITY_FILES = (
    "LICENSE",
    "README.md",
    "package.json",
    "language-configuration.json",
    "syntaxes/qvr.tmLanguage.json",
    "out/extension.js",
    "out/extension.js.map",
)


def check_grammar_pin() -> int:
    """Require Zed's immutable parser revision to match every current parser input.

    Fetch the revision before running this check in a shallow checkout. Matching
    the input bytes also makes the current-parser query tests apply to Zed.
    """
    zed = ROOT / "editors/zed-extension-qvr"
    manifest = tomllib.loads((zed / "extension.toml").read_text())
    grammar = manifest["grammars"]["qvr"]
    revision = grammar["rev"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        sys.stderr.write("Zed grammar rev must be a full immutable commit SHA\n")
        return 1
    failures = []
    for relative in (
        "grammar.js",
        "src/parser.c",
        "src/scanner.c",
        "src/node-types.json",
        "src/grammar.json",
        "src/tree_sitter/parser.h",
        "src/tree_sitter/alloc.h",
        "src/tree_sitter/array.h",
        "queries/highlights.scm",
    ):
        path = f"grammars/qvr/{relative}"
        result = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode or result.stdout != (ROOT / path).read_bytes():
            failures.append(path)
    if (zed / "languages/qvr/highlights.scm").read_bytes() != (
        ROOT / "grammars/qvr/queries/highlights.scm"
    ).read_bytes():
        failures.append("Zed highlight query mirror")
    if failures:
        sys.stderr.write(
            f"Zed grammar pin {revision} is missing or stale: {', '.join(failures)}\n"
        )
        return 1
    print(f"OK Zed grammar {revision}: parser inputs and highlight queries match")
    return 0


def main(*, full: bool = False) -> int:
    manifest = json.loads((EXTENSION / "package.json").read_text())
    version = manifest["version"]
    expected = EXTENSION / f"vscode-qvr-{version}.vsix"
    archives = sorted(EXTENSION.glob("*.vsix"))
    if archives != [expected]:
        rendered = ", ".join(path.name for path in archives) or "none"
        sys.stderr.write(
            f"expected only {expected.name}; found {rendered}. "
            "Run `npm ci && npm run package` in editors/vscode-qvr.\n"
        )
        return 1
    failures: list[str] = []
    with ZipFile(expected) as archive:
        parity_files = list(PARITY_FILES)
        if full:
            parity_files.extend(
                name.removeprefix("extension/")
                for name in archive.namelist()
                if name.startswith("extension/node_modules/") and not name.endswith("/")
            )
        for relative in parity_files:
            # vsce normalizes conventional metadata filenames in the archive.
            archived_name = {
                "LICENSE": "LICENSE.txt",
                "README.md": "readme.md",
            }.get(relative, relative)
            archived = archive.read(f"extension/{archived_name}")
            current_path = EXTENSION / relative
            if not current_path.is_file():
                failures.append(f"{relative} (missing)")
                continue
            current = current_path.read_bytes()
            if archived != current:
                failures.append(relative)
    if failures:
        sys.stderr.write(
            f"{expected.name} has stale files: {', '.join(failures)}. "
            "Run `npm ci && npm run package` in editors/vscode-qvr.\n"
        )
        return 1
    print(f"OK {expected.name}: {len(parity_files)} source/archive files match")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        check_grammar_pin()
        if "--grammar-pin" in sys.argv[1:]
        else main(full="--full" in sys.argv[1:])
    )
