"""Regression gates for distributable QVR editor support."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import re
import runpy
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
VSCODE = ROOT / "editors" / "vscode-qvr"
ZED = ROOT / "editors" / "zed-extension-qvr"


def _repository() -> dict[str, object]:
    grammar = json.loads((VSCODE / "syntaxes/qvr.tmLanguage.json").read_text())
    return grammar["repository"]


def test_textmate_recognizes_the_complete_qiec_vocabulary() -> None:
    repository = _repository()
    source = (
        "index Nat = Z | S(Nat)\n"
        "family Vec[A : Type](n : Nat) : Type\n"
        "effect State[S : Type]\n"
        "instance cell : State[Int]\n"
        "handler run for State[Int] : Int -> Int [coverage=total, implementation=foreign]\n"
        "define read() : Int !{cell} = handle cell with run in\n"
        "    let x <- perform cell.get()\n"
        "    return x\n"
    )
    declaration = re.compile(repository["keyword-declaration"]["match"])
    control = re.compile(repository["keyword-control"]["match"])
    modifier = re.compile(repository["keyword-qiec-modifier"]["match"])
    assert {
        "index",
        "family",
        "effect",
        "instance",
        "handler",
        "define",
    } <= set(declaration.findall(source))
    assert {"handle", "with", "let", "perform", "return"} <= set(
        control.findall(source)
    )
    assert {"Type", "coverage", "total"} <= set(modifier.findall(source))


def test_textmate_recognizes_every_distribution_family() -> None:
    from quivers.dsl.family_schemas import DISTRIBUTION_FAMILIES

    pattern = re.compile(_repository()["distribution-family"]["match"])
    assert all(pattern.fullmatch(name) for name in DISTRIBUTION_FAMILIES)


def test_zed_uses_an_immutable_grammar_revision() -> None:
    manifest = tomllib.loads((ZED / "extension.toml").read_text())
    revision = manifest["grammars"]["qvr"]["commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_vscode_archive_matches_release_sources() -> None:
    from tools.check_editor_release import main

    assert main() == 0


def test_vscode_forwards_live_target_configuration_to_the_lsp() -> None:
    source = (VSCODE / "src" / "extension.ts").read_text()
    compiled = (VSCODE / "out" / "extension.js").read_text()
    assert 'configurationSection: "qvr"' in source
    assert 'configurationSection: "qvr"' in compiled


def test_textual_input_registers_the_wheel_qvr_grammar() -> None:
    from textual.widgets import TextArea

    from quivers.cli.repl_tui import _enable_qvr_highlighting

    area = TextArea()
    assert _enable_qvr_highlighting(area)
    assert area.language == "qvr"


# ---------------------------------------------------------------------------
# One token corpus, classified alike by every highlighter.
# ---------------------------------------------------------------------------


_CORPUS = (
    "effect Echo\n"
    "    ping : Int -> Int\n\n"
    "define twice(x : Int) : Int !{} =\n"
    "    let y <- twice(x)\n"
    "    return y\n\n"
    "define mapped(xs : Tensor[Real]([2])) : Tensor[Real]([2]) !{} =\n"
    "    let doubled = map(xs, x -> x * 2.0)\n"
    "    let checked <- traverse(doubled, x -> twice(x))\n"
    "    return checked\n\n"
    "handler doubling for Echo : Int -> Int [coverage=total, implementation=authored]\n"
    "    ping(n : Int) resumes 1 =>\n"
    "        resume(n)\n"
    "\n"
    "object Row : FinSet 8\n"
    "program regression : Row -> Row [effects=[Sample, Score]]\n"
    "    sample location <- Normal(loc=0.0, scale=2.0)\n"
    "    observe y : Row <- Normal(loc=location, scale=1.0)\n"
    "    return y\n"
)

#: Each token of interest by position, with the class every highlighter
#: must give it: (line, text) -> class.
_EXPECTED: dict[tuple[int, str], str] = {
    (0, "effect"): "keyword",
    (3, "define"): "keyword",
    (1, "ping"): "function",
    (3, "twice"): "function",
    (4, "twice"): "function",
    (8, "map"): "function",
    (9, "traverse"): "function",
    (12, "handler"): "keyword",
    (12, "[coverage"): "keyword",
    (12, "total"): "keyword",
    (12, "implementation"): "keyword",
    (13, "ping"): "function",
    (13, "resumes"): "keyword",
    (13, "1"): "number",
    (14, "resume"): "keyword",
    (1, "Int"): "type",
    (16, "object"): "keyword",
    (17, "program"): "keyword",
    (18, "sample"): "keyword",
    (18, "Normal"): "type",
    (19, "observe"): "keyword",
    (19, "Normal"): "type",
    (20, "return"): "keyword",
}

#: The tree-sitter capture names and Pygments token families that stand
#: for each class.
_TREE_SITTER_CLASSES = {
    "keyword": {"keyword"},
    "function": {
        "function",
        "function.builtin",
        "function.call",
        "function.method",
    },
    "number": {"constant.builtin", "number"},
    "type": {"type", "type.builtin"},
}


def _repl_classes() -> dict[tuple[int, str], str]:
    from quivers.cli.repl_highlight import tokenize

    return {
        (span.line, span.text): span.token
        for span in tokenize(_CORPUS)
        if (span.line, span.text) in _EXPECTED
    }


def _pygments_classes() -> dict[tuple[int, str], str]:
    from pygments.token import Keyword, Name, Number

    from quivers.dsl.pygments_lexer import QvrLexer

    families = (
        (Keyword, "keyword"),
        (Number, "number"),
        (Name.Class, "type"),
        (Name.Builtin, "function"),
        (Name.Function, "function"),
    )
    out: dict[tuple[int, str], str] = {}
    for index, token, text in QvrLexer().get_tokens_unprocessed(_CORPUS):
        line = _CORPUS.count("\n", 0, index)
        if (line, text) not in _EXPECTED:
            continue
        for family, name in families:
            if token in family:
                out[(line, text)] = name
                break
    return out


def _docs_hook_namespace() -> dict[str, object]:
    """Load the MkDocs lexer while containing its deliberate monkey-patch."""
    import pygments.lexers

    targets = [pygments.lexers]
    try:
        import markdown.extensions.codehilite as codehilite

        targets.append(codehilite)
    except ImportError:
        pass
    try:
        import pymdownx.highlight as highlight

        targets.append(highlight)
    except ImportError:
        pass
    originals = {
        (module, "get_lexer_by_name"): getattr(module, "get_lexer_by_name", None)
        for module in targets
    }
    try:
        return runpy.run_path(str(ROOT / "docs/hooks/register_qvr_lexer.py"))
    finally:
        for (module, name), value in originals.items():
            if value is None:
                if hasattr(module, name):
                    delattr(module, name)
            else:
                setattr(module, name, value)


def _docs_classes() -> dict[tuple[int, str], str]:
    from pygments.token import Keyword, Name, Number

    lexer = _docs_hook_namespace()["QvrLexer"]()
    families = (
        (Keyword, "keyword"),
        (Number, "number"),
        (Name.Class, "type"),
        (Name.Builtin, "function"),
        (Name.Function, "function"),
    )
    out: dict[tuple[int, str], str] = {}
    for index, token, text in lexer.get_tokens_unprocessed(_CORPUS):
        line = _CORPUS.count("\n", 0, index)
        if (line, text) not in _EXPECTED:
            continue
        for family, name in families:
            if token in family:
                out[(line, text)] = name
                break
    return out


def _tree_sitter_classes() -> dict[tuple[int, str], str]:
    import tree_sitter

    from quivers.dsl.pygments_lexer import _load_parser

    parser, language, _ = _load_parser()
    source = _CORPUS.encode()
    tree = parser.parse(source)
    query = tree_sitter.Query(
        language, (ROOT / "grammars/qvr/queries/highlights.scm").read_text()
    )
    captures = tree_sitter.QueryCursor(query).captures(tree.root_node)
    out: dict[tuple[int, str], str] = {}
    for capture, nodes in captures.items():
        for node in nodes:
            key = (
                node.start_point.row,
                source[node.start_byte : node.end_byte].decode(),
            )
            if key not in _EXPECTED:
                continue
            for name, names in _TREE_SITTER_CLASSES.items():
                if capture in names:
                    out[key] = name
    return out


def _textmate_classes() -> dict[tuple[int, str], str]:
    repository = _repository()
    rules = {
        "keyword": ("keyword-control", "keyword-declaration", "keyword-qiec-modifier"),
        "function": ("builtin-function",),
        "type": ("builtin-type", "distribution-family"),
    }
    out: dict[tuple[int, str], str] = {}
    for (line, text), expected in _EXPECTED.items():
        word = text.lstrip("[")
        for name, rule_names in rules.items():
            if any(
                re.fullmatch(repository[rule]["match"].replace("\\b", ""), f"({word})")
                or re.search(repository[rule]["match"], word)
                for rule in rule_names
            ):
                out[(line, text)] = name
                break
    return out


def test_highlighters_classify_the_corpus_alike() -> None:
    """The REPL tokenizer, the Pygments lexer, and the tree-sitter query
    give every token of the corpus the same class, and the TextMate
    grammar agrees on every keyword and builtin it names."""
    assert _repl_classes() == _EXPECTED
    assert _pygments_classes() == _EXPECTED
    assert _docs_classes() == _EXPECTED
    assert _tree_sitter_classes() == _EXPECTED
    textmate = _textmate_classes()
    for key, expected in _EXPECTED.items():
        if key in textmate:
            assert textmate[key] == expected, key
    assert {key for key, value in _EXPECTED.items() if value == "keyword"} <= set(
        textmate
    )


def test_docs_highlighter_uses_the_authoritative_lexer() -> None:
    """MkDocs must not carry a second, drift-prone QVR vocabulary."""
    from quivers.dsl.pygments_lexer import QvrLexer

    assert _docs_hook_namespace()["QvrLexer"] is QvrLexer


def test_docs_highlighter_check_tolerates_missing_docs_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dev test environment need not install the documentation toolchain."""
    from quivers.dsl.pygments_lexer import QvrLexer

    real_import = builtins.__import__

    def _without_docs_extras(name, *args, **kwargs):
        if name.startswith(("markdown", "pymdownx")):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_docs_extras)
    assert _docs_hook_namespace()["QvrLexer"] is QvrLexer
