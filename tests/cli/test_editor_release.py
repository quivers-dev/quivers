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
COUNT_FAMILIES = (
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedBinomial",
    "ZeroInflatedBetaBinomial",
    "HurdleNegativeBinomial",
)
SHAPE_FAMILIES = (
    "SkewNormal",
    "ShiftedLogNormal",
    "Frechet",
    "GeneralizedExtremeValue",
)


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


def test_textmate_recognizes_every_checked_pure_builtin() -> None:
    from quivers.dsl.pure_builtins import PURE_BUILTINS

    pattern = re.compile(_repository()["builtin-function"]["match"])
    assert all(pattern.fullmatch(name) for name in PURE_BUILTINS)


@pytest.mark.parametrize("name", ["observation_weight", "scale_weight"])
def test_checked_weight_builtins_are_repl_completion_candidates(name: str) -> None:
    from quivers.cli.repl_complete import all_completions
    from quivers.cli.repl_session import ReplSession

    assert any(
        item.text == name and item.kind == "function"
        for item in all_completions(ReplSession(), name[:5])
    )


@pytest.mark.parametrize(
    "name",
    (
        *COUNT_FAMILIES,
        *SHAPE_FAMILIES,
        "DiscreteWeibull",
        "ZeroInflatedAsymmetricLaplace",
    ),
)
def test_response_families_are_repl_and_lsp_completion_candidates(name: str) -> None:
    from lsprotocol import types as lsp

    from quivers.cli.repl_complete import all_completions
    from quivers.cli.repl_session import ReplSession
    from quivers.lsp.server import build_server
    from tests.transpile.fixtures._generate_families import (
        _INLINE_OBSERVE,
        _inline_observe_source,
    )

    prefix = name[:10]
    assert any(
        item.text == name and item.kind == "type"
        for item in all_completions(ReplSession(), prefix)
    )
    source = _inline_observe_source(name, _INLINE_OBSERVE[name])
    fixture = ROOT / f"tests/transpile/fixtures/families/{name.lower()}.qvr"
    assert fixture.read_text() == source
    server = build_server()
    server.text_document_publish_diagnostics = lambda params: None
    uri = f"file:///tmp/{name}.qvr"
    server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN](
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri=uri, language_id="qvr", version=1, text=source
            )
        )
    )
    line = source.splitlines()[4]
    completion = server.protocol.fm.features[lsp.TEXT_DOCUMENT_COMPLETION](
        lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=4, character=line.index(name) + len(prefix)),
        )
    )
    assert any(
        item.label == name and item.kind == lsp.CompletionItemKind.Class
        for item in completion.items
    )


def test_zed_uses_an_immutable_grammar_revision() -> None:
    manifest = tomllib.loads((ZED / "extension.toml").read_text())
    revision = manifest["grammars"]["qvr"]["rev"]
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
    "\n"
    "define checked(w : Real) : Real !{} =\n"
    "    return observation_weight(w)\n"
    "define scaled(w : Real, ell : Real) : Real !{} =\n"
    "    return weight_value(scale_weight(w, weight(ell)))\n"
    "\n"
    "object CountRow : FinSet 2\n"
    "program counts : CountRow -> CountRow\n"
    "    observe a : CountRow <- ZeroInflatedNegativeBinomial(zero_prob=0.2, total_count=2.0, probs=0.5)\n"
    "    observe b : CountRow <- ZeroInflatedBinomial(zero_prob=0.2, total_count=4.0, probs=0.5)\n"
    "    observe c : CountRow <- ZeroInflatedBetaBinomial(zero_prob=0.2, total_count=4.0, concentration1=2.0, concentration0=2.0)\n"
    "    observe d : CountRow <- HurdleNegativeBinomial(zero_prob=0.2, total_count=2.0, probs=0.5)\n"
    "    observe e : CountRow <- SkewNormal(loc=0.2, scale=1.1, skewness=0.7)\n"
    "    observe f : CountRow <- ShiftedLogNormal(loc=0.2, scale=0.5, shift=0.1)\n"
    "    observe g : CountRow <- Frechet(scale=1.3, concentration=3.0)\n"
    "    observe h : CountRow <- GeneralizedExtremeValue(loc=0.2, scale=1.1, concentration=0.1)\n"
    "    observe i : CountRow <- DiscreteWeibull(probs=0.4, concentration=1.3)\n"
    "    observe j : CountRow <- ZeroInflatedAsymmetricLaplace(zero_prob=0.2, loc=0.1, scale=0.7, quantile=0.35)\n"
    "    return a\n"
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
    (23, "observation_weight"): "function",
    (25, "scale_weight"): "function",
    (29, "ZeroInflatedNegativeBinomial"): "type",
    (30, "ZeroInflatedBinomial"): "type",
    (31, "ZeroInflatedBetaBinomial"): "type",
    (32, "HurdleNegativeBinomial"): "type",
    (33, "SkewNormal"): "type",
    (34, "ShiftedLogNormal"): "type",
    (35, "Frechet"): "type",
    (36, "GeneralizedExtremeValue"): "type",
    (37, "DiscreteWeibull"): "type",
    (38, "ZeroInflatedAsymmetricLaplace"): "type",
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
    assert {(23, "observation_weight"), (25, "scale_weight")} <= set(textmate)
    assert {(29 + index, name) for index, name in enumerate(COUNT_FAMILIES)} <= set(
        textmate
    )
    assert {(33 + index, name) for index, name in enumerate(SHAPE_FAMILIES)} <= set(
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


def test_zed_manifest_has_a_compiled_language_server_adapter() -> None:
    manifest = tomllib.loads((ZED / "extension.toml").read_text())
    assert set(manifest["grammars"]["qvr"]) == {"repository", "rev", "path"}
    assert "binary" not in manifest["language_servers"]["qvr-lsp"]
    cargo = tomllib.loads((ZED / "Cargo.toml").read_text())
    assert cargo["package"]["version"] == manifest["version"]
    assert cargo["lib"]["crate-type"] == ["cdylib"]


def test_grammar_pin_check_rejects_a_stale_parser(monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    from tools import check_editor_release as checker

    def git_show(args, **kwargs):
        relative = args[-1].split(":", 1)[1]
        data = (ROOT / relative).read_bytes()
        if relative.endswith("parser.c"):
            data += b"stale"
        return SimpleNamespace(returncode=0, stdout=data)

    monkeypatch.setattr(checker.subprocess, "run", git_show)
    assert checker.check_grammar_pin() == 1
    assert "parser.c" in capsys.readouterr().err
