"""Run every QVR example in `docs/examples/source/` through every
backend's target compiler / parser, asserting the transpiled bytes
are syntactically valid in the target language.

This is the gallery-level extension of
[`test_external_syntax.py`][tests.transpile.test_external_syntax]:
that test only exercises the canonical Beta-Bernoulli fixture; this
one drives every example in the documentation gallery through every
backend whose syntax-check tool is on PATH.

Each (backend, example) cell either:

- emits non-empty bytes that the target syntax check accepts, or
- raises `UnsupportedConstruct` from a cell registered in
  `_EXPECTED_UNSUPPORTED` (with the expected kind-prefix).

A cell missing from the registry that nevertheless raises is a
regression. A registered cell that no longer raises is a closed
gap — drop the entry.

The four-tier verification hierarchy:

1. Walker structural assertions ([`test_structural.py`][tests.transpile.test_structural]).
2. Mapping composition laws ([`test_lens_laws.py`][tests.transpile.test_lens_laws]).
3. Target compiler acceptance (this test for the gallery; [`test_external_syntax.py`][tests.transpile.test_external_syntax] for the canonical fixture).
4. Measure equivalence in Docker
   ([`test_numeric_equivalence.py`][tests.transpile.test_numeric_equivalence]).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quivers.dsl.parser import parse
from tests.transpile._tools import require_tool
from quivers.transpile import UnsupportedConstruct, transpile


_GALLERY = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"


def _gallery_examples() -> list[Path]:
    return sorted(_GALLERY.glob("*.qvr"))


# Per-backend external syntax checker: (binary, argv-builder, stdin?).
# Backends with no canonical lint-only tool are skipped (Church has
# no standard interpreter we can lint against; Gen and Turing share
# Julia's `Meta.parse`).
#: Placeholder in an ``argv`` template standing for the path of the
#: file the emitted source is written to.
_SOURCE_PATH = "<source-path>"

#: `(binary, argv, suffix)`. A `suffix` of None feeds the emitted
#: source to the checker on stdin. Otherwise the source is written to
#: a file with that suffix and `_SOURCE_PATH` in `argv` is replaced by
#: its path. The suffix is not cosmetic: `node` picks a module format
#: by extension and refuses a name it does not recognise.
_SYNTAX_CHECKS: dict[str, tuple[str, list[str], str | None]] = {
    "stan": ("stanc", ["stanc", "--info", "-"], None),
    "numpyro": (
        "python",
        ["python", "-c", "import ast, sys; ast.parse(sys.stdin.read())"],
        None,
    ),
    "pyro": (
        "python",
        ["python", "-c", "import ast, sys; ast.parse(sys.stdin.read())"],
        None,
    ),
    "pymc": (
        "python",
        ["python", "-c", "import ast, sys; ast.parse(sys.stdin.read())"],
        None,
    ),
    "edward2": (
        "python",
        ["python", "-c", "import ast, sys; ast.parse(sys.stdin.read())"],
        None,
    ),
    # `node --check` opens its argument as a file. Handed
    # `/dev/stdin` it resolves that through `/proc/<pid>/fd/0`, which
    # is a pipe when the source arrives on stdin, and a pipe is not
    # something it can open. The source goes to a real file, named
    # here by the `_SOURCE_PATH` placeholder.
    "webppl": ("node", ["node", "--check", _SOURCE_PATH], ".js"),
    "turing": (
        "julia",
        [
            "julia",
            "--startup-file=no",
            "--quiet",
            "-e",
            "src = read(stdin, String); Meta.parseall(src)",
        ],
        None,
    ),
    "gen": (
        "julia",
        [
            "julia",
            "--startup-file=no",
            "--quiet",
            "-e",
            "src = read(stdin, String); Meta.parseall(src)",
        ],
        None,
    ),
}


# Cells where the gallery example exercises a construct outside the
# backend's support tier. Each entry pins the expected kind-prefix
# of the raise, so a closed gap surfaces as a test failure
# (`pytest.raises` matches no entry) and a regression surfaces as a
# different-kind raise.
#
# Five boundary classes are represented:
#
# 1. Composition and contraction declarations still have no executable
#    QIEC entry, so those examples fail before target capability analysis.
# 2. Executable search computations (`schema_chart_parser` and the
#    deduction calls in `montague_nli`, `multimodal_tlg`,
#    `quantifier_scope`, and `type_logical`) reach the QIEC boundary,
#    which refuses them on every target for want of a search runtime.
# 3. The term autoencoder reaches the same boundary with typed structural
#    computations, which require a neural attachment no target provides.
#
# The `sum` builtin is deliberately absent from this registry. It
# lowers to each target's own sum-axis reduction (`jnp.sum(...,
# axis=-1)`, `torch.sum(..., dim=-1)`, `pymc.math.sum(..., axis=-1)`,
# `tf.reduce_sum(..., axis=-1)`), so the factor_analysis and ppca
# Python cells transpile rather than raise. Those eight emissions match
# the QVR joint to within 3e-5 across the gallery point set, so the gap
# is closed rather than merely silenced.
#
# Key: (backend, example-stem). Value: kind-prefix the raised
# `UnsupportedConstruct.kinds` must match.
_EXPECTED_UNSUPPORTED: dict[tuple[str, str], str] = {
    # 1. Standalone QIEC computations and structural declarations.
    ("edward2", "schema_chart_parser"): "qiec:capability:search",
    ("gen", "schema_chart_parser"): "qiec:capability:search",
    ("numpyro", "schema_chart_parser"): "qiec:capability:search",
    ("pymc", "schema_chart_parser"): "qiec:capability:search",
    ("pyro", "schema_chart_parser"): "qiec:capability:search",
    ("stan", "schema_chart_parser"): "qiec:capability:search",
    ("turing", "schema_chart_parser"): "qiec:capability:search",
    ("webppl", "schema_chart_parser"): "qiec:capability:search",
    ("edward2", "pmf"): "composition_decl",
    ("gen", "pmf"): "composition_decl",
    ("numpyro", "pmf"): "composition_decl",
    ("pymc", "pmf"): "composition_decl",
    ("pyro", "pmf"): "composition_decl",
    ("stan", "pmf"): "composition_decl",
    ("turing", "pmf"): "composition_decl",
    ("webppl", "pmf"): "composition_decl",
    ("edward2", "tensor_contraction"): "composition_decl",
    ("gen", "tensor_contraction"): "composition_decl",
    ("numpyro", "tensor_contraction"): "composition_decl",
    ("pymc", "tensor_contraction"): "composition_decl",
    ("pyro", "tensor_contraction"): "composition_decl",
    ("stan", "tensor_contraction"): "composition_decl",
    ("turing", "tensor_contraction"): "composition_decl",
    ("webppl", "tensor_contraction"): "composition_decl",
    ("edward2", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("gen", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("numpyro", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("pymc", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("pyro", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("stan", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("turing", "term_autoencoder"): "qiec:capability:neural-attachment",
    ("webppl", "term_autoencoder"): "qiec:capability:neural-attachment",
    # 2. A program calling a deduction is refused at the QIEC boundary
    #    on every target: the deduction enumerates its derivations
    #    through a search handler no target runtime carries.
    **{
        (backend, stem): "qiec:capability:search"
        for stem in (
            "montague_nli",
            "multimodal_tlg",
            "quantifier_scope",
            "type_logical",
        )
        for backend in (
            "bugs",
            "church",
            "edward2",
            "gen",
            "jags",
            "numpyro",
            "pymc",
            "pyro",
            "stan",
            "turing",
            "webppl",
        )
    },
    # 2b. The weighted-grammar examples score a deduction's goal
    #     weight from a program, which reaches the same boundary.
    **{
        (backend, stem): "qiec:capability:search"
        for stem in ("ccg", "custom_rules", "pmcfg")
        for backend in (
            "bugs",
            "church",
            "edward2",
            "gen",
            "jags",
            "numpyro",
            "pymc",
            "pyro",
            "stan",
            "turing",
            "webppl",
        )
    },
    # 3b. A recurrent cell adds two rank-1 operands. Neither BUGS nor
    #     JAGS lifts an infix operator over an axis, and the
    #     elementwise result exists only as a named array built one
    #     index at a time inside a loop of its own, which a let
    #     binding lowered to a single scalar expression has nowhere to
    #     put. Every other target renders these two examples.
}

# 4. Neural morphisms (`param_source=mlp`) compute their mean with a
#    network whose weights are model-internal: they appear in neither
#    the wire form nor the sample sites, so no backend can reconstruct
#    the mean. The transpiler raises rather than emit a meanless
#    observation. Every syntax-check backend is affected.
for _neural_example in (
    "bnn",
    "deep_markov",
    "seq2seq",
    "transformer_lm",
    "vae",
):
    for _syntax_backend in _SYNTAX_CHECKS:
        _EXPECTED_UNSUPPORTED[(_syntax_backend, _neural_example)] = "param-source:mlp"

# 5. `scan(cell)` denotes one draw per sequence position over
#    intermediate states the program never names, and the sequence
#    axis is not a declared object: it arrives with the data. There
#    is no loop bound to emit and no name to bind the per-position
#    states to, so the expansion pass refuses rather than emit a
#    program that reads the cell as a free input and scores a measure
#    the source does not state. The refusal is raised before any
#    renderer runs, so it is the same kind on every target.
for _scan_example in (
    "bidirectional_rnn_lm",
    "gru_lm",
    "lstm_lm",
    "vanilla_rnn_lm",
):
    for _syntax_backend in _SYNTAX_CHECKS:
        _EXPECTED_UNSUPPORTED[(_syntax_backend, _scan_example)] = "scan:no-lowering"


# 6. An ungrouped `marginalize` over a plated `observe` shares one
#    latent across the body's rows, so its density accumulates the
#    rows and reduces once. These targets reduce each row on its
#    own, which gives every row a draw the source never declares, so
#    they refuse rather than emit a different measure; `gen`
#    marginalizes through the same emission as `turing`.
for _ungrouped_backend in (
    "bugs",
    "church",
    "edward2",
    "gen",
    "jags",
    "pymc",
    "turing",
):
    if _ungrouped_backend in _SYNTAX_CHECKS:
        _EXPECTED_UNSUPPORTED[(_ungrouped_backend, "hmm")] = (
            "marginalize:ungrouped-over-plate"
        )


@pytest.mark.parametrize("example", _gallery_examples(), ids=lambda p: p.stem)
@pytest.mark.parametrize("backend", sorted(_SYNTAX_CHECKS))
def test_gallery_example_compiles(example: Path, backend: str, tmp_path: Path) -> None:
    """Transpile a gallery example to `backend` and run its target
    compiler / parser as a syntax check."""
    binary, argv, suffix = _SYNTAX_CHECKS[backend]
    require_tool(binary)

    source = example.read_text()
    cell = (backend, example.stem)
    expected_unsupported = _EXPECTED_UNSUPPORTED.get(cell)
    if expected_unsupported is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(parse(source), target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_unsupported) for k in kinds), (
            f"{backend!r} on {example.name}: expected raise with "
            f"kind prefix {expected_unsupported!r}, got {kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_UNSUPPORTED`) or a different gap fired "
            f"(fix the renderer)."
        )
        return

    emitted = transpile(parse(source), target=backend)

    if suffix is None:
        run_argv, stdin_bytes = argv, emitted
    else:
        script = tmp_path / f"{example.stem}{suffix}"
        script.write_bytes(emitted)
        run_argv = [str(script) if a == _SOURCE_PATH else a for a in argv]
        stdin_bytes = b""
    completed = subprocess.run(
        run_argv,
        input=stdin_bytes,
        capture_output=True,
        timeout=60.0,
    )
    assert completed.returncode == 0, (
        f"{backend!r} compiler rejected {example.name}: "
        f"stdout={completed.stdout.decode('utf-8', errors='replace')!r} "
        f"stderr={completed.stderr.decode('utf-8', errors='replace')!r}\n"
        f"emitted source:\n{emitted.decode('utf-8', errors='replace')}"
    )
