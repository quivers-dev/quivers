"""Gallery-wide compile, fit, and doc-block sweep.

Every ``docs/examples/source/*.qvr`` example compiles. Each deduction
example fits its corpus: a MAP fit drives the negative log partition
function down, and NUTS on the lifted Bayesian model runs to completion
with finite log-density and positive acceptance. Every monadic example's
SVI + NUTS contract is the ``Try it`` block of its page, executed
verbatim by :func:`test_gallery_try_it_blocks_execute`, so the docs and
the sweep share one source of truth.

How this stays in sync with the gallery
---------------------------------------

The suite follows the contents of ``docs/examples/`` in two ways:

* The ``stem`` parameter list is computed at collection time by
  :func:`_all_example_stems`, which globs
  ``docs/examples/source/*.qvr``. Adding (or deleting) a ``.qvr``
  file adds or removes a parametrised test case, and a deduction
  example must register the corpus it parses in
  ``_DEDUCTION_CORPORA``.

* The ``Try it`` code blocks inside ``docs/examples/*.md`` are
  extracted by :func:`test_gallery_try_it_blocks_execute` and
  executed under a sandboxed namespace. An HTML comment
  (``<!-- pytest: skip -->``) excludes illustrative pseudo-code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from quivers.dsl import load
from quivers.dsl.parser import parse_file
from quivers.dsl.compiler import Compiler
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import (
    adam_fit_deduction,
    nuts_program_from_deduction,
)


# The corpus each deduction example parses, keyed by its stem: the
# deduction's name and the sentences its rules and lexicon license.
_DEDUCTION_CORPORA: dict[str, tuple[str, list[list[str]]]] = {
    "ccg": (
        "CCG",
        [
            ["the", "cat", "sleeps"],
            ["the", "dog", "whom", "Mary", "saw", "sleeps"],
            ["the", "cat", "that", "sleeps", "today", "purrs"],
            ["Mary", "probably", "sleeps"],
            ["Mary", "saw", "yesterday", "the", "cat"],
        ],
    ),
    "custom_rules": (
        "AB",
        [
            ["the", "dog", "runs"],
            ["the", "dog", "whom", "Mary", "saw", "runs"],
            ["the", "dog", "that", "runs", "today", "barks"],
        ],
    ),
    "multimodal_tlg": (
        "MMTLG",
        [["every", "linguist", "that", "admires", "Kim", "sleeps"]],
    ),
    "type_logical": ("Lambek", [["Kim", "gives", "Lee", "books"]]),
    "pcfg": ("PCFG", [["the", "cat", "sleeps"]]),
    "wmcfg": (
        "WMCFG",
        [
            ["mer", "em", "Hans", "es", "huus", "hälfed", "aastriiche"],
            [
                "mer",
                "d'chind",
                "em",
                "Hans",
                "es",
                "huus",
                "lönd",
                "hälfe",
                "aastriiche",
            ],
        ],
    ),
    "montague_nli": ("Montague", [["every", "dog", "barks"]]),
    "quantifier_scope": ("QScope", [["every", "linguist", "admires", "some", "model"]]),
}


def _all_example_stems() -> list[str]:
    return sorted(p.stem for p in Path("docs/examples/source").glob("*.qvr"))


@pytest.mark.parametrize("stem", _all_example_stems())
def test_gallery_example_compiles(stem: str) -> None:
    """Every gallery example parses and compiles, and every deduction it
    declares has a registered corpus."""
    path = Path(f"docs/examples/source/{stem}.qvr")
    mod = parse_file(str(path))
    Compiler(mod).compile_env()
    prog = load(str(path))
    if prog.deductions:
        assert stem in _DEDUCTION_CORPORA, (
            f"{stem} declares the deduction(s) {sorted(prog.deductions)} but "
            "registers no corpus in _DEDUCTION_CORPORA"
        )
        name, _ = _DEDUCTION_CORPORA[stem]
        assert name in prog.deductions


@pytest.mark.parametrize("stem", sorted(_DEDUCTION_CORPORA))
def test_gallery_deduction_fits(stem: str) -> None:
    """MAP + NUTS contract for one deduction example on its corpus."""
    prog = load(f"docs/examples/source/{stem}.qvr")
    ded_name, corpus = _DEDUCTION_CORPORA[stem]
    ded = prog.deductions[ded_name]
    # MAP fit: log Z should rise (loss should fall).
    history = adam_fit_deduction(
        ded,
        corpus,
        steps=60,
        lr=5e-2,
        prior_scale=1.0,
    )
    assert history, f"{stem}: the MAP fit recorded no loss"
    assert history[-1] <= history[0] + 1.0, (
        f"{stem}: MAP loss did not decrease ({history[0]:.2f} -> {history[-1]:.2f})"
    )
    # NUTS on the lifted Bayesian model.
    model, x, obs = nuts_program_from_deduction(
        ded,
        corpus,
        prior_scale=1.0,
    )
    kernel = NUTSKernel(
        step_size=0.05,
        max_tree_depth=3,
        target_accept=0.8,
    )
    mc = MCMC(kernel, num_warmup=8, num_samples=8, num_chains=1)
    torch.manual_seed(0)
    res = mc.run(model, x, obs)
    assert torch.isfinite(res.log_densities).all(), (
        f"{stem}: NUTS chain contains non-finite log densities"
    )
    assert float(res.acceptance_rates.mean()) > 0.05, (
        f"{stem}: NUTS acceptance too low ({float(res.acceptance_rates.mean()):.2f})"
    )


# ---------------------------------------------------------------------------
# Try-it doc-block executor — keeps the docs in sync with the gallery.
# ---------------------------------------------------------------------------


_TRY_IT_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
_SKIP_MARKER = "<!-- pytest: skip -->"


def _extract_try_it_blocks(md_text):
    """Pull out every fenced ``python ... `` block under a
    ``## Try it`` heading, dropping blocks immediately preceded
    by an HTML ``<!-- pytest: skip -->`` opt-out comment."""
    out = []
    try_it_pos = md_text.find("## Try it")
    if try_it_pos < 0:
        return out
    tail = md_text[try_it_pos:]
    nxt = re.search(r"\n## [^\n]", tail)
    body = tail[: nxt.start()] if nxt else tail
    for m in _TRY_IT_RE.finditer(body):
        start = m.start()
        prev = body.rfind("\n", 0, start - 1)
        prev_line = body[prev + 1 : start - 1] if prev >= 0 else ""
        if _SKIP_MARKER in prev_line:
            continue
        out.append(m.group(1))
    return out


_DOC_MD_FILES = sorted(
    p.name
    for p in Path("docs/examples").glob("*.md")
    if p.name not in {"index.md", "README.md"}
)


@pytest.mark.slow
@pytest.mark.parametrize("doc_name", _DOC_MD_FILES)
def test_gallery_try_it_blocks_execute(doc_name):
    """Execute every ``## Try it`` Python block in the doc. A block
    that names a missing helper or breaks under the current compiler
    fails the suite, keeping the docs honest about what the framework
    supports today. The blocks of a page share one namespace and run
    in order, so a later block sees the imports and bindings an
    earlier one established, exactly as a reader stepping through the
    page would. Blocks may opt out via a ``<!-- pytest: skip -->``
    HTML comment immediately above the fenced block."""
    path = Path(f"docs/examples/{doc_name}")
    blocks = _extract_try_it_blocks(path.read_text())
    assert blocks, f"{doc_name}: every example page carries a Try-it block"
    ns = {"__name__": f"_try_it_{path.stem}"}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{doc_name}::block-{i}", "exec"), ns)
        except SystemExit:
            pass
