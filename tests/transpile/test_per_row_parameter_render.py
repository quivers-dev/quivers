"""A per-row value filling a family parameter is rendered at its row.

The emitted text of the three per-row fixtures that
`test_per_row_parameter_numeric.py` scores in the probe tier, pinned
without Docker:

- Turing reciprocates a per-row Gamma rate inside the broadcast family
  call with the broadcast `inv.`, since `inv` of a vector is a
  `MethodError`, and keeps the scalar `inv` on a scalar rate.
- JAGS reads a per-row host datum at the loop's row, as BUGS does.
- The plan names the axis of a `factor` let after its binder's object
  even when another object shares its cardinality, so WebPPL reads a
  per-row let inside a marginalize atom at the observation's row.
"""

from __future__ import annotations

import pathlib

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.ir import IRDeterministic, IRMarginalize, IRNode
from quivers.transpile.plan import Lower

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "steps"


def _emit(stem: str, target: str) -> str:
    """A step fixture transpiled to `target`, without spaces or newlines.

    Parameters
    ----------
    stem : str
        The fixture's file stem under ``fixtures/steps``.
    target : str
        The transpile target.

    Returns
    -------
    str
        The emitted program with whitespace removed.
    """
    source = (_FIXTURES / f"{stem}.qvr").read_text()
    text = transpile(parse(source), target=target).decode()
    return "".join(text.split())


def _deterministics(nodes: tuple[IRNode, ...]) -> dict[str, IRDeterministic]:
    """Every deterministic binding of a plan body, marginalize scopes included.

    Parameters
    ----------
    nodes : tuple[IRNode, ...]
        The body.

    Returns
    -------
    dict[str, IRDeterministic]
        Each binding by name.
    """
    found: dict[str, IRDeterministic] = {}
    for node in nodes:
        if isinstance(node, IRDeterministic):
            found[node.name] = node
        elif isinstance(node, IRMarginalize):
            found.update(_deterministics(node.scope))
    return found


def test_turing_broadcasts_the_reciprocal_of_a_per_row_rate() -> None:
    emitted = _emit("observe_per_row_gamma_rate", "turing")
    assert "product_distribution(Gamma.(nu,inv.(rate)))" in emitted
    assert "Gamma(2,inv(0.5))" in emitted
    assert "inv(rate)" not in emitted


def test_jags_reads_a_per_row_datum_at_the_row() -> None:
    emitted = _emit("observe_per_row_datum_scale", "jags")
    assert "y[m_School]~dnorm(mu,1/(se[m_School]*se[m_School]))" in emitted
    assert "1/(se*se)" not in emitted


def test_jags_and_bugs_index_the_datum_alike() -> None:
    jags = _emit("observe_per_row_datum_scale", "jags")
    bugs = _emit("observe_per_row_datum_scale", "bugs")
    relation = "y[m_School]~dnorm(mu,1/(se[m_School]*se[m_School]))"
    assert relation in jags
    assert relation in bugs


def test_a_factor_let_is_planned_over_its_binders_object() -> None:
    source = (_FIXTURES / "marginalize_factor_let_rows.qvr").read_text()
    module = expand_composite_lets(parse(source), target="webppl")
    plan = Lower().forward(module, target="webppl")
    bindings = _deterministics(plan.body)
    assert [str(d.name) for d in bindings["tau_prior"].plate.batch_dims] == ["Tau"]
    assert [str(d.name) for d in bindings["after"].plate.batch_dims] == ["Step"]
    assert [str(d.name) for d in bindings["rate"].plate.batch_dims] == ["Step"]


def test_webppl_scores_a_marginalize_atom_at_the_row() -> None:
    emitted = _emit("marginalize_factor_let_rows", "webppl")
    assert emitted.count("_qvr_poisson_score({mu:rate[n]},y_n)") == 4
    assert "{mu:rate}" not in emitted
