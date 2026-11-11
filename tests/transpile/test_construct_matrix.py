"""Tier 4: (construct fixture × backend) compatibility matrix.

For every fixture under `fixtures/{statements, steps,
let_expressions, options, axes}` and every registered backend, the
test asserts one of three pre-declared outcomes:

1. **Construct out of tier**: the fixture's statement kind is not in
   the backend's `_SUPPORT_TIER` frozenset; the pipeline MUST raise
   `UnsupportedConstruct` with at least one of the predicted kinds.
2. **Orthogonal concern**: the fixture's statement kind IS in the
   tier, but the fixture also exercises a feature the backend
   cannot represent (a family without a target_name; a let-expression
   shape with no method-call surface; a downstream IR node kind the
   renderer hasn't wired). Each such cell is pinned in
   `_EXPECTED_ORTHOGONAL_RAISES` with the kind-prefix the raise
   must carry. A closed gap surfaces as a test failure
   (pytest.raises matches no entry); a regression surfaces as a
   different-kind raise.
3. **Render**: not in either bucket; the pipeline MUST emit
   non-empty bytes.

Thus every cell is a positive assertion, with no xfail-on-exception
dispatch.
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import (
    UnsupportedConstruct,
    available_targets,
    transpile,
    unsupported_for,
)
from quivers.transpile._api import (
    CATEGORICAL_METADATA_IGNORABLE,
    CHURCH_LIKE,
    QIEC_SURFACE,
    STAN_LIKE,
    STRUCTURAL_QIEC,
)
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


# Backend → support tier. Each tier is a frozenset of the QVR
# `Statement` discriminators the backend's pipeline (frontend
# expand + Lower + Renderer) accepts.
_SUPPORT_TIER: dict[str, frozenset[str]] = {
    "stan": STAN_LIKE,
    "numpyro": STAN_LIKE,
    "pyro": STAN_LIKE,
    "pymc": STAN_LIKE,
    "edward2": STAN_LIKE,
    "church": CHURCH_LIKE,
    "webppl": CHURCH_LIKE,
    "turing": STAN_LIKE,
    "gen": STAN_LIKE,
    "bugs": STAN_LIKE,
    "jags": STAN_LIKE,
}


# Some categorical fixtures introduce a non-supported statement
# alongside the program_decl. The construct test for those fixtures
# expects `UnsupportedConstruct`.
def _expected_unsupported_kinds(fixture: _load.Fixture, backend: str) -> set[str]:
    """Return the set of kinds the backend should report as
    unsupported for ``fixture``.

    Walks the parsed module and collects every statement discriminator
    that falls outside the backend's support tier.
    [`CATEGORICAL_METADATA_IGNORABLE`][quivers.transpile.CATEGORICAL_METADATA_IGNORABLE]
    kinds are accepted alongside a `program_decl` per the
    [`unsupported_for`][quivers.transpile.unsupported_for] contract,
    so they're not flagged when a program is present.
    """
    module = parse(fixture.source)
    tier = _SUPPORT_TIER[backend]
    kinds = {str(stmt.kind) for stmt in module.statements}
    has_program = "program_decl" in kinds
    effective_tier = tier | QIEC_SURFACE
    if has_program:
        effective_tier |= CATEGORICAL_METADATA_IGNORABLE
    if "signature_decl" in kinds and kinds & {"encoder_decl", "decoder_decl"}:
        effective_tier |= STRUCTURAL_QIEC
    return {k for k in kinds if k not in effective_tier}


# Cells where the fixture's statement kind IS in the backend's
# support tier, but a different concern (the resolved distribution
# family, a let-expression shape, or a downstream IR node) provokes
# `UnsupportedConstruct` before the construct gate fires. Each entry
# pins the expected kind-prefix of the raise.
#
# Key: (backend, fixture-category, fixture-name).
# Value: the prefix the raised `UnsupportedConstruct.kinds` must
# match (`"family:"`, `"let-expr:"`, `"node:"`, `"declare:"`, etc.).
#
# A closed gap surfaces as a test failure because `pytest.raises`
# succeeds but the assertion-of-prefix is unreachable; remove the
# entry. A regression surfaces because the raised kinds match a
# different prefix; either update the entry or fix the renderer.
_EXPECTED_ORTHOGONAL_RAISES: dict[tuple[str, str, str], str] = {
    # Static graphical targets preserve only the closed, monomorphic,
    # effect-free scalar QIEC fragment.
    ("stan", "statements", "qiec_effectful_computation"): "qiec:",
    # A program calling an effectful helper: a graph language has no
    # statement that runs a computation, and Stan cannot define a
    # helper that performs.
    ("bugs", "steps", "call_step"): "call:graph:",
    ("jags", "steps", "call_step"): "call:graph:",
    ("stan", "steps", "call_step"): "qiec:capability:",
    ("bugs", "statements", "qiec_effectful_computation"): "qiec:",
    ("jags", "statements", "qiec_effectful_computation"): "qiec:",
    # A program calling a deduction, `parse(D, sentence)` followed by
    # `chart.goal_weight()`, is refused at the QIEC boundary on every
    # target: the deduction enumerates its derivations through a search
    # handler no target runtime carries.
    **{
        (backend, "let_expressions", "let_expr_method_call"): "qiec:"
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
    # BUGS / JAGS dialects ship no MatrixNormal surface, so the
    # axes/matrix_kronecker fixture trips the family-target-name
    # check before the construct gate.
    ("bugs", "axes", "matrix_kronecker"): "family:",
    ("jags", "axes", "matrix_kronecker"): "family:",
    # A grouped block whose rows fibre into its groups sums each atom's
    # rows per group before the reduction, and BUGS has no statement
    # that adds the per-group log-density to the joint.
    ("bugs", "steps", "marginalize_step"): "marginalize:grouped-fibration",
}

# A marginalize block with no index into the observe's plate shares one
# latent across the rows; a target that samples each row's latent on
# its own would score a different measure, so it refuses the block.
for _backend in ("bugs", "church", "edward2", "gen", "jags", "pymc", "turing"):
    for _fixture in (
        "marginalize_body_let_chain",
        "marginalize_body_plated_let",
        "marginalize_latent_gather",
        "marginalize_latent_value",
        "marginalize_let_probs",
        "marginalize_let_vector_probs",
    ):
        _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", _fixture)] = (
            "marginalize:ungrouped-over-plate"
        )
# A scan's sequence extent is runtime data, not a declared object a target
# can use as a loop bound, so every target refuses a scan by its cell's
# name, whatever bindings the cell reads.
for _backend in _BACKENDS:
    for _fixture in (
        "scan_cell_reads_vector",
        "scan_cell_reads_matrix",
        "scan_cell_reads_matrix_data",
        "scan_initial_binding",
    ):
        _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", _fixture)] = (
            "scan:no-lowering:cell"
        )
# A learned function's network weights are model-internal, neither sites
# nor data, so every target refuses a weighted function by its network;
# latent weights are sites, but no target has a form for the network's
# layers applied to them.
for _backend in _BACKENDS:
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", "function_linear")] = (
        "param-source:linear"
    )
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", "kernel_latent_weights")] = (
        "param-source:linear:latent"
    )
# BUGS / JAGS `sum` contracts every axis of a two-axis operand, where the
# QVR reduction collapses only the last.
for _backend in ("bugs", "jags"):
    _EXPECTED_ORTHOGONAL_RAISES[
        (_backend, "let_expressions", "let_host_matrix_row")
    ] = "let-expr:axis-reduction"

# A let reduction collapses only the last axis of its argument, and BUGS /
# JAGS have no partial reduction of a two-axis table.
for _backend in ("bugs", "jags"):
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "let_expressions", "let_reduce_matrix")] = (
        "let-expr:axis-reduction"
    )
# No target spells the reduction of a rank-three factor's last axis inside
# its row loop: read at one row, the factor still has two axes.
for _backend in ("bugs", "church", "gen", "jags", "stan", "turing"):
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "let_expressions", "let_reduce_nested")] = (
        "let-expr:axis-reduction"
    )

# A host datum filling a vector-valued family parameter of a plated draw
# holds one vector per row; a target whose library cannot read a parameter
# vector per row refuses it rather than score a different model, and the
# families a target lacks are refused by name.
for _backend in ("bugs", "church", "gen", "jags", "pymc", "stan", "turing", "webppl"):
    for _fixture in ("observe_vector_parameter_data", "observe_probs_data"):
        _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", _fixture)] = (
            "data:per-row-vector-parameter"
        )
for _backend in ("pymc", "stan"):
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", "observe_logits_data")] = (
        "data:per-row-vector-parameter"
    )
for _backend in ("bugs", "church", "gen", "jags", "turing", "webppl"):
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", "observe_logits_data")] = (
        "family:Categorical:logits"
    )
_EXPECTED_ORTHOGONAL_RAISES[("edward2", "steps", "observe_vector_parameter_data")] = (
    "family:OrderedLogistic"
)

# Vector- and matrix-valued families a target ships no class for, or
# whose literal arguments it cannot take, are refused by name.
_EXPECTED_ORTHOGONAL_RAISES.update(
    {
        ("bugs", "steps", "sample_mvn_factor"): "arg:list-literal",
        ("church", "steps", "sample_mvn_factor"): "arg:matrix-literal",
        ("jags", "steps", "sample_mvn_factor"): "family:MultivariateNormal",
        ("bugs", "steps", "sample_mvn_sampled_loc"): "arg:matrix-literal",
        ("church", "steps", "sample_mvn_sampled_loc"): "arg:matrix-literal",
        ("jags", "steps", "sample_mvn_sampled_loc"): "family:MultivariateNormal",
        ("bugs", "steps", "sample_wishart_factor"): "arg:matrix-literal",
        ("jags", "steps", "sample_wishart_factor"): "arg:matrix-literal",
        ("church", "steps", "sample_wishart_factor"): "family:Wishart",
        ("edward2", "steps", "sample_wishart_factor"): "family:Wishart",
        ("gen", "steps", "sample_wishart_factor"): "family:Wishart",
        ("webppl", "steps", "sample_wishart_factor"): "family:no-webppl-target:Wishart",
        ("bugs", "steps", "sample_onehot_probs"): "arg:list-literal",
        ("church", "steps", "sample_onehot_probs"): "family:OneHotCategorical",
        ("gen", "steps", "sample_onehot_probs"): "family:OneHotCategorical",
        ("jags", "steps", "sample_onehot_probs"): "family:no-target-name",
        ("pymc", "steps", "sample_onehot_probs"): "family:OneHotCategorical",
        ("stan", "steps", "sample_onehot_probs"): "family:no-stan-target",
        ("turing", "steps", "sample_onehot_probs"): "family:OneHotCategorical",
        ("webppl", "steps", "sample_onehot_probs"): "family:no-webppl-target",
        ("bugs", "steps", "sample_onehot_sampled_probs"): "arg:list-literal",
        ("church", "steps", "sample_onehot_sampled_probs"): "family:OneHotCategorical",
        ("gen", "steps", "sample_onehot_sampled_probs"): "family:OneHotCategorical",
        ("jags", "steps", "sample_onehot_sampled_probs"): "arg:list-literal",
        ("pymc", "steps", "sample_onehot_sampled_probs"): "family:OneHotCategorical",
        ("stan", "steps", "sample_onehot_sampled_probs"): "family:no-stan-target",
        ("turing", "steps", "sample_onehot_sampled_probs"): "family:OneHotCategorical",
        ("webppl", "steps", "sample_onehot_sampled_probs"): "family:no-webppl-target",
        ("bugs", "steps", "sample_relaxed_onehot_probs"): "arg:list-literal",
        ("church", "steps", "sample_relaxed_onehot_probs"): (
            "family:RelaxedOneHotCategorical"
        ),
        ("gen", "steps", "sample_relaxed_onehot_probs"): (
            "family:RelaxedOneHotCategorical"
        ),
        ("jags", "steps", "sample_relaxed_onehot_probs"): "family:no-target-name",
        ("numpyro", "steps", "sample_relaxed_onehot_probs"): "family:no-target-name",
        ("pymc", "steps", "sample_relaxed_onehot_probs"): (
            "family:RelaxedOneHotCategorical"
        ),
        ("stan", "steps", "sample_relaxed_onehot_probs"): "declare:simplex",
        ("turing", "steps", "sample_relaxed_onehot_probs"): (
            "family:RelaxedOneHotCategorical"
        ),
        ("webppl", "steps", "sample_relaxed_onehot_probs"): "family:no-webppl-target",
    }
)

# An observed class code counts from zero and `dcat` reads it from one;
# BUGS has no data block to shift the codes in and no free log-density
# statement to score them otherwise.
for _category, _fixture in (
    ("let_expressions", "let_row_gather"),
    ("let_expressions", "let_row_gather_sampled"),
    ("steps", "observe_class_code"),
):
    _EXPECTED_ORTHOGONAL_RAISES[("bugs", _category, _fixture)] = "observe:class-code"
# Stan's samplers draw no discrete parameter, so a sampled class is one
# it cannot carry.
_EXPECTED_ORTHOGONAL_RAISES[("stan", "steps", "sample_class_code")] = (
    "sample:discrete-latent"
)
# A gather by a sampled class draws that class, which Stan cannot carry;
# BUGS / JAGS take no list-literal probability vector; Turing has no
# lift of a residual event axis over a plate, and WebPPL carries one
# batch axis.
for _fixture in ("observe_gather_event_vector", "observe_gather_matrix_row"):
    _EXPECTED_ORTHOGONAL_RAISES[("stan", "steps", _fixture)] = "sample:discrete-latent"
    _EXPECTED_ORTHOGONAL_RAISES[("bugs", "steps", _fixture)] = "arg:list-literal"
    _EXPECTED_ORTHOGONAL_RAISES[("jags", "steps", _fixture)] = "arg:list-literal"
_EXPECTED_ORTHOGONAL_RAISES[("turing", "steps", "observe_gather_matrix_row")] = (
    "sample:event-axis-lift"
)
_EXPECTED_ORTHOGONAL_RAISES[("webppl", "steps", "observe_gather_matrix_row")] = (
    "batch-rank:2"
)
# An ordinal outcome is scored by a family only some targets' libraries
# carry; BUGS also has no list-literal cutpoint vector.
_EXPECTED_ORTHOGONAL_RAISES[("bugs", "steps", "observe_ordinal_code")] = (
    "arg:list-literal"
)
for _backend in ("church", "edward2", "gen", "jags", "turing", "webppl"):
    _EXPECTED_ORTHOGONAL_RAISES[(_backend, "steps", "observe_ordinal_code")] = "family:"

# A site drawn or observed over more `over` axes than its family's event
# rank is plated over several axes, and WebPPL carries one batch axis.
for _fixture in (
    "observe_independent_array",
    "scalar_independent_array",
    "vector_independent_array",
):
    _EXPECTED_ORTHOGONAL_RAISES[("webppl", "axes", _fixture)] = "batch-rank:"

for _backend in _BACKENDS:
    for _fixture in ("encoder_decl", "decoder_decl"):
        _EXPECTED_ORTHOGONAL_RAISES[(_backend, "statements", _fixture)] = (
            "qiec:capability:neural-attachment"
        )


def _construct_fixtures() -> list[_load.Fixture]:
    """Every fixture across the per-construct categories."""
    return (
        _load.load_statements()
        + _load.load_steps()
        + _load.load_let_expressions()
        + _load.load_options()
        + _load.load_axes()
    )


def _transpile_with_tier_check(module, *, target: str) -> bytes:
    """Run [`unsupported_for`][quivers.transpile.unsupported_for]
    against the backend's support tier, then
    [`transpile`][quivers.transpile.transpile].

    The support-tier check is the documented public contract for
    "this module contains statement kinds the backend cannot
    represent"; the tier frozensets live in
    [`quivers.transpile._api`][quivers.transpile._api]. The
    construct-matrix test enforces the contract at the same boundary
    the production helper documents, then delegates body emission
    to the Lower + Renderer pipeline.
    """
    unsupported_for(f"qvr-{target}", module, allow=_SUPPORT_TIER[target])
    return transpile(module, target=target)


@pytest.mark.parametrize(
    "fixture", _construct_fixtures(), ids=lambda f: f"{f.category}/{f.name}"
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_construct_backend_cell(fixture: _load.Fixture, backend: str) -> None:
    """One cell of the (construct × backend) matrix."""
    expected_bad = _expected_unsupported_kinds(fixture, backend)
    module = parse(fixture.source)

    if expected_bad:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            _transpile_with_tier_check(module, target=backend)
        # The error must name at least one of the kinds we predicted.
        reported = set(exc_info.value.kinds)
        overlap = reported & expected_bad
        assert overlap, (
            f"backend {backend!r} on {fixture.name!r}: expected "
            f"UnsupportedConstruct mentioning {expected_bad}, got "
            f"{reported}"
        )
        return

    cell = (backend, fixture.category, fixture.name)
    expected_orthogonal = _EXPECTED_ORTHOGONAL_RAISES.get(cell)
    if expected_orthogonal is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            _transpile_with_tier_check(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_orthogonal) for k in kinds), (
            f"{backend!r} on {fixture.category}/{fixture.name!r}: "
            f"expected an orthogonal-concern raise with kind prefix "
            f"{expected_orthogonal!r}, got kinds={kinds!r}. Either "
            f"the renderer changed (update the entry in "
            f"`_EXPECTED_ORTHOGONAL_RAISES`) or a different gap fired "
            f"(fix the renderer)."
        )
        return

    output = _transpile_with_tier_check(module, target=backend)
    assert output, (
        f"backend {backend!r} on {fixture.name!r}: transpile returned empty bytes"
    )
