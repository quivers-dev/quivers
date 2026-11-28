"""Exact first-step oracles for live typed cyclic feature queries."""

from fractions import Fraction
from hashlib import sha256
from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import DivergentClosureError
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExecutionFailure,
    FixedCyclicRelationFeatureMomentRequest,
    FixedRelationFeatureMomentRequest,
    FixedRelationFeatureObservation,
    FixedRelationRuntimeProvider,
    ParameterId,
    discharge_module,
    dumps,
    loads,
    run_program,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_feature_moments import _module, _request, _wire_logs

_MODULE = "authored_cyclic_feature_moments"
_SOURCE = """object Letter : FinSet 2

relation feedback [algebra=markov, states=4]
    input Letter
    initial 0 [weight=unit]
    arc 0 -> 1 [labels=(0), weight=enter]
    arc 1 -> 1 [labels=(eps), weight=loop_a]
    arc 1 -> 2 [labels=(eps), weight=shared]
    arc 1 -> 2 [labels=(eps), weight=shared]
    arc 2 -> 1 [labels=(eps), weight=back]
    arc 2 -> 2 [labels=(eps), weight=loop_b]
    arc 1 -> 3 [labels=(eps), weight=exit_a]
    arc 2 -> 3 [labels=(eps), weight=exit_b]
    final 3 [weight=unit]
"""
_FEATURES = (
    (0, 2, 0),
    (0, 1, 0),
    (1, 0, 0),
    (3, 0, 0),
    (0, 4, 0),
    (0, 2, 0),
    (0, 5, 0),
    (0, 6, 0),
)

# The independent linear equations use two internal states. None is a
# stopping transition with no further feature or path continuation.
_EDGES = (
    (0, 0, "loop_a", (0, 1)),
    (0, 1, "shared", (1, 0)),
    (0, 1, "shared", (3, 0)),
    (1, 0, "back", (0, 4)),
    (1, 1, "loop_b", (0, 2)),
    (0, None, "exit_a", (0, 5)),
    (1, None, "exit_b", (0, 6)),
)


def _oracle(values, *, without_first_parallel=False):
    p, q, a, b, u, v = (
        values[name]
        for name in ("shared", "back", "loop_a", "loop_b", "exit_a", "exit_b")
    )
    forward = p if without_first_parallel else 2 * p
    determinant = (1 - a) * (1 - b) - forward * q

    def solve(rhs):
        x, y = rhs
        return (
            ((1 - b) * x + forward * y) / determinant,
            (q * x + (1 - a) * y) / determinant,
        )

    z = solve((u, v))
    if without_first_parallel:
        return z[0]
    dz = solve((2 * z[1], 0))
    moments = []
    derivatives = []
    for k, fixed in enumerate((0, 10)):
        forcing = [Fraction(0), Fraction(0)]
        dforcing = [Fraction(0), Fraction(0)]
        for source, target, name, features in _EDGES:
            value = values[name]
            tail = 1 if target is None else z[target]
            dtail = 0 if target is None else dz[target]
            forcing[source] += value * features[k] * tail
            dforcing[source] += features[k] * (
                (name == "shared") * tail + value * dtail
            )
        feature = solve(forcing)
        dforcing[0] += 2 * feature[1]
        dfeature = solve(dforcing)
        moments.append(feature[0] + fixed * z[0])
        derivatives.append(dfeature[0] + fixed * dz[0])
    return (z[0], *moments), (dz[0], *derivatives)


def _bind(log_weights):
    source = (
        _SOURCE.replace("algebra=markov", "algebra=logprob") if log_weights else _SOURCE
    )
    compiler = Compiler(parse(source), module_name=_MODULE)
    compiler.compile()
    observation = FixedRelationFeatureObservation(
        relation=compiler.relation_specs["feedback"],
        words=((0,),),
        feature_extent=3,
        arc_logs=_wire_logs(_FEATURES),
        initial_logs=_wire_logs(((0, 1, 0),)),
        final_logs=_wire_logs(((0, 7, 0),)),
    )
    base = FixedRelationFeatureMomentRequest(
        observation=observation, max_states=24, max_work=80, max_feature_cells=200
    )
    values = {
        "unit": Fraction(1),
        "enter": Fraction(1),
        "loop_a": Fraction(1, 8),
        "shared": Fraction(1, 8),
        "back": Fraction(1, 4),
        "loop_b": Fraction(1, 8),
        "exit_a": Fraction(1, 2),
        "exit_b": Fraction(1, 8),
    }
    owners = {
        name: ParameterId.derive(_MODULE, "relation_weights", name) for name in values
    }
    weights = {
        owners[name]: nn.Parameter(
            torch.tensor(
                (log(value) - (1000 if name == "enter" else 0))
                if log_weights
                else float(value),
                dtype=torch.float64,
            )
        )
        for name, value in values.items()
    }
    return base, values, owners, weights


@pytest.mark.parametrize("log_weights", [False, True])
def test_authored_cyclic_features_match_two_state_first_step_oracle(log_weights):
    base, values, owners, weights = _bind(log_weights)
    request = FixedCyclicRelationFeatureMomentRequest(request=base, max_closure_size=3)
    assert request.handler_name == base.handler_name
    assert request.result_type == base.result_type
    native = loads(dumps(_module(request)))
    wire = dumps(request)
    request = loads(wire)
    assert isinstance(request, FixedCyclicRelationFeatureMomentRequest)
    assert dumps(request) == wire
    provider = FixedRelationRuntimeProvider(
        request, weights, handler_name=request.handler_name
    )
    owner = weights[owners["shared"]]
    shift = -1000 if log_weights else 0
    for value in (Fraction(1, 8), Fraction(1, 4)):
        values["shared"] = value
        with torch.no_grad():
            owner.fill_(log(value) if log_weights else float(value))
        expected, derivatives = _oracle(values)
        actual = run_program(
            native, "features", data={}, sites={}, providers=(provider,)
        ).value
        assert actual[:3].tolist() == pytest.approx(
            [shift + log(item) for item in expected], abs=3e-12, rel=0
        )
        for output, target, derivative in zip(
            actual[:3], expected, derivatives, strict=True
        ):
            (gradient,) = torch.autograd.grad(output, owner, retain_graph=True)
            expected_gradient = derivative / target
            if log_weights:
                expected_gradient *= value
            assert gradient.item() == pytest.approx(
                float(expected_gradient), abs=3e-12, rel=0
            )
        assert actual[3].item() == -torch.inf
        assert torch.autograd.grad(actual[3], owner)[0].item() == 0

        # Wrong observer: one mark per arc ID, even if it is traversed
        # several times. Removing any one of the parallel p occurrences
        # gives the weight of walks avoiding it. The weighted union of
        # visits cannot equal this once-per-ID mutant.
        once = 4 * (expected[0] - _oracle(values, without_first_parallel=True))
        assert expected[1] > once > 0
        assert actual[1].item() > shift + log(once)

    with torch.no_grad():
        weights[owners["enter"]].fill_(-torch.inf if log_weights else 0.0)
    zero = run_program(
        native, "features", data={}, sites={}, providers=(provider,)
    ).value
    assert torch.isneginf(zero).all()

    with torch.no_grad():
        weights[owners["enter"]].fill_(shift if log_weights else 1.0)
        owner.fill_(log(2) if log_weights else 2)
    with pytest.raises(DivergentClosureError):
        request.plan(weights)()
    with pytest.raises(ExecutionFailure, match="diverg"):
        run_program(native, "features", data={}, sites={}, providers=(provider,))

    checked = discharge_module(native)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "evaluate"
            and f"foreign handler `{request.handler_name}`" in item.reason
            for item in project_for_target(native, checked, target).obstacles
        )


def test_cyclic_feature_budget_and_legacy_acyclic_wire():
    acyclic, _ = _request()
    wire = dumps(acyclic)
    assert (
        sha256(wire.encode()).hexdigest()
        == "253583c5caf25da4961f11b5c9a5fe28fe35fce1b2432798e5b1e2526f9f4272"
    )
    assert dumps(loads(wire)) == wire

    base, _, _, weights = _bind(False)
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon"):
        base.plan(weights)
    request = FixedCyclicRelationFeatureMomentRequest(request=base, max_closure_size=2)
    # The two control states and one auxiliary parallel-arc state require order three.
    with pytest.raises(RelationLimitError, match="max_closure_size"):
        request.plan(weights)


def test_one_state_geometric_raw_feature_moment():
    compiler = Compiler(
        parse("""object Letter : FinSet 2
relation repeat [algebra=markov, states=1]
    input Letter
    initial 0 [weight=unit]
    arc 0 -> 0 [labels=(eps), weight=p]
    final 0 [weight=unit]
"""),
        module_name=_MODULE,
    )
    compiler.compile()
    p = Fraction(1, 8)
    values = {
        ParameterId.derive(_MODULE, "relation_weights", "p"): nn.Parameter(
            torch.tensor(float(p), dtype=torch.float64)
        ),
        ParameterId.derive(_MODULE, "relation_weights", "unit"): torch.tensor(
            1.0, dtype=torch.float64
        ),
    }
    compiled = compiler.relation_specs["repeat"].bind(values)
    query = compiled.query(((),), max_states=8, max_work=12, max_closure_size=1)
    feature = torch.zeros((1, 1), dtype=torch.float64)
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic"):
        query.log_feature_moments(feature)
    z, moment = query.log_feature_moments(feature, allow_cyclic=True)
    assert z.exp().item() == pytest.approx(float(1 / (1 - p)))
    assert moment.exp().item() == pytest.approx(float(p / (1 - p) ** 2))
