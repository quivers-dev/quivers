"""Exact negative-binomial oracles for a composed noisy string channel."""

from fractions import Fraction
from math import comb, log
from pathlib import Path

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationTopology,
    condition,
    join,
)
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    ParameterId,
    QiecModule,
    RationalRelationSpec,
    RelationWeightBindings,
    RelationWeightComplement,
    RuntimeConfiguration,
    dumps,
    loads,
    run_named,
)


def _model():
    class Weights(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
            self.calls = 0

        def forward(self):
            self.calls += 1
            return torch.stack((self.p, 1 - self.p, self.p.new_ones(())))

    tape = StringSet(alphabet=FinSet(name="UnaryAlphabet", cardinality=1))
    weights = Weights()
    names = ("repeat_or_delete", "stop_or_keep", "unit")
    unit = (Boundary(0, 2),)
    source = RationalRelation(
        RelationTopology(
            (tape,), 0, 1, (Arc(0, 0, (0,), 0),), unit, (Boundary(0, 1),), names
        ),
        weights,
        algebra=MARKOV,
    )
    channel = RationalRelation(
        RelationTopology(
            (tape, tape),
            1,
            1,
            (Arc(0, 0, (0, None), 0), Arc(0, 0, (0, 0), 1)),
            unit,
            unit,
            names,
        ),
        weights,
        algebra=MARKOV,
    )
    joint = join(
        source,
        channel,
        left_tape=0,
        right_tape=0,
        keep_shared=True,
        max_states=64,
        max_work=128,
    )
    return tape, weights, joint


@pytest.mark.parametrize("observed_length", (0, 1, 2, 3))
def test_unbounded_deletion_channel_evidence_posterior_and_tied_gradient(
    observed_length,
):
    tape, source, joint = _model()
    assert joint.topology.tapes == (tape, tape)
    assert joint.topology.n_states == 3
    assert list(joint.module().parameters())[0] is source.p
    assert len(list(joint.module().parameters())) == 1
    m = observed_length
    observed = condition(joint, 1, (0,) * m, max_states=64, max_work=128)
    before = source.calls
    mass = observed.total_weight(max_states=64, max_work=128, max_closure_size=2)
    assert source.calls == before + 1
    p = Fraction(1, 4)
    expected = (1 - p) * (p * (1 - p)) ** m / (1 - p**2) ** (m + 1)
    expected_derivative = expected * (
        m / p - (m + 1) / (1 - p) + (m + 1) * 2 * p / (1 - p**2)
    )
    assert mass.item() == pytest.approx(float(expected), abs=1e-14)
    (gradient,) = torch.autograd.grad(mass, source.p, retain_graph=True)
    assert gradient.item() == pytest.approx(float(expected_derivative), abs=1e-13)
    for n in range(7):
        probability = observed.weight(((0,) * n,)) / mass
        if n < m:
            assert probability.item() == 0
            continue
        exact = comb(n, m) * (p**2) ** (n - m) * (1 - p**2) ** (m + 1)
        exact_derivative = exact * (2 * (n - m) / p - 2 * p * (m + 1) / (1 - p**2))
        assert probability.item() == pytest.approx(float(exact), abs=1e-14)
        (derivative,) = torch.autograd.grad(probability, source.p, retain_graph=True)
        assert derivative.item() == pytest.approx(float(exact_derivative), abs=1e-13)


@pytest.fixture(scope="module")
def authored_channel():
    page = (
        Path(__file__).resolve().parents[1] / "docs/examples/noisy-deletion-channel.md"
    ).read_text()
    source = page.split('source = """\n', 1)[1].split('\n"""', 1)[0]
    compiler = Compiler(parse(source), module_name="native-noisy-channel-test")
    module = compiler.qiec_module
    assert module is not None
    decoded = loads(dumps(module))
    assert isinstance(decoded, QiecModule)
    return compiler, decoded


@pytest.mark.parametrize("observed_length", (0, 2, 4))
def test_checked_noisy_channel_reuses_the_same_leaf_over_optimizer_steps(
    authored_channel, observed_length
):
    compiler, module = authored_channel
    p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    ids = {
        name: ParameterId.derive(module.module, "relation_weights", name)
        for name in ("repeat_or_delete", "stop_or_keep", "unit_weight")
    }
    owner = ids["repeat_or_delete"]
    complement = RelationWeightComplement(owner=owner)
    wire_complement = loads(dumps(complement))
    assert isinstance(wire_complement, RelationWeightComplement)
    assert wire_complement.owner == owner
    values: RelationWeightBindings = {
        owner: p,
        ids["stop_or_keep"]: wire_complement,
        ids["unit_weight"]: p.new_ones(()),
    }
    m, n = observed_length, observed_length + 1
    providers = []
    for score, relation, words in (
        ("evidence", "observed", ((0,) * m,)),
        ("mass", "joint", ((0,) * n, (0,) * m)),
    ):
        request = FixedTransformedRelationRequest(
            relation=compiler.transformed_relation_specs[relation],
            words=words,
            max_states=64,
            max_work=128,
            max_closure_size=4,
        )
        decoded = loads(dumps(request))
        assert isinstance(decoded, FixedTransformedRelationRequest)
        providers.append(FixedRelationRuntimeProvider.for_score(score, decoded, values))
    runtime = RuntimeConfiguration(providers=tuple(providers))
    rate = 0.01
    optimizer = torch.optim.SGD((p,), lr=rate)
    for _ in range(4):
        q = p.item()
        log_prob = run_named(module, "observation_log_prob", runtime=runtime).value
        posterior = run_named(module, "posterior_length", runtime=runtime).value
        assert isinstance(log_prob, torch.Tensor) and isinstance(
            posterior, torch.Tensor
        )
        expected_log_prob = m * log(q) - (m + 1) * log(1 + q)
        expected_posterior = comb(n, m) * (q**2) ** (n - m) * (1 - q**2) ** (m + 1)
        assert log_prob.item() == pytest.approx(expected_log_prob, abs=1e-13)
        assert posterior.item() == pytest.approx(expected_posterior, abs=1e-13)
        (sensitivity,) = torch.autograd.grad(posterior, p)
        assert sensitivity.item() == pytest.approx(
            expected_posterior * (2 * (n - m) / q - 2 * q * (m + 1) / (1 - q**2)),
            abs=1e-12,
        )
        exact_score = m / q - (m + 1) / (1 + q)
        optimizer.zero_grad()
        (-log_prob).backward()
        assert p.grad.item() == pytest.approx(-exact_score, abs=1e-12)
        optimizer.step()
        assert p.item() == pytest.approx(q + rate * exact_score, abs=1e-13)


def test_complement_requires_one_original_markov_probability_owner(authored_channel):
    compiler, _ = authored_channel
    spec = compiler.relation_specs["geometric"]
    ids = dict(zip(spec.weight_names, spec.weight_parameters, strict=True))
    p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
    owner, dependent, unit = (
        ids[name] for name in ("repeat_or_delete", "stop_or_keep", "unit_weight")
    )
    values = {
        owner: p,
        dependent: RelationWeightComplement(owner=owner),
        unit: p.new_ones(()),
    }
    relation = spec.bind(values)
    assert tuple(relation.module().parameters()) == (p,)
    invalid = values | {owner: RelationWeightComplement(owner=dependent)}
    with pytest.raises(ValueError, match="original relation owner"):
        spec.bind(invalid)
    with pytest.raises(ValueError, match="same parameter identity"):
        spec.bind(values | {dependent: p})
    with pytest.raises(TypeError, match="dense scalar leaf"):
        spec.bind(values | {dependent: 1 - p})
    log_spec = RationalRelationSpec.of(spec.topology, LOG_PROB, spec.weight_parameters)
    with pytest.raises(ValueError, match="require the Markov algebra"):
        log_spec.bind(values)
    with torch.no_grad():
        p.fill_(1.25)
    with pytest.raises(ValueError, match=r"owner in \[0, 1\]"):
        relation.weight(((0,),))
