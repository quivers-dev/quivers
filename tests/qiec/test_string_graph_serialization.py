"""Check graph wire scope, shared parameter identities, and exact inference."""

import pytest
import torch
from didactic.api import ValidationError
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV, REAL
from quivers.core.relations import Arc, Boundary, RelationTopology
from quivers.qiec import (
    COMPUTE_APPLY,
    COMPUTE_EFFECT,
    EMPTY_ROW,
    UNIT,
    ComputationId,
    ComputationType,
    CoreRuntimeProvider,
    EffectRequest,
    ExecutionFailure,
    Handle,
    LiteralValue,
    NamedComputation,
    NamedEffectInstance,
    ParameterId,
    Perform,
    QiecModule,
    RationalRelationSpec,
    RelationWeightComplement,
    RuntimeConfiguration,
    SerializationError,
    SiteProvenance,
    SourceOrigin,
    StringFactorGraphSpec,
    TreeStringGraphRequest,
    TreeStringGraphRuntimeProvider,
    discharge_module,
    dumps,
    from_data,
    instantiate_effect,
    loads,
    run_named,
    to_data,
    tree_string_graph_handler,
)
from quivers.core.relations import RationalRelation
from quivers.transpile import project_for_target

_TIED = ParameterId.derive("string-graph", "tied")
_UNIT = ParameterId.derive("string-graph", "unit")
_OTHER = ParameterId.derive("string-graph", "other")
_A = StringSet(alphabet=FinSet(name="A", cardinality=2))
_B = StringSet(alphabet=FinSet(name="B", cardinality=2))


def _prior(algebra, tied=_TIED, unit=_UNIT):
    return RationalRelationSpec.of(
        RelationTopology(
            (_A,),
            0,
            2,
            (Arc(0, 1, (0,), 0), Arc(0, 1, (1,), 1)),
            (Boundary(0, 1),),
            (Boundary(1, 1),),
            ("zero", "unit"),
        ),
        algebra,
        (tied, unit),
    )


def _coupling(algebra):
    repeated = Arc(0, 1, (0, 1), 0)
    return RationalRelationSpec.of(
        RelationTopology(
            (_A, _B),
            1,
            2,
            (repeated, repeated, Arc(0, 1, (1, 1), 1)),
            (Boundary(0, 1),),
            (Boundary(1, 1),),
            ("tied", "unit"),
        ),
        algebra,
        (_TIED, _UNIT),
    )


def _graph(algebra):
    return StringFactorGraphSpec(
        variables=(("x", "A", 2, None), ("z", "A", 2, None), ("y", "B", 2, (1,))),
        relations=(
            ("prior", _prior(algebra)),
            ("coupling", _coupling(algebra)),
            ("unused", _prior(algebra, _OTHER)),
        ),
        factors=(("prior", ("x",)), ("coupling", ("x", "y")), ("prior", ("z",))),
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_round_trip_preserves_relation_reuse_parameter_sharing_and_observations(
    algebra, monkeypatch
):
    spec = _graph(algebra)
    encoded = dumps(spec)
    decoded = loads(encoded)
    assert isinstance(decoded, StringFactorGraphSpec)
    assert decoded == spec
    assert dumps(decoded) == encoded

    bound_specs = []
    bind = RationalRelationSpec.bind

    def track(rel, parameters):
        bound_specs.append(rel)
        return bind(rel, parameters)

    monkeypatch.setattr(RationalRelationSpec, "bind", track)
    parameter = nn.Parameter(torch.empty((), dtype=torch.float64))
    unit = parameter.new_tensor(0.0 if algebra is LOG_PROB else 1.0)
    with torch.no_grad():
        parameter.fill_(-1.0 if algebra is LOG_PROB else 0.5)
    graph = decoded.bind({_TIED: parameter, _UNIT: unit})
    assert bound_specs == [spec.relations[0][1], spec.relations[1][1]]
    query = graph.tree_marginal("x", max_states=512, max_work=2048)
    for value in (0.25, 0.8):
        with torch.no_grad():
            p = parameter.new_tensor(value)
            parameter.copy_(p.log() if algebra is LOG_PROB else p)
        raw = parameter.exp() if algebra is LOG_PROB else parameter
        expected_mass = (2 * raw.square() + 1) * (raw + 1)
        intended = expected_mass.log() if algebra is LOG_PROB else expected_mass
        actual = query.partition()
        torch.testing.assert_close(actual, intended)
        torch.testing.assert_close(
            query.log_prob((0,)).exp(), 2 * raw.square() / (2 * raw.square() + 1)
        )
        (actual_gradient,) = torch.autograd.grad(actual, parameter)
        (oracle_gradient,) = torch.autograd.grad(intended, parameter)
        torch.testing.assert_close(actual_gradient, oracle_gradient)


def test_graph_complement_reaches_an_owner_used_only_through_the_binding():
    spec = _graph(MARKOV)
    p = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    values = {
        _TIED: RelationWeightComplement(owner=_OTHER),
        _OTHER: p,
        _UNIT: p.new_ones(()),
    }
    graph = spec.bind(values)
    query = graph.tree_marginal("x", max_states=512, max_work=2048)
    actual = query.partition()
    q = 1 - p
    intended = (2 * q.square() + 1) * (q + 1)
    torch.testing.assert_close(actual, intended)
    (actual_gradient,) = torch.autograd.grad(actual, p)
    (oracle_gradient,) = torch.autograd.grad(intended, p)
    torch.testing.assert_close(actual_gradient, oracle_gradient)
    with pytest.raises(ValueError, match="same parameter identity"):
        spec.bind(values | {_UNIT: p})


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("factors", (("unknown", ("x",)),), "unknown relation"),
        ("factors", (("coupling", ("x", "x")),), "one distinct variable"),
        ("factors", (("coupling", ("y", "x")),), "mistyped variable"),
        ("variables", (("x", "A", 2, None), ("x", "A", 2, None)), "must be distinct"),
        ("variables", (("x", "A", 2, (2,)),), "invalid observed symbol"),
        ("relations", (("prior", 0),), "RationalRelationSpec"),
        ("variables", (("x", "A", True, None),), "got bool"),
    ],
)
def test_decode_checks_nominal_scope_and_observations(field, value, diagnostic):
    data = to_data(_graph(MARKOV))
    fields = {entry["name"]: entry for entry in data["root"]["fields"]}
    fields[field]["value"] = to_data(value)["root"]
    with pytest.raises(SerializationError, match=diagnostic):
        from_data(data)


def test_model_checks_algebra_only_for_referenced_relations():
    spec = _graph(MARKOV)
    with pytest.raises(ValidationError, match="one shared algebra"):
        StringFactorGraphSpec(
            variables=spec.variables,
            relations=(("prior", _prior(MARKOV)), ("coupling", _coupling(LOG_PROB))),
            factors=spec.factors,
        )
    with pytest.raises(ValidationError, match="Markov or LogProb"):
        _graph(REAL)
    with pytest.raises(
        ValidationError, match="relation names must be nonempty and distinct"
    ):
        StringFactorGraphSpec(
            variables=spec.variables,
            relations=(*spec.relations, spec.relations[0]),
            factors=spec.factors,
        )


def test_binding_checks_parameter_identity_and_dtype_across_relation_boundaries():
    spec = StringFactorGraphSpec(
        variables=(("x", "A", 2, None), ("z", "A", 2, None)),
        relations=(
            ("a", _prior(MARKOV, _TIED, _TIED)),
            ("b", _prior(MARKOV, _OTHER, _OTHER)),
        ),
        factors=(("a", ("x",)), ("b", ("z",))),
    )
    x = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="missing string graph parameter"):
        spec.bind({_TIED: x})
    with pytest.raises(ValueError, match="same parameter identity"):
        spec.bind({_TIED: x, _OTHER: x})
    with pytest.raises(ValueError, match="common dtype and device"):
        spec.bind({_TIED: x, _OTHER: x.float().detach()})


def _query_module(request):
    module, handler_name, protocol = "tree_graph", "graph_score", "qvr-source/v0.19"
    handler = tree_string_graph_handler(module, handler_name, request)
    effect = instantiate_effect(
        handler.effect, module=module, lexical_path=("graph_score", "instance")
    )

    def origin(role):
        return SourceOrigin(
            module=module,
            structural_path=("graph_score", role),
            role=role,
            source_protocol=protocol,
        )

    entry = NamedComputation(
        id=ComputationId.derive(module, "evaluate"),
        name="evaluate",
        telescope=(),
        parameters=(),
        body=Handle(
            instance=effect.instance,
            handler=handler.id,
            computation=Perform(
                request=EffectRequest(
                    instance=effect.instance,
                    effect=handler.effect,
                    operation=COMPUTE_APPLY,
                    static_arguments=(),
                    arguments=(LiteralValue(value=None, type=UNIT),),
                    result_type=request.result_type,
                    origin=SiteProvenance(origin=origin("perform")),
                )
            ),
        ),
        type=ComputationType(effects=EMPTY_ROW, result=request.result_type),
        origin=origin("computation"),
    )
    return QiecModule(
        module=module,
        source_protocol=protocol,
        effects=(COMPUTE_EFFECT,),
        instances=(
            NamedEffectInstance(name="query", entry=effect, origin=origin("instance")),
        ),
        handlers=(handler,),
        computations=(entry,),
    )


def _request(algebra, word=None):
    return TreeStringGraphRequest(
        graph=_graph(algebra),
        variable="x",
        word=word,
        max_states=512,
        max_work=2048,
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
@pytest.mark.parametrize("word", [None, (0,)])
def test_wire_request_runs_through_checked_execution_with_live_weights(
    algebra, word, monkeypatch
):
    request = _request(algebra, word)
    decoded = loads(dumps(request))
    assert decoded == request
    assert isinstance(decoded, TreeStringGraphRequest)
    module = loads(dumps(_query_module(decoded)))
    assert isinstance(module, QiecModule)
    original = RationalRelation.query
    compile_calls = []

    def track_query(relation, *args, **kwargs):
        compile_calls.append(args)
        return original(relation, *args, **kwargs)

    monkeypatch.setattr(RationalRelation, "query", track_query)
    p = nn.Parameter(
        torch.tensor(-1.0 if algebra is LOG_PROB else 0.5, dtype=torch.float64)
    )
    bindings = {_TIED: p, _UNIT: p.new_tensor(0.0 if algebra is LOG_PROB else 1.0)}
    provider = TreeStringGraphRuntimeProvider(
        decoded, bindings, handler_name="graph_score"
    )
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    before = len(compile_calls)
    assert before == 2 + int(word is not None)
    for value in (0.2, 0.75):
        with torch.no_grad():
            p.fill_(value if algebra is MARKOV else p.new_tensor(value).log())
        result = run_named(module, "evaluate", runtime=runtime)
        assert result.result_type == decoded.result_type
        assert isinstance(result.value, torch.Tensor)
        raw = p.exp() if algebra is LOG_PROB else p
        if word is None:
            comparison = (2 * raw.square() + 1) * (raw + 1)
            if algebra is LOG_PROB:
                comparison = comparison.log()
        else:
            comparison = (2 * raw.square() / (2 * raw.square() + 1)).log()
        torch.testing.assert_close(result.value, comparison)
        (actual_gradient,) = torch.autograd.grad(result.value, p)
        (expected_gradient,) = torch.autograd.grad(comparison, p)
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert len(compile_calls) == before


def test_provider_rejects_different_objective_with_same_result_type():
    partition = _request(LOG_PROB)
    marginal = _request(LOG_PROB, (0,))
    assert partition.result_type == marginal.result_type
    module = _query_module(partition)
    params = {_TIED: torch.tensor(-1.0), _UNIT: torch.tensor(0.0)}
    provider = TreeStringGraphRuntimeProvider(
        marginal, params, handler_name="graph_score"
    )
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    with pytest.raises(
        ExecutionFailure, match="incompatible tree string graph handler"
    ):
        run_named(module, "evaluate", runtime=runtime)


def test_graph_runtime_query_reports_the_unavailable_native_lowering():
    module = _query_module(_request(MARKOV))
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        projection = project_for_target(module, discharged, target)
        assert [(item.computation, item.reason) for item in projection.obstacles] == [
            (
                "evaluate",
                "installs the foreign handler `graph_score`, whose clause `apply` a runtime provider implements",
            )
        ]


def test_explicit_empty_word_and_cyclic_request_have_distinct_contracts():
    params = {_TIED: torch.tensor(-1.0), _UNIT: torch.tensor(0.0)}
    empty = _request(LOG_PROB, ())
    assert empty.operation == "log_prob"
    assert empty.bind(params)().item() == -torch.inf

    coupling = _coupling(LOG_PROB)
    graph = StringFactorGraphSpec(
        variables=(("x", "A", 2, None), ("y", "B", 2, None)),
        relations=(("coupling", coupling),),
        factors=(("coupling", ("x", "y")), ("coupling", ("x", "y"))),
    )
    query = empty.with_(graph=graph)
    with pytest.raises(ValueError, match="requires an acyclic factor graph"):
        query.bind(params)
