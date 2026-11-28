"""A checked immutable reverse sweep over an ordinary nominal QIEC graph."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import product

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    BOOL,
    INT,
    REAL,
    Bind,
    Case,
    ConditionalValue,
    Evaluator,
    ExecutionFailure,
    Handle,
    If,
    KernelError,
    LetBinding,
    LetValue,
    LiteralValue,
    Local,
    Perform,
    PrimitiveApplication,
    QiecModule,
    Resume,
    Return,
    RuntimeAttachments,
    TupleValue,
    Var,
    compile_reverse_diff,
    discharge_module,
    elaborate_reverse_diff,
    exact_jvp,
    run_named,
    validate_module,
)
from quivers.qiec import evaluator as machine
from quivers.qiec.numbers import NumberDomain, host_value, machine_value
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import product_type
from quivers.transpile import UnsupportedConstruct
from quivers.transpile._entry import _emit_target
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.ir import IRProgram
from quivers.transpile.qiec_ir import (
    IRQiecHandle,
    IRQiecLetValue,
    IRQiecReturn,
    project_for_target,
)
from quivers.transpile.renderers import PyroRenderer

_DECLARATIONS = """
family ReverseReal : Type
    constructor Vertex : Real * Int -> ReverseReal

effect ReverseDiff
    seed : Int * Real -> ReverseReal
    add : Int * ReverseReal * ReverseReal -> ReverseReal
    mul : Int * ReverseReal * ReverseReal -> ReverseReal
    stop : ReverseReal -> ReverseReal

instance reverse_scope : ReverseDiff
instance alias : ReverseDiff
"""
_GRAPH = """
        let first <- perform reverse_scope.seed(0, x)
        let second <- perform reverse_scope.seed(1, y)
        let product <- perform reverse_scope.mul(2, first, second)
        let double <- perform reverse_scope.add(3, product, product)
        let answer <- perform reverse_scope.mul(4, double, first)
        perform reverse_scope.stop(answer)
"""
_POINT = (2.0, 3.0)
_EXPECTED = (24.0, 24.0, 8.0, 4.0, 2.0, 1.0)
_SINK_VERTICES = """
        let first <- perform reverse_scope.seed(0, x)
        let second <- perform reverse_scope.seed(1, y)
        let square <- perform reverse_scope.mul(2, first, first)
        let product <- perform reverse_scope.mul(3, first, second)
        let one <- perform reverse_scope.seed(4, 1.0)"""
_SINK_DECISION = """
        let outer <- perform reverse_scope.less(first, one)
        if outer then
            let inner <- perform reverse_scope.less(square, second)
            if inner then
                perform reverse_scope.stop(square)
            else
                perform reverse_scope.stop(product)
        else
            perform reverse_scope.stop(product)
"""
_BOUNDARY = "qiec-forward-comparison-boundary"


def _module(graph=_GRAPH, *, vertices=5, scope="reverse_scope", comparison=False):
    answer = "(" + " * ".join(["Real"] * (vertices + 1)) + ")"
    template = f"""
handler reverse for ReverseDiff : ReverseReal -> {answer} [coverage=total, forwards=none, implementation=foreign]
    seed resumes 1
    add resumes 1
    mul resumes 1
    stop resumes 1

define objective(x : Real, y : Real) : {answer} !{{}} =
    handle {scope} with reverse in
"""
    declarations = _DECLARATIONS
    if comparison:
        declarations = declarations.replace(
            "    stop : ReverseReal",
            "    less : ReverseReal * ReverseReal -> Bool\n    stop : ReverseReal",
        )
        template = template.replace(
            "    stop resumes", "    less resumes 1\n    stop resumes"
        )
    module = Compiler(
        parse(declarations + template + graph), module_name="nominal-reverse"
    ).qiec_module
    assert module is not None
    return module


def _lower(module, lowering=elaborate_reverse_diff):
    owner = next(
        item.entry.instance for item in module.instances if item.name == "reverse_scope"
    )
    (handler,) = module.handlers
    (constructor,) = module.constructors
    return lowering(module, owner, handler.id, constructor.id)


def _observe(module, point=_POINT):
    result = run_named(module, "objective", point)
    trace = tuple(
        (event.detail["instance"], event.detail["operation"], event.detail["address"])
        for event in result.trace
        if event.event == "operation.requested"
    )
    return host_value(result.value), trace


def _requests(module, choices=()):
    body = module.computations[0].body
    assert isinstance(body, Handle)
    current = body.computation
    values = []
    choices = iter(choices)
    while isinstance(current, Bind):
        assert isinstance(current.first, Perform)
        values.append(current.first.request)
        current = current.then
        if isinstance(current, If):
            current = current.then if next(choices) else current.otherwise
    assert isinstance(current, Perform)
    assert next(choices, None) is None
    return (*values, current.request)


def _pyro(module, discharged):
    projected = project_for_target(module, discharged, "pyro")
    assert isinstance(projected.computations[0].body, IRQiecHandle)
    assert projected.handlers[0].implementation == "authored"
    ir = IRProgram(name=projected.module, inputs=(), body=(), module=projected)
    schema = PyroRenderer().render(ir)
    namespace = {}
    exec(bytes(parser_registry().emit_pretty("python", schema)), namespace)
    return namespace["qiec_objective"]


def test_reverse_clauses_resume_once_then_combine_both_edges():
    module = _module()
    target = _lower(module)
    assert target.computations == module.computations
    expected = tuple(
        (str(request.instance), str(request.operation), request.origin.dynamic_key())
        for request in _requests(module)
    )
    assert _observe(target) == (_EXPECTED, expected)
    assert len({address for _, _, address in expected}) == 6
    resolved = {
        operation.name: operation.id for operation in module.effects[0].operations
    }
    assert expected[-1][1] == str(resolved["stop"])
    selected = {clause.operation: clause.body for clause in target.handlers[0].clauses}
    for operation in ("add", "mul"):
        body = selected[resolved[operation]]
        assert isinstance(body, Case)
        inner = body.branches[0].body
        assert isinstance(inner, Case)
        action = inner.branches[0].body
        assert isinstance(action, Bind) and isinstance(action.first, Resume)
        assert isinstance(action.then, Return) and isinstance(
            action.then.value, LetValue
        )


def test_full_reverse_vector_matches_closed_algebra_and_forward_action():
    artifact = _lower(_module())
    reference = Compiler(
        parse("""
define direct(x : Real, y : Real) : Real !{} =
    return 2.0 * x * x * y
"""),
        module_name="independent-reverse",
    ).qiec_module
    assert reference is not None
    for x, y, u, v in ((2, 3, 3, -1), (1, -4, 2, 5), (0, 8, 2, 5), (-1, 0, 5, 1)):
        found, _ = _observe(artifact, (x, y))
        expected = (2 * x * x * y, 4 * x * y, 2 * x * x, 2 * x, x, 1)
        assert found == expected
        observed = exact_jvp(reference, "direct", (x, y), (u, v))
        assert (observed.value, observed.tangent) == (
            expected[0],
            expected[1] * u + expected[2] * v,
        )
    # Neither the backwards sweep nor that reference experiment writes inputs.
    point = torch.tensor(_POINT, dtype=torch.float64, requires_grad=True)
    got, _ = _observe(artifact, tuple(point.unbind()))
    assert got == _EXPECTED and point.tolist() == list(_POINT) and point.grad is None


def test_terminal_stop_can_choose_an_earlier_vertex_with_dead_later_vertices():
    graph = _GRAPH.replace("stop(answer)", "stop(product)")
    target = _lower(_module(graph))
    assert _observe(target)[0] == (6.0, 3.0, 2.0, 1.0, 0.0, 0.0)


def test_constant_vertex_has_its_own_adjoint_and_is_not_an_input():
    graph = """
        let first <- perform reverse_scope.seed(0, x)
        let second <- perform reverse_scope.seed(1, y)
        let coefficient <- perform reverse_scope.seed(2, 4.0)
        let product <- perform reverse_scope.mul(3, first, second)
        let answer <- perform reverse_scope.mul(4, product, coefficient)
        perform reverse_scope.stop(answer)
"""
    assert _observe(_lower(_module(graph)))[0] == (24.0, 12.0, 8.0, 6.0, 4.0, 1.0)


def test_reverse_disagrees_with_a_forward_order_sweep_and_one_dropped_edge():
    x, y = _POINT
    # Each node lists its immediate parents and its exact local partials.
    parents = ((), (), ((0, y), (1, x)), ((2, 1.0), (2, 1.0)), ((3, x), (0, 2 * x * y)))

    def sweep(order, drop_duplicate=False):
        cotangents = [0.0] * len(parents)
        cotangents[-1] = 1.0
        for vertex in order:
            edges = parents[vertex]
            if drop_duplicate and vertex == 3:
                edges = edges[:1]
            for predecessor, coefficient in edges:
                cotangents[predecessor] += cotangents[vertex] * coefficient
        return (2 * x * x * y, *cotangents)

    correct = _observe(_lower(_module()))[0]
    assert correct == sweep(reversed(range(5)))
    assert sweep(range(5)) == (24.0, 12.0, 0.0, 0.0, 2.0, 1.0)
    assert sweep(reversed(range(5)), drop_duplicate=True) == (
        24.0,
        18.0,
        4.0,
        2.0,
        2.0,
        1.0,
    )
    assert correct != sweep(range(5))
    assert correct != sweep(reversed(range(5)), drop_duplicate=True)


@pytest.mark.parametrize("lowering", [elaborate_reverse_diff, compile_reverse_diff])
def test_same_interface_alias_bad_slots_and_raw_carrier_forgery_refuse(lowering):
    alias = _module(
        _GRAPH.replace("perform reverse_scope.", "perform alias."), scope="alias"
    )
    validate_module(alias)
    with pytest.raises(KernelError, match="closed named handler"):
        _lower(alias, lowering)
    bad = _module(_GRAPH.replace("mul(2, first, second)", "mul(1, first, second)"))
    with pytest.raises(KernelError, match="fresh and consecutive"):
        _lower(bad, lowering)
    swapped = _module(_GRAPH.replace("seed(0, x)", "seed(0, y)"))
    with pytest.raises(KernelError, match="in parameter order"):
        _lower(swapped, lowering)
    forged = _module(
        _GRAPH.replace(
            "mul(2, first, second)",
            "mul(2, construct Vertex(x, 0) as ReverseReal, second)",
        )
    )
    validate_module(forged)
    with pytest.raises(KernelError, match="earlier vertices"):
        _lower(forged, lowering)
    with pytest.raises(KernelError, match="one Real cotangent"):
        _lower(_module(vertices=4), lowering)


def test_wire_roundtrip_and_ordinary_discharge_keep_cotangents():
    module = _lower(_module())
    data = dumps(module)
    restored = loads(data)
    assert isinstance(restored, QiecModule) and restored == module
    assert dumps(restored) == data
    assert _observe(restored) == _observe(module)
    target = discharge_module(restored)
    assert not target.obstacles
    assert _observe(target.module) == (_EXPECTED, ())


def test_public_host_runs_maximum_vertex_shape_and_oversize_has_a_named_refusal():
    graph = _GRAPH.replace(
        "        perform reverse_scope.stop(answer)",
        "        let sixth <- perform reverse_scope.add(5, answer, answer)\n"
        "        perform reverse_scope.stop(sixth)",
    )
    edge = _lower(_module(graph, vertices=6))
    target = discharge_module(edge)
    assert not target.obstacles
    expected = (48.0, 48.0, 16.0, 8.0, 4.0, 2.0, 1.0)
    assert _observe(edge)[0] == _observe(target.module)[0] == expected
    assert _pyro(edge, target)(*_POINT) == expected
    oversize = graph.replace(
        "        perform reverse_scope.stop(sixth)",
        "        let seventh <- perform reverse_scope.mul(6, sixth, sixth)\n"
        "        perform reverse_scope.stop(seventh)",
    )
    with pytest.raises(KernelError, match="6 vertices") as error:
        _lower(_module(oversize, vertices=7))
    assert error.value.code == "qiec-reverse-limit"


def _pure_body(module):
    (entry,) = module.computations
    assert isinstance(entry.body, Return)
    assert isinstance(entry.body.value, LetValue)
    return entry.body.value


def _replace_pure_body(module, value):
    (entry,) = module.computations
    entry = entry.with_(
        body=Return(value=value), type=entry.type.with_(result=value.result_type)
    )
    result = module.with_(computations=(entry,))
    validate_module(result)
    return result


def _pure_pyro(module):
    discharged = discharge_module(module)
    assert not discharged.obstacles and discharged.module == module
    projected = project_for_target(module, discharged, "pyro")
    body = projected.computations[0].body
    assert isinstance(body, IRQiecReturn) and isinstance(body.value, IRQiecLetValue)
    ir = IRProgram(name=projected.module, inputs=(), body=(), module=projected)
    schema = PyroRenderer().render(ir)
    namespace = {}
    exec(bytes(parser_registry().emit_pretty("python", schema)), namespace)
    return namespace["qiec_objective"]


def _sink_oracle(point, choices):
    x, y = map(Fraction, point)

    def values(offsets):
        first = x + offsets[0]
        second = y + offsets[1]
        return (
            first,
            second,
            first**2 + offsets[2],
            first * second + offsets[3],
            1 + offsets[4],
        )

    def pick(vertices):
        first, second, square, _, one = vertices
        assert first != one
        decisions = (first < one,)
        if decisions[0]:
            assert square != second
            decisions += (square < second,)
        return decisions

    baseline = values((0,) * 5)
    assert pick(baseline) == choices
    output = 2 if choices == (True, True) else 3
    step = Fraction(1, 16)
    adjoints = []
    for i in range(5):
        upper = values(tuple(step if j == i else 0 for j in range(5)))
        lower = values(tuple(-step if j == i else 0 for j in range(5)))
        assert pick(upper) == pick(lower) == choices
        # Each selected expression is quadratic in any one vertex injection.
        adjoints.append((upper[output] - lower[output]) / (2 * step))
    return (baseline[output], *adjoints)


def test_lazy_named_and_flat_reverse_choose_the_same_sink_and_cotangents():
    source = _module(_SINK_VERTICES + _SINK_DECISION, comparison=True)
    native = _lower(source)
    assert native.computations == source.computations
    named = loads(dumps(native))
    compiled = loads(dumps(_lower(source, compile_reverse_diff)))
    assert isinstance(named, QiecModule) and isinstance(compiled, QiecModule)
    discharged = discharge_module(named)
    assert not discharged.obstacles
    raw_target = _pyro(named, discharged)
    flat_target = _pure_pyro(compiled)
    for point, choices in (
        ((2.0, 4.0), (False,)),
        ((0.5, 2.0), (True, True)),
        ((-0.5, 0.0), (True, False)),
    ):
        oracle = _sink_oracle(point, choices)
        requests = _requests(source, choices)
        expected_trace = tuple(
            (
                str(request.instance),
                str(request.operation),
                request.origin.dynamic_key(),
            )
            for request in requests
        )
        assert _observe(named, point) == (oracle, expected_trace)
        assert _observe(compiled, point) == (oracle, ())
        assert _observe(discharged.module, point)[0] == oracle
        assert raw_target(*point) == flat_target(*point) == oracle

    for point in ((0.0, 0.0), (1.0, 4.0), (0.5, float("inf"))):
        for module in (named, discharged.module, compiled):
            with pytest.raises(ExecutionFailure, match=_BOUNDARY):
                _observe(module, point)
        for execute in (raw_target, flat_target):
            with pytest.raises(ValueError, match=_BOUNDARY):
                execute(*point)

    for module in (named, compiled):
        with pytest.raises(UnsupportedConstruct) as refusal:
            projected = project_for_target(module, discharge_module(module), "stan")
            ir = IRProgram(name=projected.module, inputs=(), body=(), module=projected)
            _emit_target(ir, "stan")
        assert (
            "qiec:capability:strict-scalar-comparison:objective" in refusal.value.kinds
        )


def test_flat_reverse_retains_each_guard_origin_and_seeds_only_the_selected_sink():
    source = _module(_SINK_VERTICES + _SINK_DECISION, comparison=True)
    compiled = _lower(source, compile_reverse_diff)
    code = _pure_body(compiled)
    bindings = list(code.bindings)
    decision = bindings[5].value
    assert isinstance(decision, ConditionalValue)
    assert isinstance(inner := decision.then, ConditionalValue)
    assert isinstance(leaf := inner.then, TupleValue)
    for condition, request in zip(
        (decision.condition, inner.condition),
        _requests(source, (True, True))[5:7],
        strict=True,
    ):
        assert isinstance(condition, PrimitiveApplication)
        assert condition.origin == request.origin.origin

    # Use the other slot as cotangent seed while still returning the
    # selected primal. The mutated module remains structurally valid.
    wrong_slot = leaf.with_(items=(LiteralValue(value=3, type=INT), leaf.items[1]))
    bindings[5] = bindings[5].with_(
        value=decision.with_(then=inner.with_(then=wrong_slot))
    )
    mutant = _replace_pure_body(compiled, code.with_(bindings=tuple(bindings)))
    point = (0.5, 2.0)
    oracle = _sink_oracle(point, (True, True))
    wrong = (0.25, 2.0, 0.5, 0.0, 1.0, 0.0)
    assert _observe(compiled, point)[0] == oracle != wrong
    assert _observe(mutant, point)[0] == _pure_pyro(mutant)(*point) == wrong


def test_flat_reverse_does_not_eagerly_check_a_skipped_comparison():
    source = _module(_SINK_VERTICES + _SINK_DECISION, comparison=True)
    compiled = _lower(source, compile_reverse_diff)
    code = _pure_body(compiled)
    bindings = list(code.bindings)
    decision = bindings[5].value
    assert isinstance(decision, ConditionalValue)
    assert isinstance(inner := decision.then, ConditionalValue)
    eager_guard = Local(name="eagerGuard", type=BOOL)
    eager_decision = LetValue(
        bindings=(LetBinding(binder=eager_guard, value=inner.condition),),
        body=decision.with_(then=inner.with_(condition=Var(local=eager_guard))),
        result_type=decision.result_type,
    )
    bindings[5] = bindings[5].with_(value=eager_decision)
    eager = _replace_pure_body(compiled, code.with_(bindings=tuple(bindings)))
    point = (2.0, 4.0)
    assert _observe(compiled, point)[0] == _sink_oracle(point, (False,))
    with pytest.raises(ExecutionFailure, match=_BOUNDARY):
        _observe(eager, point)
    with pytest.raises(ValueError, match=_BOUNDARY):
        _pure_pyro(eager)(*point)


@pytest.mark.parametrize("lowering", [elaborate_reverse_diff, compile_reverse_diff])
def test_sink_selection_refuses_same_typed_alias_and_nonterminal_actions(lowering):
    graph = _SINK_VERTICES + _SINK_DECISION
    aliased = _module(
        graph.replace("perform reverse_scope.", "perform alias."),
        scope="alias",
        comparison=True,
    )
    validate_module(aliased)
    with pytest.raises(KernelError, match="one closed named handler"):
        _lower(aliased, lowering)

    after_comparison = _module(
        graph.replace(
            "                perform reverse_scope.stop(square)",
            "                let later <- perform reverse_scope.mul(5, first, square)\n"
            "                perform reverse_scope.stop(later)",
        ),
        comparison=True,
    )
    validate_module(after_comparison)
    with pytest.raises(KernelError, match="must branch immediately"):
        _lower(after_comparison, lowering)


def test_flat_reverse_preserves_entry_and_binary_origins_without_operation_events():
    source = _module()
    named = _lower(source)
    compiled = _lower(source, compile_reverse_diff)
    restored = loads(dumps(compiled))
    assert restored == compiled and dumps(restored) == dumps(compiled)
    assert compiled.entries == source.entries
    (entry,) = source.computations
    assert compiled.computations[0].with_(body=entry.body) == entry
    assert not (compiled.effects or compiled.instances or compiled.handlers)

    body = _pure_body(restored)
    for binding, request in zip(
        body.bindings[2:5], _requests(source)[2:5], strict=True
    ):
        assert isinstance(binding.value, PrimitiveApplication)
        assert binding.value.origin == request.origin.origin
    requests = _requests(source)
    origins = {request.origin.origin: index for index, request in enumerate(requests)}

    def arithmetic(value):
        if isinstance(value, PrimitiveApplication):
            yield (value.name, origins[value.origin])
            for argument in value.arguments:
                yield from arithmetic(argument)

    assert sorted(
        row for binding in body.bindings[5:] for row in arithmetic(binding.value)
    ) == [
        ("add_real", 0),
        ("add_real", 2),
        ("mul_real", 2),
        ("mul_real", 2),
        ("mul_real", 4),
        ("mul_real", 4),
    ]
    trace = _observe(named)[1]
    assert tuple(event[2] for event in trace) == tuple(
        request.origin.dynamic_key() for request in requests
    )

    draw = _pure_pyro(restored)
    points = list(product((-9, -1, 0, 3, 11, 13, 20, 23), repeat=2))
    for dx, dy in points:
        x, y = Fraction(dx, 4), Fraction(dy, 4)
        # Independent symbolic partials for w=xy, z=w+w, f=xz, each
        # intermediate treated as a free input to its remaining computation.
        oracle = (2 * x * x * y, 4 * x * y, 2 * x * x, 2 * x, x, 1)
        point = (float(x), float(y))
        assert _observe(restored, point) == (oracle, ())
        assert _observe(named, point) == (oracle, trace)
        assert draw(*point) == oracle


@pytest.mark.parametrize(
    ("fault", "oracle"),
    [
        ("drop", (24.0, 18.0, 4.0, 2.0, 2.0, 1.0)),
        ("duplicate", (24.0, 30.0, 12.0, 6.0, 2.0, 1.0)),
    ],
)
def test_compiled_edge_occurrences_are_observable_in_native_and_pyro(fault, oracle):
    compiled = _lower(_module(), compile_reverse_diff)
    code = _pure_body(compiled)
    bindings = list(code.bindings)
    for i, binding in enumerate(bindings[5:], start=5):
        value = binding.value
        if (
            isinstance(value, PrimitiveApplication)
            and value.name == "add_real"
            and value.arguments[0] == value.arguments[1]
        ):
            edge = value.arguments[0]
            # The two reverse edge weights into the same preceding vertex
            # are equal. Modify just their sum in the executed artifact.
            mutant = edge if fault == "drop" else value.with_(arguments=(value, edge))
            bindings[i] = binding.with_(value=mutant)
            break
    else:
        pytest.fail("reverse sweep did not expose the two operand occurrences")
    changed = _replace_pure_body(compiled, code.with_(bindings=tuple(bindings)))
    assert _observe(compiled)[0] == _EXPECTED != oracle
    assert _observe(changed)[0] == _pure_pyro(changed)(*_POINT) == oracle


class _CountingEvaluator(Evaluator):
    def _value(self, value, environment):
        self.value_visits += 1
        return super()._value(value, environment)


def _measured(module, point, monkeypatch, primitives=("add_real", "mul_real")):
    (entry,) = module.computations
    numbers = NumberDomain()
    evaluator = _CountingEvaluator(RuntimeAttachments(numbers=numbers), module=module)
    evaluator.value_visits = 0
    operations = []
    implementations = machine.IMPLEMENTATIONS

    def count_operation(name):
        def counted(*terms):
            operations.append(name)
            return implementations[name](*terms)

        return counted

    context = {
        parameter: machine_value(parameter.type, value, numbers)
        for parameter, value in zip(entry.parameters, point, strict=True)
    }
    registry = validate_module(module)
    with monkeypatch.context() as isolated:
        isolated.setattr(
            machine,
            "IMPLEMENTATIONS",
            {
                **implementations,
                **{name: count_operation(name) for name in primitives},
            },
        )
        result = evaluator.evaluate_validated(entry.body, registry, context)
    return host_value(result), tuple(operations), evaluator.value_visits


def test_flat_reverse_evaluates_one_primal_and_adjoint_sweep(monkeypatch):
    compiled = _lower(
        _module(_SINK_VERTICES + _SINK_DECISION, comparison=True),
        compile_reverse_diff,
    )
    observed = ("add_real", "mul_real", "eq_int", "select_real", "strict_lt_real")
    for point, choices in (((2.0, 4.0), (False,)), ((0.5, 2.0), (True, True))):
        value, operations, _ = _measured(compiled, point, monkeypatch, observed)
        assert value == _sink_oracle(point, choices)
        assert Counter(operations) == {
            "add_real": 4,
            "mul_real": 6,
            "eq_int": 5,
            "select_real": 5,
            "strict_lt_real": len(choices),
        }
        assert operations[: 2 + len(choices)] == ("mul_real",) * 2 + (
            "strict_lt_real",
        ) * len(choices)


def _text_chain(size, family):
    source = [
        "        let v0 <- perform reverse_scope.seed(0, x)",
        "        let v1 <- perform reverse_scope.seed(1, y)",
    ]
    for i in range(2, size):
        operation = "add" if family == "repeated" else "mul"
        other = i - 1 if family == "repeated" else (0 if family == "fanout" else 1)
        source.append(
            f"        let v{i} <- perform reverse_scope.{operation}({i}, v{i - 1}, v{other})"
        )
    selected = 1 if family == "earlier" else size - 1
    return _module(
        "\n"
        + "\n".join((*source, f"        perform reverse_scope.stop(v{selected})"))
        + "\n",
        vertices=size,
    )


def _primal_only(compiled, size):
    code = _pure_body(compiled)
    assert isinstance(code.body, TupleValue)
    primal = code.with_(
        bindings=code.bindings[:size], body=code.body.items[0], result_type=REAL
    )
    return _replace_pure_body(compiled, primal)


@pytest.mark.parametrize("family", ["repeated", "fanout", "chain", "earlier"])
def test_compiled_reverse_executes_linear_work_on_source_families(family, monkeypatch):
    for size in (2, 4, 16, 64):
        compiled = _lower(_text_chain(size, family), compile_reverse_diff)
        primal, value_ops, value_visits = _measured(
            _primal_only(compiled, size), (1.0, 1.0), monkeypatch
        )
        result, reverse_ops, reverse_visits = _measured(
            compiled, (1.0, 1.0), monkeypatch
        )
        binary = size - 2
        edges = 2 * binary
        products = 0 if family == "repeated" else edges
        adds = len(reverse_ops) - binary - products
        assert len(value_ops) == binary
        assert reverse_ops.count("mul_real") == (
            0 if family == "repeated" else 3 * binary
        )
        assert 0 <= adds <= edges
        assert len(reverse_ops) <= 5 * len(value_ops)
        assert value_visits == size + 2 * binary + 2
        assert reverse_visits == 3 * size + 2 * binary + 3 * adds + 2 * products + 3
        assert reverse_visits <= 5 * value_visits

        selected = 1 if family == "earlier" else size - 1
        if family == "repeated":
            objective = 2 ** (size - 2)
            expected = (objective, 0, *(2 ** (size - i - 1) for i in range(1, size)))
        else:
            dx = binary if family == "fanout" else 0
            dy = size - 1 if family == "chain" else 1
            expected = (1, dx, dy, *(int(i <= selected) for i in range(2, size)))
        assert primal == expected[0]
        assert result == expected
    assert _pure_pyro(compiled)(1.0, 1.0) == expected


def test_work_counter_refuses_a_numerically_inert_quadratic_sweep(monkeypatch):
    size = 24
    compiled = _lower(_text_chain(size, "chain"), compile_reverse_diff)
    body = _pure_body(compiled)
    add = next(
        binding.value
        for binding in body.bindings
        if isinstance(binding.value, PrimitiveApplication)
        and binding.value.name == "add_real"
    )
    zero = LiteralValue(type=REAL, value=0.0)
    inert = add.with_(arguments=(zero, zero))
    padding = tuple(
        LetBinding(binder=Local(name=f"padding{i}", type=REAL), value=inert)
        for i in range(size * size)
    )
    mutated = _replace_pure_body(
        compiled, body.with_(bindings=(*body.bindings, *padding))
    )
    expected, operations, visits = _measured(compiled, (1.0, 1.0), monkeypatch)
    actual, bad_operations, bad_visits = _measured(mutated, (1.0, 1.0), monkeypatch)
    assert actual == expected
    assert len(bad_operations) == len(operations) + size * size > 5 * (size - 2)
    assert bad_visits == visits + 3 * size * size


def _core_chain(size, sink_depth=0):
    module = _module(comparison=bool(sink_depth))
    (handler,) = module.handlers
    (entry,) = module.computations
    source = entry.body
    assert isinstance(source, Handle)
    first = source.computation
    assert isinstance(first, Bind) and isinstance(first.then, Bind)
    second = first.then
    numeric_type = second.binder.type
    parameters = (Local(name="reverseValue0", type=REAL), entry.parameters[1])
    assert isinstance(first.first, Perform)
    first_seed = first.first.with_(
        request=first.first.request.with_(
            arguments=(first.first.request.arguments[0], Var(local=parameters[0]))
        )
    )
    slots = (
        first.binder,
        second.binder,
        *(Local(name=f"vertex{i}", type=numeric_type) for i in range(2, size)),
    )
    multiply = _requests(module)[2]
    stop = _requests(module)[-1]
    current = Perform(request=stop.with_(arguments=(Var(local=slots[-1]),)))
    if sink_depth:
        (effect,) = module.effects
        less = next(op.id for op in effect.operations if op.name == "less")
        for i in reversed(range(sink_depth)):
            origin = stop.origin.origin.with_(
                structural_path=("testing", "guard", str(i))
            )
            request = stop.with_(
                operation=less,
                arguments=(Var(local=slots[0]), Var(local=slots[1])),
                result_type=BOOL,
                origin=stop.origin.with_(origin=origin),
            )
            fallback = stop.origin.origin.with_(
                structural_path=("testing", "stop", str(i))
            )
            other = Perform(
                request=stop.with_(
                    arguments=(Var(local=slots[1]),),
                    origin=stop.origin.with_(origin=fallback),
                )
            )
            test = Local(name=f"condition{i}", type=BOOL)
            current = Bind(
                binder=test,
                first=Perform(request=request),
                then=If(condition=Var(local=test), then=current, otherwise=other),
            )
    for i in reversed(range(2, size)):
        origin = multiply.origin.with_(
            origin=multiply.origin.origin.with_(
                structural_path=("testing", "chain", str(i))
            )
        )
        request = multiply.with_(
            arguments=(
                LiteralValue(type=INT, value=i),
                Var(local=slots[i - 1]),
                Var(local=slots[1]),
            ),
            origin=origin,
        )
        current = Bind(binder=slots[i], first=Perform(request=request), then=current)
    result = product_type(*((REAL,) * (size + 1)))
    return module.with_(
        computations=(
            entry.with_(
                body=source.with_(
                    computation=first.with_(
                        first=first_seed, then=second.with_(then=current)
                    )
                ),
                parameters=parameters,
                type=entry.type.with_(result=result),
            ),
        ),
        handlers=(handler.with_(output_type=result),),
    )


def test_compiled_core_and_pyro_admit_maximum_with_named_oversize_refusal():
    size = 256
    source = _core_chain(size)
    compiled = _lower(source, compile_reverse_diff)
    assert all(
        binding.binder.name != "reverseValue0"
        for binding in _pure_body(compiled).bindings
    )
    answer = (1.0, 0.0, float(size - 1), *((1.0,) * (size - 2)))
    assert _observe(compiled, (2.0, 1.0)) == (answer, ())
    assert _pure_pyro(compiled)(2.0, 1.0) == answer
    with pytest.raises(KernelError, match="256 vertices") as error:
        _lower(_core_chain(size + 1), compile_reverse_diff)
    assert error.value.code == "qiec-reverse-compile-limit"


@pytest.mark.parametrize(
    ("maximum", "lowering"),
    [(6, elaborate_reverse_diff), (256, compile_reverse_diff)],
)
def test_reverse_public_emission_at_sink_depth_limit(maximum, lowering):
    source = _core_chain(maximum, sink_depth=15)
    model = _lower(source, lowering)
    if lowering is compile_reverse_diff:
        execute = _pure_pyro(model)
    else:
        discharged = discharge_module(model)
        assert not discharged.obstacles
        execute = _pyro(model, discharged)
    point = (0.5, 1.0)
    all_guards = (1.0, 0.0, maximum - 1.0, *((1.0,) * (maximum - 2)))
    first_false = (1.0, 0.0, 1.0, *((0.0,) * (maximum - 2)))
    assert _observe(model, point)[0] == execute(*point) == all_guards
    assert _observe(model, (2.0, 1.0))[0] == execute(2.0, 1.0) == first_false
    with pytest.raises(KernelError, match="15 nested") as error:
        _lower(_core_chain(maximum, sink_depth=16), lowering)
    assert error.value.code == "qiec-reverse-sink-limit"


def _balanced_sinks(comparisons):
    def children(count, indent, index):
        if not count:
            return [f"{indent}perform reverse_scope.stop(square)"]
        middle = (count - 1) // 2
        return [
            f"{indent}let test{index} <- perform reverse_scope.less(first, one)",
            f"{indent}if test{index} then",
            *children(middle, indent + "    ", 2 * index),
            f"{indent}else",
            *children(count - 1 - middle, indent + "    ", 2 * index + 1),
        ]

    return "\n" + "\n".join(children(comparisons, "        ", 1)) + "\n"


def test_reverse_sink_comparison_budget_counts_untaken_branches():
    admitted = _module(_SINK_VERTICES + _balanced_sinks(256), comparison=True)
    compiled = _lower(admitted, compile_reverse_diff)
    named = _lower(admitted)
    discharged = discharge_module(named)
    assert not discharged.obstacles
    point = (0.5, 2.0)
    oracle = (0.25, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert _observe(compiled, point)[0] == _pure_pyro(compiled)(*point) == oracle
    assert (
        _observe(discharged.module, point)[0]
        == _pyro(named, discharged)(*point)
        == oracle
    )
    over = _module(_SINK_VERTICES + _balanced_sinks(257), comparison=True)
    for lowering in (elaborate_reverse_diff, compile_reverse_diff):
        with pytest.raises(KernelError, match="256 total comparisons") as error:
            _lower(over, lowering)
        assert error.value.code == "qiec-reverse-sink-limit"
