"""A checked immutable reverse sweep over an ordinary nominal QIEC graph."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Bind,
    Case,
    Handle,
    KernelError,
    LetValue,
    Perform,
    QiecModule,
    Resume,
    Return,
    discharge_module,
    elaborate_reverse_diff,
    exact_jvp,
    run_named,
    validate_module,
)
from quivers.qiec.numbers import host_value
from quivers.qiec.serialization import dumps, loads
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.ir import IRProgram
from quivers.transpile.qiec_ir import IRQiecHandle, project_for_target
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


def _module(graph=_GRAPH, *, vertices=5, scope="reverse_scope"):
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
    module = Compiler(
        parse(_DECLARATIONS + template + graph), module_name="nominal-reverse"
    ).qiec_module
    assert module is not None
    return module


def _lower(module):
    owner = next(
        item.entry.instance for item in module.instances if item.name == "reverse_scope"
    )
    (handler,) = module.handlers
    (constructor,) = module.constructors
    return elaborate_reverse_diff(module, owner, handler.id, constructor.id)


def _observe(module, point=_POINT):
    result = run_named(module, "objective", point)
    trace = tuple(
        (event.detail["instance"], event.detail["operation"], event.detail["address"])
        for event in result.trace
        if event.event == "operation.requested"
    )
    return host_value(result.value), trace


def _requests(module):
    body = module.computations[0].body
    assert isinstance(body, Handle)
    current = body.computation
    values = []
    while isinstance(current, Bind):
        assert isinstance(current.first, Perform)
        values.append(current.first.request)
        current = current.then
    assert isinstance(current, Perform)
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


def test_same_interface_alias_bad_slots_and_raw_carrier_forgery_refuse():
    alias = _module(
        _GRAPH.replace("perform reverse_scope.", "perform alias."), scope="alias"
    )
    validate_module(alias)
    with pytest.raises(KernelError, match="closed named handler"):
        _lower(alias)
    bad = _module(_GRAPH.replace("mul(2, first, second)", "mul(1, first, second)"))
    with pytest.raises(KernelError, match="fresh and consecutive"):
        _lower(bad)
    swapped = _module(_GRAPH.replace("seed(0, x)", "seed(0, y)"))
    with pytest.raises(KernelError, match="in parameter order"):
        _lower(swapped)
    forged = _module(
        _GRAPH.replace(
            "mul(2, first, second)",
            "mul(2, construct Vertex(x, 0) as ReverseReal, second)",
        )
    )
    validate_module(forged)
    with pytest.raises(KernelError, match="earlier vertices"):
        _lower(forged)
    with pytest.raises(KernelError, match="one Real cotangent"):
        _lower(_module(vertices=4))


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
