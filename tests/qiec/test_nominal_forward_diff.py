"""Independent perturbation, scope, trace and export checks for forward Diff."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    Bind,
    Case,
    ExecutionFailure,
    ForwardDiffInstance,
    Handle,
    If,
    KernelError,
    Local,
    Perform,
    PrimitiveApplication,
    QiecModule,
    Resume,
    Return,
    Var,
    discharge_module,
    elaborate_forward_diff,
    exact_mixed_derivative,
    run_named,
    validate_module,
)
from quivers.qiec import evaluator as machine
from quivers.qiec.numbers import NumberDomain, host_value
from quivers.qiec.primitives import PRIMITIVES
from quivers.qiec.serialization import dumps, loads
from quivers.transpile import UnsupportedConstruct
from quivers.transpile._entry import _emit_target
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.ir import IRProgram
from quivers.transpile.qiec_ir import IRQiecHandle, project_for_target
from quivers.transpile.renderers import PyroRenderer

_DECLARATIONS = """
family OuterReal : Type
    constructor OuterJet : Real * Real -> OuterReal

family InnerReal : Type
    constructor InnerJet : OuterReal * OuterReal -> InnerReal

effect OuterDiff
    seed : Real * Real -> OuterReal
    add : OuterReal * OuterReal -> OuterReal
    mul : OuterReal * OuterReal -> OuterReal
    stop_primal : OuterReal -> Real
    stop_tangent : OuterReal -> Real

effect InnerDiff
    seed : OuterReal * OuterReal -> InnerReal
    add : InnerReal * InnerReal -> InnerReal
    mul : InnerReal * InnerReal -> InnerReal
    stop_primal : InnerReal -> OuterReal
    stop_tangent : InnerReal -> OuterReal

instance outer : OuterDiff
instance inner : InnerDiff
instance alias : OuterDiff

handler forward_outer for OuterDiff : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    seed resumes 1
    add resumes 1
    mul resumes 1
    stop_primal resumes 1
    stop_tangent resumes 1

handler forward_inner for InnerDiff : Real -> Real [coverage=total, forwards=none, introduces=!{outer}, implementation=foreign]
    seed resumes 1
    add resumes 1
    mul resumes 1
    stop_primal resumes 1
    stop_tangent resumes 1
"""
_MIXED = """
define mixed(a : Real, b : Real, u : Real, v : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        let zero <- perform outer.seed(0.0, 0.0)
        let y <- perform outer.seed(b, 0.0)
        let dv <- perform outer.seed(v, 0.0)
        handle inner with forward_inner in
            let ax <- perform inner.seed(x, zero)
            let by <- perform inner.seed(y, dv)
            let xy <- perform inner.mul(ax, by)
            let inner_tangent <- perform inner.stop_tangent(xy)
            perform outer.stop_tangent(inner_tangent)
"""
_SINGLE = """
define mixed(a : Real, b : Real, u : Real, v : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        let y <- perform outer.seed(b, v)
        let xy <- perform outer.mul(x, y)
        perform outer.stop_tangent(xy)
"""
_POINT = (2.0, 3.0, 5.0, 7.0)


def _module(body, declarations=_DECLARATIONS):
    module = Compiler(
        parse(declarations + body), module_name="nominal-forward-diff"
    ).qiec_module
    assert module is not None
    return module


def _scopes(module, selection=("outer", "inner")):
    instances = {item.name: item.entry.instance for item in module.instances}
    handlers = {item.name: item.id for item in module.handlers}
    constructors = {item.name: item.id for item in module.constructors}
    return tuple(
        ForwardDiffInstance(
            instances[name],
            handlers[f"forward_{'outer' if name == 'alias' else name}"],
            constructors[f"{'Outer' if name == 'alias' else name.title()}Jet"],
        )
        for name in selection
    )


def _value(module, point=_POINT):
    return host_value(run_named(module, "mixed", point).value)


def _stops(module, point=_POINT):
    stop_ids = {
        str(op.id): op.name
        for effect in module.effects
        for op in effect.operations
        if op.name.startswith("stop_")
    }
    events = run_named(module, "mixed", point).trace
    return tuple(
        (
            event.detail["instance"],
            stop_ids[str(event.detail["operation"])],
            event.detail["address"],
        )
        for event in events
        if event.event == "operation.requested"
        and str(event.detail["operation"]) in stop_ids
    )


def _request_sites(term, branch=None):
    while isinstance(term, Bind | Handle):
        if isinstance(term, Bind):
            if isinstance(term.first, Perform):
                yield term.first.request
            term = term.then
        else:
            term = term.computation
    if isinstance(term, Perform):
        yield term.request
    if isinstance(term, If) and branch is not None:
        yield from _request_sites(term.then if branch else term.otherwise)


def test_nested_nominal_diff_retains_parent_tangent_and_stopping_addresses():
    module = _module(_MIXED)
    outer, inner = _scopes(module)
    lowered = elaborate_forward_diff(module, (outer, inner))
    assert lowered.computations == module.computations
    generated = {handler.id: handler for handler in lowered.handlers}
    assert all(
        generated[scope.handler].implementation == "authored"
        for scope in (outer, inner)
    )
    expected = tuple(_request_sites(module.computations[0].body))[-2:]
    assert tuple(request.instance for request in expected) == (
        inner.instance,
        outer.instance,
    )
    assert _stops(lowered) == tuple(
        (str(request.instance), "stop_tangent", request.origin.dynamic_key())
        for request in expected
    )
    assert tuple(
        (event.detail["instance"], event.detail["address"])
        for event in run_named(lowered, "mixed", _POINT).trace
        if event.event == "operation.requested"
    ) == tuple(
        (str(request.instance), request.origin.dynamic_key())
        for request in _request_sites(module.computations[0].body)
    )
    reference = _module(
        "define polynomial(a : Real, b : Real) : Real !{} =\n    return a * b\n",
        declarations="",
    )
    for a, b, u, v in ((2, 3, 5, 7), (4, -3, -2, 8), (0, 8, 1, 2), (3, 0, 0, 8)):
        exact = exact_mixed_derivative(reference, "polynomial", (a, b), (u, 0), (0, v))
        assert exact.mixed == u * v
        assert _value(lowered, (a, b, u, v)) == exact.mixed


@pytest.mark.parametrize(
    ("change", "expected", "stops"),
    [
        (
            ("inner.stop_tangent", "inner.stop_primal"),
            15,
            ("stop_primal", "stop_tangent"),
        ),
        (
            ("outer.stop_tangent", "outer.stop_primal"),
            14,
            ("stop_tangent", "stop_primal"),
        ),
    ],
)
def test_stop_projects_the_named_coefficient_only(change, expected, stops):
    module = _module(_MIXED.replace(*change))
    lowered = elaborate_forward_diff(module, _scopes(module))
    assert _value(lowered) == expected
    assert tuple(event[1] for event in _stops(lowered)) == stops


def test_add_and_two_occurrences_of_the_same_operand_preserve_both_paths():
    square = _MIXED.replace(
        "            let xy <- perform inner.mul(ax, by)",
        "            let total <- perform inner.add(ax, by)\n"
        "            let xy <- perform inner.mul(total, total)",
    )
    module = _module(square)
    assert _value(elaborate_forward_diff(module, _scopes(module))) == 70
    source = _module(_SINGLE)
    assert _value(elaborate_forward_diff(source, _scopes(source, ("outer",)))) == 29


def test_executable_shared_perturbation_mutant_loses_the_mixed_coefficient():
    confused = _SINGLE.replace(
        "        perform outer.stop_tangent(xy)\n",
        "        let first <- perform outer.stop_tangent(xy)\n"
        "        let reseeded <- perform outer.seed(first, 0.0)\n"
        "        perform outer.stop_tangent(reseeded)\n",
    )
    wrong_module = _module(confused)
    wrong = elaborate_forward_diff(wrong_module, _scopes(wrong_module, ("outer",)))
    correct_module = _module(_MIXED)
    correct = elaborate_forward_diff(correct_module, _scopes(correct_module))
    # Both argument seeds used the same square-zero variable in the mutant.
    # Its product contains only the primal and a single first coefficient.
    assert _value(wrong) == 0
    assert _value(correct) == 35


def test_same_interface_wrong_named_instance_and_distinct_carrier_fail_closed():
    aliased = _module(
        _SINGLE.replace("handle outer", "handle alias").replace(
            "perform outer.", "perform alias."
        )
    )
    outer, alias = _scopes(aliased, ("outer", "alias"))
    validate_module(aliased)
    with pytest.raises(KernelError, match="matching lexical installation"):
        elaborate_forward_diff(aliased, (outer,))
    with pytest.raises(KernelError, match="distinct nominal"):
        elaborate_forward_diff(aliased, (outer, alias))
    with pytest.raises(QiecDiagnosticError, match="type"):
        _module(_MIXED.replace("inner.mul(ax, by)", "inner.mul(ax, y)"))


def test_typed_carrier_forgery_and_projection_refuse():
    constructed = _SINGLE.replace(
        "let x <- perform outer.seed(a, u)",
        "let x = construct OuterJet(a, u) as OuterReal",
    )
    module = _module(constructed)
    validate_module(module)
    with pytest.raises(KernelError, match="constructed and read"):
        elaborate_forward_diff(module, _scopes(module, ("outer",)))
    destructured = _SINGLE.replace(
        "        perform outer.stop_tangent(xy)",
        "        case xy motive => Real\n"
        "            OuterJet(primal, tangent) =>\n"
        "                return tangent",
    )
    module = _module(destructured)
    validate_module(module)
    with pytest.raises(KernelError, match="Case is outside"):
        elaborate_forward_diff(module, _scopes(module, ("outer",)))


def test_checked_nominal_operations_must_match_the_carrier_contract():
    declarations = _DECLARATIONS.replace(
        "add : OuterReal * OuterReal -> OuterReal",
        "add : OuterReal * OuterReal -> Real",
    )
    module = _module(_MIXED, declarations)
    validate_module(module)
    with pytest.raises(KernelError, match="coefficient types"):
        elaborate_forward_diff(module, _scopes(module))


def test_an_ordinary_typed_export_of_a_live_forward_number_refuses():
    declarations = _DECLARATIONS.replace(
        "forward_outer for OuterDiff : Real -> Real",
        "forward_outer for OuterDiff : OuterReal -> OuterReal",
    )
    source = """
define leak(a : Real, u : Real) : OuterReal !{} =
    handle outer with forward_outer in
        perform outer.seed(a, u)

define mixed(a : Real, u : Real) : Real !{} =
    return 0.0
"""
    module = _module(source, declarations)
    items = {item.name: item for item in module.computations}
    owner, leak = items["mixed"], items["leak"]
    leaked = Local(name="escaped_number", type=leak.type.result)
    body = Bind(binder=leaked, first=leak.body, then=owner.body)
    forged = module.with_(computations=(owner.with_(body=body),))
    validate_module(forged)
    with pytest.raises(KernelError, match="stop before leaving"):
        elaborate_forward_diff(forged, _scopes(forged, ("outer",)))


def test_ordinary_wire_and_discharge_preserve_the_nested_result():
    module = _module(_MIXED)
    lowered = elaborate_forward_diff(module, _scopes(module))
    text = dumps(lowered)
    restored = loads(text)
    assert isinstance(restored, QiecModule) and restored == lowered
    assert dumps(restored) == text
    assert _stops(restored) == _stops(lowered)
    assert _value(restored) == 35
    target = discharge_module(restored)
    assert not target.obstacles
    assert _value(target.module) == 35
    assert _stops(target.module) == ()


def _padded(count):
    prefix = "    handle outer with forward_outer in"
    lines = (
        "        let unused_" + str(i) + " <- perform outer.seed(0.0, 0.0)"
        for i in range(count)
    )
    return _MIXED.replace(prefix, "\n".join((prefix, *lines)), 1)


def test_public_pyro_projection_runs_the_raw_handlers_at_the_admitted_size():
    source = _module(_padded(6))
    lowered = elaborate_forward_diff(source, _scopes(source))
    target = discharge_module(lowered)
    projected = project_for_target(lowered, target, "pyro")
    assert isinstance(projected.computations[0].body, IRQiecHandle)
    ir = IRProgram(name=projected.module, inputs=(), body=(), module=projected)
    schema = PyroRenderer().render(ir)
    namespace = {}
    exec(bytes(parser_registry().emit_pretty("python", schema)), namespace)
    assert namespace["qiec_mixed"](*_POINT) == _value(lowered) == 35.0

    oversize = _module(_padded(7))
    with pytest.raises(KernelError, match="16 binds and handlers") as error:
        elaborate_forward_diff(oversize, _scopes(oversize))
    assert error.value.code == "qiec-forward-limit"


def test_three_checked_lexical_instances_receive_the_specialization_limit():
    declarations = (
        _DECLARATIONS
        + """
family DeepReal : Type
    constructor DeepJet : InnerReal * InnerReal -> DeepReal

effect DeepDiff
    seed : InnerReal * InnerReal -> DeepReal
    add : DeepReal * DeepReal -> DeepReal
    mul : DeepReal * DeepReal -> DeepReal
    stop_primal : DeepReal -> InnerReal
    stop_tangent : DeepReal -> InnerReal

instance deep : DeepDiff

handler forward_deep for DeepDiff : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    seed resumes 1
    add resumes 1
    mul resumes 1
    stop_primal resumes 1
    stop_tangent resumes 1
"""
    )
    body = """
define mixed(a : Real, u : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        handle inner with forward_inner in
            let y <- perform inner.seed(x, x)
            handle deep with forward_deep in
                let z <- perform deep.seed(y, y)
                let deep_tangent <- perform deep.stop_tangent(z)
                let inner_tangent <- perform inner.stop_tangent(deep_tangent)
                perform outer.stop_tangent(inner_tangent)
"""
    module = _module(body, declarations)
    validate_module(module)
    with pytest.raises(KernelError, match="2 nested instances") as error:
        elaborate_forward_diff(module, _scopes(module, ("outer", "inner", "deep")))
    assert error.value.code == "qiec-forward-limit"


_COMPARISON_DECLARATIONS = (
    _DECLARATIONS.replace(
        "    stop_primal : OuterReal",
        "    less : OuterReal * OuterReal -> Bool\n    stop_primal : OuterReal",
    )
    .replace(
        "    stop_primal : InnerReal",
        "    less : InnerReal * InnerReal -> Bool\n    stop_primal : InnerReal",
    )
    .replace(
        "    stop_primal resumes 1", "    less resumes 1\n    stop_primal resumes 1"
    )
)
_COMPARISON = _MIXED.replace(
    """            let xy <- perform inner.mul(ax, by)
            let inner_tangent <- perform inner.stop_tangent(xy)
            perform outer.stop_tangent(inner_tangent)""",
    """            let first <- perform inner.less(ax, by)
            if first then
                let xy <- perform inner.mul(ax, by)
                let inner_tangent <- perform inner.stop_tangent(xy)
                perform outer.stop_tangent(inner_tangent)
            else
                let total <- perform inner.add(ax, by)
                let square <- perform inner.mul(total, total)
                let inner_tangent <- perform inner.stop_tangent(square)
                perform outer.stop_tangent(inner_tangent)""",
)
_DEAD_COMPARISON = """
define mixed(a : Real, b : Real, u : Real, v : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        let y <- perform outer.seed(b, v)
        let first <- perform outer.less(x, y)
        if first then
            let xy <- perform outer.mul(x, y)
            perform outer.stop_tangent(xy)
        else
            let diagonal <- perform outer.less(x, x)
            if diagonal then
                return 0.0
            else
                return 0.0
"""
_BOUNDARY = "qiec-forward-comparison-boundary"


def _emitted_forward(module, discharged, target="pyro"):
    projected = project_for_target(module, discharged, target)
    if target == "pyro":
        assert isinstance(projected.computations[0].body, IRQiecHandle)
    ir = IRProgram(name=projected.module, inputs=(), body=(), module=projected)
    namespace = {}
    exec(_emit_target(ir, target), namespace)
    return namespace["qiec_mixed"]


def _comparison_forms(source, selection=("outer", "inner")):
    lowered = elaborate_forward_diff(source, _scopes(source, selection))
    restored = loads(dumps(lowered))
    assert isinstance(restored, QiecModule) and restored == lowered
    target = discharge_module(restored)
    assert target.obstacles == ()
    return restored, target.module, _emitted_forward(restored, target)


def _piecewise_mixed(a, b, u, v):
    # The four rational points stay in the same strict region. The
    # centered mixed difference is exact for both degree-two polynomials.
    step = abs(a - b) / (4 * (1 + abs(u) + abs(v)))

    def primal(x, y):
        assert (x < y) == (a < b)
        return x * y if x < y else (x + y) ** 2

    return (
        primal(a + step * u, b + step * v)
        - primal(a + step * u, b - step * v)
        - primal(a - step * u, b + step * v)
        + primal(a - step * u, b - step * v)
    ) / (4 * step**2)


def test_named_comparison_selects_the_exact_nested_branch_and_keeps_request_addresses():
    source = _module(_COMPARISON, _COMPARISON_DECLARATIONS)
    original, discharged, emitted = _comparison_forms(source)
    operands = tuple(product((Fraction(-3, 2), Fraction(0), Fraction(3, 2)), repeat=2))
    directions = (
        (Fraction(0), Fraction(0)),
        (Fraction(-1, 2), Fraction(3)),
        (Fraction(5), Fraction(-7, 2)),
    )
    points = [(a, b, u, v) for a, b in operands if a != b for u, v in directions]
    delta = Fraction(1, 2**22)
    points.extend(
        (
            (Fraction(2), 2 + delta, Fraction(5), Fraction(7)),
            (2 + delta, Fraction(2), Fraction(5), Fraction(7)),
        )
    )
    assert not PRIMITIVES["strict_lt_real"].total
    for a, b, u, v in points:
        expected = _piecewise_mixed(a, b, u, v)
        point = tuple(map(float, (a, b, u, v)))
        native = run_named(original, "mixed", point)
        assert (
            host_value(native.value)
            == _value(discharged, point)
            == emitted(*point)
            == expected
        )
        requests = tuple(_request_sites(source.computations[0].body, branch=a < b))
        assert tuple(
            (
                event.detail["instance"],
                event.detail["operation"],
                event.detail["address"],
            )
            for event in native.trace
            if event.event == "operation.requested"
        ) == tuple(
            (
                str(request.instance),
                str(request.operation),
                request.origin.dynamic_key(),
            )
            for request in requests
        )
    assert _stops(discharged) == ()


@pytest.mark.parametrize(
    "point",
    [
        (2.0, 2.0, 0.0, 0.0),
        (0.0, -0.0, 4.0, -6.0),
        (float("inf"), 2.0, 5.0, 7.0),
        (2.0, float("nan"), 5.0, 7.0),
    ],
)
def test_reached_equality_or_nonfinite_primals_refuse_in_all_three_forms(point):
    source = _module(_COMPARISON, _COMPARISON_DECLARATIONS)
    original, discharged, emitted = _comparison_forms(source)
    for module in (original, discharged):
        with pytest.raises(ExecutionFailure, match=_BOUNDARY):
            _value(module, point)
    with pytest.raises(ValueError, match=_BOUNDARY):
        emitted(*point)


def test_untaken_diagonal_is_lazy_and_the_same_diagonal_refuses_when_reached():
    source = _module(_DEAD_COMPARISON, _COMPARISON_DECLARATIONS)
    original, discharged, emitted = _comparison_forms(source, ("outer",))
    # The true branch has d(ab)=(u b + a v). The other branch reaches x<x,
    # even though both of its result values would have been identical.
    assert _value(original) == _value(discharged) == emitted(*_POINT) == 29.0
    reached = (3.0, 2.0, 5.0, 7.0)
    for module in (original, discharged):
        with pytest.raises(ExecutionFailure, match=_BOUNDARY):
            _value(module, reached)
    with pytest.raises(ValueError, match=_BOUNDARY):
        emitted(*reached)
    tensor = tuple(torch.tensor(value, dtype=torch.float64) for value in _POINT)
    assert host_value(emitted(*tensor)) == 29.0


def test_discharge_checks_each_reached_guard_once_when_both_coefficients_are_read(
    monkeypatch,
):
    module = _module(_DEAD_COMPARISON, _COMPARISON_DECLARATIONS)
    (entry,) = module.computations
    handle = entry.body
    assert isinstance(handle, Handle) and isinstance(handle.computation, Bind)
    x = handle.computation
    y = x.then
    assert isinstance(y, Bind) and isinstance(y.then, Bind)
    first = y.then
    assert isinstance(first.then, If)
    decision = first.then
    second, multiply = decision.otherwise, decision.then
    assert isinstance(second, Bind) and isinstance(second.first, Perform)
    assert isinstance(second.then, If)
    assert isinstance(multiply, Bind) and isinstance(multiply.first, Perform)
    left, right = Var(local=x.binder), Var(local=y.binder)
    again = second.with_(
        first=second.first.with_(
            request=second.first.request.with_(arguments=(left, right))
        ),
        then=second.then.with_(then=Return(value=left), otherwise=Return(value=right)),
    )
    selection = first.with_(
        then=decision.with_(then=again, otherwise=Return(value=right))
    )
    chosen = Local(name="chosen", type=x.binder.type)
    read = Var(local=chosen)
    square = multiply.with_(
        first=multiply.first.with_(
            request=multiply.first.request.with_(arguments=(read, read))
        )
    )
    combined = Bind(binder=chosen, first=selection, then=square)
    source = module.with_(
        computations=(
            entry.with_(
                body=handle.with_(computation=x.with_(then=y.with_(then=combined)))
            ),
        )
    )
    validate_module(source)
    original, discharged, emitted = _comparison_forms(source, ("outer",))
    native = machine.IMPLEMENTATIONS["strict_lt_real"]
    exported = emitted.__globals__["_qvr_qiec_primitives"]
    host = exported["strict_lt_real"]
    calls = []

    def measure_native(*args):
        calls.append("native")
        return native(*args)

    def measure_host(*args):
        calls.append("host")
        return host(*args)

    monkeypatch.setattr(
        machine,
        "IMPLEMENTATIONS",
        {**machine.IMPLEMENTATIONS, "strict_lt_real": measure_native},
    )
    monkeypatch.setitem(exported, "strict_lt_real", measure_host)
    for point, expected, checks in ((_POINT, 20.0, 2), ((3.0, 2.0, 5.0, 7.0), 28.0, 1)):
        for term in (original, discharged):
            calls.clear()
            assert _value(term, point) == expected
            assert calls == ["native"] * checks
        calls.clear()
        assert emitted(*point) == expected
        assert calls == ["host"] * checks


def _comparison_mutant(module, fault):
    operation = next(
        op.id
        for effect in module.effects
        if effect.ref.name == "InnerDiff"
        for op in effect.operations
        if op.name == "less"
    )
    handler = next(item for item in module.handlers if item.name == "forward_inner")
    clause = next(item for item in handler.clauses if item.operation == operation)
    layers = []
    tail = clause.body
    while isinstance(tail, Case):
        (branch,) = tail.branches
        layers.append((tail, branch))
        tail = branch.body
    assert isinstance(tail, Resume) and isinstance(tail.value, PrimitiveApplication)
    assert len(layers) == 4
    if fault == "tangent":
        left = layers[1][1].fields[1]
        right = layers[3][1].fields[1]
        value = tail.value.with_(arguments=(Var(local=left), Var(local=right)))
    else:
        primitive = PRIMITIVES["lt_real"]
        value = tail.value.with_(primitive=primitive.id, name=primitive.name)
    body = tail.with_(value=value)
    for case, branch in reversed(layers):
        body = case.with_(branches=(branch.with_(body=body),))
    altered = handler.with_(
        clauses=tuple(
            clause.with_(body=body) if item.operation == operation else item
            for item in handler.clauses
        )
    )
    result = module.with_(
        handlers=tuple(
            altered if item.id == handler.id else item for item in module.handlers
        )
    )
    validate_module(result)
    return result


def test_tangent_guard_and_total_tie_mutants_change_observable_branch_semantics():
    source = _module(_COMPARISON, _COMPARISON_DECLARATIONS)
    original = elaborate_forward_diff(source, _scopes(source))
    equality = (2.0, 2.0, 5.0, 7.0)
    assert _value(original) == 35.0
    with pytest.raises(ExecutionFailure, match=_BOUNDARY):
        _value(original, equality)
    for fault, point in (("tangent", _POINT), ("total", equality)):
        changed = _comparison_mutant(original, fault)
        target = discharge_module(changed)
        assert target.obstacles == ()
        assert (
            _value(changed, point)
            == _value(target.module, point)
            == _emitted_forward(changed, target)(*point)
            == 70.0
        )
    reversed_source = _module(
        _COMPARISON.replace("inner.less(ax, by)", "inner.less(by, ax)"),
        _COMPARISON_DECLARATIONS,
    )
    reversed_module = elaborate_forward_diff(reversed_source, _scopes(reversed_source))
    assert _value(reversed_module) == 70.0


@pytest.mark.parametrize(
    "body",
    [
        _COMPARISON.replace("if first then", "if false then"),
        _COMPARISON.split("            if first then", 1)[0] + "            return a\n",
    ],
)
def test_comparisons_must_be_directly_consumed_as_lazy_guards(body):
    source = _module(body, _COMPARISON_DECLARATIONS)
    validate_module(source)
    with pytest.raises(KernelError) as error:
        elaborate_forward_diff(source, _scopes(source))
    assert error.value.code == "qiec-forward-comparison-use"


def test_comparison_checks_nominal_carriers_and_the_named_instance():
    with pytest.raises(QiecDiagnosticError, match="InnerReal"):
        _module(
            _COMPARISON.replace("inner.less(ax, by)", "inner.less(x, y)"),
            _COMPARISON_DECLARATIONS,
        )
    aliased = _module(
        _DEAD_COMPARISON.replace("handle outer ", "handle alias ").replace(
            "perform outer.", "perform alias."
        ),
        _COMPARISON_DECLARATIONS,
    )
    validate_module(aliased)
    with pytest.raises(KernelError, match="matching lexical installation"):
        elaborate_forward_diff(aliased, _scopes(aliased, ("outer",)))


def test_branch_budget_and_precise_unavailable_target_capability():
    marker = "    handle outer with forward_outer in"

    def padded(size):
        padding = tuple(
            f"        let pad_{i} <- perform outer.seed(0.0, 0.0)" for i in range(size)
        )
        return _module(
            _COMPARISON.replace(marker, "\n".join((marker, *padding))),
            _COMPARISON_DECLARATIONS,
        )

    available = padded(1)
    lowered = elaborate_forward_diff(available, _scopes(available))
    target = discharge_module(lowered)
    assert target.obstacles == ()
    assert _emitted_forward(lowered, target)(*_POINT) == 35.0
    refused = padded(2)
    with pytest.raises(KernelError) as error:
        elaborate_forward_diff(refused, _scopes(refused))
    assert error.value.code == "qiec-forward-limit"
    for backend in ("stan", "numpyro"):
        with pytest.raises(UnsupportedConstruct) as missing:
            _emitted_forward(lowered, target, backend)
        assert "qiec:capability:strict-scalar-comparison:mixed" in missing.value.kinds


def _with_division(declarations):
    for scope in ("Outer", "Inner"):
        arithmetic = f"    mul : {scope}Real * {scope}Real -> {scope}Real"
        declarations = declarations.replace(
            arithmetic,
            arithmetic + f"\n    div : {scope}Real * {scope}Real -> {scope}Real",
        )
    return declarations.replace(
        "    mul resumes 1", "    mul resumes 1\n    div resumes 1"
    )


_DIVISION_DECLARATIONS = _with_division(_DECLARATIONS)
_DIVISION = (
    _MIXED.replace("u : Real, v : Real", "u : Real, v : Real, s : Real, t : Real")
    .replace("outer.seed(b, 0.0)", "outer.seed(b, s)")
    .replace("outer.seed(v, 0.0)", "outer.seed(v, t)")
    .replace("inner.mul(ax, by)", "inner.div(ax, by)")
)
_DIVISION_DOMAIN = "qiec-forward-division-domain"
_DIVISION_RANGE = "qiec-forward-division-range"


def _nilpotent_product(left, right):
    # Index bits denote the two different infinitesimals. A repeated bit
    # annihilates the term, including the square of a mixed coefficient.
    result = [Fraction(0) for _ in range(4)]
    for i, j in product(range(4), repeat=2):
        if not i & j:
            result[i | j] += left[i] * right[j]
    return tuple(result)


def _rational_quotient(a, b, u, v, s, t):
    # The denominator is b + n where n^3=0. The finite geometric series
    # is independent of the implemented recursive coefficient equations.
    numerator = (a, u, Fraction(0), Fraction(0))
    nilpotent = (Fraction(0), s, v, t)
    n_squared = _nilpotent_product(nilpotent, nilpotent)
    inverse = (1 / b, *(n_squared[i] / b**3 - nilpotent[i] / b**2 for i in range(1, 4)))
    quotient = _nilpotent_product(numerator, inverse)
    assert _nilpotent_product((b, s, v, t), quotient) == numerator
    return quotient


@pytest.mark.parametrize(
    "outer_tangent,inner_tangent", product((False, True), repeat=2)
)
def test_nested_division_matches_all_exact_rational_projections(
    outer_tangent, inner_tangent
):
    body = _DIVISION
    if not outer_tangent:
        body = body.replace("outer.stop_tangent", "outer.stop_primal")
    if not inner_tangent:
        body = body.replace("inner.stop_tangent", "inner.stop_primal")
    source = _module(body, _DIVISION_DECLARATIONS)
    native, discharged, exported = _comparison_forms(source)
    position = int(outer_tangent) + 2 * int(inner_tangent)
    records = tuple(_request_sites(source.computations[0].body))
    points = product(
        (Fraction(-3, 2), Fraction(0), Fraction(5, 2)),
        (Fraction(-2), Fraction(1, 2), Fraction(4)),
        (Fraction(-3), Fraction(2)),
        (Fraction(-2), Fraction(1)),
        (Fraction(-1, 2), Fraction(3, 2)),
        (Fraction(-1), Fraction(2)),
    )
    for point in points:
        expected = _rational_quotient(*point)[position]
        machine_point = tuple(map(float, point))
        raw = run_named(native, "mixed", machine_point)
        assert (
            host_value(raw.value)
            == _value(discharged, machine_point)
            == exported(*machine_point)
            == expected
        )
        assert tuple(
            (event.detail["operation"], event.detail["address"])
            for event in raw.trace
            if event.event == "operation.requested"
        ) == tuple(
            (str(request.operation), request.origin.dynamic_key())
            for request in records
        )


def test_division_checks_the_primal_followed_by_every_ground_coefficient():
    source = _module(_DIVISION, _DIVISION_DECLARATIONS)
    lowered = elaborate_forward_diff(source, _scopes(source))
    operation = next(
        op.id
        for effect in lowered.effects
        if effect.ref.name == "InnerDiff"
        for op in effect.operations
        if op.name == "div"
    )
    handler = next(item for item in lowered.handlers if item.name == "forward_inner")
    clause = next(item for item in handler.clauses if item.operation == operation)
    body = clause.body
    fields = []
    while isinstance(body, Case):
        (part,) = body.branches
        fields.append(part.fields)
        body = part.body
    assert len(fields) == 6
    ground = Var(local=fields[4][0])
    for _ in range(4):
        assert isinstance(body, Bind) and isinstance(body.first, Return)
        value = body.first.value
        assert isinstance(value, PrimitiveApplication)
        assert value.name == "strict_div_real" and value.arguments[1] == ground
        body = body.then
    assert isinstance(body, Resume)
    assert not PRIMITIVES["strict_div_real"].total


@pytest.mark.parametrize("divisor", [0.0, -0.0, float("inf"), float("nan")])
def test_divisor_ground_domain_ignores_nonzero_tangents(divisor):
    source = _module(_DIVISION, _DIVISION_DECLARATIONS)
    native, discharged, exported = _comparison_forms(source)
    point = (2.0, divisor, 5.0, 7.0, 2.0, 3.0)
    for module in (native, discharged):
        with pytest.raises(ExecutionFailure, match=_DIVISION_DOMAIN):
            _value(module, point)
    with pytest.raises(ValueError, match=_DIVISION_DOMAIN):
        exported(*point)


@pytest.mark.parametrize(
    ("point", "dtype", "first_nonfinite"),
    [
        ((1e308, 1e-308, 0.0, 0.0, 0.0, 0.0), torch.float64, 0),
        ((1.0, 0.5, 1e308, 0.0, 0.0, 0.0), torch.float64, 1),
        ((1.0, 0.5, 2e38, 0.0, 0.0, 0.0), torch.float32, 1),
    ],
)
def test_mixed_division_refuses_overflow_before_it_can_contaminate_exact_zero(
    point, dtype, first_nonfinite
):
    operands = tuple(torch.tensor(coordinate, dtype=dtype) for coordinate in point)
    numbers = NumberDomain.of(operands[0])
    coefficients = _rational_quotient(*(Fraction(float(value)) for value in operands))
    maximum = Fraction(float(torch.finfo(dtype).max))
    assert coefficients[3] == 0
    assert tuple(i for i, value in enumerate(coefficients) if abs(value) > maximum) == (
        first_nonfinite,
    )
    native, discharged, exported = _comparison_forms(
        _module(_DIVISION, _DIVISION_DECLARATIONS)
    )
    for module in (native, discharged):
        with pytest.raises(ExecutionFailure, match=_DIVISION_RANGE):
            run_named(module, "mixed", operands, numbers=numbers)
    with pytest.raises(ValueError, match=_DIVISION_RANGE):
        exported(*operands)


def test_unused_division_retains_the_domain_and_range_checks_in_discharge():
    body = _SINGLE.replace("outer.mul(x, y)", "outer.div(x, y)").replace(
        "perform outer.stop_tangent(xy)", "return a"
    )
    source = _module(body, _DIVISION_DECLARATIONS)
    native, discharged, exported = _comparison_forms(source, ("outer",))
    assert _value(native) == _value(discharged) == exported(*_POINT) == 2.0
    for point, refusal in (
        ((2.0, 0.0, 5.0, 7.0), _DIVISION_DOMAIN),
        ((1.0, 0.5, 1e308, 0.0), _DIVISION_RANGE),
    ):
        for module in (native, discharged):
            with pytest.raises(ExecutionFailure, match=refusal):
                _value(module, point)
        with pytest.raises(ValueError, match=refusal):
            exported(*point)


def test_untaken_zero_divisor_remains_lazy_under_strict_comparison():
    body = """
define mixed(a : Real, b : Real, u : Real, v : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        let y <- perform outer.seed(b, v)
        let zero <- perform outer.seed(0.0, 1.0)
        let first <- perform outer.less(x, y)
        if first then
            let quotient <- perform outer.div(x, y)
            perform outer.stop_tangent(quotient)
        else
            let quotient <- perform outer.div(x, zero)
            perform outer.stop_tangent(quotient)
"""
    source = _module(body, _with_division(_COMPARISON_DECLARATIONS))
    native, discharged, exported = _comparison_forms(source, ("outer",))
    valid, refused = (2.0, 4.0, 3.0, 2.0), (4.0, 2.0, 3.0, 2.0)
    assert _value(native, valid) == _value(discharged, valid) == exported(*valid) == 0.5
    for module in (native, discharged):
        with pytest.raises(ExecutionFailure, match=_DIVISION_DOMAIN):
            _value(module, refused)
    with pytest.raises(ValueError, match=_DIVISION_DOMAIN):
        exported(*refused)


def test_untaken_quotient_overflow_remains_lazy_under_strict_comparison():
    body = """
define mixed(a : Real, b : Real, u : Real, v : Real) : Real !{} =
    handle outer with forward_outer in
        let x <- perform outer.seed(a, u)
        let y <- perform outer.seed(b, v)
        let first <- perform outer.less(x, y)
        if first then
            return a
        else
            let quotient <- perform outer.div(x, y)
            perform outer.stop_tangent(quotient)
"""
    source = _module(body, _with_division(_COMPARISON_DECLARATIONS))
    native, discharged, exported = _comparison_forms(source, ("outer",))
    skipped, reached = (0.25, 0.5, 1e308, 0.0), (1.0, 0.5, 1e308, 0.0)
    assert (
        _value(native, skipped)
        == _value(discharged, skipped)
        == exported(*skipped)
        == 0.25
    )
    for module in (native, discharged):
        with pytest.raises(ExecutionFailure, match=_DIVISION_RANGE):
            _value(module, reached)
    with pytest.raises(ValueError, match=_DIVISION_RANGE):
        exported(*reached)


def test_direct_division_stays_defined_when_reciprocal_would_overflow():
    tiny = 2.0**-1070
    body = _SINGLE.replace("outer.mul(x, y)", "outer.div(x, y)").replace(
        "perform outer.stop_tangent(xy)", "perform outer.stop_primal(xy)"
    )
    source = _module(body, _DIVISION_DECLARATIONS)
    native, discharged, exported = _comparison_forms(source, ("outer",))
    point = (tiny, tiny, 0.0, 0.0)
    assert _value(native, point) == _value(discharged, point) == exported(*point) == 1.0
    as_tensors = tuple(torch.tensor(arg, dtype=torch.float64) for arg in point)
    assert host_value(exported(*as_tensors)) == 1.0


def test_nested_division_works_at_the_public_source_budget_and_names_other_targets():
    marker = "    handle outer with forward_outer in"
    padding = tuple(
        f"        let pad_{i} <- perform outer.seed(0.0, 0.0)" for i in range(6)
    )
    source = _module(
        _DIVISION.replace(marker, "\n".join((marker, *padding))), _DIVISION_DECLARATIONS
    )
    lowered = elaborate_forward_diff(source, _scopes(source))
    target = discharge_module(lowered)
    assert target.obstacles == ()
    point = (2.0, 4.0, 5.0, 7.0, 2.0, 3.0)
    assert (
        _value(lowered, point)
        == _value(target.module, point)
        == _emitted_forward(lowered, target)(*point)
        == -27.0 / 16.0
    )
    for backend in ("stan", "numpyro"):
        with pytest.raises(UnsupportedConstruct) as missing:
            _emitted_forward(lowered, target, backend)
        assert "qiec:capability:strict-scalar-division:mixed" in missing.value.kinds
