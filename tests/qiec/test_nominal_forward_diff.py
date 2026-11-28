"""Independent perturbation, scope, trace and export checks for forward Diff."""

from __future__ import annotations

import pytest

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    Bind,
    ForwardDiffInstance,
    Handle,
    KernelError,
    Local,
    Perform,
    QiecModule,
    discharge_module,
    elaborate_forward_diff,
    exact_mixed_derivative,
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


def _request_sites(term):
    while isinstance(term, Bind | Handle):
        if isinstance(term, Bind):
            if isinstance(term.first, Perform):
                yield term.first.request
            term = term.then
        else:
            term = term.computation
    if isinstance(term, Perform):
        yield term.request


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
