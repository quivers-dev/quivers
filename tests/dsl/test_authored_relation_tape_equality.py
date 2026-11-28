"""Native QVR source for checked sparse tape equality and projection."""

from fractions import Fraction

import pytest
import torch

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    ExecutionFailure,
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    QiecModule,
    RelationCascadeSpec,
    RuntimeConfiguration,
    UnaryRelationTransformSpec,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_auto_intersection import (
    _MODULE,
    _SOURCE,
    _fixture,
    _restricted_original_powers,
)

_DECLARATIONS = """relation report = project(synced, (2))
relation synced = auto_intersect(original, 0, 1)
relation_score equality_score : Real [operation=pair_weight, relation=report]
"""
_AUTHORED = _DECLARATIONS + "\n" + _SOURCE


def _compile(source):
    compiler = Compiler(parse(source), module_name=_MODULE)
    return compiler, compiler.qiec_module


def test_normal_source_binds_equal_then_project_with_forward_references():
    compiler, module = _compile(_AUTHORED)
    emitted = module_to_source(parse(_AUTHORED))
    assert "relation synced = auto_intersect(original, 0, 1)" in emitted
    spec = compiler.transformed_relation_specs["report"]
    assert isinstance(spec, RelationCascadeSpec)
    assert spec == _compile(emitted)[0].transformed_relation_specs["report"]
    assert (
        spec
        == _compile(_SOURCE + "\n" + _DECLARATIONS)[0].transformed_relation_specs[
            "report"
        ]
    )
    equality = compiler.transformed_relation_specs["synced"]
    assert isinstance(equality, UnaryRelationTransformSpec)
    assert equality.operation == "auto_intersect" and equality.positions == (0, 1)

    request = FixedTransformedRelationRequest(
        relation=spec, words=((1,) * 4,), max_states=400, max_work=4000
    )
    request = loads(dumps(request))
    module = loads(dumps(module))
    assert isinstance(request, FixedTransformedRelationRequest)
    assert isinstance(module, QiecModule)
    _, parameters, ids = _fixture()
    provider = FixedRelationRuntimeProvider.for_score(
        "equality_score", request, parameters
    )
    runtime = RuntimeConfiguration(providers=(provider,))
    original = parameters[ids["p"]]
    for value in (Fraction(1, 8), Fraction(1, 4)):
        with torch.no_grad():
            original.fill_(float(value))
        expected, derivative = _restricted_original_powers(value, Fraction(1, 4))
        weight = run_named(module, "equality_score", runtime=runtime).value
        (gradient,) = torch.autograd.grad(weight, original)
        assert Fraction(weight.item()) == expected
        assert Fraction(gradient.item()) == derivative

    target = project_for_target(module, discharge_module(module), "stan")
    assert any(
        item.computation == "equality_score"
        and "__relation_score_equality_score" in item.reason
        for item in target.obstacles
    )

    # The equality predicate is symmetric. Authored operand order still
    # belongs to the checked recipe identity, including in cascades.
    swapped_source = _AUTHORED.replace("original, 0, 1", "original, 1, 0")
    swapped = _compile(swapped_source)[0].transformed_relation_specs["report"]
    replacement = request.with_(relation=swapped)
    provider = FixedRelationRuntimeProvider.for_score(
        "equality_score", replacement, parameters
    )
    with pytest.raises(ExecutionFailure, match="authored topology"):
        run_named(
            module,
            "equality_score",
            runtime=RuntimeConfiguration(providers=(provider,)),
        )


def test_authored_equality_checks_nominal_tapes_and_cascade_cycles():
    with pytest.raises(QiecDiagnosticError, match="same nominal tape"):
        _compile(_AUTHORED.replace("original, 0, 1", "original, 0, 2"))
    with pytest.raises(QiecDiagnosticError, match="two distinct tape positions"):
        _compile(_AUTHORED.replace("original, 0, 1", "original, 1, 1"))
    with pytest.raises(QiecDiagnosticError, match="cyclic authored relation"):
        _compile(_AUTHORED.replace("original, 0, 1", "report, 0, 1"))
