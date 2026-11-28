"""Named operations determine closed carriers of scoped judgments."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec import (
    REAL,
    QiecModule,
    RuleError,
    ScopedInterpretationBinding,
    TypeApplication,
    dumps,
    loads,
)
from quivers.qiec.program_runtime import run_program


SOURCE = """\
object Token : FinSet 1
object Response : Real 1

family Reading : Type
    constructor reading : Real -> Reading

family Lookalike : Type
    constructor lookalike : Real -> Lookalike

effect Scale
    scale : Real -> Real

instance seed : Scale
instance view : Scale

handler scale_by for Scale : Real -> Real with (multiplier : Real) [coverage=total, forwards=none, implementation=authored]
    return result =>
        return result
    scale(value : Real) resumes 1 =>
        resume(multiplier * value)

signature MixedScope
    sorts
        Expr : object
        Atom : object
    constructors
        Merge : Atom, Expr -> Expr
        Read : Atom -> Expr
        AtomLeaf : -> Atom
        AtomOther : -> Atom
    binders
        Bind : binds (value : Atom) in (body : Expr) -> Expr

interpret meaning [structure=cartesian, effects=[seed, view], carriers=by_judgment] : MixedScope -> Real
    rule Merge = merge
    rule Read = read_atom
    rule AtomLeaf = leaf
    rule AtomOther = other
    rule Bind = introduce

define contents(item : Reading) : Real !{} =
    case item motive => Real
        reading(value) =>
            return value

define merge[G : Type](x : Reading, y : Real) : Real !{} =
    let material <- contents(x)
    return material + y

define read_atom[G : Type](item : Reading) : Real !{view} =
    let value <- contents(item)
    perform view.scale(value)

define wrong_read[G : Type](item : Lookalike) : Real !{view} =
    case item motive => Real
        lookalike(value) =>
            perform view.scale(value)

define wrong_root[G : Type](item : Reading) : Reading !{} =
    return item

define leaf[G : Type]() : Reading !{seed, view} =
    let baseline <- perform seed.scale(2.0)
    let value <- perform view.scale(baseline)
    return construct reading(value) as Reading

define other[G : Type]() : Reading !{seed, view} =
    leaf[G]()

define wrong_producer[G : Type]() : Lookalike !{seed, view} =
    let item <- leaf[G]()
    let value <- contents(item)
    return construct lookalike(value) as Lookalike

define introduce[G : Type](body : MixedScope__ScopedProof_Expr[Real, Reading, MixedScope__ScopedFrame_Atom[G, Reading]] * G) : Real !{seed, view} =
    let value <- perform seed.scale(3.0)
    let bound = construct reading(value) as Reading
    meaning__apply_Bind_0[G](body, bound)

define evaluate() : Real !{seed, view} =
    let member = construct MixedScope__ScopedHere_Atom[Real, Reading, MixedScope__ScopedEmpty]() as MixedScope__ScopedMember_Atom[Real, Reading, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Reading]]
    let variable = construct MixedScope__ScopedHypothesis_Atom[Real, Reading, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Reading]](member) as MixedScope__ScopedProof_Atom[Real, Reading, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Reading]]
    let read = construct MixedScope__ScopedRule_Read[Real, Reading, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Reading]](variable) as MixedScope__ScopedProof_Expr[Real, Reading, MixedScope__ScopedFrame_Atom[MixedScope__ScopedEmpty, Reading]]
    let bind = construct MixedScope__ScopedRule_Bind[Real, Reading, MixedScope__ScopedEmpty](read) as MixedScope__ScopedProof_Expr[Real, Reading, MixedScope__ScopedEmpty]
    let atom = construct MixedScope__ScopedRule_AtomLeaf[Real, Reading, MixedScope__ScopedEmpty]() as MixedScope__ScopedProof_Atom[Real, Reading, MixedScope__ScopedEmpty]
    let syntax = construct MixedScope__ScopedRule_Merge[Real, Reading, MixedScope__ScopedEmpty](atom, bind) as MixedScope__ScopedProof_Expr[Real, Reading, MixedScope__ScopedEmpty]
    meaning(syntax)

define execute(theta : Real, beta : Real) : Real !{} =
    handle seed with scale_by(theta) in
        handle view with scale_by(beta) in
            evaluate()

program interpreted(theta : Real, beta : Real) : Token -> Response
    let estimate <- execute(theta, beta)
    observe y <- Normal(loc=estimate, scale=1.0)
    return estimate
"""


def test_heterogeneous_scoped_proof_scores_a_live_observation() -> None:
    compiled = load_source(SOURCE)
    encoded = dumps(compiled.qiec)
    module = loads(encoded)
    assert isinstance(module, QiecModule) and module == compiled.qiec
    assert dumps(module) == encoded
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    observation = 0.7
    result = run_program(
        module,
        "interpreted",
        data={"theta": theta, "beta": beta, "y": observation},
        sites={},
        distributions=TensorBackend(),
    )

    # AtomLeaf constructs Reading(2θβ). Bind gives Read a distinct
    # Reading(3θ), whose field is observed under the view instance.
    # Merge eliminates the left Reading and adds the two Real values.
    expected = 2 * theta * beta + 3 * theta * beta
    likelihood = -((observation - expected) ** 2 + math.log(2 * math.pi)) / 2
    torch.testing.assert_close(result.value, expected)
    torch.testing.assert_close(result.log_joint, likelihood)
    expected_gradient = (observation - expected) * 5 * torch.stack((beta, theta))
    torch.testing.assert_close(
        torch.stack(torch.autograd.grad(result.log_joint, (theta, beta))),
        expected_gradient,
    )
    ordinary = compiled.run(
        "interpreted", data={"theta": 0.6, "beta": 0.7, "y": observation}
    )
    torch.testing.assert_close(ordinary.value, expected)
    torch.testing.assert_close(ordinary.log_joint, likelihood)


def test_companion_uses_checked_fold_carriers_and_refuses_nominal_substitution() -> (
    None
):
    model = Model(SOURCE, entry="interpreted")
    (binding,) = model.schema.scoped_interpretations
    module = model.checked_module
    judgments = {item.name: item.id for item in binding.presentation.judgments}
    families = {item.name: item.type_constructor for item in module.families}
    reading = TypeApplication(constructor=families["Reading"], arguments=())
    lookalike = TypeApplication(constructor=families["Lookalike"], arguments=())
    assert dict(binding.carriers) == {
        judgments["Expr"]: REAL,
        judgments["Atom"]: reading,
    }

    restored = loads(dumps(binding))
    assert isinstance(restored, ScopedInterpretationBinding) and restored == binding
    saved_module = loads(dumps(module))
    assert isinstance(saved_module, QiecModule)
    restored.validate(saved_module)
    forged = restored.with_(
        carriers=((judgments["Expr"], REAL), (judgments["Atom"], lookalike)),
    )
    with pytest.raises(RuleError, match="entry does not match the checked root"):
        forged.validate(saved_module)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            "rule AtomOther = other",
            "rule AtomOther = wrong_producer",
            "producers for judgment 'Atom' disagree.*expected Reading, got Lookalike",
        ),
        (
            "rule Read = read_atom",
            "rule Read = wrong_root",
            "disagrees with the declared root carrier.*expected Real, got Reading",
        ),
        (
            "rule Read = read_atom",
            "rule Read = wrong_read",
            "operation 'wrong_read' needs the rule's typed arguments",
        ),
        ("carriers=by_judgment", "carriers=auto", "require one carriers=by_judgment"),
        (
            ", carriers=by_judgment",
            "",
            "needs the rule's typed arguments and the declared carrier",
        ),
    ],
)
def test_heterogeneous_carriers_refuse_invalid_witnesses(
    before, after, message
) -> None:
    with pytest.raises(QiecDiagnosticError, match=message) as caught:
        load_source(SOURCE.replace(before, after, 1))
    assert caught.value.code == "qiec-interpretation"


def test_nonroot_judgment_requires_a_producer_for_carrier_inference() -> None:
    with pytest.raises(
        QiecDiagnosticError,
        match="no declared rule operation produces a closed carrier for Missing",
    ):
        load_source(
            """\
signature Incomplete
    sorts
        Expr : object
        Missing : object
    constructors
        Point : -> Expr

interpret meaning [structure=cartesian, carriers=by_judgment] : Incomplete -> Real
    rule Point = point

define point[G : Type]() : Real !{} =
    return 1.0
"""
        )


def test_carrier_inference_requires_a_closed_codomain() -> None:
    with pytest.raises(
        QiecDiagnosticError,
        match="carrier for judgment 'Atom' must be closed independently of the ambient Type binder",
    ):
        load_source(
            """\
family Pack[A : Type] : Type
    constructor empty : Pack[A]

signature Incomplete
    sorts
        Expr : object
        Atom : object
    constructors
        Point : -> Expr
        Other : -> Atom

interpret meaning [structure=cartesian, carriers=by_judgment] : Incomplete -> Real
    rule Point = point
    rule Other = other

define point[G : Type]() : Real !{} =
    return 1.0

define other[G : Type]() : Pack[G] !{} =
    return construct empty[G]() as Pack[G]
"""
        )
