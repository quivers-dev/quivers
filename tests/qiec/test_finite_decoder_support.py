"""The finite decoder law restricts and renormalizes the declared candidates."""

from __future__ import annotations

import didactic.api as dx
import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.qiec import FiniteDecoderSupport
from quivers.qiec.builtins import (
    RANDOM,
    RANDOM_EFFECT,
    RANDOM_SAMPLE,
    SCORE,
    SCORE_EFFECT,
)
from quivers.qiec.canonical import LOG_WEIGHT, sampleable_type, site_type
from quivers.qiec.effects import (
    ComputationType,
    EffectRequest,
    EffectRow,
    instantiate_effect,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import run_named
from quivers.qiec.identifiers import ComputationId, SiteProvenance, SourceOrigin
from quivers.qiec.module import NamedComputation, NamedEffectInstance, QiecModule
from quivers.qiec.program_runtime import run_program, sample_program
from quivers.qiec.programs import ProgramEntry, ProgramParameter, ProgramSite
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Local, LogDensity, Perform, Return, SiteValue, Var
from quivers.structural import Decoder, make_term


SIG = """\
signature Example
    sorts
        Tree : object [dim=4]
        Other : object [dim=4]
    constructors
        A : -> Tree
        B : -> Tree
        C : -> Tree
        Branch : Tree -> Tree
        OtherLeaf : -> Other
"""


def _decoder() -> Decoder:
    signature = load_source(SIG).signatures["Example"]
    return Decoder(
        name="finite-test",
        signature=signature,
        sort_dims={"Tree": 4, "Other": 4},
        depth=1,
        structure_fns={"Tree": lambda code: code},
        primitive_fns={},
        factor_fns={"Tree": {1: lambda code: (code * 0.5 + 0.2,)}},
        binder_select_fn=lambda _code, _context: torch.empty(0),
        data_vocab={},
    )


def _checked_module(support: FiniteDecoderSupport) -> QiecModule:
    module = support.presentation.to_module()
    origin = SourceOrigin(
        module=module.module,
        source_protocol=module.source_protocol,
        role="computation",
        structural_path=("computations", "density"),
    )
    law = Local(name="law", type=sampleable_type(support.element))
    point = Local(name="point", type=support.element)
    computation = NamedComputation(
        id=ComputationId.derive(module.module, "density"),
        name="density",
        telescope=(),
        parameters=(law, point),
        type=ComputationType(effects=EffectRow(), result=LOG_WEIGHT),
        body=Return(
            value=LogDensity(
                sampleable=Var(local=law),
                value=Var(local=point),
                origin=origin,
            )
        ),
        origin=origin,
    )
    random = instantiate_effect(RANDOM, module=module.module, lexical_path=("random",))
    score = instantiate_effect(SCORE, module=module.module, lexical_path=("score",))
    draw_origin = origin.with_(structural_path=("computations", "draw"))
    draw = NamedComputation(
        id=ComputationId.derive(module.module, "draw"),
        name="draw",
        telescope=(),
        parameters=(law,),
        type=ComputationType(
            effects=EffectRow(entries=(random,)), result=support.element
        ),
        body=Perform(
            request=EffectRequest(
                instance=random.instance,
                effect=RANDOM,
                operation=RANDOM_SAMPLE,
                static_arguments=(support.element,),
                arguments=(
                    SiteValue(label="decoded", result_type=site_type(support.element)),
                    Var(local=law),
                ),
                result_type=support.element,
                origin=SiteProvenance(origin=draw_origin),
            )
        ),
        origin=draw_origin,
    )
    entry = ProgramEntry(
        name=draw.name,
        computation=draw.id,
        parameters=(ProgramParameter(name=law.name, role="data", type=law.type),),
        return_names=("decoded",),
        return_labels=None,
        sites=(ProgramSite(name="decoded", kind="sample", family="FiniteSampleable"),),
        random_instance=random.instance,
        score_instance=score.instance,
    )
    restored = loads(
        dumps(
            module.with_(
                computations=(computation, draw),
                entries=(entry,),
                effects=(RANDOM_EFFECT, SCORE_EFFECT),
                instances=(
                    NamedEffectInstance(name="random", entry=random, origin=origin),
                    NamedEffectInstance(name="score", entry=score, origin=origin),
                ),
            )
        )
    )
    assert isinstance(restored, QiecModule)
    return restored


def test_decoder_conditional_law_has_exact_mass_and_live_gradients() -> None:
    leaf = make_term("A")
    branch = make_term("Branch", make_term("B"))
    support = FiniteDecoderSupport(
        decoder=_decoder(),
        sort="Tree",
        candidates=(leaf, branch, make_term("A")),
        max_candidates=3,
    )
    module = _checked_module(support)
    code = torch.tensor([0.1, -0.3, 1.2, 0.7], dtype=torch.float64, requires_grad=True)
    for _ in range(3):
        law = support.distribution(code)
        weighted = law.weighted_support()
        assert len(weighted) == 2
        (leaf_value, leaf_mass), (branch_value, branch_mass) = weighted
        assert isinstance(leaf_value, RuntimeConstructor)
        assert (
            isinstance(branch_value, RuntimeConstructor)
            and len(branch_value.fields) == 1
        )
        # Outside the candidate set the original decoder has B, C and
        # two other Branch terms. The depth-zero child can only be A/B/C.
        root = torch.log_softmax(code, 0)
        child = torch.log_softmax(code[:3] * 0.5 + 0.2, 0)
        exact = torch.stack((root[0], root[3] + child[1]))
        exact = exact - exact.logsumexp(0)
        torch.testing.assert_close(torch.stack((leaf_mass, branch_mass)), exact)
        density = run_named(module, "density", (law, branch_value)).value
        torch.testing.assert_close(density, exact[1])
        (found,) = torch.autograd.grad(density, code)
        (expected,) = torch.autograd.grad(exact[1], code)
        torch.testing.assert_close(found, expected)
        draw = sample_program(module, "draw", data={"law": law})
        assert draw.value in (leaf_value, branch_value)
        replay = run_program(
            module, "draw", data={"law": law}, sites={"decoded": branch_value}
        )
        assert replay.value == branch_value
        torch.testing.assert_close(replay.log_joint, exact[1])
        assert law.log_mass().item() == 0.0
        with torch.no_grad():
            code.add_(0.2 * found)


def test_decoder_support_checks_the_explicit_bound_sort_and_depth() -> None:
    decoder = _decoder()
    a = make_term("A")
    b = make_term("B")
    with pytest.raises(dx.ValidationError, match="1 to 1 candidate entries"):
        FiniteDecoderSupport(
            decoder=decoder, sort="Tree", candidates=(a, b), max_candidates=1
        )
    with pytest.raises(dx.ValidationError, match="does not inhabit sort 'Tree'"):
        FiniteDecoderSupport(
            decoder=decoder,
            sort="Tree",
            candidates=(make_term("OtherLeaf"),),
            max_candidates=1,
        )
    deep = make_term("Branch", make_term("Branch", a))
    support = FiniteDecoderSupport(
        decoder=decoder,
        sort="Tree",
        candidates=(deep,),
        max_candidates=1,
    )
    with pytest.raises(RuntimeError, match="not in candidate set"):
        support.distribution(torch.zeros(4))
    with pytest.raises(ValueError, match="one code of shape"):
        support.distribution(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="floating dtype"):
        support.distribution(torch.zeros(4, dtype=torch.int64))


def test_decoder_conditioning_refuses_zero_mass() -> None:
    support = FiniteDecoderSupport(
        decoder=_decoder(),
        sort="Tree",
        candidates=(make_term("A"),),
        max_candidates=1,
    )
    with pytest.raises(dx.ValidationError, match="without positive mass"):
        support.distribution(torch.tensor([-torch.inf, 0.0, 0.0, 0.0]))
