"""The plan is derived from the checked module, not from the source steps."""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.dsl.ast_nodes.let_expressions import LetExprBinOp, LetExprVar
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program
from quivers.transpile.ir import (
    CSIntegerInterval,
    DimStatic,
    IRArgBroadcast,
    IRArgList,
    IRArgNumber,
    IRArgRef,
    IRCall,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRObserve,
    IRReturn,
    IRSample,
    IRScore,
    LetExprAffineMap,
)
from quivers.transpile.plan import Lower

REGRESSION = """\
object Resp : FinSet 5
program prog : Resp -> Resp
    sample alpha <- Normal(loc=0.0, scale=5.0)
    sample beta <- Normal(loc=0.0, scale=5.0)
    let mu = alpha + beta * x
    score penalty = -0.5 * alpha * alpha
    observe y : Resp <- Normal(loc=mu, scale=1.0)
    return alpha
export prog
"""

GROUPED = """\
object Item : FinSet 8
object Comp : FinSet 4
program prog : Item -> Item
    sample probs <- Dirichlet(concentration=1.0) [over=Comp]
    marginalize z : Comp <- Categorical(probs=probs) [over=Item, reduction=logsumexp]
        observe r : Item <- Normal(loc=0.0, scale=1.0) [via=idx]
    return probs
export prog
"""

RELAXED = """\
object Resp : FinSet 6
object Out : Real 1
program prog : Out -> Out
    sample p <- Beta(concentration1=2.0, concentration0=2.0)
    sample rate <- Gamma(concentration=2.0, rate=1.0)
    marginalize z : Resp <- ContinuousBernoulli(probs=p)
        let gated = z * rate
        observe y : Resp <- Poisson(rate=gated)
    return rate
export prog
"""

KERNEL = """\
object Obs : Real 2
morphism kernel : Obs -> Obs [role=kernel] ~ Normal
program prog : Obs -> Obs
    sample x <- kernel
    return x
export prog
"""

DETERMINISTIC = """\
object Obs : Real 1
program prog : Obs -> Obs
    let shifted = x + 2.0
    return shifted
export prog
"""

CATEGORICAL_LOGITS = """\
object Category : FinSet 3
program prog : Category -> Category
    observe y <- Categorical(logits=[-2.0, 0.5, 1.25])
    return y
export prog
"""

CATEGORICAL_LOGITS_MORPHISM = """\
object Category : FinSet 3
morphism prior : Category -> Category ~ Categorical(logits=[-2.0, 0.5, 1.25])
program prog : Category -> Category
    observe y <- prior
    return y
export prog
"""


def _kinds(body: tuple[object, ...]) -> list[str]:
    """The node kinds of a plan body, in order."""
    return [type(node).__name__ for node in body]


def test_the_plan_reads_every_statement_off_the_computation() -> None:
    ir = Lower().forward(parse(REGRESSION))
    assert ir.name == "prog"
    assert ir.module.module
    assert _kinds(ir.body) == [
        "IRSample",
        "IRSample",
        "IRDeterministic",
        "IRScore",
        "IRObserve",
        "IRReturn",
    ]
    sample, _, mu, penalty, observe, returned = ir.body
    assert isinstance(sample, IRSample)
    assert sample.name == "alpha" and sample.family == "Normal"
    assert sample.arg_names == ("loc", "scale")
    assert isinstance(mu, IRDeterministic)
    assert mu.plate.batch_dims == (DimStatic(size=5, name="Resp"),)
    assert isinstance(penalty, IRScore)
    assert penalty.name == "penalty"
    assert isinstance(observe, IRObserve)
    assert observe.args == (IRArgRef(name="mu", indices=()), observe.args[1])
    assert observe.plate.batch_dims == (DimStatic(size=5, name="Resp"),)
    assert isinstance(returned, IRReturn)
    assert returned.names == ("alpha",)
    assert [(item.name, type(item.constraint).__name__) for item in ir.inputs] == [
        ("y", "CSReal"),
        ("x", "CSReal"),
    ]
    assert all(isinstance(item, IRDataInput) for item in ir.inputs)
    assert ir.inputs[1].plate.batch_dims == (DimStatic(size=5, name="Resp"),)


def test_target_plan_optimization_can_be_disabled() -> None:
    """The checked graph remains executable without a target plan."""

    source = parse(DETERMINISTIC)
    optimized = Lower().forward(source)
    unoptimized = Lower().forward(source, optimize=False)

    assert optimized.body
    assert unoptimized.body == ()
    assert unoptimized.name == optimized.name == "prog"
    assert unoptimized.module == optimized.module

    reference = run_program(
        lower_qvr_to_qiec(source),
        "prog",
        data={"obs": (0.0,), "x": 3.0},
        sites={},
    )
    assert reference.value == 5.0
    assert reference.log_joint == 0.0


@pytest.mark.parametrize(
    ("target", "fragment"),
    [
        ("pyro", "Categorical(logits="),
        ("numpyro", "Categorical(logits="),
        ("pymc", "logit_p="),
        ("edward2", "Categorical(logits="),
        ("stan", "categorical_logit_lpmf(y+1 |"),
    ],
)
def test_categorical_logits_keep_the_target_parameterization(
    target: str, fragment: str
) -> None:
    rendered = transpile(parse(CATEGORICAL_LOGITS), target=target).decode()
    assert fragment in rendered


def test_categorical_morphism_init_keeps_its_named_parameterization() -> None:
    ir = Lower().forward(parse(CATEGORICAL_LOGITS_MORPHISM))
    observe = ir.body[0]
    assert isinstance(observe, IRObserve)
    assert observe.arg_names == ("logits",)
    assert observe.args == (
        IRArgList(
            elements=(
                IRArgNumber(value=-2.0),
                IRArgNumber(value=0.5),
                IRArgNumber(value=1.25),
            )
        ),
    )


def test_a_kernel_initializer_reading_a_name_is_refused() -> None:
    source = CATEGORICAL_LOGITS_MORPHISM.replace(
        "logits=[-2.0, 0.5, 1.25]", "logits=scores"
    )
    with pytest.raises(
        UnsupportedConstruct, match="`logits=scores` .* is not a literal"
    ):
        Lower().forward(parse(source))


@pytest.mark.parametrize(
    "target", ["webppl", "church", "gen", "turing", "bugs", "jags"]
)
def test_categorical_logits_are_refused_by_probability_only_targets(
    target: str,
) -> None:
    with pytest.raises(UnsupportedConstruct, match="probability parameterization"):
        transpile(parse(CATEGORICAL_LOGITS), target=target)


def test_a_grouped_marginalization_states_its_index_map_on_the_observation() -> None:
    ir = Lower().forward(parse(GROUPED))
    probs, marginal, _ = ir.body
    assert isinstance(probs, IRSample)
    assert probs.args == (
        IRArgBroadcast(value=probs.args[0].value, target_shape=(4,)),  # type: ignore[union-attr]
    )
    assert isinstance(marginal, IRMarginalize)
    assert marginal.latent == "z"
    assert marginal.plate.batch_dims == (DimStatic(size=8, name="Item"),)
    # The latent's atoms are the elements of its index object `Comp`.
    assert marginal.constraint == CSIntegerInterval(lower=0, upper=3)
    (observe,) = marginal.scope
    assert isinstance(observe, IRObserve)
    assert observe.via == "idx"
    index_map = next(item for item in ir.inputs if item.name == "idx")
    assert index_map.constraint == CSIntegerInterval(lower=0, upper=7)


def test_a_relaxed_latent_marginalizes_one_atom_pair_per_element() -> None:
    ir = Lower().forward(parse(RELAXED))
    marginal = next(node for node in ir.body if isinstance(node, IRMarginalize))
    assert marginal.family == "ContinuousBernoulli"
    assert marginal.plate.batch_dims == (DimStatic(size=6, name="Resp"),)
    gated, observe = marginal.scope
    assert isinstance(gated, IRDeterministic)
    assert gated.plate.batch_dims == (DimStatic(size=6, name="Resp"),)
    assert isinstance(observe, IRObserve)
    assert observe.via is None


def test_a_kernel_morphism_binds_its_heads_before_the_draw() -> None:
    ir = Lower().forward(parse(KERNEL))
    assert _kinds(ir.body) == [
        "IRDeterministic",
        "IRDeterministic",
        "IRDeterministic",
        "IRSample",
        "IRReturn",
    ]
    location, raw, scale, draw, _ = ir.body
    assert isinstance(location, IRDeterministic)
    assert location.name == "x_loc"
    assert isinstance(location.expr, LetExprAffineMap)
    assert location.expr.transform == "identity"
    assert isinstance(raw, IRDeterministic) and raw.name == "x_scale_raw"
    assert isinstance(scale, IRDeterministic) and scale.name == "x_scale"
    assert isinstance(draw, IRSample)
    assert draw.args == (
        IRArgRef(name="x_loc", indices=()),
        IRArgRef(name="x_scale", indices=()),
    )
    names = [item.name for item in ir.inputs]
    assert names == ["obs", "kernel_param_weight", "kernel_param_bias"]
    weight = ir.inputs[1]
    assert [
        dim.size for dim in weight.plate.batch_dims if isinstance(dim, DimStatic)
    ] == [4, 2]


def test_an_unknown_let_call_is_refused_under_its_kind() -> None:
    source = REGRESSION.replace("alpha + beta * x", "helper(alpha)")
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(source), target="stan")
    assert caught.value.kinds == ["let:call:unknown:helper"]


def test_a_program_template_draw_is_planned_in_place() -> None:
    """A draw from a program template plans the template's steps in the
    caller: its draw of ``z`` is the site ``theta$z``, spelled
    ``theta__z``, under a plate of the object the draw names, and its
    return binds ``theta``."""
    source = """\
object School : FinSet 3
object Effect : Real 1
program effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect
program prog : School -> Effect
    sample theta <- effects(0.5, School)
    return theta
export prog
"""
    ir = Lower().forward(parse(source))
    names = [getattr(node, "name", None) for node in ir.body]
    assert names[:2] == ["theta__z", "theta"]
    sample = ir.body[0]
    assert isinstance(sample, IRSample)
    assert sample.family == "Normal"
    assert [dim.name for dim in sample.plate.batch_dims] == ["School"]


CALLS = """\
object Obs : FinSet 4

define shift(x : Real, k : Real) : Real !{} =
    return x + k

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

instance random : Random

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y : Obs <- Normal(loc=c, scale=0.5)
    return c
export prog
"""


def test_a_pure_call_is_inlined_and_an_effectful_call_is_a_call() -> None:
    ir = Lower().forward(parse(CALLS))
    assert _kinds(ir.body) == [
        "IRSample",
        "IRDeterministic",
        "IRCall",
        "IRObserve",
        "IRReturn",
    ]
    _, shifted, call, observe, _ = ir.body
    assert isinstance(shifted, IRDeterministic)
    assert shifted.name == "b"
    assert isinstance(shifted.expr, LetExprBinOp)
    assert shifted.expr.op == "+"
    assert isinstance(call, IRCall)
    assert call.name == "c"
    assert call.callee == "noisy"
    assert call.static_arguments == ()
    assert call.arguments == (LetExprVar(name="b"),)
    assert call.random_instance.startswith("qiec:effect-instance:")
    assert call.score_instance.startswith("qiec:effect-instance:")
    assert call.plate.batch_dims == ()
    assert isinstance(observe, IRObserve)
    assert observe.args[0] == IRArgRef(name="c", indices=())
