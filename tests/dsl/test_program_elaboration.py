"""Programs elaborate to computations that score as the runtime does.

A ``program`` becomes a named computation over the module's canonical
``random`` and ``score`` instances, with typed data, observation, and
index map parameters, and one helper computation per marginalization
block. Run on the reference machine with every site replayed, its log
joint is the same number the torch runtime's trace accumulates.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as td

from quivers.continuous.morphisms import SampledComposition
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec import (
    INT,
    REAL,
    Bind,
    Call,
    DistributionValue,
    Handle,
    NewInstance,
    Perform,
    QiecModule,
    SegmentSum,
    SiteValue,
    dumps,
    loads,
    tensor_type,
)
from quivers.qiec.program_runtime import program_entry, run_program
from quivers.qiec.types import IndexLiteral, render_static
from quivers.qiec.kinds import NAT

BETA_BERNOULLI = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample theta <- Beta(concentration1=2.0, concentration0=2.0)
    observe y <- Bernoulli(probs=theta)
    return theta
export prog
"""

LET_STEP = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = a + b
    observe y <- Normal(loc=mu, scale=0.5)
    return mu
export prog
"""

IID_PLATE = """\
object N : FinSet 3
object Obs : FinSet 8
program prog : Obs -> Obs
    sample xs : N <- Normal(loc=0.0, scale=1.0)
    observe y : N <- Normal(loc=xs, scale=0.5)
    return xs
export prog
"""

GROUPED = """\
object Component : FinSet 3
object Item : FinSet 2
object Resp : FinSet 4
program prog(concentration : Real) : Resp -> Resp
    sample probs <- Dirichlet(concentration=concentration) [over=Component]
    sample mu : Component <- Normal(loc=0.0, scale=5.0)
    sample sigma : Component <- HalfNormal(scale=1.0)
    marginalize cls : Component <- Categorical(probs=probs) [over=Item, reduction=logsumexp]
        observe r : Resp <- Normal(loc=mu[cls], scale=sigma[cls]) [via=idx]
    return probs
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

SCORED = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample x <- Normal(loc=0.0, scale=1.0)
    score penalty = x * x
    return x
export prog
"""

CATEGORICAL_LOGITS = """\
object Category : FinSet 3
program prog : Category -> Category
    sample logits : Category <- Normal(loc=0.0, scale=1.0)
    observe y <- Categorical(logits=logits)
    return y
export prog
"""


def _module(source: str) -> QiecModule:
    """Lower a program module.

    Parameters
    ----------
    source : str
        QVR text.

    Returns
    -------
    QiecModule
        The checked module.
    """
    return lower_qvr_to_qiec(parse(source), file_path="program.qvr")


def _tensor(value: object) -> torch.Tensor:
    """A one-dimensional tensor for a scalar or a sequence.

    Parameters
    ----------
    value : object
        A number, a Boolean, or a sequence of numbers.

    Returns
    -------
    torch.Tensor
        The values as integers when every one is an integer, as the
        checked type of an index reads them, else as floats.
    """
    if isinstance(value, bool | int | float):
        return torch.tensor([float(value)])
    items = list(value)  # type: ignore[call-overload]
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        return torch.tensor(items, dtype=torch.long)
    return torch.tensor([float(item) for item in items])


def _classic_log_joint(
    source: str,
    clamps: dict[str, object],
    *,
    scalars: dict[str, float] | None = None,
) -> float:
    """The torch runtime's log joint at clamped sites and observations.

    Parameters
    ----------
    source : str
        QVR text.
    clamps : dict[str, object]
        Every site and observation value, by name.
    scalars : dict[str, float] | None
        The program's scalar parameters, when it declares any.

    Returns
    -------
    float
        The trace's accumulated log joint.
    """
    program = Compiler(parse(source)).compile()
    if scalars:
        monadic = getattr(program, "prog")(**scalars).morphism
    else:
        monadic = program._morphism
    torch.manual_seed(0)
    observations = {name: _tensor(value) for name, value in clamps.items()}
    log_joint = trace(monadic, torch.zeros(1, 1), observations=observations).log_joint
    assert log_joint is not None
    return float(log_joint.sum())


def test_a_program_becomes_a_computation_over_canonical_instances() -> None:
    module = _module(BETA_BERNOULLI)
    entry = program_entry(module, "prog")
    computation = next(
        item for item in module.computations if item.id == entry.computation
    )
    assert [parameter.name for parameter in computation.parameters] == ["y"]
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("y", "observation")
    ]
    assert [(site.name, site.kind, site.family) for site in entry.sites] == [
        ("theta", "sample", "Beta"),
        ("y", "observe", "Bernoulli"),
    ]
    instances = {instance.name: instance for instance in module.instances}
    assert instances["random"].entry.instance == entry.random_instance
    assert instances["score"].entry.instance == entry.score_instance
    assert {entry.instance for entry in computation.type.effects.entries} == {
        entry.random_instance,
        entry.score_instance,
    }
    assert isinstance(computation.body, Bind)
    assert isinstance(computation.body.first, Perform)
    assert loads(dumps(module)) == module


def test_plates_type_the_sampled_tensors() -> None:
    module = _module(IID_PLATE)
    computation = next(item for item in module.computations if item.name == "prog")
    assert computation.parameters[0].type == tensor_type(
        REAL, (IndexLiteral(value=3, sort=NAT),)
    )
    assert computation.type.result == tensor_type(
        REAL, (IndexLiteral(value=3, sort=NAT),)
    )
    entry = program_entry(module, "prog")
    assert [axis.name for axis in entry.sites[0].batch] == ["N"]


def test_a_marginalization_block_becomes_an_enumerated_helper() -> None:
    module = _module(GROUPED)
    names = [item.name for item in module.computations]
    assert names == ["prog", "prog__cls_marginal"]
    helper = module.computations[1]
    assert [parameter.name for parameter in helper.parameters] == [
        "probs",
        "mu",
        "sigma",
        "idx",
        "r",
    ]
    assert helper.parameters[3].type == tensor_type(
        INT, (IndexLiteral(value=4, sort=NAT),)
    )
    assert isinstance(helper.body, NewInstance)
    assert isinstance(helper.body.body, Handle)
    handler_names = {handler.name for handler in module.handlers}
    assert {"enumerate_marginal", "collect_marginal"} <= handler_names
    program = module.computations[0]
    calls = []
    node = program.body
    while isinstance(node, Bind):
        if isinstance(node.first, Call):
            calls.append(node.first.name)
        node = node.then
    assert calls == ["prog__cls_marginal"]
    inner = helper.body.body.computation
    assert isinstance(inner, Bind)
    latent = inner.first
    assert isinstance(latent, Perform)
    distribution = latent.request.arguments[1]
    assert isinstance(distribution, DistributionValue)
    assert [axis.name for axis in distribution.plate.batch] == ["Item"]
    weights = inner.then.body.computation.first.request.arguments[0]  # type: ignore[union-attr]
    assert isinstance(weights, SegmentSum)


@pytest.mark.parametrize(
    ("source", "clamps", "sites"),
    [
        (BETA_BERNOULLI, {"y": True}, {"theta": 0.3}),
        (LET_STEP, {"y": 0.7}, {"a": 0.2, "b": -0.4}),
        (IID_PLATE, {"y": (0.1, 0.2, 0.3)}, {"xs": (0.5, -0.5, 1.0)}),
        (
            CATEGORICAL_LOGITS,
            {"y": 2},
            {"logits": (-2.0, 0.5, 1.25)},
        ),
        (SCORED, {}, {"x": 0.8}),
    ],
)
def test_reference_runs_agree_with_the_torch_runtime(
    source: str, clamps: dict[str, object], sites: dict[str, object]
) -> None:
    module = _module(source)
    run = run_program(module, "prog", data=clamps, sites=sites)
    expected = _classic_log_joint(source, {**clamps, **sites})
    assert run.log_joint == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_named_categorical_logits_match_torch_on_signed_scores() -> None:
    logits = torch.tensor((-2.0, 0.5, 1.25))
    run = run_program(
        _module(CATEGORICAL_LOGITS),
        "prog",
        data={"y": 2},
        sites={"logits": tuple(float(item) for item in logits)},
    )
    expected = td.Normal(0.0, 1.0).log_prob(logits).sum()
    expected = expected + td.Categorical(logits=logits).log_prob(torch.tensor(2))
    assert run.log_joint == pytest.approx(float(expected), rel=1e-5, abs=1e-6)


def test_categorical_rejects_probs_and_logits_together() -> None:
    source = """\
object Category : FinSet 3
program prog : Category -> Category
    sample probs : Category <- HalfNormal(scale=1.0)
    sample logits : Category <- Normal(loc=0.0, scale=1.0)
    observe y <- Categorical(probs=probs, logits=logits)
    return y
export prog
"""
    with pytest.raises(QiecDiagnosticError, match="exactly one"):
        _module(source)


def test_grouped_marginalization_agrees_with_the_torch_runtime_and_closed_form() -> (
    None
):
    module = _module(GROUPED)
    data = {
        "concentration": 1.5,
        "r": (0.1, 0.5, -0.3, 2.0),
        "idx": (0, 0, 1, 1),
    }
    sites = {"probs": (0.2, 0.3, 0.5), "mu": (0.0, 1.0, -1.0), "sigma": (1.0, 0.5, 2.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    clamps = {key: value for key, value in data.items() if key != "concentration"}
    classic = _classic_log_joint(
        GROUPED, {**clamps, **sites}, scalars={"concentration": 1.5}
    )
    probs, mu, sigma = (torch.tensor(sites[name]) for name in ("probs", "mu", "sigma"))
    responses, index = torch.tensor(data["r"]), torch.tensor(data["idx"])
    closed = (
        td.Dirichlet(torch.full((3,), 1.5)).log_prob(probs)
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + td.HalfNormal(1.0).log_prob(sigma).sum()
    )
    per_row = td.Normal(mu.unsqueeze(0), sigma.unsqueeze(0)).log_prob(
        responses.unsqueeze(-1)
    )
    for item in range(2):
        closed = closed + torch.logsumexp(
            probs.log() + per_row[index == item].sum(0), dim=0
        )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    assert run.log_joint == pytest.approx(classic, rel=1e-5)
    assert run.value == sites["probs"]


def test_a_kernel_morphism_reads_its_parameter_map() -> None:
    torch.manual_seed(1)
    program = Compiler(parse(KERNEL)).compile()
    monadic = program._morphism
    parameters = dict(monadic.named_parameters())
    weight = next(
        value for name, value in parameters.items() if name.endswith("weight")
    )
    bias = next(value for name, value in parameters.items() if name.endswith("bias"))
    torch.manual_seed(0)
    classic = trace(
        monadic,
        torch.tensor([[0.3, -0.2]]),
        observations={"x": torch.tensor([0.5, 0.1])},
    ).log_joint
    assert classic is not None
    module = _module(KERNEL)
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("obs", "domain"),
        ("kernel_param_weight", "weight"),
        ("kernel_param_bias", "bias"),
    ]
    run = run_program(
        module,
        "prog",
        data={
            "obs": (0.3, -0.2),
            "kernel_param_weight": tuple(
                tuple(float(v) for v in row) for row in weight.detach()
            ),
            "kernel_param_bias": tuple(float(v) for v in bias.detach()),
        },
        sites={"x": (0.5, 0.1)},
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)


def test_scores_add_as_weights() -> None:
    module = _module(SCORED)
    run = run_program(module, "prog", data={}, sites={"x": 0.8})
    assert run.log_joint == pytest.approx(
        -0.5 * 0.8**2 - 0.5 * math.log(2 * math.pi) + 0.64
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (
            "    sample (a, b) <- Normal(loc=0.0, scale=1.0)\n    return a\n",
            "a family draw binds one name",
        ),
        (
            "    observe y <- Normal(loc=0.0, scale=1.0) [via=idx]\n    return y\n",
            "outside a grouped",
        ),
        (
            "    marginalize z <- Normal(loc=0.0, scale=1.0) [reduction=mean]\n"
            "        observe y <- Normal(loc=z, scale=1.0)\n    return z\n",
            "reduction",
        ),
    ],
)
def test_forms_outside_the_elaboration_are_reported_at_the_step(
    body: str, fragment: str
) -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n" + body + "export prog\n"
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert captured.value.code == "qiec-program"
    assert fragment in captured.value.message
    assert captured.value.line == 3


_PAIRED = """\
object X : FinSet 2
object R : FinSet 2
program sub : X -> R * R
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=a, scale=1.0)
    return (a, b)
program main : X -> R
    sample (u, v) <- sub
    let s = u + v
    sample w <- Normal(loc=s, scale=1.0)
    return w
export main
"""


def test_a_program_draw_runs_the_program_in_place_under_the_pattern() -> None:
    """``sample (u, v) <- sub`` runs ``sub``'s steps in the caller: the
    names it returns become the pattern's, so its sites are ``u`` and
    ``v``, and the joint is the two programs' together."""
    module = _module(_PAIRED)
    run = run_program(module, "main", data={}, sites={"u": 0.3, "v": 0.5, "w": 1.0})
    expected = (
        td.Normal(0.0, 1.0).log_prob(torch.tensor(0.3))
        + td.Normal(0.3, 1.0).log_prob(torch.tensor(0.5))
        + td.Normal(0.8, 1.0).log_prob(torch.tensor(1.0))
    )
    assert run.log_joint == pytest.approx(float(expected), rel=1e-6)
    assert run.value == 1.0
    entry = next(item for item in module.entries if item.name == "main")
    assert [site.name for site in entry.sites] == ["u", "v", "w"]


def test_a_program_draw_under_one_name_prefixes_the_program_locals() -> None:
    """``sample pair <- sub`` binds ``pair`` to the pair ``sub`` returns
    and names ``sub``'s locals ``pair$a`` and ``pair$b``, as the torch
    runtime names a template's inlined latents."""
    source = _PAIRED.replace(
        "    sample (u, v) <- sub\n    let s = u + v\n",
        "    sample pair <- sub\n    let s = pair[0] + pair[1]\n",
    )
    module = _module(source)
    run = run_program(
        module, "main", data={}, sites={"pair$a": 0.3, "pair$b": 0.5, "w": 1.0}
    )
    assert run.log_joint == pytest.approx(
        float(
            td.Normal(0.0, 1.0).log_prob(torch.tensor(0.3))
            + td.Normal(0.3, 1.0).log_prob(torch.tensor(0.5))
            + td.Normal(0.8, 1.0).log_prob(torch.tensor(1.0))
        ),
        rel=1e-6,
    )


def test_a_pattern_of_the_wrong_arity_is_reported_at_the_step() -> None:
    source = _PAIRED.replace("sample (u, v) <- sub", "sample (u, v, t) <- sub")
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert captured.value.code == "qiec-program"
    assert "destructures 3 names" in captured.value.message
    assert "returns 2" in captured.value.message
    assert captured.value.line == 8


_TEMPLATE = """\
object School : FinSet 8
object Effect : Real 1
program school_effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect
program pooled : School -> Effect
    sample theta <- school_effects(0.6, School)
    sample sigma <- LogNormal(loc=0.0, scale=0.5)
    observe y : School <- Normal(loc=theta, scale=sigma)
    return theta
export pooled
"""


def test_a_template_draw_instantiates_the_template_and_agrees_with_torch() -> None:
    """A draw from a program template substitutes the objects and scalars
    it names, renames the template's local ``z`` to ``theta$z`` and its
    return to ``theta``, and scores the same joint the torch runtime
    does under those names; the template itself is a computation over
    the extent of ``K``."""
    module = _module(_TEMPLATE)
    template = next(
        item for item in module.computations if item.name == "school_effects"
    )
    assert [binder.name for binder in template.telescope] == ["K"]
    z = [0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.1]
    y = [0.3, -0.1, 0.4, 0.2, 0.6, -0.5, 0.1, 0.0]
    run = run_program(
        module,
        "pooled",
        data={"y": tuple(y)},
        sites={"theta$z": tuple(z), "sigma": 0.8},
    )
    theta = 0.6 * torch.tensor(z, dtype=torch.float64)
    expected = (
        td.Normal(0.0, 1.0).log_prob(torch.tensor(z, dtype=torch.float64)).sum()
        + td.LogNormal(0.0, 0.5).log_prob(torch.tensor(0.8, dtype=torch.float64))
        + td.Normal(theta, 0.8).log_prob(torch.tensor(y, dtype=torch.float64)).sum()
    )
    assert run.log_joint == pytest.approx(float(expected), rel=1e-6)
    compiled = Compiler(parse(_TEMPLATE)).compile().morphism
    traced = trace(
        compiled,
        torch.zeros(1, 1, dtype=torch.float64),
        observations={
            "theta$z": torch.tensor(z, dtype=torch.float64),
            "sigma": torch.tensor(0.8, dtype=torch.float64),
            "y": torch.tensor(y, dtype=torch.float64),
        },
    )
    assert float(traced.log_joint) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("parameter", ["probs", "logits"])
def test_a_template_draw_preserves_categorical_parameterization(
    parameter: str,
) -> None:
    """Alpha-renaming a template's local parameter must retain the source
    keyword that distinguishes categorical probabilities from logits."""
    source = f"""\
object Class : FinSet 3
program choice(K : FinSet) : K -> K
    sample scores : K <- Normal(loc=0.0, scale=1.0)
    sample z <- Categorical({parameter}=scores)
    return z
program main : Class -> Class
    sample outcome <- choice(Class)
    return outcome
export main
"""
    Compiler(parse(source)).compile()


_MORPHISM_TEMPLATE = """\
object Subj : FinSet 5
object UnitSpace : Real 1
morphism my_prior : Subj -> UnitSpace [role=kernel] ~ Normal(loc=0.0, scale=1.0)
program with_prior(G : FinSet, prior : Mor[Subj, UnitSpace]) : G -> Real 1
    sample v : G <- prior
    return v
program demo : Subj -> Subj
    sample by_subj <- with_prior(Subj, my_prior)
    return by_subj
export demo
"""


def test_a_template_over_a_morphism_draws_through_the_morphism_it_is_given() -> None:
    """A morphism parameter takes a declared morphism at the draw: the
    template's `sample v : G <- prior` becomes a draw through
    `my_prior` over `Subj`, named `by_subj` in the caller. The template
    itself is no entry point, since a morphism is no static argument."""
    module = _module(_MORPHISM_TEMPLATE)
    assert [item.name for item in module.entries] == ["demo"]
    entry = program_entry(module, "demo")
    assert [(site.name, site.family) for site in entry.sites] == [("by_subj", "Normal")]
    values = (0.1, 0.2, -0.3, 0.0, 0.5)
    run = run_program(
        module, "demo", data={}, sites={"by_subj": tuple((v,) for v in values)}
    )
    expected = td.Normal(0.0, 1.0).log_prob(torch.tensor(values, dtype=torch.float64))
    assert run.log_joint == pytest.approx(float(expected.sum()), rel=1e-6)
    compiled = Compiler(parse(_MORPHISM_TEMPLATE)).compile().morphism
    assert not tuple(compiled.named_parameters())
    traced = trace(
        compiled,
        torch.tensor([0]),
        observations={
            "by_subj": torch.tensor(
                tuple((value,) for value in values), dtype=torch.float64
            )
        },
    )
    assert traced.log_joint is not None
    assert float(traced.log_joint.sum()) == pytest.approx(
        float(run.log_joint), rel=1e-6
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(
            _MORPHISM_TEMPLATE.replace(
                "with_prior(Subj, my_prior)", "with_prior(Subj, 2.0)"
            )
        )
    assert "takes a declared morphism" in captured.value.message


def test_a_template_draw_with_the_wrong_arguments_is_reported() -> None:
    source = _TEMPLATE.replace(
        "sample theta <- school_effects(0.6, School)",
        "sample theta <- school_effects(School, 0.6)",
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert captured.value.code == "qiec-program"
    assert "takes a number or a bound name" in captured.value.message


def test_an_input_without_a_plate_gets_a_static_extent() -> None:
    """A vector input nothing shapes becomes an index parameter of the
    program, which a run reads off the data and the log joint agrees
    with torch on."""
    module = _module(
        "object Obs : FinSet 3\nprogram prog : Obs -> Obs\n"
        "    sample x <- Dirichlet(concentration=alpha)\n    return x\nexport prog\n"
    )
    computation = next(item for item in module.computations if item.name == "prog")
    assert [binder.name for binder in computation.telescope] == ["alpha_extent"]
    entry = program_entry(module, "prog")
    assert entry.telescope == computation.telescope
    (alpha,) = computation.parameters
    assert render_static(alpha.type) == "Tensor[Real]([alpha_extent])"
    assert render_static(computation.type.result) == "Tensor[Real]([alpha_extent])"
    run = run_program(
        module,
        "prog",
        data={"alpha": (1.0, 2.0, 3.0)},
        sites={"x": (0.2, 0.3, 0.5)},
    )
    expected = td.Dirichlet(torch.tensor([1.0, 2.0, 3.0])).log_prob(
        torch.tensor([0.2, 0.3, 0.5])
    )
    assert run.value == (0.2, 0.3, 0.5)
    assert run.log_joint == pytest.approx(expected.item(), rel=1e-6)
    with pytest.raises(KeyError, match="alpha"):
        run_program(module, "prog", data={"alpha": 1.0}, sites={"x": (1.0,)})


CALLS = """\
object Obs : FinSet 4

define shift(x : Real, by : Real) : Real !{} =
    return x + by

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

instance random : Random

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y <- Normal(loc=c, scale=0.5)
    return c
export prog

define twice(y : Real) : Real !{random, score} =
    let first <- prog(y)
    let second <- prog(y)
    return first + second
"""


def test_programs_and_computations_call_each_other() -> None:
    module = _module(CALLS)
    program = next(item for item in module.computations if item.name == "prog")
    calls = []
    node = program.body
    while isinstance(node, Bind):
        if isinstance(node.first, Call):
            calls.append(node.first.name)
        node = node.then
    assert calls == ["shift", "noisy"]
    twice = next(item for item in module.computations if item.name == "twice")
    assert isinstance(twice.body, Bind)
    assert isinstance(twice.body.first, Call)
    assert twice.body.first.name == "prog"
    entry = program_entry(module, "prog")
    assert [site.name for site in entry.sites] == ["a", "y"]
    run = run_program(module, "prog", data={"y": 2.4}, sites={"a": 0.3, "noise": 2.35})
    expected = (
        -0.5 * 0.3**2
        - 0.5 * (0.05 / 0.1) ** 2
        - math.log(0.1)
        - 0.5 * (0.05 / 0.5) ** 2
        - math.log(0.5)
        - 1.5 * math.log(2 * math.pi)
    )
    assert run.value == 2.35
    assert run.log_joint == pytest.approx(expected)


def test_a_call_step_round_trips_through_the_emitter_and_runs_in_place() -> None:
    """A program calling a computation that only draws runs the callee's
    steps in place: its site ``noise`` is the program's ``c$noise``, and
    the compiled joint is the one the reference run scores under the
    callee's own label."""
    module = parse(CALLS)
    source = module_to_source(module)
    assert "    let b <- shift(a, 2.0)\n" in source
    assert module_to_source(parse(source)) == source
    program = Compiler(module).compile()
    assert isinstance(program.morphism, MonadicProgram)
    checked = _module(CALLS)
    reference = run_program(
        checked, "prog", data={"y": 2.4}, sites={"a": 0.3, "noise": 2.35}
    )
    # Double precision, so the numbers the machine scores are the ones
    # the reference run scored.
    joint = program.morphism.log_joint(
        torch.zeros(1, 1),
        {
            "y": torch.tensor(2.4, dtype=torch.float64),
            "a": torch.tensor([0.3], dtype=torch.float64),
            "c$noise": torch.tensor([2.35], dtype=torch.float64),
        },
    )
    assert joint.shape == (1,)
    assert float(joint[0]) == pytest.approx(reference.log_joint)
    torch.manual_seed(0)
    drawn = program.morphism.rsample(
        torch.zeros(1, 1), observations={"y": torch.tensor(2.4)}
    )
    assert drawn.shape == (1, 1)
    assert torch.isfinite(drawn).all()


def test_open_extents_reach_marginal_helpers_and_callers() -> None:
    """A captured input with an open extent keeps its shape inside the
    marginalization helper, a program passing the input on takes the
    extent as its own, and an authored computation names the extent
    with an index literal."""
    module = _module(
        "object Obs : FinSet 3\n"
        "program prog : Obs -> Obs\n"
        "    sample probs <- Dirichlet(concentration=alpha)\n"
        "    marginalize z <- Categorical(probs=probs)\n"
        "        observe y <- Normal(loc=mu[z], scale=1.0)\n"
        "    return probs\n"
        "export prog\n"
        "program outer : Obs -> Obs\n"
        "    let inner <- prog(alpha, mu, y)\n"
        "    return inner\n"
        "export outer\n"
        "define fixed(alpha : Tensor[Real]([3]), mu : Tensor[Real]([3]), y : Real)"
        " : Tensor[Real]([3]) !{random, score} =\n"
        "    let inner <- prog[3](alpha, mu, y)\n"
        "    return inner\n"
    )
    helper = next(item for item in module.computations if "marginal" in item.name)
    assert [binder.name for binder in helper.telescope] == ["alpha_extent"]
    outer = next(item for item in module.computations if item.name == "outer")
    assert [binder.name for binder in outer.telescope] == ["alpha_extent"]
    assert isinstance(outer.body, Bind)
    assert isinstance(outer.body.first, Call)
    assert [render_static(a) for a in outer.body.first.static_arguments] == [
        "alpha_extent"
    ]
    fixed = next(item for item in module.computations if item.name == "fixed")
    assert fixed.telescope == ()
    assert isinstance(fixed.body, Bind)
    assert isinstance(fixed.body.first, Call)
    assert [render_static(a) for a in fixed.body.first.static_arguments] == ["3"]
    data = {"alpha": (1.0, 2.0, 3.0), "y": 1.4, "mu": (0.0, 1.0, 2.0)}
    sites = {"probs": (0.2, 0.3, 0.5)}
    expected = td.Dirichlet(torch.tensor([1.0, 2.0, 3.0])).log_prob(
        torch.tensor([0.2, 0.3, 0.5])
    ).item() + math.log(
        sum(
            weight * math.exp(td.Normal(float(k), 1.0).log_prob(torch.tensor(1.4)))
            for k, weight in enumerate((0.2, 0.3, 0.5))
        )
    )
    for name in ("prog", "outer"):
        run = run_program(module, name, data=data, sites=sites)
        assert run.log_joint == pytest.approx(expected, rel=1e-6)


REDUCED = """\
object Component : FinSet 3
object Item : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs <- Dirichlet(concentration=1.5) [over=Component]
    sample mu : Component <- Normal(loc=0.0, scale=5.0)
    sample sigma : Component <- HalfNormal(scale=1.0)
    marginalize cls : Component <- Categorical(probs=probs) [over=Item, reduction=REDUCTION]
        observe r : Resp <- Normal(loc=mu[cls], scale=sigma[cls]) [via=idx]
    return probs
export prog
"""

PRODUCT_GROUP = """\
object Component : FinSet 2
object Item : FinSet 2
object Subj : FinSet 3
object Resp : FinSet 6
program prog : Resp -> Resp
    sample probs <- Dirichlet(concentration=1.0) [over=Component]
    sample mu : Component <- Normal(loc=0.0, scale=5.0)
    sample sigma : Component <- HalfNormal(scale=1.0)
    marginalize cls : Component <- Categorical(probs=probs) [over=[Item, Subj]]
        observe r : Resp <- Normal(loc=mu[cls], scale=sigma[cls]) [via=[item_idx, subj_idx]]
    return probs
export prog
"""

NESTED_SAME_GROUP = """\
object Outer : FinSet 2
object Inner : FinSet 2
object Item : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs_outer <- Dirichlet(concentration=1.0) [over=Outer]
    sample probs_inner : Outer <- Dirichlet(concentration=1.0) [over=Inner]
    sample mu : Inner <- Normal(loc=0.0, scale=5.0)
    sample sigma : Inner <- HalfNormal(scale=1.0)
    marginalize z : Outer <- Categorical(probs=probs_outer) [over=Item]
        marginalize s : Inner <- Categorical(probs=probs_inner[z]) [over=Item]
            observe r : Resp <- Normal(loc=mu[s], scale=sigma[s]) [via=idx]
    return probs_outer
export prog
"""

NESTED_PROJECTED = """\
object Outer : FinSet 2
object Inner : FinSet 2
object Item : FinSet 2
object Subj : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs_outer <- Dirichlet(concentration=1.0) [over=Outer]
    sample probs_inner : Outer <- Dirichlet(concentration=1.0) [over=Inner]
    sample mu : Inner <- Normal(loc=0.0, scale=5.0)
    sample sigma : Inner <- HalfNormal(scale=1.0)
    marginalize z : Outer <- Categorical(probs=probs_outer) [over=Item]
        marginalize s : Inner <- Categorical(probs=probs_inner[z]) [over=[Item, Subj]]
            observe r : Resp <- Normal(loc=mu[s], scale=sigma[s]) [via=[item_idx, subj_idx]]
    return probs_outer
export prog
"""

HOISTED_DRAW = """\
object Component : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs <- Dirichlet(concentration=1.0) [over=Component]
    marginalize cls : Component <- Categorical(probs=probs)
        sample mu : Component <- Normal(loc=0.0, scale=5.0)
        observe r : Resp <- Normal(loc=mu[cls], scale=1.0)
    return probs
export prog
"""


def _sampled_types(module: QiecModule, name: str) -> dict[str, str]:
    """The sampled type of every site a program's body draws at.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    name : str
        The program's name.

    Returns
    -------
    dict[str, str]
        By site label, the rendered static argument of its
        ``Random.sample`` request.
    """
    computation = next(item for item in module.computations if item.name == name)
    found: dict[str, str] = {}
    node = computation.body
    while isinstance(node, Bind):
        first = node.first
        if isinstance(first, Perform) and first.request.arguments:
            label = first.request.arguments[0]
            if isinstance(label, SiteValue):
                found[label.label] = render_static(first.request.static_arguments[0])
        node = node.then
    return found


def _mixture_rows(mu: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """Per-row, per-class log-likelihoods of unit-scale normal responses.

    Parameters
    ----------
    mu : torch.Tensor
        One location per class.
    responses : torch.Tensor
        The responses.

    Returns
    -------
    torch.Tensor
        A ``(rows, classes)`` matrix of log densities.
    """
    return td.Normal(mu.unsqueeze(0), 1.0).log_prob(responses.unsqueeze(-1))


def _priors(
    probs: torch.Tensor, mu: torch.Tensor, concentration: float
) -> torch.Tensor:
    """The prior density of a mixture's class probabilities and components.

    Parameters
    ----------
    probs : torch.Tensor
        The class probabilities, under a symmetric Dirichlet.
    mu : torch.Tensor
        The component locations, under ``Normal(0, 5)``, whose scales
        are clamped at one under ``HalfNormal(1)``.
    concentration : float
        The Dirichlet's symmetric concentration.

    Returns
    -------
    torch.Tensor
        The summed prior log density.
    """
    classes = probs.shape[-1]
    return (
        td.Dirichlet(torch.full((classes,), concentration)).log_prob(probs).sum()
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + td.HalfNormal(1.0).log_prob(torch.ones_like(mu)).sum()
    )


@pytest.mark.parametrize("reduction", ["logsumexp", "sum", "mean"])
def test_marginalization_reductions_aggregate_the_grouped_shots(
    reduction: str,
) -> None:
    source = REDUCED.replace("REDUCTION", reduction)
    module = _module(source)
    data = {"r": (0.1, 0.5, -0.3, 2.0), "idx": (0, 0, 1, 1)}
    sites = {
        "probs": (0.2, 0.3, 0.5),
        "mu": (0.0, 1.0, -1.0),
        "sigma": (1.0, 1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    responses, index = torch.tensor(data["r"]), torch.tensor(data["idx"])
    per_row = _mixture_rows(mu, responses)
    closed = _priors(probs, mu, 1.5)
    for item in range(2):
        weighted = probs.log() + per_row[index == item].sum(0)
        if reduction == "logsumexp":
            closed = closed + torch.logsumexp(weighted, dim=0)
        elif reduction == "sum":
            closed = closed + weighted.sum()
        else:
            closed = closed + weighted.mean()
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    classic = _classic_log_joint(source, {**data, **sites})
    assert run.log_joint == pytest.approx(classic, rel=1e-5)


def test_a_reduction_on_a_continuous_latent_is_refused() -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n"
        "    marginalize z <- Normal(loc=0.0, scale=1.0) [reduction=logsumexp]\n"
        "        observe y <- Normal(loc=z, scale=1.0)\n    return z\nexport prog\n"
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "finite support" in captured.value.message


def test_a_product_group_flattens_its_index_maps_row_major() -> None:
    module = _module(PRODUCT_GROUP)
    data = {
        "r": (0.1, 0.5, -0.3, 2.0, 1.0, -1.5),
        "item_idx": (0, 0, 1, 1, 0, 1),
        "subj_idx": (0, 1, 2, 0, 2, 1),
    }
    sites = {"probs": (0.4, 0.6), "mu": (0.0, 1.0), "sigma": (1.0, 1.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    responses = torch.tensor(data["r"])
    flat = torch.tensor(data["item_idx"]) * 3 + torch.tensor(data["subj_idx"])
    per_row = _mixture_rows(mu, responses)
    closed = _priors(probs, mu, 1.0)
    for group in range(6):
        closed = closed + torch.logsumexp(
            probs.log() + per_row[flat == group].sum(0), dim=0
        )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    classic = _classic_log_joint(PRODUCT_GROUP, {**data, **sites})
    assert run.log_joint == pytest.approx(classic, rel=1e-5)


def test_a_product_index_map_must_match_the_group_arity() -> None:
    source = PRODUCT_GROUP.replace("[via=[item_idx, subj_idx]]", "[via=item_idx]")
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "names 1 factor(s) but the group 'ItemxSubj' has 2" in captured.value.message


def _nested_closed_form(
    probs_outer: torch.Tensor,
    probs_inner: torch.Tensor,
    mu: torch.Tensor,
    responses: torch.Tensor,
    inner_index: torch.Tensor,
    inner_groups: int,
    projection: torch.Tensor,
) -> torch.Tensor:
    """The joint of a nested grouped mixture by explicit enumeration.

    Parameters
    ----------
    probs_outer : torch.Tensor
        The outer class prior.
    probs_inner : torch.Tensor
        One inner class prior per outer class.
    mu : torch.Tensor
        One location per inner class.
    responses : torch.Tensor
        The responses.
    inner_index : torch.Tensor
        Each response's inner group.
    inner_groups : int
        The number of inner groups.
    projection : torch.Tensor
        Each inner group's outer group.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    per_row = _mixture_rows(mu, responses)
    total = td.Dirichlet(torch.ones(2)).log_prob(probs_outer) + _priors(
        probs_inner, mu, 1.0
    )
    for group in range(int(projection.max()) + 1):
        shots = []
        for z in range(2):
            inner_total = torch.zeros(())
            for inner_group in range(inner_groups):
                if int(projection[inner_group]) != group:
                    continue
                rows = per_row[inner_index == inner_group].sum(0)
                inner_total = inner_total + torch.logsumexp(
                    probs_inner[z].log() + rows, dim=0
                )
            shots.append(probs_outer[z].log() + inner_total)
        total = total + torch.logsumexp(torch.stack(shots), dim=0)
    return total


def test_a_nested_grouped_marginalization_adds_per_group_marginals() -> None:
    module = _module(NESTED_SAME_GROUP)
    data = {"r": (0.1, 0.5, -0.3, 2.0), "idx": (0, 0, 1, 1)}
    sites = {
        "probs_outer": (0.3, 0.7),
        "probs_inner": ((0.5, 0.5), (0.1, 0.9)),
        "mu": (0.0, 1.0),
        "sigma": (1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    closed = _nested_closed_form(
        torch.tensor(sites["probs_outer"]),
        torch.tensor(sites["probs_inner"]),
        torch.tensor(sites["mu"]),
        torch.tensor(data["r"]),
        torch.tensor(data["idx"]),
        2,
        torch.arange(2),
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_nested_product_group_projects_onto_the_outer_group() -> None:
    module = _module(NESTED_PROJECTED)
    data = {
        "r": (0.1, 0.5, -0.3, 2.0),
        "item_idx": (0, 0, 1, 1),
        "subj_idx": (0, 1, 0, 1),
    }
    sites = {
        "probs_outer": (0.3, 0.7),
        "probs_inner": ((0.5, 0.5), (0.1, 0.9)),
        "mu": (0.0, 1.0),
        "sigma": (1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    flat = torch.tensor(data["item_idx"]) * 2 + torch.tensor(data["subj_idx"])
    closed = _nested_closed_form(
        torch.tensor(sites["probs_outer"]),
        torch.tensor(sites["probs_inner"]),
        torch.tensor(sites["mu"]),
        torch.tensor(data["r"]),
        flat,
        4,
        torch.tensor([0, 0, 1, 1]),
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_nested_group_unrelated_to_the_outer_is_refused() -> None:
    source = (
        NESTED_PROJECTED.replace("[over=[Item, Subj]]", "[over=Subj]")
        .replace("[via=[item_idx, subj_idx]]", "[via=subj_idx]")
        .replace("object Subj : FinSet 2", "object Subj : FinSet 3")
        .replace(
            "Categorical(probs=probs_inner[z])",
            "Categorical(probs=probs_outer)",
        )
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "projected onto it" in captured.value.message


def test_a_draw_inside_a_block_is_drawn_once_before_it() -> None:
    module = _module(HOISTED_DRAW)
    entry = program_entry(module, "prog")
    assert [site.name for site in entry.sites] == ["probs", "mu", "cls", "r"]
    data = {"r": (0.1, 0.5, -0.3, 2.0)}
    sites = {"probs": (0.4, 0.6), "mu": (0.0, 1.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    per_row = _mixture_rows(mu, torch.tensor(data["r"]))
    closed = (
        td.Dirichlet(torch.ones(2)).log_prob(probs)
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + torch.logsumexp(probs.log() + per_row.sum(0), dim=0)
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_draw_reading_the_latent_inside_a_block_is_refused() -> None:
    source = HOISTED_DRAW.replace(
        "sample mu : Component <- Normal(loc=0.0, scale=5.0)",
        "sample mu <- Normal(loc=cls, scale=5.0)",
    ).replace("Normal(loc=mu[cls], scale=1.0)", "Normal(loc=mu, scale=1.0)")
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "draw per value of the latent" in captured.value.message


@pytest.mark.parametrize(
    ("draw", "shape"),
    [
        ("sample pc <- Dirichlet(concentration=1.0)", (3,)),
        ("sample pc : Cat <- Dirichlet(concentration=1.0)", (3,)),
        ("sample pc : Item <- Dirichlet(concentration=[1.0, 2.0])", (5, 2)),
        ("sample pc : Item <- Dirichlet(concentration=1.0) [over=Cat]", (5, 3)),
        ("sample pc <- Dirichlet(concentration=[1.0, 2.0, 3.0, 4.0])", (4,)),
    ],
)
def test_a_vector_family_gathers_its_spread_literals(
    draw: str, shape: tuple[int, ...]
) -> None:
    source = (
        "object Cat : FinSet 3\nobject Item : FinSet 5\n"
        f"program prog : Cat -> Cat\n    {draw}\n    return pc\nexport prog\n"
    )
    module = _module(source)
    sampled = _sampled_types(module, "prog")["pc"]
    assert sampled == f"Tensor[Real]([{', '.join(map(str, shape))}])"


def test_an_integer_atom_beside_an_integer_family_shares_its_element() -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n"
        "    sample rate <- Gamma(concentration=2.0, rate=1.0)\n"
        "    sample y <- Mixture(weights=[0.3, 0.7], components=[PointMass(value=0.0), Poisson(rate=rate)])\n"
        "    return rate\nexport prog\n"
    )
    module = _module(source)
    assert _sampled_types(module, "prog")["y"] == "Int"
    run = run_program(module, "prog", data={}, sites={"rate": 1.5, "y": 0})
    closed = td.Gamma(2.0, 1.0).log_prob(torch.tensor(1.5)) + math.log(
        0.3 + 0.7 * math.exp(-1.5)
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


FINITE_KERNEL = """\
object Cls : FinSet 3
object Obs : Real 2
morphism kernel : Cls -> Obs [role=kernel] ~ Normal
program prog : Cls -> Obs
    sample x <- kernel
    return x
export prog
"""


def test_a_kernel_over_a_finite_domain_reads_its_table() -> None:
    torch.manual_seed(1)
    program = Compiler(parse(FINITE_KERNEL)).compile()
    monadic = program._morphism
    table = next(
        value
        for name, value in dict(monadic.named_parameters()).items()
        if name.endswith("table")
    )
    torch.manual_seed(0)
    classic = trace(
        monadic, torch.tensor([2]), observations={"x": torch.tensor([0.5, 0.1])}
    ).log_joint
    assert classic is not None
    module = _module(FINITE_KERNEL)
    assert loads(dumps(module)) == module
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("cls", "domain"),
        ("kernel_param_table", "table"),
    ]
    rows = tuple(tuple(float(v) for v in row) for row in table.detach())
    run = run_program(
        module,
        "prog",
        data={"cls": 2, "kernel_param_table": rows},
        sites={"x": (0.5, 0.1)},
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)
    loc = torch.tensor(rows[2][:2])
    scale = torch.tensor(rows[2][2:]).exp()
    closed = td.Normal(loc, scale).log_prob(torch.tensor([0.5, 0.1])).sum()
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


NETWORK_KERNEL = """\
object Feature : Real 2
object Target : Real 1
object Resp : FinSet 6
morphism net : Feature -> Target [param_source=mlp(5, 3)] ~ Normal
program prog : Resp -> Target
    observe y : Resp <- net(x)
    return y
export prog
"""


def _nested(value: torch.Tensor) -> object:
    """A tensor as nested tuples of floats.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    object
        Nested tuples down to floats, or a float for a scalar.
    """
    if value.dim() == 0:
        return float(value)
    return tuple(_nested(row) for row in value)


def _network_data(
    parameters: dict[str, torch.Tensor], prefix: str, morphism: str, depth: int
) -> dict[str, object]:
    """The program inputs a torch multilayer perceptron's weights fill.

    Parameters
    ----------
    parameters : dict[str, torch.Tensor]
        The classic program's named parameters.
    prefix : str
        The parameter prefix of the perceptron's ``net`` sequential.
    morphism : str
        The morphism the inputs belong to.
    depth : int
        The number of hidden layers.

    Returns
    -------
    dict[str, object]
        The hidden layers' and the heads' weights and biases by input
        name.
    """
    data: dict[str, object] = {}
    for layer in range(depth):
        data[f"{morphism}_param_layer{layer}_weight"] = _nested(
            parameters[f"{prefix}{2 * layer}.weight"]
        )
        data[f"{morphism}_param_layer{layer}_bias"] = _nested(
            parameters[f"{prefix}{2 * layer}.bias"]
        )
    data[f"{morphism}_param_weight"] = _nested(
        parameters[f"{prefix}{2 * depth}.weight"]
    )
    data[f"{morphism}_param_bias"] = _nested(parameters[f"{prefix}{2 * depth}.bias"])
    return data


def test_a_network_kernel_reads_its_layers_at_every_row() -> None:
    torch.manual_seed(1)
    program = Compiler(parse(NETWORK_KERNEL)).compile()
    monadic = program._morphism
    assert isinstance(monadic, MonadicProgram)
    parameters = {name: value.detach() for name, value in monadic.named_parameters()}
    assert [tuple(value.shape) for value in parameters.values()] == [
        (5, 2),
        (5,),
        (3, 5),
        (3,),
        (2, 3),
        (2,),
    ]
    torch.manual_seed(0)
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)
    # One run of the program: its six rows come from the data.
    classic = trace(monadic, torch.zeros(1, 1), observations={"y": y, "x": x}).log_joint
    assert classic is not None
    classic = classic.detach()
    module = _module(NETWORK_KERNEL)
    assert loads(dumps(module)) == module
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("x", "data"),
        ("net_param_layer0_weight", "weight"),
        ("net_param_layer0_bias", "bias"),
        ("net_param_layer1_weight", "weight"),
        ("net_param_layer1_bias", "bias"),
        ("net_param_weight", "weight"),
        ("net_param_bias", "bias"),
        ("y", "observation"),
    ]
    assert render_static(entry.parameters[-1].type) == "Tensor[Real]([6, 1])"
    assert render_static(entry.parameters[0].type) == "Tensor[Real]([6, 2])"
    data = _network_data(parameters, "_step_y._family.param_source.net.", "net", 2)
    run = run_program(
        module, "prog", data={"x": _nested(x), "y": _nested(y), **data}, sites={}
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)
    layers = "_step_y._family.param_source.net."
    hidden = torch.tanh(
        x @ parameters[layers + "0.weight"].T + parameters[layers + "0.bias"]
    )
    hidden = torch.tanh(
        hidden @ parameters[layers + "2.weight"].T + parameters[layers + "2.bias"]
    )
    heads = hidden @ parameters[layers + "4.weight"].T + parameters[layers + "4.bias"]
    closed = td.Normal(heads[:, :1], heads[:, 1:].exp().clamp(min=1e-7)).log_prob(y)
    assert run.log_joint == pytest.approx(float(closed.sum()), rel=1e-5)


@pytest.mark.parametrize(
    ("options", "widths"),
    [
        ("[param_source=mlp]", ((64, 2), (64,), (64, 64), (64,), (2, 64), (2,))),
        ("[param_source=mlp, hidden_dim=8]", ((8, 2), (8,), (2, 8), (2,))),
        (
            "[param_source=mlp, hidden_dim=[4, 6]]",
            ((4, 2), (4,), (6, 4), (6,), (2, 6), (2,)),
        ),
    ],
)
def test_hidden_widths_follow_the_option_block(
    options: str, widths: tuple[tuple[int, ...], ...]
) -> None:
    source = NETWORK_KERNEL.replace("[param_source=mlp(5, 3)]", options)
    module = _module(source)
    entry = program_entry(module, "prog")
    shapes = tuple(
        render_static(parameter.type)
        for parameter in entry.parameters
        if parameter.role in {"weight", "bias"}
    )
    assert shapes == tuple(
        f"Tensor[Real]([{', '.join(str(dimension) for dimension in shape)}])"
        for shape in widths
    )


def test_a_network_kernel_rejects_a_fractional_width() -> None:
    source = NETWORK_KERNEL.replace("mlp(5, 3)", "mlp(2.5)")
    with pytest.raises(
        QiecDiagnosticError, match="hidden width that is not an integer"
    ):
        _module(source)


TOWER = """\
object Token : FinSet 5
object Latent : Real 3
object HeadOut : Real 2
object Wide : Real 4
morphism emb : Token -> Latent [role=embed]
morphism head : Latent -> HeadOut [replicate=2, param_source=mlp(3)] ~ Normal
morphism out : Wide -> Latent ~ Normal
define tower = emb >> fan(head) >> out
program prog : Token -> Latent
    sample h <- tower
    return h
export prog
"""


def test_an_embedding_fan_and_bundle_elaborate_as_sites() -> None:
    torch.manual_seed(3)
    program = Compiler(parse(TOWER)).compile()
    monadic = program._morphism
    assert isinstance(monadic, MonadicProgram)
    parameters = {name: value.detach() for name, value in monadic.named_parameters()}
    module = _module(TOWER)
    assert loads(dumps(module)) == module
    entry = program_entry(module, "prog")
    assert [(site.name, site.family) for site in entry.sites] == [
        ("h_chain_1", "Normal"),
        ("h_chain_2_par_0_3", "Normal"),
        ("h_chain_2_par_1_4", "Normal"),
        ("h", "Normal"),
    ]
    assert [parameter.name for parameter in entry.parameters] == [
        "token",
        "emb_param_table",
        "head_0_param_layer0_weight",
        "head_0_param_layer0_bias",
        "head_0_param_weight",
        "head_0_param_bias",
        "head_1_param_layer0_weight",
        "head_1_param_layer0_bias",
        "head_1_param_weight",
        "head_1_param_bias",
        "out_param_weight",
        "out_param_bias",
    ]
    root = "_step_h."
    data: dict[str, object] = {
        "token": 2,
        "emb_param_table": _nested(
            torch.cat(
                [
                    parameters[root + "left.left.centers"],
                    parameters[root + "left.left.log_sigma"],
                ],
                dim=-1,
            )
        ),
        "out_param_weight": _nested(
            parameters[root + "right.param_source.linear.weight"]
        ),
        "out_param_bias": _nested(parameters[root + "right.param_source.linear.bias"]),
    }
    for index in range(2):
        data.update(
            _network_data(
                parameters,
                f"{root}left.right._components.{index}.param_source.net.",
                f"head_{index}",
                1,
            )
        )
    embedded = torch.tensor([0.1, -0.3, 0.5])
    first = torch.tensor([0.2, 0.4])
    second = torch.tensor([-0.1, 0.9])
    output = torch.tensor([0.3, 0.1, -0.2])
    run = run_program(
        module,
        "prog",
        data=data,
        sites={
            "h_chain_1": _nested(embedded),
            "h_chain_2_par_0_3": _nested(first),
            "h_chain_2_par_1_4": _nested(second),
            "h": _nested(output),
        },
    )
    chain = monadic.get_submodule("_step_h")
    assert isinstance(chain, SampledComposition)
    embed, fan, out = chain.factors
    bundle = torch.cat([first, second]).unsqueeze(0)
    classic = (
        embed.log_prob(torch.tensor([2]), embedded.unsqueeze(0))
        + fan.log_prob(embedded.unsqueeze(0), bundle)
        + out.log_prob(bundle, output.unsqueeze(0))
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)
    assert run.value == _nested(output)


def test_a_stack_copies_its_morphisms_and_a_product_splits_its_input() -> None:
    source = """\
object Left, Right : FinSet 3
object Latent : Real 2
object Wide : Real 4
morphism left_embed : Left -> Latent [role=embed]
morphism right_embed : Right -> Latent [role=embed]
morphism deep : Latent -> Latent [param_source=mlp(2)] ~ Normal
morphism join : Wide -> Latent ~ Normal
define tower = left_embed >> stack(deep, 2)
define backbone = (tower @ right_embed) >> join
program prog : Left * Right -> Latent
    sample h <- backbone
    return h
export prog
"""
    module = _module(source)
    entry = program_entry(module, "prog")
    names = [parameter.name for parameter in entry.parameters]
    assert names[:2] == ["left", "left_embed_param_table"]
    assert "deep_param_weight" in names
    assert "deep_copy1_param_weight" in names
    assert "right" in names
    assert "right_embed_param_table" in names
    assert [site.name for site in entry.sites] == [
        "h_chain_1_par_0_chain_3",
        "h_chain_1_par_0_chain_4",
        "h_chain_1_par_0_2",
        "h_chain_1_par_1_5",
        "h",
    ]
    types = _sampled_types(module, "prog")
    assert types["h_chain_1_par_0_chain_3"] == "Tensor[Real]([2])"
    assert types["h"] == "Tensor[Real]([2])"


def test_an_embedding_under_a_plate_reads_a_row_per_position() -> None:
    source = """\
object Token : FinSet 4
object Latent : Real 2
object Pos : FinSet 3
morphism emb : Token -> Latent [role=embed]
program prog : Pos -> Latent
    sample e : Pos <- emb(tokens)
    return e
export prog
"""
    module = _module(source)
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("tokens", "data"),
        ("emb_param_table", "table"),
    ]
    assert _sampled_types(module, "prog")["e"] == "Tensor[Real]([3, 2])"
    table = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.5, -0.5],
            [4.0, 5.0, 1.0, 1.0],
            [6.0, 7.0, 0.0, 0.2],
        ]
    )
    draws = torch.tensor([[0.1, 1.2], [2.3, 2.9], [0.0, 1.0]])
    run = run_program(
        module,
        "prog",
        data={"tokens": (0, 1, 0), "emb_param_table": _nested(table)},
        sites={"e": _nested(draws)},
    )
    rows = table[torch.tensor([0, 1, 0])]
    closed = td.Normal(rows[:, :2], rows[:, 2:].exp()).log_prob(draws).sum()
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


KERNEL_HEADS = """\
object In : Real 2
object Out : {codomain}
morphism kernel : In -> Out ~ {family}
program prog : In -> Out
    sample x <- kernel
    return x
export prog
"""


@pytest.mark.parametrize(
    ("family", "codomain", "value", "site"),
    [
        ("Gamma", "Real 1", torch.tensor([1.3]), (1.3,)),
        ("Poisson", "Real 1", torch.tensor([2.0]), (2,)),
        ("Beta", "Real 1", torch.tensor([0.4]), (0.4,)),
        ("Exponential", "Real 2", torch.tensor([0.3, 0.9]), (0.3, 0.9)),
        ("LogitNormal", "Real 2", torch.tensor([0.3, 0.9]), (0.3, 0.9)),
        ("StudentT", "Real 2", torch.tensor([0.3, -0.7]), (0.3, -0.7)),
        ("Dirichlet", "Real 3", torch.tensor([0.2, 0.5, 0.3]), (0.2, 0.5, 0.3)),
        ("Categorical", "FinSet 4", torch.tensor([2]), 2),
        ("Bernoulli", "FinSet 2", torch.tensor([1]), True),
    ],
)
def test_kernel_heads_follow_the_torch_parameterization(
    family: str, codomain: str, value: torch.Tensor, site: object
) -> None:
    source = KERNEL_HEADS.format(family=family, codomain=codomain)
    torch.manual_seed(4)
    program = Compiler(parse(source)).compile()
    monadic = program._morphism
    assert isinstance(monadic, MonadicProgram)
    parameters = {name: item.detach() for name, item in monadic.named_parameters()}
    classic = trace(
        monadic, torch.tensor([[0.3, -0.2]]), observations={"x": value.unsqueeze(0)}
    ).log_joint
    assert classic is not None
    module = _module(source)
    run = run_program(
        module,
        "prog",
        data={
            "in": (0.3, -0.2),
            "kernel_param_weight": _nested(
                parameters["_step_x.param_source.linear.weight"]
            ),
            "kernel_param_bias": _nested(
                parameters["_step_x.param_source.linear.bias"]
            ),
        },
        sites={"x": site},
    )
    assert run.log_joint == pytest.approx(float(classic.detach().sum()), rel=1e-5)


SCANNED = """\
object Input : Real 2
object Hidden : Real 3
morphism cell : Input * Hidden -> Hidden ~ Normal
define rnn = scan(cell)
program prog : Input -> Hidden
    sample h <- rnn
    return h
export prog
"""


def test_a_scan_recurs_over_a_sequence() -> None:
    torch.manual_seed(2)
    program = Compiler(parse(SCANNED)).compile()
    monadic = program._morphism
    assert isinstance(monadic, MonadicProgram)
    parameters = {name: item.detach() for name, item in monadic.named_parameters()}
    weight = parameters["_scan_h._step_h.param_source.linear.weight"]
    bias = parameters["_scan_h._step_h.param_source.linear.bias"]
    module = _module(SCANNED)
    assert loads(dumps(module)) == module
    assert [item.name for item in module.computations] == [
        "cell__scan_step",
        "prog",
        "prog__h_scan",
    ]
    entry = program_entry(module, "prog")
    assert [
        (item.name, item.role, render_static(item.type)) for item in entry.parameters
    ] == [
        ("input", "domain", "Tensor[Real]([input_extent, 2])"),
        ("cell_param_weight", "weight", "Tensor[Real]([6, 5])"),
        ("cell_param_bias", "bias", "Tensor[Real]([6])"),
    ]
    assert [binder.name for binder in entry.telescope] == ["input_extent"]
    inputs = torch.randn(4, 2)
    states = torch.randn(4, 3)
    run = run_program(
        module,
        "prog",
        data={
            "input": _nested(inputs),
            "cell_param_weight": _nested(weight),
            "cell_param_bias": _nested(bias),
        },
        sites={f"h@{position}": _nested(states[position]) for position in range(4)},
    )
    previous = torch.zeros(3)
    closed = torch.tensor(0.0)
    for position in range(4):
        raw = weight @ torch.cat([inputs[position], previous]) + bias
        closed = (
            closed + td.Normal(raw[:3], raw[3:].exp()).log_prob(states[position]).sum()
        )
        previous = states[position]
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    assert run.value == _nested(states[3])
    compiled = monadic.log_joint(
        inputs.unsqueeze(0),
        {f"h@{position}": states[position : position + 1] for position in range(4)},
    ).detach()
    assert run.log_joint == pytest.approx(float(compiled.sum()), rel=1e-5)


RECURRENT_CELL = """\
object Token : FinSet 5
object Embedded : Real 2
object Hidden : Real 3
morphism tok_embed : Token -> Embedded [role=embed]
morphism cell : Embedded * Hidden -> Hidden ~ LogitNormal
morphism lm_head : Hidden -> Token ~ Categorical
program rnn_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    sample squashed <- cell(x_t, h_prev)
    let h_new = 2.0 * squashed - 1.0
    return h_new
define backbone = tok_embed >> scan(rnn_cell)
program lm : Token -> Token
    sample h <- backbone
    observe next_token <- lm_head(h)
    return next_token
export lm
"""


def test_a_scan_step_program_carries_the_cells_inputs() -> None:
    module = _module(RECURRENT_CELL)
    assert loads(dumps(module)) == module
    assert [item.name for item in module.computations] == [
        "rnn_cell",
        "rnn_cell__scan_step",
        "lm",
        "lm__h_scan",
    ]
    step = program_entry(module, "rnn_cell__scan_step")
    assert [(item.name, item.role) for item in step.parameters] == [
        ("h$x_t", "domain"),
        ("h$h_prev", "domain"),
        ("tok_embed_param_table", "table"),
        ("cell_param_weight", "weight"),
        ("cell_param_bias", "bias"),
    ]
    assert [(site.name, site.family) for site in step.sites] == [("h$x_in", "Normal")]
    entry = program_entry(module, "lm")
    assert [
        (item.name, item.role, render_static(item.type)) for item in entry.parameters
    ] == [
        ("token", "domain", "Tensor[Int]([token_extent])"),
        ("tok_embed_param_table", "table", "Tensor[Real]([5, 4])"),
        ("cell_param_weight", "weight", "Tensor[Real]([6, 5])"),
        ("cell_param_bias", "bias", "Tensor[Real]([6])"),
        ("lm_head_param_weight", "weight", "Tensor[Real]([5, 3])"),
        ("lm_head_param_bias", "bias", "Tensor[Real]([5])"),
        ("next_token", "observation", "Int"),
    ]
    torch.manual_seed(5)
    table = torch.randn(5, 4)
    weight = torch.randn(6, 5)
    bias = torch.randn(6)
    head_weight = torch.randn(5, 3)
    head_bias = torch.randn(5)
    tokens = (1, 4, 0)
    embedded = torch.randn(3, 2)
    squashed = torch.rand(3, 3)
    run = run_program(
        module,
        "lm",
        data={
            "token": tokens,
            "tok_embed_param_table": _nested(table),
            "cell_param_weight": _nested(weight),
            "cell_param_bias": _nested(bias),
            "lm_head_param_weight": _nested(head_weight),
            "lm_head_param_bias": _nested(head_bias),
            "next_token": 3,
        },
        sites={
            **{f"h$x_in@{t}": _nested(embedded[t]) for t in range(3)},
            **{f"squashed@{t}": _nested(squashed[t]) for t in range(3)},
        },
    )
    previous = torch.zeros(3)
    closed = torch.tensor(0.0)
    for position in range(3):
        row = table[tokens[position]]
        closed = (
            closed
            + td.Normal(row[:2], row[2:].exp()).log_prob(embedded[position]).sum()
        )
        raw = weight @ torch.cat([embedded[position], previous]) + bias
        point = squashed[position].clamp(1e-7, 1 - 1e-7)
        logit = torch.log(point / (1 - point))
        closed = (
            closed
            + (
                td.Normal(raw[:3], raw[3:].exp()).log_prob(logit)
                - torch.log(point * (1 - point))
            ).sum()
        )
        previous = 2.0 * squashed[position] - 1.0
    logits = head_weight @ previous + head_bias
    closed = closed + td.Categorical(logits=logits).log_prob(torch.tensor(3))
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    assert run.value == 3


def test_a_scalar_family_broadcasts_over_a_vector_argument() -> None:
    source = """\
object Obs : Real 3
program prog : Obs -> Obs
    let mu = [0.5, -1.0, 2.0]
    sample h <- Normal(loc=mu, scale=0.5)
    return h
export prog
"""
    module = _module(source)
    assert _sampled_types(module, "prog")["h"] == "Tensor[Real]([3])"
    run = run_program(
        module, "prog", data={"obs": (0.0, 0.0, 0.0)}, sites={"h": (0.4, -0.9, 2.5)}
    )
    closed = td.Normal(torch.tensor([0.5, -1.0, 2.0]), 0.5).log_prob(
        torch.tensor([0.4, -0.9, 2.5])
    )
    assert run.log_joint == pytest.approx(float(closed.sum()), rel=1e-5)


def test_a_call_passes_the_callees_inputs_through() -> None:
    source = """\
object In : Real 2
object Out : Real 1
morphism kernel : In -> Out ~ Normal
program inner(x) : In -> Out
    sample y <- kernel(x)
    return y
program outer : In -> Out
    let y <- inner(in)
    return y
export outer
"""
    module = _module(source)
    entry = program_entry(module, "outer")
    assert [(item.name, item.role) for item in entry.parameters] == [
        ("in", "domain"),
        ("kernel_param_weight", "weight"),
        ("kernel_param_bias", "bias"),
    ]
    run = run_program(
        module,
        "outer",
        data={
            "in": (0.3, -0.2),
            "kernel_param_weight": ((1.0, 2.0), (0.0, 0.0)),
            "kernel_param_bias": (0.5, 0.0),
        },
        sites={"y": (1.0,)},
    )
    closed = td.Normal(0.3 + 2.0 * -0.2 + 0.5, 1.0).log_prob(torch.tensor(1.0))
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_scan_may_start_from_a_learned_state() -> None:
    source = SCANNED.replace("scan(cell)", "scan(cell, init=learned)")
    module = _module(source)
    entry = program_entry(module, "prog")
    assert [(item.name, item.role) for item in entry.parameters][-1] == (
        "cell_scan_init",
        "weight",
    )
    weight = torch.zeros(6, 5)
    bias = torch.tensor([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    weight[:3, 2:] = torch.eye(3)
    initial = torch.tensor([0.5, -0.5, 1.0])
    states = torch.tensor([[0.4, -0.6, 1.1], [0.5, -0.4, 0.9]])
    run = run_program(
        module,
        "prog",
        data={
            "input": ((0.0, 0.0), (0.0, 0.0)),
            "cell_param_weight": _nested(weight),
            "cell_param_bias": _nested(bias),
            "cell_scan_init": _nested(initial),
        },
        sites={"h@0": _nested(states[0]), "h@1": _nested(states[1])},
    )
    closed = (
        td.Normal(initial, math.exp(-1.0)).log_prob(states[0]).sum()
        + td.Normal(states[0], math.exp(-1.0)).log_prob(states[1]).sum()
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
