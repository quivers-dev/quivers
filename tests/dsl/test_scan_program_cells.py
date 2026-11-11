"""Scans on the compiled and the checked path.

Every scan's step runs as steps of the scanning program, once per
position. A morphism cell is drawn at the site the scan's binding
names, ``"<binding>@<n>"``; a program cell's steps read the scanning
program's bindings, their sites are reached once per position under
``"<label>@<n>"``, and an observation inside the cell is one
observation per position. Each test pins the compiled runtime against a
hand-written log density or against the reference machine running the
checked elaboration of the same source.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as td

from quivers.continuous import Draw, MonadicProgram, Scan, ScanMorphism
from quivers.continuous.families import ConditionalNormal
from quivers.dsl import CompileError, loads, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference import (
    ELBO,
    MCMC,
    SVI,
    AutoNormalGuide,
    NUTSKernel,
    bayesian_lift_parameters,
)
from quivers.inference.trace import trace
from quivers.qiec import ExecutionFailure, QiecModule
from quivers.qiec.program_runtime import program_entry, run_program
from quivers.qiec.types import render_static


def _module(source: str) -> QiecModule:
    """Lower a source to its checked module.

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


def _program(source: str) -> MonadicProgram:
    """Compile a source's exported program.

    Parameters
    ----------
    source : str
        QVR text.

    Returns
    -------
    MonadicProgram
        The compiled program.
    """
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    return program


def _host(value: torch.Tensor) -> object:
    """A tensor as the reference machine takes it.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    object
        A number for a scalar, else nested tuples of numbers.
    """
    if value.dim() == 0:
        return value.item()
    return tuple(_host(row) for row in value)


def _linear(program: MonadicProgram, path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The weight and bias of a compiled normal kernel.

    Parameters
    ----------
    program : MonadicProgram
        The compiled program.
    path : str
        The kernel's submodule path.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Its affine map's weight and bias, detached.
    """
    kernel = program.get_submodule(path)
    assert isinstance(kernel, ConditionalNormal)
    linear = kernel.get_submodule("param_source.linear")
    assert isinstance(linear, torch.nn.Linear)
    return linear.weight.detach(), linear.bias.detach()


def _normal(
    weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor
) -> td.Normal:
    """The distribution a normal kernel puts at an input.

    Parameters
    ----------
    weight : torch.Tensor
        The affine map's weight.
    bias : torch.Tensor
        Its bias.
    inputs : torch.Tensor
        The kernel's input, ``(batch, fan_in)``.

    Returns
    -------
    td.Normal
        The kernel's normal: the first half of the map's output is the
        location, the second the log scale.
    """
    raw = inputs @ weight.T + bias
    half = raw.shape[-1] // 2
    return td.Normal(raw[..., :half], raw[..., half:].exp())


# ---------------------------------------------------------------------
# Per-position sites and observations in a state-space model.
# ---------------------------------------------------------------------

STATE_SPACE = """\
object Driver : Real 2
object State : Real 3
object Obs : Real 2
morphism transition : Driver * State -> State ~ Normal
morphism emission : State -> Obs ~ Normal
program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    observe o <- emission(s)
    return s
define run_all = scan(step)
program ssm : Driver -> State
    sample h <- run_all
    return h
export ssm
"""

_TRANSITION = "_scan_h._call_h._step_s"
_EMISSION = "_scan_h._call_h._step_o"


def _state_space_closed_form(
    program: MonadicProgram,
    drivers: torch.Tensor,
    states: torch.Tensor,
    observations: torch.Tensor,
) -> torch.Tensor:
    """The state-space model's joint density, written out by hand.

    Parameters
    ----------
    program : MonadicProgram
        The compiled program, whose kernels' weights the density reads.
    drivers : torch.Tensor
        ``(batch, T, 2)``.
    states : torch.Tensor
        ``(batch, T, 3)``.
    observations : torch.Tensor
        ``(batch, T, 2)``.

    Returns
    -------
    torch.Tensor
        ``sum_t log N(s_t | transition(u_t, s_{t-1})) + log N(o_t |
        emission(s_t))`` per row, ``s_{-1} = 0``.
    """
    transition = _linear(program, _TRANSITION)
    emission = _linear(program, _EMISSION)
    previous = torch.zeros(drivers.shape[0], 3)
    total = torch.zeros(drivers.shape[0])
    for position in range(drivers.shape[1]):
        cell_input = torch.cat([drivers[:, position], previous], dim=-1)
        total = total + _normal(*transition, cell_input).log_prob(
            states[:, position]
        ).sum(-1)
        total = total + _normal(*emission, states[:, position]).log_prob(
            observations[:, position]
        ).sum(-1)
        previous = states[:, position]
    return total


def test_a_cells_sites_and_observations_are_scored_at_every_position() -> None:
    torch.manual_seed(0)
    program = _program(STATE_SPACE)
    drivers = torch.randn(2, 5, 2)
    states = torch.randn(2, 5, 3)
    observations = torch.randn(2, 5, 2)
    given = {
        **{f"s@{t}": states[:, t] for t in range(5)},
        "o": observations,
    }
    joint = program.log_joint(drivers, given)
    closed = _state_space_closed_form(program, drivers, states, observations)
    assert torch.allclose(joint, closed, rtol=1e-6, atol=1e-5)
    moved = program.log_joint(drivers, {**given, "o": observations + 1.0})
    assert not torch.allclose(moved, joint)


def test_a_cells_sites_are_traced_per_position() -> None:
    torch.manual_seed(1)
    program = _program(STATE_SPACE)
    drivers = torch.randn(3, 4, 2)
    observations = torch.randn(3, 4, 2)
    run = trace(program, drivers, {"o": observations})
    assert sorted(run.sites) == sorted(
        [f"s@{t}" for t in range(4)] + [f"o@{t}" for t in range(4)]
    )
    for t in range(4):
        assert run.sites[f"o@{t}"].is_observed
        assert torch.equal(run.sites[f"o@{t}"].value, observations[:, t])
        assert not run.sites[f"s@{t}"].is_observed
    states = torch.stack([run.sites[f"s@{t}"].value for t in range(4)], dim=1)
    closed = _state_space_closed_form(program, drivers, states, observations)
    total = sum(site.log_prob for site in run.sites.values())
    assert isinstance(total, torch.Tensor)
    assert torch.allclose(total, closed, rtol=1e-6, atol=1e-5)
    # The returned state is the last position's site.
    assert torch.equal(run.output, run.sites["s@3"].value)  # type: ignore[arg-type]


def test_per_position_and_whole_sequence_observations_agree() -> None:
    torch.manual_seed(2)
    program = _program(STATE_SPACE)
    drivers = torch.randn(2, 3, 2)
    states = torch.randn(2, 3, 3)
    observations = torch.randn(2, 3, 2)
    latents = {f"s@{t}": states[:, t] for t in range(3)}
    whole = program.log_joint(drivers, {**latents, "o": observations})
    split = program.log_joint(
        drivers, {**latents, **{f"o@{t}": observations[:, t] for t in range(3)}}
    )
    assert torch.equal(whole, split)


def test_a_cell_site_given_twice_or_past_the_sequence_is_refused() -> None:
    program = _program(STATE_SPACE)
    drivers = torch.randn(2, 3, 2)
    states = {f"s@{t}": torch.randn(2, 3) for t in range(3)}
    observations = torch.randn(2, 3, 2)
    with pytest.raises(ValueError, match="given both"):
        program.log_joint(
            drivers, {**states, "o": observations, "o@1": observations[:, 1]}
        )
    with pytest.raises(ValueError, match="never reached 'o@3'"):
        program.log_joint(drivers, {**states, "o": torch.randn(2, 4, 2)})
    with pytest.raises(ValueError, match="laid out"):
        program.log_joint(drivers, {**states, "o": torch.randn(2)})
    with pytest.raises(ValueError, match="cannot be supplied"):
        program.log_joint(drivers, {**states, "o": observations, "h": states["s@2"]})
    with pytest.raises(KeyError, match="s@2"):
        program.log_joint(
            drivers, {"s@0": states["s@0"], "s@1": states["s@1"], "o": observations}
        )


def test_the_checked_elaboration_observes_a_sequence() -> None:
    module = _module(STATE_SPACE)
    entry = program_entry(module, "ssm")
    roles = {
        item.name: (item.role, render_static(item.type)) for item in entry.parameters
    }
    assert roles["o"] == ("observation", "Tensor[Real]([driver_extent, 2])")


def test_the_reference_machine_and_the_compiled_runtime_agree() -> None:
    torch.manual_seed(3)
    program = _program(STATE_SPACE)
    module = _module(STATE_SPACE)
    drivers = torch.randn(2, 4, 2)
    states = torch.randn(2, 4, 3)
    observations = torch.randn(2, 4, 2)
    compiled = program.log_joint(
        drivers,
        {**{f"s@{t}": states[:, t] for t in range(4)}, "o": observations},
    )
    transition = _linear(program, _TRANSITION)
    emission = _linear(program, _EMISSION)
    for row in range(2):
        run = run_program(
            module,
            "ssm",
            data={
                "driver": _host(drivers[row]),
                "transition_param_weight": _host(transition[0]),
                "transition_param_bias": _host(transition[1]),
                "emission_param_weight": _host(emission[0]),
                "emission_param_bias": _host(emission[1]),
                "o": _host(observations[row]),
            },
            sites={f"s@{t}": _host(states[row, t]) for t in range(4)},
        )
        assert run.log_joint == pytest.approx(float(compiled[row]), rel=1e-5)


def test_gradients_reach_the_cells_kernels() -> None:
    program = _program(STATE_SPACE)
    drivers = torch.randn(2, 3, 2)
    given = {
        **{f"s@{t}": torch.randn(2, 3) for t in range(3)},
        "o": torch.randn(2, 3, 2),
    }
    program.log_joint(drivers, given).sum().backward()
    for path in (_TRANSITION, _EMISSION):
        weight = program.get_submodule(path).get_submodule("param_source.linear").weight
        assert weight.grad is not None and bool(weight.grad.abs().sum() > 0)


def test_inference_runs_over_the_latent_trajectory() -> None:
    torch.manual_seed(7)
    program = _program(STATE_SPACE)
    drivers = torch.randn(1, 4, 2)
    observations = {"o": torch.randn(1, 4, 2)}
    lifted, lx, lobs = bayesian_lift_parameters(
        program,
        drivers,
        observations,
        prior_scale=1.0,
        additional_latents={"s": (1, 4, 3)},
    )
    result = MCMC(
        NUTSKernel(step_size=0.01, max_tree_depth=3),
        num_warmup=5,
        num_samples=5,
        num_chains=1,
    ).run(lifted, lx, lobs)
    trajectory = result.samples["latent__s"]
    assert trajectory.shape[-1] == 12
    assert bool(torch.isfinite(trajectory).all())
    guide = AutoNormalGuide(lifted, observed_names=set(lobs))
    svi = SVI(
        lifted,
        guide,
        torch.optim.Adam(guide.parameters(), lr=1e-2),
        ELBO(num_particles=1),
    )
    losses = [svi.step(lx, lobs) for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


# ---------------------------------------------------------------------
# A cell reading bindings of the scanning program.
# ---------------------------------------------------------------------

OUTER_BINDINGS = """\
object Token : FinSet 5
object Resp : FinSet 4
object Embedded : Real 4
object Hidden : Real 4

morphism tok_embed : Token -> Embedded [role=embed]

program rnn_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    let h_new = tanh(x_t * a + h_prev * b)
    return h_new

define backbone = tok_embed >> scan(rnn_cell)

program lm : Token -> Token
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample h <- backbone
    observe next_token : Resp <- Categorical(logits=h)
    return next_token

export lm
"""


def test_a_cell_reads_the_scanning_programs_bindings() -> None:
    torch.manual_seed(4)
    program = _program(OUTER_BINDINGS)
    tokens = torch.randint(0, 5, (4, 6))
    # Four runs, each drawing the four rows of its `Resp` plate.
    assert program.rsample(tokens).shape == (4, 4)
    run = trace(program, tokens)
    assert {"a", "b", "next_token"} <= set(run.sites)
    assert [f"h$x_in@{t}" for t in range(6)] == [
        name for name in run.sites if name.startswith("h$x_in@")
    ]
    a = run.sites["a"].value
    b = run.sites["b"].value
    state = torch.zeros(4, 4)
    for t in range(6):
        state = torch.tanh(run.sites[f"h$x_in@{t}"].value * a + state * b)
        assert torch.allclose(run.sites[f"h_new@{t}"].value, state)


def test_gradients_reach_the_bindings_a_cell_reads() -> None:
    torch.manual_seed(5)
    program = _program(OUTER_BINDINGS)
    tokens = torch.randint(0, 5, (2, 3))
    run = trace(program, tokens)
    a = run.sites["a"].value.detach().clone().requires_grad_(True)
    b = run.sites["b"].value.detach().clone().requires_grad_(True)
    given = {
        name: site.value.detach()
        for name, site in run.sites.items()
        if not site.is_deterministic and name not in ("a", "b")
    }
    program.log_joint(tokens, {**given, "a": a, "b": b}).sum().backward()
    assert a.grad is not None and bool(a.grad.abs().sum() > 0)
    assert b.grad is not None and bool(b.grad.abs().sum() > 0)


def test_the_checked_elaboration_passes_the_bindings_a_cell_reads() -> None:
    module = _module(OUTER_BINDINGS)
    step = program_entry(module, "rnn_cell__scan_step")
    assert [(item.name, item.role) for item in step.parameters][-2:] == [
        ("a", "data"),
        ("b", "data"),
    ]
    entry = program_entry(module, "lm")
    assert "a" not in {item.name for item in entry.parameters}


# ---------------------------------------------------------------------
# The neighbours: a sample depending on an outer binding, a score, a
# let reading host data, and name shadowing.
# ---------------------------------------------------------------------

NEIGHBOURS = """\
object Input : Real 2
object Hidden : Real 2
program cell(x_t, h_prev) : Input * Hidden -> Hidden
    let m = h_prev * rho + x_t * gain + sigma
    sample s <- Normal(loc=m, scale=0.5)
    let total = sum(s)
    score bonus = 0.25 * total
    return s
define rnn = scan(cell)
program prog : Input -> Hidden
    sample rho <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    let m = rho * 10.0
    sample h <- rnn
    sample tail <- Normal(loc=m, scale=1.0)
    return h
export prog
"""


def _neighbours_closed_form(
    inputs: torch.Tensor,
    rho: torch.Tensor,
    sigma: torch.Tensor,
    gain: torch.Tensor,
    states: torch.Tensor,
    tail: torch.Tensor,
) -> torch.Tensor:
    """The neighbours program's joint density, written out by hand.

    Parameters
    ----------
    inputs : torch.Tensor
        ``(batch, T, 2)``.
    rho : torch.Tensor
        ``(batch,)``.
    sigma : torch.Tensor
        ``(batch,)``.
    gain : torch.Tensor
        A scalar of host data.
    states : torch.Tensor
        ``(batch, T, 2)``.
    tail : torch.Tensor
        ``(batch,)``.

    Returns
    -------
    torch.Tensor
        The priors on ``rho`` and ``sigma``, each position's draw and
        its score, and ``tail`` around the program's own ``m``.
    """
    total = td.Normal(0.0, 1.0).log_prob(rho) + td.HalfNormal(1.0).log_prob(sigma)
    previous = torch.zeros_like(states[:, 0])
    for position in range(inputs.shape[1]):
        loc = previous * rho[:, None] + inputs[:, position] * gain + sigma[:, None]
        draw = states[:, position]
        total = total + td.Normal(loc, 0.5).log_prob(draw).sum(-1)
        total = total + 0.25 * draw.sum(-1)
        previous = draw
    return total + td.Normal(rho * 10.0, 1.0).log_prob(tail)


def test_a_cell_reads_outer_samples_host_data_and_shadows_names() -> None:
    torch.manual_seed(6)
    program = _program(NEIGHBOURS)
    # One row: a draw whose parameters are all literals is one value
    # shared by every row, so a single row keeps the hand-written
    # density per row.
    inputs = torch.randn(1, 3, 2)
    rho = torch.randn(1)
    sigma = torch.rand(1) + 0.5
    gain = torch.tensor(0.7)
    states = torch.randn(1, 3, 2)
    tail = torch.randn(1)
    given = {
        "rho": rho,
        "sigma": sigma,
        "gain": gain,
        "tail": tail,
        **{f"s@{t}": states[:, t] for t in range(3)},
    }
    joint = program.log_joint(inputs, given)
    closed = _neighbours_closed_form(inputs, rho, sigma, gain, states, tail)
    assert torch.allclose(joint, closed, rtol=1e-6, atol=1e-5)
    run = trace(program, inputs, given)
    for t in range(3):
        assert run.sites[f"bonus@{t}"].log_prob.sum() != 0
    module = _module(NEIGHBOURS)
    for row in range(1):
        reference = run_program(
            module,
            "prog",
            data={"input": _host(inputs[row]), "gain": gain.item()},
            sites={
                "rho": rho[row].item(),
                "sigma": sigma[row].item(),
                "tail": tail[row].item(),
                **{f"s@{t}": _host(states[row, t]) for t in range(3)},
            },
        )
        assert reference.log_joint == pytest.approx(float(joint[row]), rel=1e-5)


def test_a_cell_without_its_host_data_names_it() -> None:
    program = _program(NEIGHBOURS)
    with pytest.raises(CompileError, match="undefined variable 'gain'"):
        program.rsample(torch.randn(2, 3, 2))


# ---------------------------------------------------------------------
# What both paths refuse.
# ---------------------------------------------------------------------

_HEAD = """\
object Input : Real 2
object Hidden : Real 2
"""


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """\
program cell(x_t, h_prev) : Input * Hidden -> Hidden
    sample s <- Normal(loc=h_prev, scale=1.0)
    return s
define rnn = scan(cell)
program prog : Input -> Hidden
    sample s <- Normal(loc=0.0, scale=1.0)
    sample h <- rnn
    return h
export prog
""",
            "reaches site 's' at every position",
        ),
        (
            """\
program cell(x_t, h_prev) : Input * Hidden -> Hidden
    observe y <- Normal(loc=h_prev, scale=1.0)
    return h_prev
define rnn = scan(cell)
program prog : Input -> Hidden
    let y = 1.0
    sample h <- rnn
    return h
export prog
""",
            "observes 'y' at every position",
        ),
        (
            """\
program layer(h_prev) : Hidden -> Hidden
    sample s <- Normal(loc=h_prev, scale=1.0)
    return s
define deep = stack(layer, 2)
program prog : Hidden -> Hidden
    sample h <- deep
    return h
export prog
""",
            "composes the program 'layer'",
        ),
        (
            """\
program layer(h_prev) : Hidden -> Hidden
    sample s <- Normal(loc=h_prev, scale=1.0)
    return s
define deep = repeat(layer, 2)
program prog : Hidden -> Hidden
    sample h <- deep
    return h
export prog
""",
            "composes the program 'layer'",
        ),
        (
            """\
program layer(h_prev) : Hidden -> Hidden
    sample s <- Normal(loc=h_prev, scale=1.0)
    return s
define deep = layer >> layer
program prog : Hidden -> Hidden
    sample h <- deep
    return h
export prog
""",
            "composes the program 'layer'",
        ),
    ],
    ids=[
        "shared-site-label",
        "observation-bound-outside",
        "stack-of-a-program",
        "repeat-of-a-program",
        "composed-programs",
    ],
)
def test_both_paths_refuse_alike(body: str, message: str) -> None:
    source = _HEAD + body
    with pytest.raises(QiecDiagnosticError, match=message) as checked:
        _module(source)
    with pytest.raises(QiecDiagnosticError, match=message) as compiled:
        loads(source)
    assert str(checked.value).split("] ", 1)[1] == str(compiled.value).split("] ", 1)[1]


def test_a_program_cannot_draw_a_scan_whole() -> None:
    program = _program(STATE_SPACE)
    (scan,) = [step for step in program.steps if isinstance(step, Scan)]
    with pytest.raises(ValueError, match="bind it with a Scan step"):
        MonadicProgram(
            program.domain,
            program.codomain,
            [Draw(names=("h",), morphism=ScanMorphism(scan.step))],
            ("h",),
        )


# ---------------------------------------------------------------------
# A scan whose cell is a morphism.
# ---------------------------------------------------------------------

MORPHISM_CELL = """\
object Input : Real 2
object Hidden : Real 3
morphism cell : Input * Hidden -> Hidden ~ Normal
define rnn = scan(cell)
program prog : Input -> Hidden
    sample h <- rnn
    return h
export prog
"""


def _scan_kernels(program: MonadicProgram) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The weight and bias of the cell each of a program's scans draws.

    Parameters
    ----------
    program : MonadicProgram
        The compiled program.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        One pair per scan, in the program's order.
    """
    kernels: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in program.steps:
        if not isinstance(step, Scan):
            continue
        body = step.step
        assert isinstance(body, MonadicProgram)
        (draw,) = [item for item in body.steps if isinstance(item, Draw)]
        kernel = body.step_module(draw)
        linear = kernel.get_submodule("param_source.linear")
        assert isinstance(linear, torch.nn.Linear)
        kernels.append((linear.weight.detach(), linear.bias.detach()))
    return kernels


def _trajectory_density(
    kernel: tuple[torch.Tensor, torch.Tensor],
    inputs: torch.Tensor,
    states: torch.Tensor,
) -> torch.Tensor:
    """The density of one trajectory under a normal cell, written out.

    Parameters
    ----------
    kernel : tuple[torch.Tensor, torch.Tensor]
        The cell's weight and bias.
    inputs : torch.Tensor
        ``(T, dim A)``.
    states : torch.Tensor
        ``(T, dim H)``.

    Returns
    -------
    torch.Tensor
        ``sum_t log N(h_t | cell(x_t, h_{t-1}))``, ``h_{-1} = 0``.
    """
    previous = torch.zeros(states.shape[-1])
    total = torch.tensor(0.0)
    for position in range(inputs.shape[0]):
        distribution = _normal(*kernel, torch.cat([inputs[position], previous]))
        total = total + distribution.log_prob(states[position]).sum()
        previous = states[position]
    return total


def test_a_morphism_cell_scan_scores_its_trajectory_on_both_paths() -> None:
    torch.manual_seed(8)
    program = _program(MORPHISM_CELL)
    module = _module(MORPHISM_CELL)
    inputs = torch.randn(4, 2)
    states = torch.randn(4, 3)
    (kernel,) = _scan_kernels(program)
    closed = float(_trajectory_density(kernel, inputs, states))
    given = {f"h@{t}": states[t : t + 1] for t in range(4)}
    run = trace(program, inputs.unsqueeze(0), given)
    assert sorted(run.sites) == [f"h@{t}" for t in range(4)]
    compiled = program.log_joint(inputs.unsqueeze(0), given).detach()
    assert float(compiled.sum()) == pytest.approx(closed, rel=1e-6)
    reference = run_program(
        module,
        "prog",
        data={
            "input": _host(inputs),
            "cell_param_weight": _host(kernel[0]),
            "cell_param_bias": _host(kernel[1]),
        },
        sites={f"h@{t}": _host(states[t]) for t in range(4)},
    )
    assert reference.log_joint == pytest.approx(closed, rel=1e-5)
    moved = {**given, "h@1": given["h@1"] + 1.0}
    assert float(program.log_joint(inputs.unsqueeze(0), moved).sum()) != pytest.approx(
        closed
    )


FAN_OF_SCANS = """\
object Input : Real 2
object Hidden : Real 2
object Both : Real 4
morphism left : Input * Hidden -> Hidden ~ Normal
morphism right : Input * Hidden -> Hidden ~ Normal
define both = fan(scan(left), scan(right))
program prog : Input -> Both
    sample h <- both
    return h
export prog
"""


def test_scans_in_parallel_branches_are_addressed_apart_and_agree() -> None:
    torch.manual_seed(9)
    program = _program(FAN_OF_SCANS)
    module = _module(FAN_OF_SCANS)
    inputs = torch.randn(3, 2)
    run = trace(program, inputs.unsqueeze(0))
    labels = sorted({name.rpartition("@")[0] for name in run.sites if "@" in name})
    assert len(labels) == 2 and all(label.startswith("h$") for label in labels)
    first, second = _scan_kernels(program)
    trajectories = {
        label: torch.stack(
            [run.sites[f"{label}@{t}"].value[0].detach() for t in range(3)]
        )
        for label in labels
    }
    left_label, right_label = labels
    closed = float(
        _trajectory_density(first, inputs, trajectories[left_label])
        + _trajectory_density(second, inputs, trajectories[right_label])
    )
    given = {
        f"{label}@{t}": trajectories[label][t : t + 1]
        for label in labels
        for t in range(3)
    }
    compiled = float(program.log_joint(inputs.unsqueeze(0), given).detach().sum())
    assert compiled == pytest.approx(closed, rel=1e-6)
    assert torch.allclose(
        run.output[0],  # type: ignore[index]
        torch.cat([trajectories[left_label][2], trajectories[right_label][2]]),
    )
    reference = run_program(
        module,
        "prog",
        data={
            "input": _host(inputs),
            "left_param_weight": _host(first[0]),
            "left_param_bias": _host(first[1]),
            "right_param_weight": _host(second[0]),
            "right_param_bias": _host(second[1]),
        },
        sites={
            f"{label}@{t}": _host(trajectories[label][t])
            for label in labels
            for t in range(3)
        },
    )
    assert reference.log_joint == pytest.approx(closed, rel=1e-5)


# ---------------------------------------------------------------------
# Site addresses cannot collide.
# ---------------------------------------------------------------------

USER_SITE_LIKE_A_SCAN = """\
object Input : Real 2
object Hidden : Real 3
morphism cell : Input * Hidden -> Hidden ~ Normal
define rnn = scan(cell)
program prog : Input -> Hidden
    sample state <- rnn
    sample h <- Normal(loc=state, scale=1.0)
    return h
export prog
"""


def test_a_scans_sites_take_the_binding_the_source_wrote() -> None:
    torch.manual_seed(10)
    program = _program(USER_SITE_LIKE_A_SCAN)
    module = _module(USER_SITE_LIKE_A_SCAN)
    inputs = torch.randn(3, 2)
    states = torch.randn(3, 3)
    user_h = torch.full((3,), 0.5)
    run = trace(program, inputs.unsqueeze(0))
    assert sorted(run.sites) == ["h", "state@0", "state@1", "state@2"]
    (kernel,) = _scan_kernels(program)
    closed = float(
        _trajectory_density(kernel, inputs, states)
        + td.Normal(states[2], 1.0).log_prob(user_h).sum()
    )
    given = {
        **{f"state@{t}": states[t : t + 1] for t in range(3)},
        "h": user_h.unsqueeze(0),
    }
    compiled = float(program.log_joint(inputs.unsqueeze(0), given).detach().sum())
    assert compiled == pytest.approx(closed, rel=1e-6)
    data = {
        "input": _host(inputs),
        "cell_param_weight": _host(kernel[0]),
        "cell_param_bias": _host(kernel[1]),
    }
    reference = run_program(
        module,
        "prog",
        data=data,
        sites={
            **{f"state@{t}": _host(states[t]) for t in range(3)},
            "h": _host(user_h),
        },
    )
    assert reference.log_joint == pytest.approx(closed, rel=1e-5)
    # One value under a scan's label is never replayed at every position.
    with pytest.raises(ExecutionFailure, match="reached more than once"):
        run_program(
            module,
            "prog",
            data=data,
            sites={"state": _host(states[2]), "h": _host(user_h)},
        )
    with pytest.raises(ValueError, match="laid out"):
        program.log_joint(
            inputs.unsqueeze(0), {"state": states[2:], "h": user_h.unsqueeze(0)}
        )


def test_two_scans_reaching_one_label_are_refused_on_both_paths() -> None:
    source = """\
object Input : Real 2
object Hidden : Real 2
program cell(x_t, h_prev) : Input * Hidden -> Hidden
    sample s <- Normal(loc=h_prev, scale=1.0)
    return s
define rnn = scan(cell)
program prog : Input -> Hidden
    sample first <- rnn
    sample second <- rnn
    return second
export prog
"""
    with pytest.raises(QiecDiagnosticError, match="share addresses") as checked:
        _module(source)
    with pytest.raises(QiecDiagnosticError, match="share addresses") as compiled:
        loads(source)
    assert str(checked.value).split("] ", 1)[1] == str(compiled.value).split("] ", 1)[1]


# ---------------------------------------------------------------------
# NUTS over a scanned trajectory against the Kalman smoother.
# ---------------------------------------------------------------------

KNOWN_STATE_SPACE = """\
object Driver : Real 1
object State : Real 1
object Obs : Real 1
program step(u_t, s_prev) : Driver * State -> State
    let m = 0.8 * s_prev + 0.3 * u_t
    sample s <- Normal(loc=m, scale=0.5)
    observe o <- Normal(loc=s, scale=0.4)
    return s
define run_all = scan(step)
program ssm : Driver -> State
    sample h <- run_all
    return h
export ssm
"""


def _rts_smoother(
    drivers: list[float], observations: list[float]
) -> tuple[list[float], list[float]]:
    """The Rauch-Tung-Striebel smoother of the known state-space model.

    ``s_t = 0.8 s_{t-1} + 0.3 u_t + N(0, 0.5^2)`` from ``s_{-1} = 0``,
    and ``o_t = s_t + N(0, 0.4^2)``.

    Parameters
    ----------
    drivers : list[float]
        ``u_t``.
    observations : list[float]
        ``o_t``.

    Returns
    -------
    tuple[list[float], list[float]]
        The posterior mean and variance of every ``s_t``.
    """
    a, b, q, r = 0.8, 0.3, 0.5, 0.4
    filtered_mean: list[float] = []
    filtered_var: list[float] = []
    predicted_mean: list[float] = []
    predicted_var: list[float] = []
    mean, var = 0.0, 0.0
    for drive, observed in zip(drivers, observations, strict=True):
        prior_mean, prior_var = a * mean + b * drive, a * a * var + q * q
        gain = prior_var / (prior_var + r * r)
        mean = prior_mean + gain * (observed - prior_mean)
        var = (1.0 - gain) * prior_var
        predicted_mean.append(prior_mean)
        predicted_var.append(prior_var)
        filtered_mean.append(mean)
        filtered_var.append(var)
    smoothed_mean = list(filtered_mean)
    smoothed_var = list(filtered_var)
    for t in range(len(drivers) - 2, -1, -1):
        gain = filtered_var[t] * a / predicted_var[t + 1]
        smoothed_mean[t] = filtered_mean[t] + gain * (
            smoothed_mean[t + 1] - predicted_mean[t + 1]
        )
        smoothed_var[t] = filtered_var[t] + gain * gain * (
            smoothed_var[t + 1] - predicted_var[t + 1]
        )
    return smoothed_mean, smoothed_var


@pytest.mark.slow
def test_nuts_recovers_the_kalman_smoother_over_a_scanned_trajectory() -> None:
    torch.manual_seed(0)
    length = 20
    program = _program(KNOWN_STATE_SPACE)
    drivers = torch.randn(1, length, 1)
    state = 0.0
    observed: list[float] = []
    for t in range(length):
        state = (
            0.8 * state + 0.3 * float(drivers[0, t, 0]) + 0.5 * float(torch.randn(()))
        )
        observed.append(state + 0.4 * float(torch.randn(())))
    observations = torch.tensor(observed).reshape(1, length, 1)
    means, variances = _rts_smoother(drivers.reshape(-1).tolist(), observed)
    lifted, lx, lobs = bayesian_lift_parameters(
        program,
        drivers,
        {"o": observations},
        prior_scale=1.0,
        additional_latents={"s": (1, length, 1)},
    )
    result = MCMC(
        NUTSKernel(max_tree_depth=4),
        num_warmup=120,
        num_samples=200,
        num_chains=1,
        seed=0,
    ).run(lifted, lx, lobs)
    draws = result.samples["latent__s"].reshape(-1, length)
    effective = result.ess["latent__s"].reshape(-1)
    for t in range(length):
        # The posterior mean's Monte Carlo error is the smoother's
        # standard deviation over the square root of the effective
        # sample size; four of them bound a correct sampler's deviation
        # at every position with high probability.
        tolerance = 4.0 * math.sqrt(variances[t]) / math.sqrt(float(effective[t]))
        assert abs(float(draws[:, t].mean()) - means[t]) <= tolerance, (
            f"position {t}: posterior mean {float(draws[:, t].mean()):.4f}, "
            f"smoother {means[t]:.4f}, tolerance {tolerance:.4f}"
        )
