"""A scan whose cell is a program, on the compiled and the checked path.

The cell's steps are steps of the scanning program: they read the
scanning program's bindings, their sites are reached once per position
under ``"<label>@<n>"``, and an observation inside the cell is one
observation per position. Each test pins the compiled runtime against a
hand-written log density or against the reference machine running the
checked elaboration of the same source.
"""

from __future__ import annotations

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
    assert program.rsample(tokens).shape == (4,)
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


def test_a_program_cannot_draw_a_scan_of_a_program_whole() -> None:
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
    module = _module(USER_SITE_LIKE_A_SCAN)
    step = next(entry for entry in module.entries if entry.name.startswith("cell__"))
    assert [site.name for site in step.sites] == ["state"]
    inputs = torch.randn(3, 2)
    states = torch.randn(3, 3)
    user_h = torch.full((3,), 0.5)
    weight = torch.randn(6, 5)
    bias = torch.randn(6)
    data = {
        "input": _host(inputs),
        "cell_param_weight": _host(weight),
        "cell_param_bias": _host(bias),
    }
    previous = torch.zeros(3)
    closed = td.Normal(states[2], 1.0).log_prob(user_h).sum()
    for t in range(3):
        closed = closed + _normal(
            weight, bias, torch.cat([inputs[t], previous])
        ).log_prob(states[t]).sum()
        previous = states[t]
    reference = run_program(
        module,
        "prog",
        data=data,
        sites={**{f"state@{t}": _host(states[t]) for t in range(3)}, "h": _host(user_h)},
    )
    assert reference.log_joint == pytest.approx(float(closed), rel=1e-5)
    # One value under a scan's label is never replayed at every position.
    with pytest.raises(ExecutionFailure, match="reached more than once"):
        run_program(
            module,
            "prog",
            data=data,
            sites={"state": _host(states[2]), "h": _host(user_h)},
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
