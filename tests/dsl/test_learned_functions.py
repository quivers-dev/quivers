"""Learned deterministic morphisms, declared ``[role=function]``.

A learned function is a network from its domain's coordinates to its
codomain's, with no noise: it contributes no site, a program calls it in a
``let``, a scan runs it as a deterministic cell, and its weights are fit as
point estimates. Each claim is checked against a hand-written torch
computation from the same weights. The compiled runtime runs in float32 and
is compared with the float64 oracle at ``rel=1e-5`` (float32 rounding over a
few dozen terms is about 1e-6 relative); the reference machine runs in
float64 and is compared at ``rel=1e-9``. Each oracle comparison is paired
with a deliberately wrong oracle that the same tolerance rejects.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
import torch.distributions as D

from quivers.cli import check_file
from quivers.continuous import (
    Euclidean,
    LearnedFunction,
    LinearSource,
    MLPSource,
    ProductSpace,
    ScanMorphism,
    function_source,
)
from quivers.dsl import CompileError, loads
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import trace
from quivers.program import Program
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

type Host = float | tuple["Host", ...]
"""A value as the reference machine takes it."""


def _host(value: torch.Tensor) -> Host:
    """A tensor as nested tuples of floats, or a float for a scalar.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    Host
        The host value.
    """
    if value.dim() == 0:
        return float(value)
    return tuple(_host(row) for row in value)


def _space(name: str, dim: int) -> Euclidean:
    """A Euclidean space of the given width."""
    return Euclidean(name=name, dim=dim)


# ---------------------------------------------------------------------------
# LearnedFunction
# ---------------------------------------------------------------------------


def test_a_learned_function_applies_its_network_and_output() -> None:
    torch.manual_seed(0)
    domain = ProductSpace(components=(_space("a", 2), _space("h", 3)))
    function = LearnedFunction(domain, _space("h", 3), LinearSource(5, 3), "tanh")
    x = torch.randn(4, 5)
    linear = function.source.linear
    expected = torch.tanh(x @ linear.weight.T + linear.bias)
    assert torch.allclose(function.rsample(x), expected)
    assert torch.allclose(function.point_mass_value(x), expected)
    assert function.base_dimension(x) == 0
    assert torch.equal(function.log_prob(x, expected), torch.zeros(4))
    # One value per domain factor, as a let call passes them.
    a, h = torch.randn(2), torch.randn(3)
    joined = torch.cat([a, h])
    assert torch.allclose(
        function.apply_values(a, h),
        torch.tanh(linear.weight @ joined + linear.bias),
    )
    # The same call without the output nonlinearity differs.
    assert not torch.allclose(
        function.apply_values(a, h), linear.weight @ joined + linear.bias
    )


def test_a_learned_function_refuses_arguments_that_miss_its_domain() -> None:
    domain = ProductSpace(components=(_space("a", 2), _space("h", 3)))
    function = LearnedFunction(domain, _space("h", 3), LinearSource(5, 3))
    with pytest.raises(ValueError, match="2 factors, but the call passes 1"):
        function.apply_values(torch.randn(5))
    with pytest.raises(ValueError, match="argument 1 has shape \\(2,\\)"):
        function.apply_values(torch.randn(2), torch.randn(2))
    with pytest.raises(ValueError, match="writes 4 coordinates"):
        LearnedFunction(domain, _space("h", 3), LinearSource(5, 4))
    with pytest.raises(ValueError, match="unknown output"):
        LearnedFunction(domain, _space("h", 3), LinearSource(5, 3), "cube")  # type: ignore[arg-type]


def test_function_source_builds_the_named_network() -> None:
    domain = _space("x", 3)
    mlp = function_source(domain, 2, "mlp(5, 4)", activation="relu")
    assert isinstance(mlp, MLPSource)
    assert [type(layer).__name__ for layer in mlp.net] == [
        "Linear",
        "ReLU",
        "Linear",
        "ReLU",
        "Linear",
    ]
    assert isinstance(function_source(domain, 2, None), LinearSource)
    with pytest.raises(ValueError, match="not 'attention'"):
        function_source(domain, 2, "attention")
    with pytest.raises(ValueError, match="has none"):
        function_source(domain, 2, "linear", activation="relu")
    with pytest.raises(ValueError, match="cannot write 2"):
        function_source(domain, 2, "identity")


def test_a_scan_over_a_learned_function_is_a_deterministic_recurrence() -> None:
    torch.manual_seed(1)
    domain = ProductSpace(components=(_space("a", 2), _space("h", 3)))
    cell = LearnedFunction(domain, _space("h", 3), LinearSource(5, 3), "tanh")
    scan = ScanMorphism(cell)
    x = torch.randn(2, 4, 2)
    linear = cell.source.linear
    h = torch.zeros(2, 3)
    states = []
    for t in range(4):
        h = torch.tanh(torch.cat([x[:, t], h], dim=-1) @ linear.weight.T + linear.bias)
        states.append(h)
    assert torch.allclose(scan.rsample(x), h, atol=1e-6)
    trajectory = scan.rsample_trajectory(x)
    assert torch.allclose(trajectory, torch.stack(states, dim=1), atol=1e-6)
    assert torch.equal(scan.log_prob(x, trajectory), torch.zeros(2))


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

_DECLARED = """\
object Feat : Real 3
object Out : Real 2
morphism f : Feat -> Out [{options}]
"""


@pytest.mark.parametrize(
    ("options", "source", "output"),
    [
        ("role=function", LinearSource, "identity"),
        ("role=function, param_source=mlp(4), activation=relu", MLPSource, "identity"),
        ("role=function, output=softplus", LinearSource, "softplus"),
    ],
)
def test_a_function_declaration_builds_its_network(
    options: str, source: type, output: str
) -> None:
    compiler = Compiler(parse(_DECLARED.format(options=options)))
    compiler.compile()
    function = compiler.morphisms["f"]
    assert isinstance(function, LearnedFunction)
    assert isinstance(function.source, source)
    assert function.output == output


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _DECLARED.format(options="role=function").replace(
                "[role=function]", "[role=function] ~ Normal"
            ),
            "takes no ``~`` initializer",
        ),
        (
            _DECLARED.format(options="role=function, param_source=attention"),
            "not 'attention'",
        ),
        (
            _DECLARED.format(options="role=function, activation=relu"),
            "the 'linear' source has none",
        ),
        (
            _DECLARED.format(options="role=function, param_source=identity"),
            "cannot write 2",
        ),
        (
            _DECLARED.format(options="role=function, output=cube"),
            "unknown output 'cube'",
        ),
        (
            _DECLARED.format(options="role=function, hidden_dim=4"),
            "option 'hidden_dim' is not read by role=function",
        ),
        (
            "object K : FinSet 3\nobject Out : Real 2\n"
            "morphism f : K -> Out [role=function]\n",
            "is not a continuous space",
        ),
        (
            "object A : Real 2\nobject B : Real 2\nobject Out : Real 4\n"
            "morphism f : A * B -> Out [role=function, param_source=identity]\n",
            "an identity function passes one input through",
        ),
    ],
)
def test_a_malformed_function_declaration_is_refused(source: str, message: str) -> None:
    with pytest.raises(CompileError, match=message):
        loads(source)


# ---------------------------------------------------------------------------
# A let call, on both paths
# ---------------------------------------------------------------------------

LET_CALL = """\
object Feat : Real 3
object Out : Real 2
morphism proj : Feat * Feat -> Out [role=function, param_source=mlp(4), activation=relu, output=tanh]
program p : Feat -> Out
    sample a <- Normal(loc=0.0, scale=1.0) [over=Feat]
    sample b <- Normal(loc=0.0, scale=1.0) [over=Feat]
    let z = 2.0 * proj(a, b) + 1.0
    observe y <- Normal(loc=z, scale=0.5)
    return z
export p
"""


def _weights(program: Program, name: str) -> dict[str, torch.Tensor]:
    """A learned function's weights by their checked input names.

    Parameters
    ----------
    program : Program
        The compiled program.
    name : str
        The function's name.

    Returns
    -------
    dict[str, torch.Tensor]
        ``<name>_param_layer<k>_weight`` and ``_bias`` per hidden layer,
        then ``<name>_param_weight`` and ``<name>_param_bias``.
    """
    morphism = program.morphism
    assert isinstance(morphism, torch.nn.Module)
    found = {
        module
        for path, module in morphism.named_modules()
        if path.split(".")[-1] == f"_function_{name}"
    }
    assert len(found) == 1, f"{name!r} is held {len(found)} times"
    (function,) = found
    assert isinstance(function, LearnedFunction)
    source = function.source
    if isinstance(source, LinearSource):
        layers = [source.linear]
    else:
        assert isinstance(source, MLPSource)
        layers = [m for m in source.net if isinstance(m, torch.nn.Linear)]
    weights: dict[str, torch.Tensor] = {}
    for depth, layer in enumerate(layers[:-1]):
        weights[f"{name}_param_layer{depth}_weight"] = layer.weight.detach()
        weights[f"{name}_param_layer{depth}_bias"] = layer.bias.detach()
    weights[f"{name}_param_weight"] = layers[-1].weight.detach()
    weights[f"{name}_param_bias"] = layers[-1].bias.detach()
    return weights


def _let_call_oracle(
    weights: Mapping[str, torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    hidden: str = "relu",
) -> float:
    """The let-call program's log joint, by hand, in float64."""
    w = {k: v.double() for k, v in weights.items()}
    act = torch.relu if hidden == "relu" else torch.tanh
    h = act(
        w["proj_param_layer0_weight"] @ torch.cat([a, b]).double()
        + w["proj_param_layer0_bias"]
    )
    z = 2.0 * torch.tanh(w["proj_param_weight"] @ h + w["proj_param_bias"]) + 1.0
    joint = (
        D.Normal(0.0, 1.0).log_prob(a.double()).sum()
        + D.Normal(0.0, 1.0).log_prob(b.double()).sum()
        + D.Normal(z, 0.5).log_prob(y.double()).sum()
    )
    return float(joint)


def test_a_let_call_scores_alike_on_both_paths_and_by_hand() -> None:
    torch.manual_seed(2)
    program = loads(LET_CALL)
    a, b, y = torch.randn(3), torch.randn(3), torch.randn(2)
    weights = _weights(program, "proj")
    recorded = trace(
        program.morphism,
        torch.zeros(1, 3),
        observations={"a": a[None], "b": b[None], "y": y[None]},
    )
    oracle = _let_call_oracle(weights, a, b, y)
    assert float(recorded.log_joint[0]) == pytest.approx(oracle, rel=1e-5)
    run = program.run(
        "p",
        data={
            "feat": (0.0, 0.0, 0.0),
            "y": _host(y),
            **{k: _host(v) for k, v in weights.items()},
        },
        sites={"a": _host(a), "b": _host(b)},
        seed=0,
    )
    assert run.log_joint == pytest.approx(oracle, rel=1e-9)
    # A network with the wrong hidden activation is told apart.
    wrong = _let_call_oracle(weights, a, b, y, hidden="tanh")
    assert abs(wrong - oracle) > 1e-5 * abs(oracle)


def test_the_checked_program_reads_a_functions_weights_as_inputs() -> None:
    program = loads(LET_CALL)
    roles = {
        parameter.name: parameter.role for parameter in program.entry("p").parameters
    }
    assert roles == {
        "feat": "domain",
        "proj_param_layer0_weight": "weight",
        "proj_param_layer0_bias": "bias",
        "proj_param_weight": "weight",
        "proj_param_bias": "bias",
        "y": "observation",
    }


def test_gradients_reach_a_functions_weights() -> None:
    torch.manual_seed(3)
    program = loads(LET_CALL)
    recorded = trace(
        program.morphism,
        torch.zeros(1, 3),
        observations={
            "a": torch.randn(1, 3),
            "b": torch.randn(1, 3),
            "y": torch.randn(1, 2),
        },
    )
    recorded.log_joint.sum().backward()
    parameters = dict(program.morphism.named_parameters())  # type: ignore[union-attr]
    assert set(parameters) == {
        "_function_proj.source.net.0.weight",
        "_function_proj.source.net.0.bias",
        "_function_proj.source.net.2.weight",
        "_function_proj.source.net.2.bias",
    }
    assert all(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in parameters.values()
    )


@pytest.mark.parametrize(
    ("let", "message"),
    [
        ("let z = proj(a)", "takes 2 argument\\(s\\), one per factor of its domain"),
        (
            "let z = proj(a, c)",
            "argument 1 of the learned function 'proj' has type Real",
        ),
        (
            "let z = factor i : K in proj(a, b)",
            "called inside a factor or lambda body",
        ),
    ],
)
def test_a_call_that_misses_the_domain_is_refused(let: str, message: str) -> None:
    source = LET_CALL.replace(
        "object Out : Real 2", "object Out : Real 2\nobject K : FinSet 2"
    ).replace(
        "    let z = 2.0 * proj(a, b) + 1.0",
        "    sample c <- Normal(loc=0.0, scale=1.0)\n    " + let,
    )
    with pytest.raises(QiecDiagnosticError, match=message):
        loads(source)


def test_drawing_from_a_function_is_refused() -> None:
    source = LET_CALL.replace(
        "let z = 2.0 * proj(a, b) + 1.0", "sample z <- proj(a, b)"
    )
    with pytest.raises(CompileError, match="'proj' is deterministic, so no step draws"):
        loads(source)


# ---------------------------------------------------------------------------
# A deterministic recurrent cell inside a program cell, scanned
# ---------------------------------------------------------------------------

RECURRENT = """\
object Driver : Real 2
object Hidden : Real 3
object Obs : Real 1
morphism cell : Driver * Hidden -> Hidden [role=function, param_source=mlp(4), output=tanh]
morphism readout : Hidden -> Obs [role=function]
program step(u_t, h_prev) : Driver * Hidden -> Hidden
    let h = cell(u_t, h_prev)
    let m = readout(h)
    observe o <- Normal(loc=m, scale=0.5)
    return h
define run_all = scan(step)
program rnn : Driver -> Hidden
    sample last <- run_all
    return last
export rnn
"""


def _recurrent_oracle(
    weights: Mapping[str, torch.Tensor],
    drivers: torch.Tensor,
    observations: torch.Tensor,
    readout_bias_shift: float = 0.0,
) -> float:
    """The recurrent program's log joint for one run, by hand, in float64."""
    w = {k: v.double() for k, v in weights.items()}
    h = torch.zeros(3, dtype=torch.float64)
    total = torch.zeros((), dtype=torch.float64)
    for t in range(drivers.shape[0]):
        joined = torch.cat([drivers[t].double(), h])
        hidden = torch.tanh(
            w["cell_param_layer0_weight"] @ joined + w["cell_param_layer0_bias"]
        )
        h = torch.tanh(w["cell_param_weight"] @ hidden + w["cell_param_bias"])
        m = w["readout_param_weight"] @ h + w["readout_param_bias"] + readout_bias_shift
        total = total + D.Normal(m, 0.5).log_prob(observations[t].double()).sum()
    return float(total)


def test_a_scanned_function_cell_scores_alike_on_both_paths_and_by_hand() -> None:
    torch.manual_seed(4)
    program = loads(RECURRENT)
    drivers = torch.randn(2, 5, 2)
    observations = torch.randn(2, 5)
    compiled = program.morphism.log_joint(drivers, {"o": observations[..., None]})  # type: ignore[union-attr]
    weights = {**_weights(program, "cell"), **_weights(program, "readout")}
    for row in range(2):
        oracle = _recurrent_oracle(weights, drivers[row], observations[row])
        assert float(compiled[row]) == pytest.approx(oracle, rel=1e-5)
        run = program.run(
            "rnn",
            data={
                "driver": _host(drivers[row]),
                "o": _host(observations[row]),
                **{k: _host(v) for k, v in weights.items()},
            },
            sites={},
            seed=0,
        )
        assert run.log_joint == pytest.approx(oracle, rel=1e-9)
        wrong = _recurrent_oracle(weights, drivers[row], observations[row], 0.01)
        assert abs(wrong - oracle) > 1e-5 * abs(oracle)


def test_a_scanned_function_cell_has_no_latent_sites() -> None:
    torch.manual_seed(6)
    program = loads(RECURRENT)
    recorded = trace(
        program.morphism, torch.randn(1, 4, 2), observations={"o": torch.randn(1, 4, 1)}
    )
    scored = sorted(name for name, site in recorded.sites.items() if site.is_observed)
    assert scored == [f"o@{t}" for t in range(4)]
    latent = [
        name
        for name, site in recorded.sites.items()
        if not site.is_observed and not site.is_deterministic
    ]
    assert latent == []


def test_the_issue_reproducer_chains_an_embedding_into_a_function_scan(
    tmp_path: Path,
) -> None:
    source = """\
object Token : FinSet 5
object Embedded : Real 3
object Hidden : Real 4
morphism tok_embed : Token -> Embedded [role=embed]
morphism cell : Embedded * Hidden -> Hidden [role=function, param_source=mlp(8), output=tanh]
define backbone = tok_embed >> scan(cell)
program lm : Token -> Hidden
    sample h <- backbone
    return h
export lm
"""
    path = tmp_path / "lm.qvr"
    path.write_text(source)
    assert [d for d in check_file(path) if d.severity == "error"] == []
    program = loads(source)
    torch.manual_seed(5)
    recorded = trace(program.morphism, torch.randint(0, 5, (1, 6)))
    # The embedding draws one site per position; the cell draws none.
    drawn = [name for name in recorded.sites if name.startswith("h$x_in@")]
    assert len(drawn) == 6
    names = {name for name, _ in program.morphism.named_parameters()}  # type: ignore[union-attr]
    assert any("cell" in name or "source.net" in name for name in names)


# ---------------------------------------------------------------------------
# Transpilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("network", ["", ", param_source=mlp(4)"])
def test_every_target_refuses_a_weighted_function_by_its_network(network: str) -> None:
    source = LET_CALL.replace(
        "[role=function, param_source=mlp(4), activation=relu, output=tanh]",
        f"[role=function{network}]",
    )
    kind = "param-source:mlp" if network else "param-source:linear"
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(source), target=target)
        assert kind in caught.value.kinds, (target, caught.value.kinds)


FIT = """\
object Feat : Real 3
object Out : Real 2
morphism f : Feat -> Out [role=function]
program regression : Feat -> Out
    let m = f(u)
    observe y <- Normal(loc=m, scale=1.0)
    return m
export regression
"""


def test_maximum_likelihood_fits_a_linear_function_to_least_squares() -> None:
    """Under unit Gaussian noise the likelihood's maximizer is the OLS fit."""
    torch.manual_seed(7)
    program = loads(FIT)
    model = program.morphism
    assert isinstance(model, torch.nn.Module)
    runs = 40
    u = torch.randn(runs, 3)
    y = u @ torch.tensor([[1.0, -2.0], [0.5, 0.0], [0.0, 3.0]]) + torch.tensor(
        [0.3, -0.7]
    )
    y = y + 0.1 * torch.randn(runs, 2)
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = -model.log_joint(torch.zeros(runs, 3), {"u": u, "y": y}).sum()
        loss.backward()
        return loss

    for _ in range(5):
        optimizer.step(closure)
    design = torch.cat([u, torch.ones(runs, 1)], dim=1).double()
    solution = torch.linalg.lstsq(design, y.double()).solution
    weights = _weights(program, "f")
    fitted = torch.cat(
        [weights["f_param_weight"].T, weights["f_param_bias"][None]], dim=0
    ).double()
    # LBFGS on a convex quadratic in float32 reaches the optimum to about
    # 1e-5; 1e-3 leaves room for the float32 objective's rounding.
    assert torch.allclose(fitted, solution, atol=1e-3)
    # A fit missing the bias column lands elsewhere.
    without_bias = torch.linalg.lstsq(u.double(), y.double()).solution
    assert not torch.allclose(fitted[:3], without_bias, atol=1e-3)
