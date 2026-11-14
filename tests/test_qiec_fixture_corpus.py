"""The QIEC fixture corpus, end to end.

Each fixture under ``tests/fixtures/qiec/`` exercises one part of the
unified language, and this module holds every stable pass to it
separately: parse and emit, lowering and independent validation,
serialization, IR conversion, reference evaluation, target lowering with
its refusals, the external syntax check of every emitted program, and the
Python target runtimes that run in this process. Mutations show the
features are load-bearing: a dropped case, call, score, or observation
changes what the checker or the reference machine says.

The container probes for the other targets live in the probe tier
(`tests/transpile/test_qiec_fixture_equivalence.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import pathlib
import subprocess

import pytest
import torch

from quivers.dsl import loads, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.effects.checked_program import CheckedProgram
from quivers.qiec import (
    ExecutionFailure,
    QiecModule,
    dumps,
    loads as loads_module,
    parse_static_arguments,
    run_named,
    validate_module,
)
from quivers.qiec.entries import HostValue
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import RuntimeConfiguration, RuntimeSelection
from quivers.qiec.program_runtime import run_deduction, run_program, sample_program
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.qiec_ir import lower_qiec_ir
from tests._target_runtimes import pymc_log_density, pyro_log_density
from tests.transpile._tools import require_tool
from tests.transpile.test_gallery_compiles import _SOURCE_PATH, _SYNTAX_CHECKS

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "qiec"

#: The fixtures by letter.
_CORPUS: dict[str, str] = {
    "A": "recursive_indexed_traversal",
    "B": "authored_state_handler",
    "C": "effectful_probabilistic_helper",
    "D": "grouped_marginalization",
    "E": "ambiguous_deduction",
    "F": "network_decoder",
}

#: The targets each fixture transpiles to, and the kind each other target
#: refuses it under.
_ADMITTING: dict[str, frozenset[str]] = {
    "A": frozenset(
        {"pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church"}
    ),
    "B": frozenset(
        {"pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church"}
    ),
    "C": frozenset(
        {"pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church"}
    ),
    "D": frozenset(available_targets()) - {"bugs"},
    "E": frozenset(),
    "F": frozenset(),
}
_REFUSALS: dict[str, dict[str, str]] = {
    "A": {
        "stan": "qiec:capability:case:fold",
        "bugs": "qiec:capability:arithmetic:fold",
        "jags": "qiec:capability:arithmetic:fold",
    },
    "B": {
        "stan": "qiec:capability:handle:counted",
        "bugs": "qiec:capability:arithmetic:bump",
        "jags": "qiec:capability:arithmetic:bump",
    },
    "C": {
        "stan": "qiec:capability:case:account",
        "bugs": "call:graph:account",
        "jags": "call:graph:account",
    },
    "D": {"bugs": "marginalize:grouped-index-map"},
    "E": {
        target: "qiec:capability:search:Ambiguous__run"
        for target in available_targets()
    },
    "F": {target: "param-source:mlp" for target in available_targets()},
}

type Host = float | int | list[float] | list[int]
"""A number or a list of numbers, as a point carries it."""

_C_DATA: dict[str, list[float] | list[int]] = {
    "idx": [0, 1, 2, 0, 1, 2],
    "y": [1.0, 1.5, 0.5, 1.2, 1.4, 0.7],
}
_C_POINTS: list[dict[str, Host]] = [
    {"mu": 0.1, "tau": 0.8, "theta": [0.0, 0.3, -0.4], "predicted": 0.25},
    {"mu": -0.6, "tau": 1.3, "theta": [0.5, -0.2, 0.1], "predicted": -1.0},
    {"mu": 0.9, "tau": 0.4, "theta": [-0.3, 0.7, 0.2], "predicted": 1.4},
]
_D_DATA: dict[str, list[float] | list[int]] = {
    "idx": [0, 0, 1, 1, 2, 2],
    "y": [-1.0, -0.5, 2.5, 3.0, 0.2, -0.2],
}
_D_POINTS: list[dict[str, Host]] = [
    {"probs": [0.3, 0.7], "mu": [-1.0, 3.0]},
    {"probs": [0.6, 0.4], "mu": [0.5, 2.0]},
    {"probs": [0.1, 0.9], "mu": [-2.0, 1.0]},
]
_E_SENTENCE = ("the", "dog", "saw", "the", "cat", "with", "the", "telescope")


def _source(letter: str) -> str:
    """A fixture's text.

    Parameters
    ----------
    letter : str
        The fixture's letter.

    Returns
    -------
    str
        The QVR source.
    """
    return (_FIXTURES / f"{_CORPUS[letter]}.qvr").read_text()


def _module(letter: str, source: str | None = None) -> QiecModule:
    """A fixture's checked module.

    Parameters
    ----------
    letter : str
        The fixture's letter.
    source : str | None
        Source text standing in for the fixture's, for a mutation.

    Returns
    -------
    QiecModule
        The lowered and validated module.
    """
    text = _source(letter) if source is None else source
    module = lower_qvr_to_qiec(
        parse(text), module_name=_CORPUS[letter], file_path=f"{letter}.qvr"
    )
    validate_module(module)
    return module


def _sites(point: Mapping[str, Host]) -> dict[str, HostValue]:
    """A point as the reference machine takes it.

    Parameters
    ----------
    point : Mapping[str, Host]
        The site values, plate rows as lists.

    Returns
    -------
    dict[str, HostValue]
        The same values with every list a tuple, the host form of a
        plate row.
    """
    return {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in point.items()
    }


def _c_joint(point: Mapping[str, Host]) -> float:
    """Fixture C's joint at a point, by torch's densities.

    Parameters
    ----------
    point : Mapping[str, Host]
        The site values.

    Returns
    -------
    float
        The log joint of the sites, the observations, the observed
        measurement, and the predicted one.
    """
    normal, half = torch.distributions.Normal, torch.distributions.HalfNormal
    mu = torch.tensor(point["mu"], dtype=torch.float64)
    tau = torch.tensor(point["tau"], dtype=torch.float64)
    theta = torch.tensor(point["theta"], dtype=torch.float64)
    predicted = torch.tensor(point["predicted"], dtype=torch.float64)
    y = torch.tensor(_C_DATA["y"], dtype=torch.float64)
    idx = torch.tensor(_C_DATA["idx"])
    total = (
        normal(0.0, 1.0).log_prob(mu)
        + half(1.0).log_prob(tau)
        + normal(mu, tau).log_prob(theta).sum()
        + normal(theta.exp()[idx], 0.5).log_prob(y).sum()
        + normal(mu, 1.0).log_prob(torch.tensor(2.0, dtype=torch.float64))
        + normal(mu, 1.0).log_prob(predicted)
    )
    return float(total)


def _d_joint(point: Mapping[str, Host]) -> float:
    """Fixture D's joint at a point, by an independent logsumexp.

    Parameters
    ----------
    point : Mapping[str, Host]
        The site values.

    Returns
    -------
    float
        The log joint with the class of each item summed out.
    """
    distributions = torch.distributions
    probs = torch.tensor(point["probs"], dtype=torch.float64)
    mu = torch.tensor(point["mu"], dtype=torch.float64)
    y = torch.tensor(_D_DATA["y"], dtype=torch.float64)
    idx = torch.tensor(_D_DATA["idx"])
    rows = distributions.Normal(mu[None, :], 1.0).log_prob(y[:, None])
    per_item = torch.zeros(3, 2, dtype=torch.float64).index_add(0, idx, rows)
    total = (
        distributions.Dirichlet(torch.full((2,), 2.0, dtype=torch.float64)).log_prob(
            probs
        )
        + distributions.Normal(0.0, 3.0).log_prob(mu).sum()
        + torch.logsumexp(probs.log()[None, :] + per_item, dim=-1).sum()
    )
    return float(total)


# ---------------------------------------------------------------------------
# The stable passes, over every fixture.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("letter", sorted(_CORPUS))
def test_fixture_is_canonical_and_round_trips(letter: str) -> None:
    source = _source(letter)
    module = parse(source)
    assert module_to_source(module) == source
    assert module_to_source(parse(module_to_source(module))) == source


@pytest.mark.parametrize("letter", sorted(_CORPUS))
def test_fixture_lowers_validates_serializes_and_projects(letter: str) -> None:
    module = _module(letter)
    assert loads_module(dumps(module)) == module
    ir = lower_qiec_ir(module)
    # A deduction's helper computations serve the reference machine and
    # reach the IR only when a transpiled program calls one.
    reachable = {
        item.name
        for item in module.computations
        if not item.origin.structural_path
        or item.origin.structural_path[0] != "deductions"
        or item.name.endswith("__run")
    }
    assert {item.name for item in ir.computations} >= reachable - {
        item.name for item in module.computations if item.name.endswith("__run")
    }
    assert {item.name for item in ir.computations} <= {
        item.name for item in module.computations
    }


@pytest.mark.parametrize("letter", sorted(_CORPUS))
def test_fixture_transpiles_or_is_refused_under_its_kind(letter: str) -> None:
    module = parse(_source(letter))
    for target in available_targets():
        if target in _ADMITTING[letter]:
            emitted = transpile(module, target=target)
            assert emitted, f"{target} emitted nothing for fixture {letter}"
            continue
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(module, target=target)
        expected = _REFUSALS[letter][target]
        assert any(kind.startswith(expected) for kind in caught.value.kinds), (
            f"{target} refused fixture {letter} under {caught.value.kinds!r}, "
            f"not {expected!r}"
        )


@pytest.mark.parametrize("letter", sorted(_CORPUS))
def test_fixture_emissions_pass_the_target_compilers(
    letter: str, tmp_path: pathlib.Path
) -> None:
    """Every emitted program parses under its language's own tool."""
    module = parse(_source(letter))
    for target in sorted(_ADMITTING[letter] & set(_SYNTAX_CHECKS)):
        binary, argv, suffix = _SYNTAX_CHECKS[target]
        require_tool(binary)
        emitted = transpile(module, target=target)
        if suffix is None:
            run_argv, stdin = argv, emitted
        else:
            script = tmp_path / f"{_CORPUS[letter]}_{target}{suffix}"
            script.write_bytes(emitted)
            run_argv = [str(script) if a == _SOURCE_PATH else a for a in argv]
            stdin = b""
        completed = subprocess.run(
            run_argv, input=stdin, capture_output=True, timeout=120.0
        )
        assert completed.returncode == 0, (
            f"{target} rejected fixture {letter}: "
            f"{completed.stderr.decode('utf-8', errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Fixture A: recursive indexed computation.
# ---------------------------------------------------------------------------


def test_a_recursion_over_vec_refines_the_tail_and_is_called() -> None:
    module = _module("A")
    assert run_named(module, "total", (1.5, 2.0, 0.25)).value == 6.75
    statics = parse_static_arguments(module, "fold", ("n=S(S(Z))",))
    pair = run_named(module, "triple", (1.0, 2.0, 3.0)).value
    assert isinstance(pair, RuntimeConstructor) and pair.fields[0] == 1.0
    tail = pair.fields[1]
    assert run_named(module, "fold", (tail, 10.0), static_arguments=statics).value == (
        15.0
    )
    with pytest.raises(ExecutionFailure) as failure:
        run_named(module, "fold", (tail, 10.0))
    assert failure.value.diagnostic.code == "qiec-run-static"
    fold = next(item for item in module.computations if item.name == "fold")
    assert "fold" in dumps(fold.body)


def test_a_python_targets_run_the_recursion() -> None:
    module = parse(_source("A"))
    for target in ("pyro", "pymc"):
        namespace: dict[str, Callable[[float, float, float], float]] = {}
        exec(transpile(module, target=target).decode("utf-8"), namespace)  # noqa: S102
        assert namespace["qiec_total"](1.5, 2.0, 0.25) == 6.75


def test_a_dropping_the_refinement_is_ill_typed() -> None:
    broken = _source("A").replace(
        "            let rest <- fold[k](tail, acc + head)",
        "            let rest <- fold[m](tail, acc + head)",
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("A", broken)
    assert caught.value.code.startswith("qiec-")


# ---------------------------------------------------------------------------
# Fixture B: authored state handler.
# ---------------------------------------------------------------------------


def test_b_authored_clauses_handle_nested_instances_and_call_a_computation() -> None:
    module = _module("B")
    assert run_named(module, "counted", (3,)).value == 7
    assert run_named(module, "nested", (3,)).value == 14
    events = [event.event for event in run_named(module, "counted", (3,)).trace]
    assert "operation.handled" in events
    handler = module.handlers[0]
    assert handler.name == "run_state"
    assert {
        (clause.operation.digest[:8], clause.grade.value) for clause in handler.clauses
    }


def test_b_an_instance_cannot_escape_its_scope() -> None:
    escaped = (
        _source("B") + "\ndefine escaped(seed : Int) : Int !{} =\n"
        "    with instance cell : State[Int] in\n"
        "        perform cell.put(seed)\n"
        "        return seed\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("B", escaped)
    assert caught.value.code == "qiec-instance-escape"
    assert "local instance cell escapes its scope" in caught.value.message


def test_b_a_linear_clause_cannot_resume_twice() -> None:
    twice = _source("B").replace(
        "    get resumes 1 =>\n        resume(7)\n",
        "    get resumes 1 =>\n        let first <- resume(7)\n        resume(first)\n",
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("B", twice)
    assert "resum" in caught.value.message


# ---------------------------------------------------------------------------
# Fixture C: the nontrivial statistical model.
# ---------------------------------------------------------------------------


def test_c_reference_replay_scores_both_constructors() -> None:
    module = _module("C")
    for point in _C_POINTS:
        data = {"idx": tuple(_C_DATA["idx"]), "y": tuple(_C_DATA["y"])}
        run = run_program(module, "hierarchy", data=data, sites=_sites(point))
        assert run.value == point["predicted"]
        assert run.log_joint == pytest.approx(_c_joint(point), rel=1e-6)


def test_c_prediction_draws_the_missing_measurement() -> None:
    module = _module("C")
    data = {"idx": tuple(_C_DATA["idx"]), "y": tuple(_C_DATA["y"])}
    latents = {k: v for k, v in _C_POINTS[0].items() if k != "predicted"}
    drawn = sample_program(module, "hierarchy", data=data, sites=_sites(latents))
    replayed = run_program(
        module,
        "hierarchy",
        data=data,
        sites={**_sites(latents), "predicted": drawn.value},
    )
    assert drawn.log_joint == pytest.approx(replayed.log_joint)
    assert drawn.value != _C_POINTS[0]["predicted"]


def test_c_compiles_to_a_checked_program_that_scores_the_reference() -> None:
    program = loads(_source("C")).morphism
    assert isinstance(program, CheckedProgram)
    point = _C_POINTS[1]
    joint = program.log_joint(
        torch.zeros(1, 1),
        {
            "idx": torch.tensor(_C_DATA["idx"]),
            "y": torch.tensor(_C_DATA["y"], dtype=torch.float64),
            "mu": torch.tensor(point["mu"], dtype=torch.float64),
            "tau": torch.tensor(point["tau"], dtype=torch.float64),
            "theta": torch.tensor(point["theta"], dtype=torch.float64),
            "predicted": torch.tensor(point["predicted"], dtype=torch.float64),
        },
    )
    assert float(joint[0]) == pytest.approx(_c_joint(point), rel=1e-6)


@pytest.mark.parametrize("target", ["pyro", "pymc"])
def test_c_python_targets_score_the_reference_joint(target: str) -> None:
    module = parse(_source("C"))
    emitted = transpile(module, target=target)
    for point in _C_POINTS:
        if target == "pyro":
            scored = pyro_log_density(
                emitted, _C_DATA, point, integer_data=frozenset({"idx"})
            )
        else:
            scored = pymc_log_density(emitted, _C_DATA, point)
        assert scored == pytest.approx(_c_joint(point), rel=1e-6)


def test_c_dropping_the_case_loses_the_refinement() -> None:
    source = _source("C")
    without_case = source.replace(
        "    case m motive (t : Status) => Real\n"
        "        Present(value) =>\n"
        "            perform score.add(log_prob(Normal(location, spread), value))\n"
        "            return value\n"
        "        Absent =>\n"
        '            let drawn <- perform random.sample[Real](site("predicted"), Normal(location, spread))\n'
        "            return drawn\n",
        "    perform score.add(log_prob(Normal(location, spread), m))\n    return m\n",
    )
    assert without_case != source
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("C", without_case)
    assert caught.value.code.startswith("qiec-")
    swapped = source.replace(
        "        Absent =>\n            let drawn",
        "        Present(value) =>\n            let drawn",
    )
    with pytest.raises(QiecDiagnosticError):
        _module("C", swapped)


def test_c_calling_the_helper_from_a_pure_computation_leaves_the_row_open() -> None:
    open_row = (
        _source("C") + "\ndefine quiet(location : Real) : Real !{} =\n"
        "    let v <- account[Missing](construct Absent() as Measurement(Missing), location, 1.0)\n"
        "    return v\n"
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("C", open_row)
    assert caught.value.code == "qiec-row"
    assert "random : Random" in caught.value.message


def test_c_dropping_the_score_is_a_row_error() -> None:
    without_score = _source("C").replace(
        "            perform score.add(log_prob(Normal(location, spread), value))\n",
        "",
    )
    with pytest.raises(QiecDiagnosticError) as caught:
        _module("C", without_score)
    assert caught.value.code == "qiec-row"
    assert "score : Score" in caught.value.message


def test_c_widening_the_score_changes_the_reference_joint() -> None:
    widened = _source("C").replace(
        "            perform score.add(log_prob(Normal(location, spread), value))\n",
        "            perform score.add(log_prob(Normal(location, spread * 2.0), value))\n",
    )
    assert widened != _source("C")
    module = _module("C", widened)
    data = {"idx": tuple(_C_DATA["idx"]), "y": tuple(_C_DATA["y"])}
    point = _C_POINTS[0]
    run = run_program(module, "hierarchy", data=data, sites=_sites(point))
    narrow = float(
        torch.distributions.Normal(point["mu"], 1.0).log_prob(torch.tensor(2.0))
    )
    wide = float(
        torch.distributions.Normal(point["mu"], 2.0).log_prob(torch.tensor(2.0))
    )
    assert run.log_joint == pytest.approx(_c_joint(point) - narrow + wide, rel=1e-6)


# ---------------------------------------------------------------------------
# Fixture D: exact grouped marginalization.
# ---------------------------------------------------------------------------


def test_d_reference_torch_and_oracle_agree() -> None:
    module = _module("D")
    program = loads(_source("D")).morphism
    for point in _D_POINTS:
        run = run_program(
            module,
            "mixture",
            data={"idx": tuple(_D_DATA["idx"]), "y": tuple(_D_DATA["y"])},
            sites=_sites(point),
        )
        oracle = _d_joint(point)
        assert run.log_joint == pytest.approx(oracle, rel=1e-6)
        joint = program.log_joint(
            torch.zeros(1, 1),
            {
                "probs": torch.tensor(point["probs"]),
                "mu": torch.tensor(point["mu"]),
                "idx": torch.tensor(_D_DATA["idx"]),
                "y": torch.tensor(_D_DATA["y"]),
            },
        )
        assert float(joint[0]) == pytest.approx(oracle, rel=1e-5)


def test_d_elaborates_the_block_to_an_enumerated_helper() -> None:
    module = _module("D")
    names = {item.name for item in module.computations}
    assert any("mixture" in name and name != "mixture" for name in names), names
    kinds = {handler.name for handler in module.handlers}
    assert any("enumerate" in name or "marginal" in name for name in kinds), kinds


@pytest.mark.parametrize("target", ["pyro", "pymc"])
def test_d_python_targets_score_the_oracle_up_to_a_constant(target: str) -> None:
    emitted = transpile(parse(_source("D")), target=target)
    offsets: list[float] = []
    for point in _D_POINTS:
        if target == "pyro":
            scored = pyro_log_density(
                emitted, _D_DATA, point, integer_data=frozenset({"idx"})
            )
        else:
            scored = pymc_log_density(emitted, _D_DATA, point)
        offsets.append(scored - _d_joint(point))
    assert max(offsets) - min(offsets) < 1e-6, offsets


def test_d_stan_lowers_the_marginal_statically() -> None:
    require_tool("stanc")
    emitted = transpile(parse(_source("D")), target="stan")
    text = emitted.decode("utf-8")
    assert "log_sum_exp" in text
    completed = subprocess.run(
        ["stanc", "--info", "-"], input=emitted, capture_output=True, timeout=120.0
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")


def test_d_dropping_the_observation_changes_the_reference_joint() -> None:
    module = _module(
        "D",
        _source("D").replace(
            "        observe y : Resp <- Normal(loc=mu[z], scale=1.0) [via=idx]\n",
            "        observe y : Resp <- Normal(loc=mu[z], scale=2.0) [via=idx]\n",
        ),
    )
    point = _D_POINTS[0]
    run = run_program(
        module,
        "mixture",
        data={"idx": tuple(_D_DATA["idx"]), "y": tuple(_D_DATA["y"])},
        sites=_sites(point),
    )
    assert run.log_joint != pytest.approx(_d_joint(point), rel=1e-6)


# ---------------------------------------------------------------------------
# Fixture E: logic and model composition.
# ---------------------------------------------------------------------------


def test_e_the_deduction_sums_both_attachments_under_choose() -> None:
    module = _module("E")
    run = run_deduction(module, "Ambiguous", tokens=list(_E_SENTENCE))
    assert run.weight == pytest.approx(math.log(2.0))
    shots = [
        event
        for event in run.result.trace
        if event.event == "resumption.invoked" and event.detail.get("shot", 0) >= 1
    ]
    assert shots, "the search handler resumed a choice more than once"
    addresses = {
        (event.detail.get("instance"), event.detail.get("shot")) for event in shots
    }
    assert len(addresses) > 1
    chart = loads(_source("E")).deductions["Ambiguous"](list(_E_SENTENCE))
    assert float(chart.goal_weight().detach()) == pytest.approx(math.log(2.0))


def test_e_the_program_adds_the_converted_weight_to_its_score() -> None:
    module = _module("E")
    run = run_program(
        module, "attach", data={"sentence": _E_SENTENCE}, sites={"bias": 0.5}
    )
    prior = float(torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(0.5)))
    assert run.value == pytest.approx(math.log(2.0) + 0.5)
    assert run.log_joint == pytest.approx(math.log(2.0) + 0.5 + prior)
    unambiguous = run_program(
        module,
        "attach",
        data={"sentence": ("the", "dog", "saw", "the", "cat")},
        sites={"bias": 0.5},
    )
    assert unambiguous.value == pytest.approx(0.5)
    entry = next(item for item in module.entries if item.name == "attach")
    assert [parameter.name for parameter in entry.parameters] == ["sentence"]


def test_e_every_target_refuses_the_search() -> None:
    """The deduction's search has no target form, so every target refuses
    the program under the search capability of the deduction's entry."""
    module = parse(_source("E"))
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(module, target=target)
        assert caught.value.kinds == ["qiec:capability:search:Ambiguous__run"], (
            target,
            caught.value.kinds,
        )


# ---------------------------------------------------------------------------
# Fixture F: neural and model composition.
# ---------------------------------------------------------------------------


def test_f_backpropagates_to_the_decoder_and_matches_the_reference() -> None:
    program = loads(_source("F"))
    model = program.morphism
    torch.manual_seed(0)
    x = torch.randn(6, 2)
    y = torch.randn(6, 1)
    joint = model.log_joint(
        torch.zeros(1, 1), {"tau": torch.tensor([0.7]), "x": x, "y": y}
    )
    joint.sum().backward()
    parameters = dict(model.named_parameters())
    assert len(parameters) == 4
    assert all(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        for parameter in parameters.values()
    )
    module = _module("F")
    entry = next(item for item in module.entries if item.name == "regression")
    weights = {
        "decoder_param_layer0_weight": "_step_y._family.param_source.net.0.weight",
        "decoder_param_layer0_bias": "_step_y._family.param_source.net.0.bias",
        "decoder_param_weight": "_step_y._family.param_source.net.2.weight",
        "decoder_param_bias": "_step_y._family.param_source.net.2.bias",
    }
    assert {parameter.name for parameter in entry.parameters} == {
        "x",
        "y",
        *weights,
    }
    data: dict[str, HostValue] = {
        "x": tuple(tuple(row) for row in x.tolist()),
        "y": tuple(tuple(row) for row in y.tolist()),
    }
    for name, attribute in weights.items():
        value = parameters[attribute].detach().tolist()
        data[name] = (
            tuple(tuple(row) for row in value)
            if isinstance(value[0], list)
            else tuple(value)
        )
    run = run_program(module, "regression", data=data, sites={"tau": 0.7})
    assert run.log_joint == pytest.approx(float(joint.detach()), rel=1e-4)


def test_f_every_target_refuses_the_network_under_its_kind() -> None:
    module = parse(_source("F"))
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(module, target=target)
        reported = {k.partition(": ")[0] for k in caught.value.kinds}
        assert "param-source:mlp" in reported


def test_f_the_reference_answers_the_prelude_draw() -> None:
    module = _module("F")
    entry = next(item for item in module.entries if item.name == "regression")
    assert [site.name for site in entry.sites] == ["tau", "y"]
    assert (
        RuntimeConfiguration(selections=(RuntimeSelection(provider="core"),)).label
        == "core"
    )
