"""A marginalized chain: ``scan(cell, init=p)`` over a cell that returns its marginalized state.

The chain ``z_0 ~ p``, ``z_t ~ transition(z_{t-1})``, with each position's
observations scored at ``z_t``, has every state summed out. Two oracles,
written here and independent of each other and of the library, give its
log marginal likelihood: a brute-force sum over every path of states
(short chains), and a log-space forward algorithm in float64 torch (long
chains). Tolerances are fixed in advance:

* the compiled runtime runs in float32 and is compared with the float64
  oracles at ``abs=1e-4`` for chains of at most ten positions (the
  float32 rounding of a sum of terms of order one) and ``rel=1e-5`` for
  long chains (float32 relative precision, accumulated over the
  positions);
* the reference machine runs in float64 and is compared at ``rel=1e-9``.

Each comparison is paired with deliberately wrong oracles (a wrong first
state, transposed transitions, emissions from the previous state, the
first state emitting) that the same tolerance rejects.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler import CompileError
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference import trace

HMM = """
object State : FinSet {K}
object Symbol : FinSet {V}
object StartVec : Real {K}

program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(hmm_step, init=start)

program hmm : Symbol -> StartVec
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export hmm
"""

IO_HMM = """
object State : FinSet 3
object Cov : Real 1
object StartVec : Real 3

program io_step(u, z_prev) : Cov * State -> State
    let row = logits[z_prev] + 2.0 * sum(u)
    marginalize z : State <- Categorical(logits=row)
        let mu = means[z]
        observe y <- Normal(loc=mu, scale=0.5)
    return z

define chain = scan(io_step, init=start{REVERSE})

program io_hmm : Cov -> StartVec
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample logits : State <- Normal(loc=0.0, scale=1.0) [over=State, iid_over=State]
    sample means <- Normal(loc=0.0, scale=2.0) [over=State]
    sample path <- chain
    return start

export io_hmm
"""


def _zero(value: float) -> torch.Tensor:
    """A float64 location, so a family built from two numbers is float64."""
    return torch.tensor(value, dtype=torch.float64)


def _hmm(states: int, symbols: int) -> str:
    return HMM.replace("{K}", str(states)).replace("{V}", str(symbols))


def _io_hmm(*, reverse: bool) -> str:
    return IO_HMM.replace("{REVERSE}", ", reverse=true" if reverse else "")


# ---------------------------------------------------------------------------
# Oracles


def _enumerated(
    start: torch.Tensor, weight: Callable[[int, int, int], float], length: int
) -> float:
    """log sum over every path z_0 .. z_T of p(z_0) prod_t exp(weight(t, z_{t-1}, z_t))."""
    states = int(start.shape[0])
    terms: list[float] = []
    for path in itertools.product(range(states), repeat=length + 1):
        term = math.log(float(start[path[0]]))
        for t in range(length):
            term += weight(t, path[t], path[t + 1])
        terms.append(term)
    peak = max(terms)
    return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))


def _log_probs(probs: torch.Tensor) -> torch.Tensor:
    """The log probabilities ``Categorical(probs=...)`` assigns, row by row."""
    return D.Categorical(probs=probs).logits


def _forward(start: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The log-space forward algorithm over weights (T, K, K) in visit order."""
    log_alpha = _log_probs(start)
    for table in weights:
        log_alpha = torch.logsumexp(log_alpha[:, None] + table, dim=0)
    return torch.logsumexp(log_alpha, dim=0)


def _hmm_weights(
    transition: torch.Tensor, emission: torch.Tensor, symbols: list[int]
) -> torch.Tensor:
    """W[t, i, j] = log transition[i, j] + log emission[j, symbol_t]."""
    return torch.stack(
        [_log_probs(transition) + _log_probs(emission)[:, s][None, :] for s in symbols]
    )


def _hmm_prior(
    start: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor
) -> torch.Tensor:
    return (
        D.Dirichlet(torch.ones_like(start)).log_prob(start)
        + D.Dirichlet(torch.ones_like(transition)).log_prob(transition).sum()
        + D.Dirichlet(torch.ones_like(emission)).log_prob(emission).sum()
    )


def _parameters(
    states: int, symbols: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    start = torch.softmax(
        torch.randn(states, generator=generator, dtype=torch.float64), dim=-1
    )
    transition = torch.softmax(
        torch.randn(states, states, generator=generator, dtype=torch.float64), dim=-1
    )
    emission = torch.softmax(
        torch.randn(states, symbols, generator=generator, dtype=torch.float64), dim=-1
    )
    return start, transition, emission


def _symbols(count: int, symbols: int, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, symbols, (count, length), generator=generator)


def _nested(listed: list | float) -> tuple | float:
    return (
        tuple(_nested(item) for item in listed) if isinstance(listed, list) else listed
    )


def _host(value: torch.Tensor) -> tuple | float:
    return _nested(value.tolist())


def _sites(
    start: torch.Tensor, transition: torch.Tensor, emission: torch.Tensor
) -> dict:
    return {
        "start": _host(start),
        "transition": _host(transition),
        "emission": _host(emission),
    }


def _compiled_hmm_log_joint(
    source: str, sequences: torch.Tensor, values: dict[str, torch.Tensor]
) -> torch.Tensor:
    model = loads(source).morphism
    return model.log_joint(sequences, {k: v.float() for k, v in values.items()})


# ---------------------------------------------------------------------------
# The discrete HMM, observing each position's symbol


def test_compiled_runtime_matches_path_enumeration() -> None:
    torch.manual_seed(0)
    start, transition, emission = _parameters(3, 4, seed=1)
    sequences = _symbols(3, 4, 6, seed=2)
    got = _compiled_hmm_log_joint(
        _hmm(3, 4),
        sequences,
        {"start": start, "transition": transition, "emission": emission},
    )
    prior = float(_hmm_prior(start, transition, emission))
    for row, symbols in enumerate(sequences.tolist()):

        def weight(t: int, i: int, j: int, symbols: list[int] = symbols) -> float:
            return math.log(float(transition[i, j])) + math.log(
                float(emission[j, symbols[t]])
            )

        expected = prior + _enumerated(start, weight, len(symbols))
        assert float(got[row]) == pytest.approx(expected, abs=1e-4)


def test_wrong_oracles_are_rejected_at_the_same_tolerance() -> None:
    start, transition, emission = _parameters(3, 4, seed=1)
    symbols = _symbols(1, 4, 6, seed=2)[0].tolist()
    got = float(
        _compiled_hmm_log_joint(
            _hmm(3, 4),
            torch.tensor([symbols]),
            {"start": start, "transition": transition, "emission": emission},
        )[0]
    )
    prior = float(_hmm_prior(start, transition, emission))
    right = prior + float(_forward(start, _hmm_weights(transition, emission, symbols)))
    assert got == pytest.approx(right, abs=1e-4)
    wrong = {
        "uniform first state": _forward(
            torch.full_like(start, 1.0 / 3.0),
            _hmm_weights(transition, emission, symbols),
        ),
        "transposed transitions": _forward(
            start, _hmm_weights(transition.T.contiguous(), emission, symbols)
        ),
        "emission from the previous state": _forward(
            start,
            torch.stack(
                [
                    torch.log(transition) + torch.log(emission[:, s])[:, None]
                    for s in symbols
                ]
            ),
        ),
        # The first state emitting the first symbol, with no state before it.
        "first state emits": torch.logsumexp(
            torch.log(start)
            + torch.log(emission[:, symbols[0]])
            + _forward_from_each(transition, emission, symbols[1:]),
            dim=0,
        ),
    }
    for label, value in wrong.items():
        assert abs(got - (prior + float(value))) > 1e-4, label


def _forward_from_each(
    transition: torch.Tensor, emission: torch.Tensor, symbols: list[int]
) -> torch.Tensor:
    """log p(symbols | z_1 = i) for each i, by a backward recursion."""
    log_beta = torch.zeros(transition.shape[0], dtype=transition.dtype)
    for s in reversed(symbols):
        log_beta = torch.logsumexp(
            torch.log(transition) + (torch.log(emission[:, s]) + log_beta)[None, :],
            dim=1,
        )
    return log_beta


def test_reference_machine_matches_path_enumeration() -> None:
    start, transition, emission = _parameters(2, 3, seed=5)
    symbols = _symbols(1, 3, 7, seed=6)[0].tolist()
    program = loads(_hmm(2, 3))
    run = program.run(
        "hmm",
        data={"symbol": tuple(symbols)},
        sites=_sites(start, transition, emission),
        seed=0,
    )

    def weight(t: int, i: int, j: int) -> float:
        return math.log(float(transition[i, j])) + math.log(
            float(emission[j, symbols[t]])
        )

    expected = float(_hmm_prior(start, transition, emission)) + _enumerated(
        start, weight, len(symbols)
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-9)
    transposed = float(_hmm_prior(start, transition, emission)) + float(
        _forward(start, _hmm_weights(transition.T.contiguous(), emission, symbols))
    )
    assert run.log_joint != pytest.approx(transposed, rel=1e-9)


def test_a_first_state_with_zero_mass_somewhere_is_summed_in_log_space() -> None:
    """A first-state distribution of zeros and a one starts the chain at one state.

    ``log 0`` is minus infinity on both paths, and the forward algorithm's
    log-sum-exp leaves those states out rather than producing ``nan``.
    """
    source = _hmm(3, 4).replace(
        "    sample start <- Dirichlet(concentration=1.0) [over=State]\n",
        "    let start = [0.0, 1.0, 0.0]\n",
    )
    _, transition, emission = _parameters(3, 4, seed=7)
    symbols = _symbols(1, 4, 5, seed=8)[0].tolist()
    prior = float(
        D.Dirichlet(torch.ones(3, 3, dtype=torch.float64)).log_prob(transition).sum()
        + D.Dirichlet(torch.ones(3, 4, dtype=torch.float64)).log_prob(emission).sum()
    )

    def weight(t: int, i: int, j: int) -> float:
        return math.log(float(transition[i, j])) + math.log(
            float(emission[j, symbols[t]])
        )

    by_paths = prior + _enumerated_from(1, weight, 3, len(symbols))
    program = loads(source)
    run = program.run(
        "hmm",
        data={"symbol": tuple(symbols)},
        sites={"transition": _host(transition), "emission": _host(emission)},
        seed=0,
    )
    assert run.log_joint == pytest.approx(by_paths, rel=1e-9)
    compiled = program.morphism.log_joint(
        torch.tensor([symbols]),
        {"transition": transition.float(), "emission": emission.float()},
    )
    assert float(compiled[0]) == pytest.approx(by_paths, abs=1e-4)
    uniform = prior + float(
        _forward(
            torch.full((3,), 1.0 / 3.0, dtype=torch.float64),
            _hmm_weights(transition, emission, symbols),
        )
    )
    assert abs(float(compiled[0]) - uniform) > 1e-4


def _enumerated_from(
    first: int, weight: Callable[[int, int, int], float], states: int, length: int
) -> float:
    """log sum over every path from the state ``first`` of exp(sum_t weight(t, ...))."""
    terms: list[float] = []
    for path in itertools.product(range(states), repeat=length):
        full = (first, *path)
        terms.append(math.fsum(weight(t, full[t], full[t + 1]) for t in range(length)))
    peak = max(terms)
    return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))


def test_the_trace_records_one_score_for_the_chain_and_no_state_sites() -> None:
    model = loads(_hmm(2, 3)).morphism
    recorded = trace(model, _symbols(2, 3, 5, seed=3))
    assert set(recorded.sites) == {"start", "transition", "emission", "path"}
    start = recorded.sites["start"].value.double()
    transition = recorded.sites["transition"].value.double()
    emission = recorded.sites["emission"].value.double()
    for row, symbols in enumerate(_symbols(2, 3, 5, seed=3).tolist()):
        expected = float(
            _forward(start[row], _hmm_weights(transition[row], emission[row], symbols))
        )
        assert float(recorded.sites["path"].value.reshape(2)[row]) == pytest.approx(
            expected, abs=1e-4
        )


# ---------------------------------------------------------------------------
# An input-output chain: transitions read the covariate, Gaussian emissions


def _io_values(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "start": torch.softmax(
            torch.randn(3, generator=generator, dtype=torch.float64), dim=-1
        ),
        "logits": torch.randn(3, 3, generator=generator, dtype=torch.float64),
        "means": 2.0 * torch.randn(3, generator=generator, dtype=torch.float64),
    }


def _io_weights(
    values: dict[str, torch.Tensor], u: torch.Tensor, y: torch.Tensor, reverse: bool
) -> torch.Tensor:
    order = range(len(u) - 1, -1, -1) if reverse else range(len(u))
    return torch.stack(
        [
            torch.log_softmax(values["logits"] + 2.0 * u[t], dim=-1)
            + D.Normal(values["means"], 0.5).log_prob(y[t])[None, :]
            for t in order
        ]
    )


def _io_prior(values: dict[str, torch.Tensor]) -> float:
    return float(
        D.Dirichlet(torch.ones(3, dtype=torch.float64)).log_prob(values["start"])
        + D.Normal(_zero(0.0), 1.0).log_prob(values["logits"]).sum()
        + D.Normal(_zero(0.0), 2.0).log_prob(values["means"]).sum()
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_input_output_chain_matches_both_oracles_on_both_paths(reverse: bool) -> None:
    values = _io_values(seed=11)
    generator = torch.Generator().manual_seed(12)
    u = torch.randn(5, generator=generator, dtype=torch.float64)
    y = 2.0 * torch.randn(5, generator=generator, dtype=torch.float64)
    weights = _io_weights(values, u, y, reverse)
    by_forward = _io_prior(values) + float(_forward(values["start"], weights))
    by_paths = _io_prior(values) + _enumerated(
        values["start"], lambda t, i, j: float(weights[t, i, j]), len(u)
    )
    assert by_forward == pytest.approx(by_paths, rel=1e-12)
    program = loads(_io_hmm(reverse=reverse))
    compiled = program.morphism.log_joint(
        u.float().reshape(1, 5, 1),
        {**{k: v.float() for k, v in values.items()}, "y": y.float().reshape(1, 5)},
    )
    assert float(compiled[0]) == pytest.approx(by_paths, abs=1e-4)
    run = program.run(
        "io_hmm",
        data={"cov": tuple((item,) for item in u.tolist()), "y": tuple(y.tolist())},
        sites={k: _host(v) for k, v in values.items()},
        seed=0,
    )
    assert run.log_joint == pytest.approx(by_paths, rel=1e-9)
    # Visiting the positions in the other order is another model.
    other = _io_prior(values) + float(
        _forward(values["start"], _io_weights(values, u, y, not reverse))
    )
    assert abs(float(compiled[0]) - other) > 1e-4
    assert run.log_joint != pytest.approx(other, rel=1e-9)


# ---------------------------------------------------------------------------
# Long chains: log space keeps the marginal where probabilities underflow


def _probability_space_marginal(
    start: torch.Tensor,
    transition: torch.Tensor,
    emission: torch.Tensor,
    symbols: list[int],
) -> float:
    alpha = start.clone()
    for s in symbols:
        alpha = (alpha @ transition) * emission[:, s]
    return float(alpha.sum())


def test_compiled_runtime_on_a_chain_whose_probabilities_underflow() -> None:
    start, transition, emission = _parameters(3, 6, seed=21)
    symbols = _symbols(1, 6, 1500, seed=22)[0].tolist()
    assert _probability_space_marginal(start, transition, emission, symbols) == 0.0
    expected = float(_hmm_prior(start, transition, emission)) + float(
        _forward(start, _hmm_weights(transition, emission, symbols))
    )
    assert expected < -745.0
    got = float(
        _compiled_hmm_log_joint(
            _hmm(3, 6),
            torch.tensor([symbols]),
            {"start": start, "transition": transition, "emission": emission},
        )[0]
    )
    assert math.isfinite(got)
    assert got == pytest.approx(expected, rel=1e-5)
    transposed = float(_hmm_prior(start, transition, emission)) + float(
        _forward(start, _hmm_weights(transition.T.contiguous(), emission, symbols))
    )
    assert got != pytest.approx(transposed, rel=1e-5)


def test_reference_machine_on_a_chain_whose_probabilities_underflow() -> None:
    start, transition, emission = _parameters(2, 6, seed=23)
    symbols = _symbols(1, 6, 600, seed=24)[0].tolist()
    assert _probability_space_marginal(start, transition, emission, symbols) == 0.0
    expected = float(_hmm_prior(start, transition, emission)) + float(
        _forward(start, _hmm_weights(transition, emission, symbols))
    )
    program = loads(_hmm(2, 6))
    run = program.run(
        "hmm",
        data={"symbol": tuple(symbols)},
        sites=_sites(start, transition, emission),
        seed=0,
    )
    assert math.isfinite(run.log_joint)
    assert run.log_joint == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Gradients


def test_gradients_match_autograd_on_the_oracle() -> None:
    start, transition, emission = _parameters(3, 4, seed=31)
    symbols = _symbols(1, 4, 8, seed=32)[0].tolist()
    model = loads(_hmm(3, 4)).morphism
    leaves = {
        "start": start.float().requires_grad_(True),
        "transition": transition.float().requires_grad_(True),
        "emission": emission.float().requires_grad_(True),
    }
    model.log_joint(torch.tensor([symbols]), leaves).sum().backward()
    oracle = {
        name: value.clone().requires_grad_(True)
        for name, value in (
            ("start", start),
            ("transition", transition),
            ("emission", emission),
        )
    }
    (
        _hmm_prior(oracle["start"], oracle["transition"], oracle["emission"])
        + _forward(
            oracle["start"],
            _hmm_weights(oracle["transition"], oracle["emission"], symbols),
        )
    ).backward()
    for name, leaf in leaves.items():
        expected = oracle[name].grad
        assert expected is not None and leaf.grad is not None
        torch.testing.assert_close(
            leaf.grad.double(), expected, rtol=1e-4, atol=1e-4, msg=name
        )
    # The gradient of the transposed model is another vector.
    transposed = transition.clone().requires_grad_(True)
    _forward(start, _hmm_weights(transposed.T, emission, symbols)).backward()
    assert transposed.grad is not None and leaves["transition"].grad is not None
    assert not torch.allclose(
        leaves["transition"].grad.double(), transposed.grad, rtol=1e-4, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Recovery

TRUE_START = torch.tensor([0.5, 0.5])
TRUE_TRANSITION = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
TRUE_EMISSION = torch.tensor([[0.6, 0.3, 0.05, 0.05], [0.05, 0.05, 0.3, 0.6]])


def _simulate(count: int, length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(count):
        state = int(torch.multinomial(TRUE_START, 1, generator=generator))
        row = []
        for _ in range(length):
            state = int(
                torch.multinomial(TRUE_TRANSITION[state], 1, generator=generator)
            )
            row.append(
                int(torch.multinomial(TRUE_EMISSION[state], 1, generator=generator))
            )
        rows.append(row)
    return torch.tensor(rows)


def test_maximum_likelihood_recovers_the_transition_and_emission_rows() -> None:
    """Fit a 2-state, 4-symbol chain by maximum likelihood and compare with the truth.

    Four sequences of 200 symbols. The chain's stationary distribution is
    (2/3, 1/3), so state 2 is visited about 267 times. Were the states
    observed, the sampling sd of an emission probability of state 2 would
    be at most sqrt(0.6 * 0.4 / 267) = 0.030 and of a transition
    probability from it sqrt(0.8 * 0.2 / 267) = 0.024; with the states
    summed out the information is lower, and the tolerances are four of
    those sds: every emission probability within 0.12 of the truth and
    every transition probability within 0.10. The Dirichlet(1) priors are
    flat, so the maximum of the log joint over the simplices is the
    maximum likelihood. The likelihood is invariant to relabelling the
    states, so the fit is compared after matching its states to the
    truth's by which emits symbol 0 more; the fit starts from a seeded
    perturbation of the label-symmetric point, a stationary point. The
    first state's distribution is weakly identified from four sequences
    and is not compared. The check has power: the starting point fails it.
    """
    torch.manual_seed(0)
    sequences = _simulate(4, 200, seed=41)
    model = loads(_hmm(2, 4)).morphism
    generator = torch.Generator().manual_seed(42)
    logits = {
        "start": torch.zeros(2),
        "transition": torch.zeros(2, 2),
        "emission": 0.5 * torch.randn(2, 4, generator=generator),
    }
    initial_emission = torch.softmax(logits["emission"], dim=-1)
    for value in logits.values():
        value.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        list(logits.values()), max_iter=80, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        values = {name: torch.softmax(value, dim=-1) for name, value in logits.items()}
        loss = -model.log_joint(sequences, values).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        transition = torch.softmax(logits["transition"], dim=-1)
        emission = torch.softmax(logits["emission"], dim=-1)
    order = [0, 1] if float(emission[0, 0]) > float(emission[1, 0]) else [1, 0]
    transition = transition[order][:, order]
    emission = emission[order]
    assert float((emission - TRUE_EMISSION).abs().max()) < 0.12
    assert float((transition - TRUE_TRANSITION).abs().max()) < 0.10
    # The starting point is outside the tolerance.
    assert float((initial_emission - TRUE_EMISSION).abs().max()) >= 0.12


# ---------------------------------------------------------------------------
# Refusals, on the compiled runtime and in the checked elaboration


def _refused(source: str, kind: str) -> None:
    with pytest.raises(CompileError, match=kind):
        loads(source)
    with pytest.raises(QiecDiagnosticError, match=kind):
        lower_qvr_to_qiec(parse(source))


BASE = _hmm(2, 3)


@pytest.mark.parametrize(
    ("edit", "kind"),
    [
        (
            (
                "Categorical(probs=transition[z_prev])",
                "Dirichlet(concentration=transition[z_prev])",
            ),
            "scan:marginal:family:hmm_step",
        ),
        (
            (
                "Categorical(probs=transition[z_prev])",
                "Categorical(probs=transition[z_prev]) [reduction=sum]",
            ),
            "scan:marginal:reduction:hmm_step",
        ),
        (
            (
                "        observe symbol <- Categorical(probs=emission[z])\n",
                "        observe symbol <- Categorical(probs=emission[z])\n"
                "        sample w <- Normal(loc=0.0, scale=1.0)\n",
            ),
            "scan:marginal:step:hmm_step",
        ),
        (
            (
                "    marginalize z : State",
                "    sample q <- Normal(loc=0.0, scale=1.0)\n    marginalize z : State",
            ),
            "scan:marginal:step:hmm_step",
        ),
        ((", init=start)", ")"), "scan:marginal:init:hmm_step"),
        (("init=start)", "init=start, outputs=all)"), "scan:marginal:outputs:hmm_step"),
        (
            ("    return start\n\nexport", "    return path\n\nexport"),
            "scan:marginal:read:path",
        ),
        (
            (
                "    sample path <- chain\n",
                "    sample path <- chain\n    let twice = 2.0 * path\n",
            ),
            "scan:marginal:read:path",
        ),
        (
            ("    sample path <- chain\n", "    observe path <- chain\n"),
            "scan:marginal:observe:hmm_step",
        ),
        (
            (
                "define chain = scan(hmm_step, init=start)",
                "define chain = scan(hmm_step, init=start) >> echo",
            ),
            "scan:marginal:chain:hmm_step",
        ),
        (
            ("    sample path <- chain\n", "    sample path <- hmm_step\n"),
            "scan:marginal:entry:hmm_step",
        ),
        (
            (
                "observe symbol <- Categorical(probs=emission[z])",
                "observe symbol : State <- Categorical(probs=emission[z])",
            ),
            "scan:marginal:observe:hmm_step",
        ),
    ],
)
def test_forms_outside_the_forward_algorithm_are_refused_by_name(
    edit: tuple[str, str], kind: str
) -> None:
    old, new = edit
    assert old in BASE
    source = BASE.replace(old, new)
    if "echo" in new:
        source = source.replace(
            "object StartVec : Real 2",
            "object StartVec : Real 2\nmorphism echo : State -> State ~ Categorical",
        )
    _refused(source, kind)


def test_a_cell_that_returns_another_object_is_refused() -> None:
    source = BASE.replace(
        "program hmm_step(symbol, z_prev) : Symbol * State -> State",
        "program hmm_step(symbol, z_prev) : Symbol * State -> Symbol",
    )
    _refused(source, "scan:marginal:state:hmm_step")


def test_the_reference_machine_enumerates_a_categorical_given_by_logits() -> None:
    """``marginalize c <- Categorical(logits=w)`` sums over the classes of ``w``."""
    source = """
object Class : FinSet 3
object Weights : Real 3

program mixture : Class -> Weights
    sample w <- Normal(loc=0.0, scale=1.0) [over=Class]
    sample means <- Normal(loc=0.0, scale=3.0) [over=Class]
    marginalize c : Class <- Categorical(logits=w)
        let mu = means[c]
        observe y <- Normal(loc=mu, scale=1.0)
    return w

export mixture
"""
    w = torch.tensor([0.2, -1.0, 0.5], dtype=torch.float64)
    means = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    run = loads(source).run(
        "mixture",
        data={"y": 0.7},
        sites={"w": _host(w), "means": _host(means)},
        seed=0,
    )
    prior = float(
        D.Normal(_zero(0.0), 1.0).log_prob(w).sum()
        + D.Normal(_zero(0.0), 3.0).log_prob(means).sum()
    )
    expected = prior + torch.logsumexp(
        torch.log_softmax(w, dim=0)
        + D.Normal(means, 1.0).log_prob(torch.tensor(0.7, dtype=torch.float64)),
        dim=0,
    )
    assert run.log_joint == pytest.approx(float(expected), rel=1e-9)
    by_probs = prior + torch.logsumexp(
        torch.log(w.abs())
        + D.Normal(means, 1.0).log_prob(torch.tensor(0.7, dtype=torch.float64)),
        dim=0,
    )
    assert run.log_joint != pytest.approx(float(by_probs), rel=1e-9)
