"""Programs that call computations with authored handlers are fitted.

Each program is compared with the one a reader would write by hand, with every
request replaced by its clause's answer, and with the reference machine, which
runs the handlers. The two encodings have one log joint, one gradient, and, for
one seed, the same draws.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.effects.checked_program import CheckedProgram
from quivers.inference import (
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
    AutoNormalGuide,
)
from quivers.qiec.entries import HostValue
from tests._handler_programs import HANDLERS, random_body

_F64 = torch.float64
_Y4 = (0.4, -0.3, 0.9, 0.2)

STORE = """
effect Store
    recall : Unit -> Real
    remember : Real -> Unit

instance store : Store

handler fixed_store for Store : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    recall resumes 1 =>
        resume(2.0)
    remember(v : Real) resumes 1 =>
        resume(unit)

define width_from_store(base : Real) : Real !{} =
    handle store with fixed_store in
        perform store.remember(base)
        let w <- perform store.recall()
        return w + base

object Trial : FinSet 4
object Val : Real 1

program uses_handler : Trial -> Val
    sample base <- HalfNormal(scale=1.0)
    let w <- width_from_store(base)
    observe y : Trial <- Normal(loc=0.0, scale=w)
    return base

export uses_handler
"""

STORE_BY_HAND = """
object Trial : FinSet 4
object Val : Real 1

program uses_handler : Trial -> Val
    sample base <- HalfNormal(scale=1.0)
    let w = 2.0 + base
    observe y : Trial <- Normal(loc=0.0, scale=w)
    return base

export uses_handler
"""


def _program(source: str) -> MonadicProgram:
    """A source's compiled program."""
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram), getattr(program, "refusal", program)
    return program


def _joint(
    program: MonadicProgram,
    sites: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
) -> torch.Tensor:
    """A program's log joint at some sites."""
    return program.log_joint(torch.zeros(1, 1), {**sites, **data}).sum()


def _reference(
    source: str, entry: str, sites: dict[str, float], data: dict[str, HostValue]
) -> float:
    """The reference machine's log joint at the same sites."""
    return float(loads(source).run(entry, data=data, sites=sites, seed=0).log_joint)


def _check(
    source: str,
    by_hand: str,
    entry: str,
    points: tuple[dict[str, float], ...],
    data: dict[str, HostValue] | None = None,
) -> None:
    """Both encodings and the reference machine agree on value and gradient."""
    host = {"y": _Y4} if data is None else data
    tensors = {name: torch.as_tensor(value, dtype=_F64) for name, value in host.items()}
    kept, direct = _program(source), _program(by_hand)
    for point in points:
        leaves = {
            name: torch.tensor([value], dtype=_F64, requires_grad=True)
            for name, value in point.items()
        }
        kept_joint = _joint(kept, leaves, tensors)
        direct_joint = _joint(direct, leaves, tensors)
        assert float(kept_joint.detach()) == pytest.approx(
            float(direct_joint.detach()), rel=1e-9
        )
        assert float(kept_joint.detach()) == pytest.approx(
            _reference(source, entry, point, host), rel=1e-7
        )
        kept_gradient = torch.autograd.grad(kept_joint, list(leaves.values()))
        direct_gradient = torch.autograd.grad(direct_joint, list(leaves.values()))
        for name, left, right in zip(leaves, kept_gradient, direct_gradient):
            assert float(left) == pytest.approx(float(right), rel=1e-9, abs=1e-12)
            step = 1e-6
            moved = {
                sign: _joint(
                    kept,
                    {
                        other: value.detach() + (sign * step if other == name else 0.0)
                        for other, value in leaves.items()
                    },
                    tensors,
                )
                for sign in (1, -1)
            }
            difference = float(moved[1] - moved[-1]) / (2.0 * step)
            assert float(left) == pytest.approx(difference, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# A program that calls a handled computation
# ---------------------------------------------------------------------------


def test_the_stored_width_agrees_with_the_inlined_program() -> None:
    _check(STORE, STORE_BY_HAND, "uses_handler", ({"base": 0.4}, {"base": 1.7}))


def test_dropping_the_answer_changes_the_joint() -> None:
    changed = STORE.replace("resume(2.0)", "resume(2.5)")
    sites = {"base": torch.tensor([0.4])}
    data = {"y": torch.tensor(_Y4)}
    assert float(_joint(_program(changed), sites, data)) != pytest.approx(
        float(_joint(_program(STORE_BY_HAND), sites, data)), rel=1e-3
    )


@pytest.mark.parametrize("kernel", ["nuts", "hmc"])
def test_one_seed_draws_one_posterior_from_both_encodings(kernel: str) -> None:
    """The two programs are one density, so a sampler makes the same draws."""

    def draws(source: str) -> torch.Tensor:
        sampler = (
            NUTSKernel(max_tree_depth=5)
            if kernel == "nuts"
            else HMCKernel(step_size=0.1, num_steps=8)
        )
        result = MCMC(
            sampler, num_warmup=100, num_samples=100, num_chains=1, seed=0
        ).run(_program(source), torch.zeros(1, 1), {"y": torch.tensor(_Y4)})
        return result.samples["base"]

    kept, direct = draws(STORE), draws(STORE_BY_HAND)
    assert kept.std() > 0
    torch.testing.assert_close(kept, direct, rtol=1e-5, atol=1e-5)


def test_svi_optimizes_one_objective_for_both_encodings() -> None:
    def losses(source: str) -> list[float]:
        torch.manual_seed(0)
        program = _program(source)
        data = {"y": torch.tensor(_Y4)}
        guide = AutoNormalGuide(program, observed_names={"y"})
        svi = SVI(program, guide, torch.optim.Adam(guide.parameters(), lr=0.05), ELBO())
        return [float(svi.step(torch.zeros(1, 1), data)) for _ in range(30)]

    kept, direct = losses(STORE), losses(STORE_BY_HAND)
    assert kept == pytest.approx(direct, rel=1e-5)
    assert kept[0] != kept[-1]


# ---------------------------------------------------------------------------
# Each way of resuming
# ---------------------------------------------------------------------------

_HEAD = """
object Row : FinSet 4

program fitted : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
"""
_TAIL = """
    observe y : Row <- Normal(loc=m, scale=1.5)
    return y

export fitted
"""
_POINTS = ({"a": 0.31, "b": -0.72}, {"a": -0.45, "b": 0.18})

_FORMS: dict[str, tuple[str, str]] = {
    "grade 1 with a parameter": (
        """
define called(a : Real, b : Real) : Real !{} =
    handle st with cell(a) in
        let s <- perform st.get()
        perform st.put(s * b)
        let t <- perform st.get()
        return t + s
""",
        "let m = a * b + a + 0.5 * a * b",
    ),
    "grade 0 under an if": (
        """
define guard(x : Real) : Real !{ab} =
    if x > 0.0 then
        return x
    else
        let y <- perform ab.stop(x)
        return y

define called(a : Real, b : Real) : Real !{} =
    handle ab with stopped in
        let x <- guard(a)
        return x * b
""",
        "let m = select(a > 0.0, a * b, 0.25 * a)",
    ),
    "aff": (
        """
define called(a : Real, b : Real) : Real !{} =
    handle mb with checked in
        let x <- perform mb.check(a)
        return x * b
""",
        "let m = select(a > 0.0, a * b - 0.125, 1.0 - a)",
    ),
    "omega": (
        """
define called(a : Real, b : Real) : Real !{} =
    handle ch with mixed in
        let x <- perform ch.flip(a)
        return x * x * b + x
""",
        "let m = 0.25 * (a * a * b + a) + 0.75 * (a * a * b - a)",
    ),
    "out of tail position": (
        """
define called(a : Real, b : Real) : Real !{} =
    handle af with doubled in
        let x <- perform af.after(a)
        return x * b
""",
        "let m = 0.5 * a * b + 0.5 * a",
    ),
    "a product answer": (
        """
handler final for State[Real] : Real -> Real * Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, s)
    get resumes 1 =>
        resume(s)
    put(v : Real) resumes 1 =>
        resume() with (v)

define stepped(a : Real, b : Real) : Real * Real !{} =
    handle st with final(a) in
        let s <- perform st.get()
        perform st.put(s + b)
        return 2.0 * s

define called(a : Real, b : Real) : Real !{} =
    let p <- stepped(a, b)
    return p[0] * p[1]
""",
        "let m = 2.0 * a * (a + b)",
    ),
}


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_a_form_is_fit_as_the_program_written_by_hand(form: str) -> None:
    definitions, by_hand = _FORMS[form]
    _check(
        HANDLERS + definitions + _HEAD + "    let m <- called(a, b)" + _TAIL,
        _HEAD + f"    {by_hand}" + _TAIL,
        "fitted",
        _POINTS,
    )


def test_a_product_result_is_read_by_component_in_the_program() -> None:
    definitions, _ = _FORMS["a product answer"]
    source = (
        HANDLERS
        + definitions
        + _HEAD
        + "    let p <- stepped(a, b)\n    let m = p[0] * p[1]"
        + _TAIL
    )
    _check(source, _HEAD + "    let m = 2.0 * a * (a + b)" + _TAIL, "fitted", _POINTS)


def test_a_choice_among_the_entries_of_a_tensor_is_a_log_sum() -> None:
    """A clause that resumes once per entry is the marginal of the choice."""
    source = """
effect Choose
    choose : Tensor[Real]([3]) -> Int

handler marginal for Choose : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    choose(logits : Tensor[Real]([3])) resumes omega =>
        let options = [0, 1, 2]
        let answers <- traverse(options, k -> resume(k))
        let weights = log_softmax(logits)
        return logsumexp([weights[0] + answers[0], weights[1] + answers[1], weights[2] + answers[2]])

instance component : Choose

define mixture(y : Real, logits : Tensor[Real]([3]), mean : Tensor[Real]([3])) : Real !{} =
    handle component with marginal in
        let k <- perform component.choose(logits)
        return 0.0 - 0.5 * (y - mean[k]) * (y - mean[k])

object Component : FinSet 3
object Row : FinSet 1

program mixed : Row -> Component
    sample logits : Component <- Normal(loc=0.0, scale=1.0)
    sample mean : Component <- Normal(loc=0.0, scale=2.0)
    let l <- mixture(1.25, logits, mean)
    score s = l
    return mean

export mixed
"""
    program = _program(source)
    logits = torch.tensor([0.3, -0.4, 0.1], dtype=_F64, requires_grad=True)
    mean = torch.tensor([-1.0, 0.5, 2.0], dtype=_F64, requires_grad=True)
    joint = _joint(program, {"logits": logits, "mean": mean}, {})
    oracle = (
        torch.distributions.Normal(0.0, 1.0).log_prob(logits).sum()
        + torch.distributions.Normal(0.0, 2.0).log_prob(mean).sum()
        + torch.logsumexp(
            torch.log_softmax(logits, 0) - 0.5 * (1.25 - mean).square(), 0
        )
    )
    assert float(joint.detach()) == pytest.approx(float(oracle.detach()), rel=1e-9)
    assert float(
        loads(source)
        .run(
            "mixed",
            data={},
            sites={"logits": (0.3, -0.4, 0.1), "mean": (-1.0, 0.5, 2.0)},
            seed=0,
        )
        .log_joint
    ) == pytest.approx(float(oracle.detach()), rel=1e-9)
    for left, right in zip(
        torch.autograd.grad(joint, (logits, mean)),
        torch.autograd.grad(oracle, (logits, mean)),
        strict=True,
    ):
        torch.testing.assert_close(left, right, rtol=1e-9, atol=1e-12)


def test_a_traversed_callee_with_a_handler_is_a_comprehension() -> None:
    source = (
        HANDLERS
        + """
define scored(x : Real, a : Real) : Real !{} =
    handle st with cell(a) in
        let s <- perform st.get()
        perform st.put(s + x)
        let t <- perform st.get()
        return t * x

object Row : FinSet 4

program fitted(t : Tensor[Real]([Row])) : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- traverse(t, x -> scored(x, a))
    observe y : Row <- Normal(loc=m, scale=1.5)
    return y

export fitted
"""
    )
    by_hand = """
object Row : FinSet 4

program fitted(t : Tensor[Real]([Row])) : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m = (a + t) * t + 0.5 * (a + t)
    observe y : Row <- Normal(loc=m, scale=1.5)
    return y

export fitted
"""
    _check(
        source,
        by_hand,
        "fitted",
        ({"a": 0.3}, {"a": -0.8}),
        {"y": _Y4, "t": (0.1, -0.6, 1.2, 0.7)},
    )


def test_a_callee_that_draws_under_a_handler_keeps_its_sites() -> None:
    source = (
        """
instance random : Random
instance score : Score
"""
        + HANDLERS
        + """
define drawn(a : Real) : Real !{random} =
    handle st with cell(a) in
        let z <- perform random.sample[Real](site("z"), Normal(a, 1.0))
        let s <- perform st.get()
        perform st.put(s + z)
        let w <- perform random.sample[Real](site("w"), Normal(s, 2.0))
        let t <- perform st.get()
        return t + w

object Row : FinSet 4

program fitted : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- drawn(a)
    observe y : Row <- Normal(loc=m, scale=1.5)
    return y

export fitted
"""
    )
    program = _program(source)
    values = {"a": 0.3, "m$z": -0.4, "m$w": 0.9}
    joint = _joint(
        program,
        {name: torch.tensor([value], dtype=_F64) for name, value in values.items()},
        {"y": torch.tensor(_Y4, dtype=_F64)},
    )
    a, z, w = values.values()
    location = (a + z) + w + 0.5 * (a + z)

    def normal(value: object, loc: float, scale: float) -> float:
        return float(
            torch.distributions.Normal(torch.tensor(loc, dtype=_F64), scale)
            .log_prob(torch.as_tensor(cast(float, value), dtype=_F64))
            .sum()
        )

    expected = (
        normal(a, 0.0, 1.0)
        + normal(z, a, 1.0)
        + normal(w, a, 2.0)
        + normal(_Y4, location, 1.5)
    )
    assert float(joint) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# One analysis, with its effects and without
# ---------------------------------------------------------------------------

#: A discourse in which each sentence may introduce a referent, a definite reads
#: the last one, presupposition failure gives a floor, and an indefinite is a
#: choice between two values that a listener averages.
DISCOURSE = """
effect Store
    latest : Unit -> Real
    introduce : Real -> Unit

effect Failure
    undefined : Unit -> Real

effect Choice
    either : Real * Real -> Real

instance referents : Store
instance presupposition : Failure
instance indefinite : Choice

handler remembering for Store : Real -> Real with (r : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    latest resumes 1 =>
        resume(r)
    introduce(v : Real) resumes 1 =>
        resume() with (v)

handler floored for Failure : Real -> Real with (floor : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    undefined resumes 0 =>
        return floor

handler averaged for Choice : Real -> Real with (weight : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    either(left : Real, right : Real) resumes omega =>
        let l <- resume(left)
        let r <- resume(right)
        return weight * l + (1.0 - weight) * r

define definite(threshold : Real) : Real !{referents, presupposition} =
    let x <- perform referents.latest()
    let checked <- above(x, threshold)
    return checked

define above(x : Real, threshold : Real) : Real !{presupposition} =
    if x > threshold then
        return x
    else
        let y <- perform presupposition.undefined()
        return y

define sentence(a : Real, b : Real) : Real !{referents, presupposition, indefinite} =
    let x <- perform indefinite.either(a, b)
    perform referents.introduce(x)
    let d <- definite(0.0)
    return 2.0 * d + x

define judged(a : Real, b : Real, w : Real) : Real !{} =
    handle indefinite with averaged(w) in
        handle presupposition with floored(-1.0) in
            handle referents with remembering(0.0) in
                let m <- sentence(a, b)
                return m

object Row : FinSet 4

program judgments : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample w <- Beta(concentration1=2.0, concentration0=2.0)
    let m <- judged(a, b, w)
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export judgments
"""

DISCOURSE_ELIMINATED = """
object Row : FinSet 4

program judgments : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample w <- Beta(concentration1=2.0, concentration0=2.0)
    let left = select(a > 0.0, 3.0 * a, -1.0)
    let right = select(b > 0.0, 3.0 * b, -1.0)
    let m = w * left + (1.0 - w) * right
    observe y : Row <- Normal(loc=m, scale=1.0)
    return y

export judgments
"""


def test_an_analysis_kept_in_effects_is_the_analysis_without_them() -> None:
    _check(
        DISCOURSE,
        DISCOURSE_ELIMINATED,
        "judgments",
        (
            {"a": 0.8, "b": -0.3, "w": 0.35},
            {"a": -0.4, "b": 0.6, "w": 0.7},
            {"a": -0.4, "b": -1.6, "w": 0.2},
        ),
    )


def test_moving_the_choice_inside_the_failure_is_another_analysis() -> None:
    """The order of the handlers is the scope of the effects."""
    swapped = DISCOURSE.replace(
        """    handle indefinite with averaged(w) in
        handle presupposition with floored(-1.0) in
            handle referents""",
        """    handle presupposition with floored(-1.0) in
        handle indefinite with averaged(w) in
            handle referents""",
    )
    sites = {"a": 0.8, "b": -0.3, "w": 0.35}
    outer = _reference(DISCOURSE, "judgments", sites, {"y": _Y4})
    inner = _reference(swapped, "judgments", sites, {"y": _Y4})
    assert inner != pytest.approx(outer, rel=1e-3)
    joint = _joint(
        _program(swapped),
        {name: torch.tensor([value], dtype=_F64) for name, value in sites.items()},
        {"y": torch.tensor(_Y4, dtype=_F64)},
    )
    assert float(joint) == pytest.approx(inner, rel=1e-7)


# ---------------------------------------------------------------------------
# Random handler programs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 2, 3, 4, 5, 8, 10, 11, 14, 16])
def test_a_random_handler_program_compiles_to_its_reference_joint(seed: int) -> None:
    source = (
        HANDLERS
        + random_body(seed)
        + """
object Row : FinSet 4

program fitted : Row -> Row
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let m <- handled(a, b)
    observe y : Row <- Normal(loc=m, scale=2.0)
    return y

export fitted
"""
    )
    program = _program(source)
    for point in ({"a": 0.3137, "b": -0.7171}, {"a": -0.9049, "b": 0.8117}):
        joint = _joint(
            program,
            {name: torch.tensor([value], dtype=_F64) for name, value in point.items()},
            {"y": torch.tensor(_Y4, dtype=_F64)},
        )
        assert float(joint) == pytest.approx(
            _reference(source, "fitted", point, {"y": _Y4}), rel=1e-9
        )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

_REFUSED = {
    "a foreign handler": (
        """
handler outer for Maybe : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    check(v : Real) resumes 1

define called(a : Real, b : Real) : Real !{} =
    handle mb with outer in
        let x <- perform mb.check(a)
        return x + b
""",
        "calls `called`, which installs the foreign handler `outer`, whose clause "
        "`check` a runtime provider implements",
    ),
    "a handled request in a recursive call": (
        """
define loop(n : Int, x : Real) : Real !{rd} =
    if n <= 0 then
        return x
    else
        let r <- perform rd.ask()
        loop(n - 1, x + r)

define called(a : Real, b : Real) : Real !{} =
    handle rd with reader(a) in
        let x <- loop(3, b)
        return x
""",
        "calls `called`, which reaches a recursive call of `loop` that requests "
        "from the handler `reader`",
    ),
}


@pytest.mark.parametrize("refusal", sorted(_REFUSED))
def test_a_call_that_is_not_discharged_is_refused_naming_the_clause(
    refusal: str,
) -> None:
    definitions, reason = _REFUSED[refusal]
    source = HANDLERS + definitions + _HEAD + "    let m <- called(a, b)" + _TAIL
    program = loads(source).morphism
    assert isinstance(program, CheckedProgram)
    message = str(program.refusal)
    assert "`let m <- called(...)`" in message
    assert reason in message


_UNHANDLED = {
    "a forwarding handler": (
        """
handler half for State[Real] : Real -> Real [coverage=partial, forwards=unknown, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(1.0)

define called(a : Real, b : Real) : Real !{st} =
    handle st with half in
        perform st.put(a)
        return b
""",
        "performs on instance `st : State[Real]`, which the program's row cannot "
        "carry (handler `half` has no clause `put` and leaves it to an outer handler)",
    ),
    "an effect of a clause body": (
        """
handler logging for Maybe : Real -> Real [coverage=total, forwards=none, introduces=!{wr}, implementation=authored]
    return x =>
        return x
    check(v : Real) resumes 1 =>
        perform wr.tell(v)
        resume(v)

define called(a : Real, b : Real) : Real !{wr} =
    handle mb with logging in
        let x <- perform mb.check(a)
        return x + b
""",
        "performs on instance `wr : Tell`, which the program's row cannot carry "
        "(clause `check` of handler `logging` requests it)",
    ),
}


@pytest.mark.parametrize("unhandled", sorted(_UNHANDLED))
def test_an_instance_a_program_cannot_handle_is_refused_naming_the_clause(
    unhandled: str,
) -> None:
    definitions, reason = _UNHANDLED[unhandled]
    source = HANDLERS + definitions + _HEAD + "    let m <- called(a, b)" + _TAIL
    with pytest.raises(QiecDiagnosticError) as raised:
        loads(source)
    assert reason in str(raised.value)
