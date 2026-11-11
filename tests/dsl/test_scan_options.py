"""Scans that return every state (``outputs=all``) and run right to left (``reverse=true``).

The oracle is a hand-written torch loop over the recurrence. The compiled
runtime runs in float32 and is compared with the float64 oracle at
``rel=1e-5``; the reference machine runs in float64 and is compared at
``rel=1e-9``. Each oracle comparison has a deliberately wrong variant
(the forward order for a reversed scan, a trajectory paired in visit
order rather than input order) that the same check must reject.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.dsl import loads
from quivers.dsl.compiler import CompileError
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import ParseError, parse
from quivers.inference import trace

T = 4

COMPILED_REL = 1e-5
REFERENCE_REL = 1e-9

STACKED = """
object In : Real 2
object State : Real 2

program lower(x_t, h_prev) : In * State -> State
    let m_lower = 0.5 * h_prev + x_t
    sample a <- Normal(loc=m_lower, scale=0.5)
    return a

program upper(a_t, g_prev) : State * State -> State
    let m_upper = 0.4 * g_prev + a_t
    sample b <- Normal(loc=m_upper, scale=0.4)
    observe y <- Normal(loc=b, scale=0.3)
    return b

define deep = scan(lower, outputs=all) >> scan(upper)

program model : In -> State
    sample top <- deep
    return top

export model
"""


def _stacked_oracle(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    upper_reads_final_state: bool = False,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    previous = torch.zeros(2, dtype=torch.float64)
    for t in range(T):
        total = total + D.Normal(0.5 * previous + x[t], 0.5).log_prob(a[t]).sum()
        previous = a[t]
    previous = torch.zeros(2, dtype=torch.float64)
    for t in range(T):
        read = a[T - 1] if upper_reads_final_state else a[t]
        total = total + D.Normal(0.4 * previous + read, 0.4).log_prob(b[t]).sum()
        total = total + D.Normal(b[t], 0.3).log_prob(y[t]).sum()
        previous = b[t]
    return total


def test_stacked_scans_compiled_runtime_match_the_oracle() -> None:
    program = loads(STACKED)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, T, 2, generator=generator)
    y = torch.randn(1, T, 2, generator=generator)
    torch.manual_seed(1)
    recorded = trace(program.morphism, x, observations={"y": y})
    a = [recorded.sites[f"a@{t}"].value.reshape(2).double() for t in range(T)]
    b = [recorded.sites[f"b@{t}"].value.reshape(2).double() for t in range(T)]
    expected = _stacked_oracle(a, b, x[0].double(), y[0].double())
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=COMPILED_REL)
    assert torch.equal(recorded.output, recorded.sites[f"b@{T - 1}"].value)  # type: ignore[arg-type]


def test_stacked_scans_reference_machine_matches_the_oracle() -> None:
    program = loads(STACKED)
    generator = torch.Generator().manual_seed(2)
    x = torch.randn(T, 2, generator=generator, dtype=torch.float64)
    y = torch.randn(T, 2, generator=generator, dtype=torch.float64)
    a = [torch.randn(2, generator=generator, dtype=torch.float64) for _ in range(T)]
    b = [torch.randn(2, generator=generator, dtype=torch.float64) for _ in range(T)]
    sites: dict[str, object] = {f"a@{t}": tuple(a[t].tolist()) for t in range(T)}
    sites.update({f"b@{t}": tuple(b[t].tolist()) for t in range(T)})
    run = program.run(
        "model",
        data={
            "in": tuple(tuple(row) for row in x.tolist()),
            "y": tuple(tuple(row) for row in y.tolist()),
        },
        sites=sites,
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_stacked_oracle(a, b, x, y)), rel=REFERENCE_REL
    )
    assert run.value == pytest.approx(tuple(b[T - 1].tolist()), rel=REFERENCE_REL)


def test_the_upper_scan_reads_the_lower_trajectory_not_its_final_state() -> None:
    # An upper scan fed the lower scan's final state at every position
    # would condition b@t on a@{T-1}; the oracle with that substitution
    # must disagree with the compiled joint.
    program = loads(STACKED)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, T, 2, generator=generator)
    y = torch.randn(1, T, 2, generator=generator)
    torch.manual_seed(4)
    recorded = trace(program.morphism, x, observations={"y": y})
    a = [recorded.sites[f"a@{t}"].value.reshape(2).double() for t in range(T)]
    b = [recorded.sites[f"b@{t}"].value.reshape(2).double() for t in range(T)]
    right = _stacked_oracle(a, b, x[0].double(), y[0].double())
    wrong = _stacked_oracle(
        a, b, x[0].double(), y[0].double(), upper_reads_final_state=True
    )
    assert float(recorded.log_joint) == pytest.approx(float(right), rel=COMPILED_REL)
    assert float(recorded.log_joint) != pytest.approx(float(wrong), rel=1e-3)


REVERSED = """
object In : Real 1
object State : Real 1

program cell(x_t, h_prev) : In * State -> State
    let m = 0.8 * h_prev + x_t
    sample h <- Normal(loc=m, scale=0.5)
    observe o <- Normal(loc=h, scale=0.3)
    return h

define backward = scan(cell, reverse=true, outputs=all)

program model : In -> State
    sample hs <- backward
    return hs

export model
"""


def _reversed_oracle(
    visited: list[torch.Tensor],
    x: torch.Tensor,
    o: torch.Tensor,
    *,
    order: list[int] | None = None,
) -> torch.Tensor:
    """Score a right-to-left chain; ``visited[k]`` is the ``k``-th state drawn."""
    positions = order if order is not None else list(range(T - 1, -1, -1))
    total = torch.zeros((), dtype=torch.float64)
    previous = torch.zeros((), dtype=torch.float64)
    for k, p in enumerate(positions):
        total = total + D.Normal(0.8 * previous + x[p], 0.5).log_prob(visited[k])
        total = total + D.Normal(visited[k], 0.3).log_prob(o[p])
        previous = visited[k]
    return total


def test_reversed_scan_compiled_runtime_matches_the_oracle() -> None:
    program = loads(REVERSED)
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(1, T, 1, generator=generator)
    o = torch.randn(1, T, 1, generator=generator)
    torch.manual_seed(6)
    recorded = trace(program.morphism, x, observations={"o": o})
    visited = [recorded.sites[f"h@{k}"].value.reshape(()).double() for k in range(T)]
    xs, os_ = x.reshape(T).double(), o.reshape(T).double()
    expected = _reversed_oracle(visited, xs, os_)
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=COMPILED_REL)
    # The forward reading of the same draws is a different joint.
    forward = _reversed_oracle(visited, xs, os_, order=list(range(T)))
    assert float(recorded.log_joint) != pytest.approx(float(forward), rel=1e-3)


def test_reversed_trajectory_is_laid_out_in_input_order() -> None:
    program = loads(REVERSED)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, T, 1, generator=generator)
    o = torch.randn(1, T, 1, generator=generator)
    torch.manual_seed(8)
    recorded = trace(program.morphism, x, observations={"o": o})
    output = recorded.output
    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (1, T, 1)
    for k in range(T):
        site = recorded.sites[f"h@{k}"].value.reshape(1, 1)
        assert torch.equal(output[:, T - 1 - k], site)


def test_reversed_scan_reads_each_observation_at_its_input_position() -> None:
    # `o` given as one sequence and `o@k` given per visit agree when
    # `o@k` holds the value at input position T - 1 - k.
    program = loads(REVERSED)
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(1, T, 1, generator=generator)
    o = torch.randn(1, T, 1, generator=generator)
    states = {f"h@{k}": torch.randn(1, 1, generator=generator) for k in range(T)}
    whole = program.morphism.log_joint(x, {**states, "o": o})
    split = program.morphism.log_joint(
        x, {**states, **{f"o@{k}": o[:, T - 1 - k] for k in range(T)}}
    )
    misread = program.morphism.log_joint(
        x, {**states, **{f"o@{k}": o[:, k] for k in range(T)}}
    )
    assert float(whole) == pytest.approx(float(split), rel=COMPILED_REL)
    assert float(whole) != pytest.approx(float(misread), rel=1e-3)


def test_reversed_scan_reference_machine_matches_the_oracle() -> None:
    program = loads(REVERSED)
    generator = torch.Generator().manual_seed(10)
    x = torch.randn(T, generator=generator, dtype=torch.float64)
    o = torch.randn(T, generator=generator, dtype=torch.float64)
    visited = [
        torch.randn((), generator=generator, dtype=torch.float64) for _ in range(T)
    ]
    run = program.run(
        "model",
        data={
            "in": tuple((float(v),) for v in x),
            "o": tuple((float(v),) for v in o),
        },
        sites={f"h@{k}": (float(visited[k]),) for k in range(T)},
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_reversed_oracle(visited, x, o)), rel=REFERENCE_REL
    )
    expected = tuple((float(visited[T - 1 - p]),) for p in range(T))
    assert run.value == expected


BIDIRECTIONAL = """
object In : Real 1
object State : Real 1
object Pair : Real 2

program forward_cell(x_t, h_prev) : In * State -> State
    let m_f = 0.8 * h_prev + x_t
    sample f <- Normal(loc=m_f, scale=0.5)
    return f

program backward_cell(x_t, h_prev) : In * State -> State
    let m_g = 0.6 * h_prev - x_t
    sample g <- Normal(loc=m_g, scale=0.7)
    return g

program tag_cell(p_t, s_prev) : Pair * State -> State
    let m_s = p_t[0] - p_t[1] + 0.3 * s_prev
    sample s <- Normal(loc=m_s, scale=0.6)
    observe y <- Normal(loc=s, scale=0.3)
    return s

define both = fan(scan(forward_cell, outputs=all), scan(backward_cell, outputs=all, reverse=true))
define tagger = both >> scan(tag_cell, outputs=all)

program model : In -> State
    sample tags <- tagger
    return tags

export model
"""


def _bidirectional_oracle(
    values: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    pair_in_visit_order: bool = False,
) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64)
    forward: list[torch.Tensor] = []
    previous = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        f = values[f"f@{t}"]
        total = total + D.Normal(0.8 * previous + x[t], 0.5).log_prob(f)
        forward.append(f)
        previous = f
    backward: list[torch.Tensor] = [torch.zeros((), dtype=torch.float64)] * T
    previous = torch.zeros((), dtype=torch.float64)
    for k in range(T):
        p = T - 1 - k
        g = values[f"g@{k}"]
        total = total + D.Normal(0.6 * previous - x[p], 0.7).log_prob(g)
        backward[p] = g
        previous = g
    if pair_in_visit_order:
        backward = backward[::-1]
    previous = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        s = values[f"s@{t}"]
        total = total + D.Normal(
            forward[t] - backward[t] + 0.3 * previous, 0.6
        ).log_prob(s)
        total = total + D.Normal(s, 0.3).log_prob(y[t])
        previous = s
    return total


def test_bidirectional_compiled_runtime_matches_the_oracle() -> None:
    program = loads(BIDIRECTIONAL)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(1, T, 1, generator=generator)
    y = torch.randn(1, T, 1, generator=generator)
    torch.manual_seed(12)
    recorded = trace(program.morphism, x, observations={"y": y})
    values = {
        f"{label}@{t}": recorded.sites[f"{label}@{t}"].value.reshape(()).double()
        for label in "fgs"
        for t in range(T)
    }
    xs, ys = x.reshape(T).double(), y.reshape(T).double()
    expected = _bidirectional_oracle(values, xs, ys)
    assert float(recorded.log_joint) == pytest.approx(float(expected), rel=COMPILED_REL)
    misaligned = _bidirectional_oracle(values, xs, ys, pair_in_visit_order=True)
    assert float(recorded.log_joint) != pytest.approx(float(misaligned), rel=1e-3)


def test_bidirectional_reference_machine_matches_the_oracle() -> None:
    program = loads(BIDIRECTIONAL)
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(T, generator=generator, dtype=torch.float64)
    y = torch.randn(T, generator=generator, dtype=torch.float64)
    values = {
        f"{label}@{t}": torch.randn((), generator=generator, dtype=torch.float64)
        for label in "fgs"
        for t in range(T)
    }
    run = program.run(
        "model",
        data={
            "in": tuple((float(v),) for v in x),
            "y": tuple((float(v),) for v in y),
        },
        sites={name: (float(value),) for name, value in values.items()},
        seed=0,
    )
    assert run.log_joint == pytest.approx(
        float(_bidirectional_oracle(values, x, y)), rel=REFERENCE_REL
    )
    assert run.value == tuple((float(values[f"s@{t}"]),) for t in range(T))


def test_the_two_paths_agree_on_a_batch_of_runs() -> None:
    # Two rows are two independent runs; each row's compiled joint is the
    # reference machine's joint at the same values.
    program = loads(BIDIRECTIONAL)
    generator = torch.Generator().manual_seed(14)
    x = torch.randn(2, T, 1, generator=generator)
    y = torch.randn(2, T, 1, generator=generator)
    values = {
        f"{label}@{t}": torch.randn(2, 1, generator=generator)
        for label in "fgs"
        for t in range(T)
    }
    compiled = program.morphism.log_joint(x, {**values, "y": y})
    assert tuple(compiled.shape) == (2,)
    for row in range(2):
        run = program.run(
            "model",
            data={
                "in": tuple((float(v),) for v in x[row, :, 0]),
                "y": tuple((float(v),) for v in y[row, :, 0]),
            },
            sites={name: (float(value[row, 0]),) for name, value in values.items()},
            seed=0,
        )
        assert run.log_joint == pytest.approx(float(compiled[row]), rel=COMPILED_REL)


# -- refusals ---------------------------------------------------------------

HEAD = """
object Token : FinSet 5
object Embedded : Real 2
object Hidden : Real 2
morphism tok_embed : Token -> Embedded [role=embed]
morphism c1 : Embedded * Hidden -> Hidden ~ Normal
morphism c2 : Hidden * Hidden -> Hidden ~ Normal
morphism head : Hidden -> Token ~ Categorical
"""


def _program(define: str) -> str:
    return (
        HEAD
        + f"define chain = {define}\n"
        + """
program lm : Token -> Hidden
    sample s <- chain
    return s

export lm
"""
    )


def test_a_scan_over_a_final_state_is_refused() -> None:
    with pytest.raises(CompileError, match="one value rather than a sequence"):
        loads(_program("tok_embed >> scan(c1) >> scan(c2)"))


def test_a_morphism_after_every_state_outside_a_scan_is_refused() -> None:
    with pytest.raises(CompileError, match="every-state sequence"):
        loads(_program("tok_embed >> scan(c1, outputs=all) >> head"))


def test_pairing_a_sequence_with_one_value_is_refused() -> None:
    with pytest.raises(CompileError, match="pair sequences with sequences"):
        loads(
            _program(
                "fan(tok_embed >> scan(c1, outputs=all), tok_embed >> scan(c1)) "
                ">> scan(c2)"
            )
        )


def test_a_morphism_between_two_scans_runs_at_every_position() -> None:
    program = loads(_program("tok_embed >> scan(c1, outputs=all) >> scan(c2)"))
    run = trace(program.morphism, torch.randint(0, 5, (1, T)))
    assert {f"s@{t}" for t in range(T)} <= set(run.sites)
    assert {f"s$s_chain_1@{t}" for t in range(T)} <= set(run.sites)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("outputs=every", "outputs are 'last'"),
        ("reverse=true, reverse=false", "given twice"),
        ("outputs=all, outputs=last", "given twice"),
        ("init=h0, init=h1", "given twice"),
    ],
)
def test_malformed_scan_options_are_refused(option: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse(f"define s = scan(cell, {option})\n")


def test_scan_options_round_trip_through_the_emitter() -> None:
    source = "define s = scan(cell, init=h0, outputs=all, reverse=true)\n"
    module = parse(source)
    (statement,) = module.statements
    assert getattr(statement, "expr").outputs == "all"
    assert getattr(statement, "expr").reverse is True
    assert parse(module_to_source(module)) == module
    plain = parse("define s = scan(cell)\n")
    assert "scan(cell)" in module_to_source(plain)


# -- ScanMorphism ---------------------------------------------------------


def _cell() -> ConditionalNormal:
    torch.manual_seed(15)
    a = Euclidean(name="input", dim=2)
    h = Euclidean(name="hidden", dim=3)
    return ConditionalNormal(ProductSpace(components=(a, h)), h)


def _cell_oracle(
    cell: ConditionalNormal,
    x: torch.Tensor,
    trajectory: torch.Tensor,
    order: list[int],
) -> torch.Tensor:
    total = torch.zeros(x.shape[0])
    previous = torch.zeros(x.shape[0], 3)
    for t in order:
        total = total + cell.log_prob(
            torch.cat([x[:, t], previous], -1), trajectory[:, t]
        )
        previous = trajectory[:, t]
    return total


def test_scan_morphism_reverse_scores_the_right_to_left_chain() -> None:
    cell = _cell()
    scan = ScanMorphism(cell, reverse=True, outputs="all")
    x = torch.randn(3, T, 2)
    torch.manual_seed(16)
    trajectory = scan.rsample(x)
    assert tuple(trajectory.shape) == (3, T, 3)
    expected = _cell_oracle(cell, x, trajectory, list(range(T - 1, -1, -1)))
    torch.testing.assert_close(scan.log_prob(x, trajectory), expected)
    forward = _cell_oracle(cell, x, trajectory, list(range(T)))
    assert not torch.allclose(scan.log_prob(x, trajectory), forward, rtol=1e-3)


def test_scan_morphism_last_output_is_the_state_after_the_last_position_visited() -> (
    None
):
    cell = _cell()
    x = torch.randn(2, T, 2)
    for reverse, index in ((False, T - 1), (True, 0)):
        every = ScanMorphism(cell, outputs="all", reverse=reverse)
        last = ScanMorphism(cell, outputs="last", reverse=reverse)
        dimension = every.base_dimension(x)
        assert dimension is not None
        base = torch.randn(2, dimension)
        trajectory = every.push_base(x, base)
        torch.testing.assert_close(last.push_base(x, base), trajectory[:, index])
        value, density = every.push_base_scored(x, base)
        torch.testing.assert_close(value, trajectory)
        torch.testing.assert_close(density, every.log_prob(x, trajectory))


def test_scan_morphism_rejects_an_unknown_output() -> None:
    with pytest.raises(ValueError, match="outputs"):
        ScanMorphism(_cell(), outputs="every")  # type: ignore[arg-type]


def test_a_standalone_scan_define_compiles_to_a_scan_morphism() -> None:
    source = (
        HEAD + "define states = scan(c1, outputs=all, reverse=true)\nexport states\n"
    )
    program = loads(source)
    morphism = program.morphism
    assert isinstance(morphism, ScanMorphism)
    assert morphism.outputs == "all" and morphism.reverse
