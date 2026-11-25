"""Each oracle against a second computation, and against its own wrong variants."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
import torch

from tests.evidence.oracles import (
    closed_forms,
    derivations,
    exact,
    finite_effects,
    gradients,
    paths,
    trees,
)
from tests.evidence.oracles.derivations import Variable
from tests.evidence.oracles.paths import Arc, Machine, Mark
from tests.evidence.problems import DEDUCTIONS, trees_of

i, j, k = Variable("i"), Variable("j"), Variable("k")


def bracketings(words: int) -> list[derivations.Tree]:
    """Every binary bracketing of a row of words.

    Parameters
    ----------
    words : int
        The number of words.

    Returns
    -------
    list of Tree
        The trees.
    """
    return derivations.derivations(
        [(f"w{p}", ("span", p, p + 1)) for p in range(words)],
        [("join", (("span", i, k), ("span", k, j)), ("span", i, j), "r")],
        ("span", 0, words),
    )


@pytest.mark.parametrize("words", [1, 2, 3, 4, 5, 6])
def test_bracketings_are_counted_by_catalan_numbers(words: int) -> None:
    """Compare the trees of a row of words with the Catalan numbers.

    Parameters
    ----------
    words : int
        The number of words.
    """
    assert len(bracketings(words)) == closed_forms.catalan(words - 1)


def test_known_families_have_their_closed_form_counts() -> None:
    """Compare the trees of three problems with closed forms."""
    assert len(trees_of("skip")) == closed_forms.fibonacci(6)
    assert len(trees_of("diamond")) == 2**3
    assert len(trees_of("edit_lattice")) == closed_forms.delannoy(2, 3)
    assert len(trees_of("ambiguous")) == closed_forms.catalan(4)
    assert len(trees_of("impossible")) == 0


def test_equal_axioms_are_two_occurrences() -> None:
    """Require two axioms with one item to give two trees each time they are used."""
    found = derivations.derivations(
        [("one", ("NP",)), ("two", ("NP",))],
        [("pair", (("NP",), ("NP",)), ("S",), None)],
        ("S",),
    )
    assert len(found) == 4
    assert len(set(found)) == 4


def test_a_variable_bound_to_none_is_bound() -> None:
    """Require a variable bound to ``None`` to match only ``None``."""
    pattern = ("p", Variable("x"), Variable("x"))
    assert derivations.bind(pattern, ("p", None, 3), {}) is None
    assert derivations.bind(pattern, ("p", None, None), {}) == {"x": None}
    assert derivations.bind(("p", 1), ("p", True), {}) is None


def test_cycles_are_refused_and_bounded_on_request() -> None:
    """Require a cyclic system to be refused without a height bound."""
    axioms = [("a", ("A",))]
    rules = [("loop", (("A",),), ("A",), "c")]
    with pytest.raises(derivations.CyclicDerivations):
        derivations.derivations(axioms, rules, ("A",))
    for height in range(5):
        found = derivations.derivations(axioms, rules, ("A",), height=height)
        assert len(found) == height + 1


def test_sums_agree_with_the_rational_recurrence() -> None:
    """Compare the sum over parse trees with the cubic recurrence."""
    binary = {(0, 1, 1): Fraction(1, 3), (1, 1, 1): Fraction(1, 4), (0, 0, 1): Fraction(1, 5)}
    lexical = {(1, 0): Fraction(1, 2), (0, 0): Fraction(1, 7), (1, 1): Fraction(2, 9)}
    for sentence in itertools.product((0, 1), repeat=4):
        axioms = [
            (f"{p}:{a}", ("span", a, p, p + 1))
            for p, token in enumerate(sentence)
            for a, t in lexical
            if t == token
        ]
        rules = [
            (f"{a}{b}{c}", (("span", b, i, k), ("span", c, k, j)), ("span", a, i, j), f"{a}{b}{c}")
            for a, b, c in binary
        ]
        weights = {f"{a}{b}{c}": w for (a, b, c), w in binary.items()}
        weights |= {name: lexical[(int(name[-1]), sentence[int(name[0])])] for name, _ in axioms}
        enumerated = derivations.total(
            derivations.derivations(axioms, rules, ("span", 0, 0, 4)),
            weights,
            Fraction(0),
            Fraction(1),
        )
        assert enumerated == exact.inside(binary, lexical, {}, sentence, 2)[0]


def deduplicated(found: list[derivations.Tree]) -> int:
    """A wrong count that merges trees with equal items.

    Parameters
    ----------
    found : list of Tree
        The trees.

    Returns
    -------
    int
        The number of trees once leaves are read by item.
    """

    def strip(tree: derivations.Tree) -> derivations.Tree:
        """A tree with its leaves named by their items.

        Parameters
        ----------
        tree : Tree
            The tree.

        Returns
        -------
        Tree
            The stripped tree.
        """
        name, item, weight, children = tree
        if not children and weight == name:
            return ("leaf", item, None, ())
        return (name, item, weight, tuple(strip(child) for child in children))

    return len({strip(tree) for tree in found})


def test_wrong_variants_of_the_derivation_oracle_are_rejected() -> None:
    """Require merged occurrences, a maximum, and an omitted tree to be told apart."""
    found = derivations.derivations(
        [("one", ("NP",)), ("two", ("NP",))],
        [("pair", (("NP",), ("NP",)), ("S",), None)],
        ("S",),
    )
    assert deduplicated(found) != len(found)
    rational = DEDUCTIONS["equal_proofs"].rational()
    proofs = trees_of("equal_proofs")
    assert derivations.best(proofs, rational) != derivations.total(
        proofs, rational, Fraction(0), Fraction(1)
    )
    assert derivations.total(proofs[:-1], rational, Fraction(0), Fraction(1)) != (
        derivations.total(proofs, rational, Fraction(0), Fraction(1))
    )


def test_a_nonlinear_observer_is_summed_per_tree() -> None:
    """Compare the expected square of proof values with its hand value."""
    values = {"minus": Fraction(-1), "plus": Fraction(1)}
    proofs = [("minus", ("v",), "minus", ()), ("plus", ("v",), "plus", ())]
    found = derivations.nonlinear_fold(
        proofs,
        dict.fromkeys(values, Fraction(1)),
        lambda name, children: values[name],
        lambda value: value * value,
    )
    assert found == 2
    aggregate = sum(values.values(), Fraction(0)) ** 2
    assert aggregate != found


def test_logarithmic_gradients_agree_with_automatic_differentiation() -> None:
    """Compare expected counts with autograd through the oracle's own sum."""
    problem = DEDUCTIONS["diamond"]
    proofs = trees_of("diamond")
    rational = problem.rational()

    def log_weight(point: dict[str, torch.Tensor]) -> torch.Tensor:
        """The log weight as a function of log weights.

        Parameters
        ----------
        point : dict of str to torch.Tensor
            The log weights.

        Returns
        -------
        torch.Tensor
            The log weight.
        """
        weights = {name: value.exp() for name, value in point.items()}
        zero = torch.zeros((), dtype=torch.float64)
        return derivations.total(proofs, weights, zero, zero + 1).log()

    point = {name: math.log(float(value)) for name, value in rational.items()}
    found = gradients.autograd(log_weight, point)
    for name in rational:
        exact_value = derivations.logarithmic_gradient(proofs, rational, name)
        assert found[name] == pytest.approx(float(exact_value), rel=1e-14, abs=1e-15)
        slope = gradients.richardson(
            lambda p: float(log_weight({n: torch.tensor(v, dtype=torch.float64) for n, v in p.items()})),
            point,
            name,
            step=2.0**-6,
        )
        assert abs(slope.value - float(exact_value)) <= slope.error


def line_machine(parallel: bool) -> Machine:
    """A machine that writes ``a`` and then nothing, maybe by two arcs.

    Parameters
    ----------
    parallel : bool
        Whether the empty step has two parallel arcs.

    Returns
    -------
    Machine
        The machine.
    """
    arcs = [
        Arc(name="a", source=0, target=1, labels=("a", "x"), weight="p"),
        Arc(name="e", source=1, target=2, labels=(None, "y"), weight="q"),
    ]
    if parallel:
        arcs.append(Arc(name="f", source=1, target=2, labels=(None, "y"), weight="q"))
    return Machine(
        arcs=tuple(arcs),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=2, weight="one"),),
    )


def epsilon_pair() -> tuple[Machine, Machine]:
    """Two machines that each read or write nothing on the shared tape.

    Returns
    -------
    tuple of (Machine, Machine)
        The machines, ``a:x`` then ``ε:y`` and ``x:u`` then ``ε... ``.
    """
    first = Machine(
        arcs=(
            Arc(name="a", source=0, target=1, labels=("a", "m"), weight="p"),
            Arc(name="e", source=1, target=2, labels=("b", None), weight="q"),
        ),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=2, weight="one"),),
    )
    second = Machine(
        arcs=(
            Arc(name="m", source=0, target=1, labels=("m", "u"), weight="r"),
            Arc(name="e", source=1, target=2, labels=(None, "v"), weight="s"),
        ),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=2, weight="one"),),
    )
    return first, second


WEIGHTS = {"p": Fraction(1, 2), "q": Fraction(1, 3), "r": Fraction(2, 5), "s": Fraction(3, 4), "one": Fraction(1)}


def naive_composition(left: Machine, right: Machine) -> Machine:
    """A composition of two machines without an epsilon filter.

    Moves on the shared tape that read nothing on one side are taken in
    every interleaving, so one pair of paths is counted several times.

    Parameters
    ----------
    left : Machine
        The first machine.
    right : Machine
        The second machine.

    Returns
    -------
    Machine
        The product machine.
    """
    states = 10
    arcs = []
    for a in left.arcs:
        for b in right.arcs:
            if a.labels[1] is not None and a.labels[1] == b.labels[0]:
                arcs.append(
                    Arc(
                        name=f"{a.name}{b.name}",
                        source=a.source * states + b.source,
                        target=a.target * states + b.target,
                        labels=(a.labels[0], b.labels[1]),
                        weight=f"{a.weight}*{b.weight}",
                    )
                )
    for a in left.arcs:
        if a.labels[1] is None:
            for state in range(3):
                arcs.append(
                    Arc(
                        name=f"{a.name}-{state}",
                        source=a.source * states + state,
                        target=a.target * states + state,
                        labels=(a.labels[0], None),
                        weight=a.weight,
                    )
                )
    for b in right.arcs:
        if b.labels[0] is None:
            for state in range(3):
                arcs.append(
                    Arc(
                        name=f"-{state}{b.name}",
                        source=state * states + b.source,
                        target=state * states + b.target,
                        labels=(None, b.labels[1]),
                        weight=b.weight,
                    )
                )
    return Machine(
        arcs=tuple(arcs),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=2 * states + 2, weight="one"),),
    )


def product_weights() -> dict[str, Fraction]:
    """The weights of the product arcs.

    Returns
    -------
    dict of str to Fraction
        Products of pairs of weights by their names.
    """
    answer = dict(WEIGHTS)
    for x, y in itertools.product(WEIGHTS, repeat=2):
        answer[f"{x}*{y}"] = WEIGHTS[x] * WEIGHTS[y]
    return answer


def test_composition_without_a_filter_is_rejected_in_counting() -> None:
    """Require the relation oracle to reject repeated epsilon interleavings.

    In the Boolean reading the two agree, so an idempotent run is not a
    witness on its own.
    """
    left, right = epsilon_pair()
    reference = paths.compose_relations(
        paths.relation(left, WEIGHTS, tapes=2), paths.relation(right, WEIGHTS, tapes=2)
    )
    naive = paths.relation(naive_composition(left, right), product_weights(), tapes=2)
    assert set(naive) == set(reference)
    assert naive != reference
    assert {key: value > 0 for key, value in naive.items()} == {
        key: value > 0 for key, value in reference.items()
    }


def test_parallel_epsilon_arcs_are_two_paths() -> None:
    """Require two parallel arcs that write nothing to be two paths."""
    single = paths.relation(line_machine(False), WEIGHTS | {"q": Fraction(1, 3)}, tapes=2)
    double = paths.relation(line_machine(True), WEIGHTS | {"q": Fraction(1, 3)}, tapes=2)
    assert len(paths.accepting_paths(line_machine(True))) == 2
    assert double == {key: 2 * value for key, value in single.items()}


def test_merging_states_with_different_futures_is_rejected() -> None:
    """Require a machine with two states merged to relate other strings."""
    apart = Machine(
        arcs=(
            Arc(name="a", source=0, target=1, labels=("a",), weight="p"),
            Arc(name="b", source=0, target=2, labels=("b",), weight="p"),
            Arc(name="c", source=1, target=3, labels=("c",), weight="q"),
            Arc(name="d", source=2, target=3, labels=("d",), weight="q"),
        ),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=3, weight="one"),),
    )
    merged = apart.with_(
        arcs=tuple(
            arc.with_(source=1 if arc.source == 2 else arc.source, target=1 if arc.target == 2 else arc.target)
            for arc in apart.arcs
        )
    )
    assert paths.relation(apart, WEIGHTS, tapes=1) != paths.relation(merged, WEIGHTS, tapes=1)


def test_cyclic_machines_are_refused_without_a_bound() -> None:
    """Require a cyclic machine to be refused and a bounded one to be counted."""
    loop = Machine(
        arcs=(Arc(name="l", source=0, target=0, labels=("a",), weight="p"),),
        initial=(Mark(state=0, weight="one"),),
        final=(Mark(state=0, weight="one"),),
    )
    with pytest.raises(paths.CyclicMachine):
        paths.accepting_paths(loop)
    found = paths.accepting_paths(loop, max_arcs=6)
    assert len(found) == 7
    partial = paths.total(found, WEIGHTS, Fraction(0), Fraction(1))
    assert closed_forms.geometric(WEIGHTS["p"]) - partial == WEIGHTS["p"] ** 7 * 2


def test_conditional_masses_sum_to_one() -> None:
    """Require a normalized relation to have mass one for every input."""
    table = paths.relation(line_machine(True), WEIGHTS, tapes=2)
    for x in {x for x, _ in table}:
        mass = sum(v for (a, _), v in paths.conditional(table).items() if a == x)
        assert mass == 1


@pytest.mark.parametrize("nodes", [2, 3, 4, 5])
def test_spanning_trees_are_counted_and_summed_twice(nodes: int) -> None:
    """Compare tree enumeration with Cayley's count and the matrix-tree theorem.

    Parameters
    ----------
    nodes : int
        The number of nodes.
    """
    found = trees.spanning_trees(nodes, 0)
    assert len(found) == nodes ** (nodes - 2)
    weights = {
        (x, y): Fraction(1 + (3 * x + 5 * y) % 7, 1 + (x + 2 * y) % 5)
        for x in range(nodes)
        for y in range(nodes)
        if x != y and y != 0
    }
    total = trees.partition(found, weights, Fraction(0), Fraction(1))
    assert total == exact.tree_partition(nodes, 0, weights)
    marginals = trees.edge_marginals(found, weights)
    for child in range(1, nodes):
        assert sum(v for (x, y), v in marginals.items() if y == child) == 1
    if nodes > 2:
        flipped = {(y, x): w for (x, y), w in weights.items() if x != 0}
        flipped |= {(0, y): weights[(0, y)] for y in range(1, nodes)}
        assert exact.tree_partition(nodes, 0, flipped) != total
        assert exact.tree_partition(nodes, 1, {**weights, **{(y, 0): Fraction(1) for y in range(1, nodes)}}) != total


def test_exact_closures_agree_with_their_series_and_closed_form() -> None:
    """Compare the rational inverse with the closed form and with partial sums."""
    stay, forth, back, rest = Fraction(1, 4), Fraction(1, 2), Fraction(1, 3), Fraction(1, 5)
    matrix = [[stay, forth], [back, rest]]
    closed = closed_forms.two_state_star(stay, forth, back, rest)
    assert [list(row) for row in closed] == exact.linear_star(matrix)
    power = exact.identity(2)
    partial = exact.identity(2)
    for _ in range(60):
        power = [[sum((power[r][m] * matrix[m][c] for m in range(2)), Fraction(0)) for c in range(2)] for r in range(2)]
        partial = [[partial[r][c] + power[r][c] for c in range(2)] for r in range(2)]
    gap = max(abs(float(closed[r][c] - partial[r][c])) for r in range(2) for c in range(2))
    assert 0 < gap < 1e-12
    with pytest.raises(closed_forms.Divergent):
        closed_forms.two_state_star(Fraction(1, 2), Fraction(1), Fraction(1, 2), Fraction(1, 2))


def test_quadratic_least_root_is_the_limit_from_zero() -> None:
    """Require Kleene iteration from zero to increase to the least root."""
    for branch in (Fraction(1, 5), Fraction(3, 5)):
        x = Fraction(0)
        for _ in range(200):
            nxt = branch * x * x + (1 - branch)
            assert nxt >= x
            x = Fraction(round(float(nxt) * 2**60), 2**60)
        root = closed_forms.least_quadratic_root(branch)
        assert abs(float(x - root)) < 1e-9


def polya(draws: finite_effects.Draws, heads: int, tails: int) -> int:
    """Draw from a Polya urn and score a fixed sequence.

    Parameters
    ----------
    draws : Draws
        The run.
    heads : int
        The ones in the sequence.
    tails : int
        The zeros.

    Returns
    -------
    int
        Whether the run drew the sequence.
    """
    a, b = Fraction(2), Fraction(3)
    sequence = [1] * heads + [0] * tails
    for position, wanted in enumerate(sequence):
        value = draws.choose(f"x{position}", (b / (a + b), a / (a + b)))
        if value != wanted:
            return 0
        a, b = (a + 1, b) if value else (a, b + 1)
    return 1


def test_beta_counts_are_the_law_of_an_urn() -> None:
    """Compare the Beta closed form with an enumerated Polya urn."""
    for heads, tails in ((0, 0), (1, 0), (2, 1), (3, 2)):
        law = finite_effects.law(lambda d: polya(d, heads, tails))
        assert law.get(1, Fraction(0)) == closed_forms.beta_binomial(
            heads, tails, Fraction(2), Fraction(3)
        )
        assert closed_forms.dirichlet_categorical(
            (tails, heads), (Fraction(3), Fraction(2))
        ) == closed_forms.beta_binomial(heads, tails, Fraction(2), Fraction(3))


def test_two_state_chain_agrees_with_the_forward_recursion() -> None:
    """Compare the closed form of a symmetric chain with the recursion."""
    start, stay, emit = Fraction(1, 3), Fraction(3, 4), Fraction(4, 5)
    for symbols in itertools.product((0, 1), repeat=4):
        recursion = exact.forward(
            (start, 1 - start),
            ((stay, 1 - stay), (1 - stay, stay)),
            ((emit, 1 - emit), (1 - emit, emit)),
            symbols,
        )
        assert closed_forms.two_state_hmm(start, stay, emit, symbols) == recursion


def test_edit_distances_and_alignment_counts() -> None:
    """Compare the closed forms with hand values."""
    assert closed_forms.edit_distance("kitten", "sitting") == 3
    assert closed_forms.edit_distance("", "abc") == 3
    assert [closed_forms.delannoy(n, n) for n in range(5)] == [1, 3, 13, 63, 321]


def test_contraction_is_a_sum_over_assignments() -> None:
    """Compare the contraction oracle with a matrix product written out."""
    left = {(r, c): Fraction(r + 2 * c + 1, 7) for r in range(2) for c in range(3)}
    right = {(r, c): Fraction(3 * r + c + 1, 11) for r in range(3) for c in range(2)}
    found = exact.contraction(
        ["ij", "jk"], "ik", {"i": 2, "j": 3, "k": 2}, [left, right], Fraction(0), Fraction(1)
    )
    for r, c in itertools.product(range(2), repeat=2):
        assert found[(r, c)] == sum(left[(r, m)] * right[(m, c)] for m in range(3))


def test_boundary_derivatives_are_their_hand_values() -> None:
    """Compare the boundary cases with values computed here."""
    by_name = {case.name: case for case in gradients.BOUNDARY_CASES}
    closure = by_name["path_count_at_a_zero_edge"]
    a, b = Fraction(0), Fraction(0)
    assert closure.value == float(b / (1 - a))
    assert closure.gradient == (float(b / (1 - a) ** 2), float(1 / (1 - a)))
    assert by_name["sum_with_a_zero_summand"].value == 3 * (1 + 2 * 0)
    tie = by_name["tie"]
    assert sum(tie.gradient) == 1 and max(tie.gradient) == 1


def test_differences_estimate_their_own_error() -> None:
    """Require the extrapolated difference to be within its error estimate."""
    slope = gradients.richardson(lambda p: math.exp(3 * p["x"]), {"x": 0.25}, "x", step=0.01)
    assert abs(slope.value - 3 * math.exp(0.75)) <= slope.error
    coarse = gradients.richardson(lambda p: math.exp(3 * p["x"]), {"x": 0.25}, "x", step=0.5)
    assert coarse.error > slope.error


def fair(draws: finite_effects.Draws, shared: bool) -> int:
    """Two fair coins, or one coin read twice.

    Parameters
    ----------
    draws : Draws
        The run.
    shared : bool
        Whether the second coin is the first one.

    Returns
    -------
    int
        Whether the two readings agree.
    """
    half = (Fraction(1, 2), Fraction(1, 2))
    first = draws.choose("first", half)
    second = first if shared else draws.choose("second", half)
    return int(first == second)


def test_shared_and_independent_draws_have_different_laws() -> None:
    """Require the effect oracle to tell one draw from two."""
    assert finite_effects.law(lambda d: fair(d, False))[1] == Fraction(1, 2)
    assert finite_effects.law(lambda d: fair(d, True))[1] == Fraction(1)
    with pytest.raises(finite_effects.RepeatedAddress):
        finite_effects.runs(
            lambda d: d.choose("x", (Fraction(1),)) + d.choose("x", (Fraction(1),))
        )


def test_every_sequence_of_draws_is_run_once() -> None:
    """Require one run per sequence of values."""
    found = finite_effects.runs(
        lambda d: d.choose("a", (Fraction(1, 2),) * 2) + d.choose("b", (Fraction(1, 3),) * 3)
    )
    assert len(found) == 6
    assert len({outcome.values for outcome in found}) == 6
    assert sum(outcome.weight for outcome in found) == 1
    assert finite_effects.joint(
        lambda d: d.choose("a", (Fraction(1, 4), Fraction(3, 4))), {"a": 1}
    ) == Fraction(3, 4)
