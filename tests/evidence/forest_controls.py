"""Checks and deliberately wrong versions of packed forests.

The forest's derivations are compared with the derivation oracle (O1):
the attachment grammar's trees with their occurrences, reachability over
parallel edges whose occurrences must stay apart, and a cyclic graph
unfolded within a height. Folds are compared with sums over the oracle's
trees: linear folds with the two-pass value, a nonlinear fold with the
alternatives ``-1`` and ``1``, and expected embeddings with a fold
computed on each tree, including an encoder of terms.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import torch

from quivers.core.algebras import LOG_PROB, MAX_PLUS, REAL
from quivers.qiec.contracts import (
    OccurrenceId,
    ParameterId,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.rules import (
    Derivation,
    HostInterpretation,
    Item,
    PremiseValue,
    presentation_of,
)
from quivers.stochastic.forest import (
    DerivationForest,
    ForestRefusal,
    ForestRefused,
    build_forest,
    edge_weight,
    encoder_interpretation,
    expected_embedding,
    fold_derivations,
    forest_fold,
    nonlinear_fold,
)
from quivers.structural.encoder import Encoder, PerOpFn, make_default_op_fn
from quivers.structural.signature import (
    Constructor,
    Signature,
    Sort,
    Term,
)
from tests.evidence import graphs
from tests.evidence import presentations as fixture
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import derivations as oracle

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar weights."""

type Inputs = dict[ParameterId | OccurrenceId, torch.Tensor]
"""Input weights."""


def _require(condition: bool, what: str) -> None:
    """Fail a check with an infinite deviation.

    Parameters
    ----------
    condition : bool
        What has to hold.
    what : str
        The failure.

    Raises
    ------
    Mismatch
        If the condition does not hold.
    """
    if not condition:
        raise Mismatch("value", math.inf, what)


def _weights_only(forest: DerivationForest) -> HostInterpretation[None]:
    """The interpretation that folds every derivation to nothing.

    Parameters
    ----------
    forest : DerivationForest
        The forest.

    Returns
    -------
    HostInterpretation
        One constant operation per rule and slot.
    """

    def nothing(node: Derivation, premises: tuple[PremiseValue[None], ...]) -> None:
        """Fold a node to nothing.

        Parameters
        ----------
        node : Derivation
            The node.
        premises : tuple of PremiseValue
            Its premises' folds.
        """
        del node, premises

    return HostInterpretation(
        presentation=forest.presentation,
        operations={key: nothing for key in forest.presentation.constructors_by_id},
    )


def _derivations(
    forest: DerivationForest,
    goal: Item,
    inputs: Inputs,
    *,
    height: int | None = None,
    limit: int = 10_000,
) -> list[tuple[Derivation, torch.Tensor]]:
    """Every derivation of an item and its weight in the reals.

    Parameters
    ----------
    forest : DerivationForest
        The forest.
    goal : Item
        The item.
    inputs : Inputs
        The weights.
    height : int or None
        The bound on height.
    limit : int
        The declared limit.

    Returns
    -------
    list of tuple of Derivation and torch.Tensor
        The derivations and weights.
    """
    return [
        (derivation, weight)
        for derivation, weight, _ in fold_derivations(
            forest,
            goal,
            REAL,
            inputs,
            _weights_only(forest),
            limit=limit,
            like=LIKE,
            height=height,
        )
    ]


def _real_inputs() -> Inputs:
    """The attachment grammar's weights in the reals.

    Returns
    -------
    Inputs
        The weights.
    """
    inputs: Inputs = {
        ParameterId.derive(fixture.NAME, name): torch.tensor(
            float(value), dtype=torch.float64
        )
        for name, value in fixture.WEIGHTS.items()
    }
    for name, value in fixture.LEAF_WEIGHTS.items():
        inputs[fixture.occurrence_id(name)] = torch.tensor(
            float(value), dtype=torch.float64
        )
    return inputs


def _goal() -> Item:
    """The attachment grammar's goal.

    Returns
    -------
    Item
        ``span(S, 0, 5)``.
    """
    return Item(family=fixture.SPAN, arguments=fixture.GOAL)


def _trees() -> list[oracle.Tree]:
    """The oracle's trees of the goal.

    Returns
    -------
    list of Tree
        Every tree.
    """
    return oracle.derivations(
        fixture.oracle_axioms(),
        fixture.oracle_rules(),
        fixture.GOAL,
        fixture.oracle_sides(),
    )


def check_derivations() -> None:
    """Require the forest's derivations of the goal to be the oracle's trees (O1).

    Each derivation, with its occurrences, is one oracle tree and its
    weight that tree's; the count, the linear fold and the best value
    agree with the oracle and with the two-pass run.

    Raises
    ------
    Mismatch
        If a tree is missing or extra, or a weight, count or fold differs.
    """
    forest = build_forest(fixture.PRESENTATION, fixture.OCCURRENCES)
    goal = _goal()
    inputs = _real_inputs()
    listed = _derivations(forest, goal, inputs)
    trees = _trees()
    shapes = sorted((fixture.shape(derivation) for derivation, _ in listed), key=repr)
    _require(shapes == sorted(trees, key=repr), f"{len(shapes)} trees for {len(trees)}")
    _require(forest.count(goal) == len(trees), "the count differs")
    weights = fixture.oracle_weights()
    for derivation, weight in listed:
        compare_number(
            float(weight),
            oracle.total(
                [fixture.shape(derivation)], weights, Fraction(0), Fraction(1)
            ),
            Accuracy(domain="linear", operations=32, magnitude=1.0),
            "a derivation's weight",
        )
    log_inputs: Inputs = {key: value.log() for key, value in inputs.items()}
    for algebra, expected in (
        (LOG_PROB, math.log(oracle.total(trees, weights, Fraction(0), Fraction(1)))),
        (
            MAX_PLUS,
            math.log(
                max(
                    oracle.total([tree], weights, Fraction(0), Fraction(1))
                    for tree in trees
                )
            ),
        ),
    ):
        folded = forest_fold(
            forest,
            goal,
            algebra,
            lambda edge, algebra=algebra: edge_weight(
                forest, edge, algebra, log_inputs, like=LIKE
            ),
            like=LIKE,
        )
        compare_number(
            float(folded),
            expected,
            Accuracy(domain="log", operations=64),
            f"the {algebra.subject.name} fold",
        )


def _edge_row(index: int) -> str:
    """The oracle's name of an edge occurrence.

    Parameters
    ----------
    index : int
        The edge's position.

    Returns
    -------
    str
        Its name.
    """
    return f"edge{index}"


def _graph_oracle(
    edges: Sequence[tuple[int, int]], goal: int, height: int | None = None
) -> list[oracle.Tree]:
    """The oracle's trees of ``reach(goal)`` over a graph.

    Parameters
    ----------
    edges : Sequence of tuple of int and int
        The edges.
    goal : int
        The node.
    height : int or None
        The oracle's bound on rule levels.

    Returns
    -------
    list of Tree
        Every tree.
    """
    i, j = oracle.Variable("i"), oracle.Variable("j")
    rules: tuple[oracle.RuleRow, ...] = (
        ("step", (("reach", i), ("edge", i, j)), ("reach", j), "step"),
    )
    axioms: list[oracle.AxiomRow] = [("start", ("reach", 0))]
    axioms += [(_edge_row(index), ("edge", a, b)) for index, (a, b) in enumerate(edges)]
    return oracle.derivations(axioms, rules, ("reach", goal), height=height)


def _graph_shape(derivation: Derivation) -> oracle.Tree:
    """A reachability derivation in the oracle's form.

    Parameters
    ----------
    derivation : Derivation
        The derivation.

    Returns
    -------
    Tree
        The tree.
    """
    form = "reach" if derivation.item.family == graphs.REACH else "edge"
    item = (form, *derivation.item.arguments)
    if derivation.occurrence is not None:
        name = (
            "start"
            if derivation.occurrence == OccurrenceId.derive(graphs.NAME, "start")
            else next(
                _edge_row(index)
                for index in range(64)
                if graphs.edge_id(index) == derivation.occurrence
            )
        )
        return (name, item, name, ())
    children = tuple(
        _graph_shape(child)
        for child in derivation.premises
        if isinstance(child, Derivation)
    )
    return ("step", item, "step", children)


def check_occurrences() -> None:
    """Require parallel occurrences of one edge to stay distinct derivations (M30).

    With two occurrences of each of ``0 -> 1`` and ``1 -> 2``, node 2 has
    four derivations, as the oracle counts them, each naming its
    occurrences.

    Raises
    ------
    Mismatch
        If the forest's trees are not the oracle's.
    """
    edges = [(0, 1), (0, 1), (1, 2), (1, 2)]
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    listed = _derivations(forest, graphs.node(2), {})
    trees = _graph_oracle(edges, 2)
    found = sorted((_graph_shape(derivation) for derivation, _ in listed), key=repr)
    if found != sorted(trees, key=repr):
        raise Mismatch(
            "value",
            float(abs(len(found) - len(trees)) or 1),
            f"{len(found)} derivations for the oracle's {len(trees)}",
        )
    _require(forest.count(graphs.node(2)) == 4, "the count is not four")


def check_cyclic_height() -> None:
    """Require a cyclic forest unfolded within a height to give the oracle's trees.

    On ``0 -> 1 -> 0 -> ...`` node 1 has one derivation per odd number
    of steps; without a bound the forest refuses.

    Raises
    ------
    Mismatch
        If a bounded unfolding differs from the oracle's, or the unbounded
        one is not refused.
    """
    edges = [(0, 1), (1, 0)]
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    _require(not forest.acyclic, "the cycle was not found")
    for levels in range(1, 7):
        trees = _graph_oracle(edges, 1, height=levels)
        listed = _derivations(forest, graphs.node(1), {}, height=levels + 1)
        found = sorted((_graph_shape(derivation) for derivation, _ in listed), key=repr)
        _require(
            found == sorted(trees, key=repr),
            f"height {levels}: {len(found)} trees for {len(trees)}",
        )
    try:
        forest.count(graphs.node(1))
    except ForestRefused as refused:
        _require(refused.code is ForestRefusal.UNBOUNDED, f"refused as {refused.code}")
        return
    raise Mismatch("refusal", math.inf, "a cyclic forest was counted without a bound")


def check_nonlinear() -> None:
    """Require a nonlinear fold to sum over derivations, not apply to their sum (M35).

    Two paths of weights ``-1`` and ``1``: the sum of their squares is
    two, the square of their sum zero.

    Raises
    ------
    Mismatch
        If the fold is not two, or the linear fold is not zero.
    """
    edges = graphs.diamonds(1)
    inputs: Inputs = {**graphs.edge_weights(edges, (-1.0, 1.0, 1.0, 1.0))}
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    goal = graphs.node(3)
    squared = nonlinear_fold(
        forest,
        goal,
        REAL,
        inputs,
        lambda derivation, weight: weight * weight,
        limit=2,
        like=LIKE,
    )
    compare_number(
        float(squared), 2.0, Accuracy(domain="linear", operations=8), "sum of squares"
    )
    linear = forest_fold(
        forest,
        goal,
        REAL,
        lambda edge: edge_weight(forest, edge, REAL, inputs, like=LIKE),
        like=LIKE,
    )
    compare_number(float(linear), 0.0, Accuracy(domain="linear", operations=8), "sum")


def check_limit() -> None:
    """Require a goal with more derivations than the limit to be refused, not cut.

    Five diamonds give 32 paths: a limit of 32 lists them all, 31 refuses.

    Raises
    ------
    Mismatch
        If the limit is not honoured exactly.
    """
    edges = graphs.diamonds(5)
    forest = build_forest(graphs.PRESENTATION, graphs.occurrences(edges))
    goal = graphs.node(15)
    _require(len(_derivations(forest, goal, {}, limit=32)) == 32, "32 were not listed")
    try:
        _derivations(forest, goal, {}, limit=31)
    except ForestRefused as refused:
        _require(refused.code is ForestRefusal.TOO_MANY, f"refused as {refused.code}")
        return
    raise Mismatch("refusal", math.inf, "a limit of 31 returned derivations")


type Operation = Callable[
    [Derivation, tuple[PremiseValue[torch.Tensor], ...]], torch.Tensor
]
"""An operation of a fold into vectors."""

_MATRICES = {
    name: torch.randn(
        (2, 4), generator=torch.Generator().manual_seed(index), dtype=torch.float64
    )
    for index, name in enumerate(sorted(set(fixture.RULE_NAMES.values())))
}
"""A fixed matrix per rule of the attachment grammar."""

_WORDS = {
    name: torch.tensor([float(index), 1.0], dtype=torch.float64)
    for index, name in enumerate(sorted(fixture.LEAF_WEIGHTS))
}
"""A fixed vector per word."""


def _rule_operation(name: str) -> Operation:
    """The operation ``tanh(M_r [left; right])`` of one binary rule.

    Parameters
    ----------
    name : str
        The rule.

    Returns
    -------
    Operation
        The operation.
    """

    def operation(
        node: Derivation, premises: tuple[PremiseValue[torch.Tensor], ...]
    ) -> torch.Tensor:
        """Embed a rule node from its premises' embeddings.

        Parameters
        ----------
        node : Derivation
            The node.
        premises : tuple of PremiseValue
            The two premises' embeddings.

        Returns
        -------
        torch.Tensor
            The embedding.
        """
        del node
        left, right = premises
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        return torch.tanh(_MATRICES[name] @ torch.cat([left, right]))

    return operation


def _word_operation(
    node: Derivation, premises: tuple[PremiseValue[torch.Tensor], ...]
) -> torch.Tensor:
    """Embed a word by its occurrence.

    Parameters
    ----------
    node : Derivation
        The input node.
    premises : tuple of PremiseValue
        None.

    Returns
    -------
    torch.Tensor
        The word's vector.
    """
    del premises
    assert node.occurrence is not None
    return _WORDS[fixture.NAMES[node.occurrence]]


def _embed_tree(tree: oracle.Tree) -> torch.Tensor:
    """The same fold on an oracle tree.

    Parameters
    ----------
    tree : Tree
        The tree.

    Returns
    -------
    torch.Tensor
        Its embedding.
    """
    name, _, _, children = tree
    if not children:
        return _WORDS[name]
    left, right = (_embed_tree(child) for child in children)
    return torch.tanh(_MATRICES[name] @ torch.cat([left, right]))


def check_expected_embedding() -> None:
    """Require the expected embedding to be the weighted mean over the oracle's trees.

    Each word embeds by its occurrence and each rule as
    ``tanh(M_r [left; right])``, which is not linear, so the mean is taken
    over derivations; it is compared with the same fold computed on every
    oracle tree, weighted by the tree's weight.

    Raises
    ------
    Mismatch
        If the expectation differs.
    """
    forest = build_forest(fixture.PRESENTATION, fixture.OCCURRENCES)
    operations: dict[ConstructorId, Operation] = {
        key: _rule_operation(name) for key, name in fixture.RULE_NAMES.items()
    }
    operations[fixture.WORD.id] = _word_operation
    interpretation = HostInterpretation(
        presentation=fixture.PRESENTATION, operations=operations
    )
    found = expected_embedding(
        forest,
        _goal(),
        REAL,
        _real_inputs(),
        interpretation,
        probability=lambda weight: weight,
        limit=1000,
        like=LIKE,
    )
    weights = fixture.oracle_weights()
    trees = _trees()
    masses = [
        float(oracle.total([tree], weights, Fraction(0), Fraction(1))) for tree in trees
    ]
    expected = sum(
        (mass * _embed_tree(tree) for mass, tree in zip(masses, trees, strict=True)),
        torch.zeros(2, dtype=torch.float64),
    ) / sum(masses)
    deviation = float((found - expected).abs().max())
    if deviation > 1e-12:
        raise Mismatch("value", deviation, f"expected embedding {found.tolist()}")


TREES = Signature(
    name="Trees",
    sorts_t=(Sort(name="T", kind="object"),),
    constructors_t=(
        Constructor(name="Leaf", domain=(), codomain="T"),
        Constructor(name="Node", domain=("T", "T"), codomain="T"),
    ),
)
"""Binary trees, whose presentation derives ``T`` from itself."""


def _encoder() -> Encoder:
    """An encoder of binary trees into four dimensions.

    Returns
    -------
    Encoder
        The encoder, with fixed parameters.
    """
    torch.manual_seed(0)
    functions: dict[str, PerOpFn] = {}
    modules: list[torch.nn.Module] = []
    for op, dims in (("Leaf", ()), ("Node", (4, 4))):
        module, function = make_default_op_fn(op, dims, 4)
        modules.append(module)
        functions[op] = PerOpFn(
            op=op,
            mode="plain",
            args=tuple(f"x{i}" for i in range(len(dims))),
            fn=function,
        )
    return Encoder(
        name="trees",
        signature=TREES,
        sort_dims={"T": 4},
        op_fns=functions,
        var_init_fns={},
        data_embedders={},
        modules_owned=modules,
    ).double()


def _terms(height: int) -> list[Term]:
    """Every binary tree of at most ``height`` levels, a leaf having one.

    Parameters
    ----------
    height : int
        The bound.

    Returns
    -------
    list of Term
        The trees.
    """
    if height < 1:
        return []
    smaller = _terms(height - 1)
    return [Term(op="Leaf", args=())] + [
        Term(op="Node", args=(left, right)) for left in smaller for right in smaller
    ]


def check_encoder_embedding() -> None:
    """Require an encoder's expected embedding over a cyclic forest within a height.

    The presentation of binary trees derives ``T`` from itself; within
    height three its derivations are the five trees of three levels, and
    the mean of the encoder's fold over them equals the mean of the
    encoder applied to each tree, enumerated on its own.

    Raises
    ------
    Mismatch
        If the count or the mean differs.
    """
    encoder = _encoder()
    presentation = presentation_of(TREES)
    forest = build_forest(presentation, [])
    goal = Item(family=presentation.judgments[0].id, arguments=())
    terms = _terms(3)
    _require(forest.count(goal, height=3) == len(terms), "the count of trees differs")
    with torch.no_grad():
        found = expected_embedding(
            forest,
            goal,
            REAL,
            {},
            encoder_interpretation(encoder, presentation),
            probability=lambda weight: weight,
            limit=len(terms),
            like=LIKE,
            height=3,
        )
        expected = torch.stack([encoder(term) for term in terms]).mean(0)
    deviation = float((found - expected).abs().max())
    if deviation > 1e-12:
        raise Mismatch("value", deviation, f"encoder mean {found.tolist()}")


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="forest", question="Q23", run=run)
    for name, run in (
        ("forest/derivations", check_derivations),
        ("forest/occurrences", check_occurrences),
        ("forest/cyclic-height", check_cyclic_height),
        ("forest/nonlinear", check_nonlinear),
        ("forest/limit", check_limit),
        ("forest/expected-embedding", check_expected_embedding),
        ("forest/encoder-embedding", check_encoder_embedding),
    )
}
"""Every check of packed forests by name."""

FOREST = "quivers.stochastic.forest"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="forest-occurrences-merged",
        defect_class="provenance",
        provenance="two occurrences of one item at one slot and binding become one edge",
        target=f"{FOREST}:_add_input",
        rewrites=(
            Rewrite(
                old="edges.setdefault(item, []).append(edge)",
                new=(
                    "if all(\n"
                    "        other.constructor != edge.constructor\n"
                    "        or other.binding != edge.binding\n"
                    "        for other in edges.get(item, [])\n"
                    "    ):\n"
                    "        edges.setdefault(item, []).append(edge)"
                ),
            ),
        ),
        rejected_by=("forest/occurrences",),
        min_effect=1.0,
    ),
    Mutation(
        name="forest-nonlinear-fold-on-aggregate",
        defect_class="observer",
        provenance="the nonlinear fold is applied to the sum of the weights",
        target=f"{FOREST}:nonlinear_fold",
        also_in=("tests.evidence.forest_controls",),
        rewrites=(
            Rewrite(
                old="total = _join(algebra, total, observe(derivation, weight))",
                new="total = _join(algebra, total, weight)",
            ),
            Rewrite(old="return total", new="return observe(terms[0][0], total)"),
        ),
        rejected_by=("forest/nonlinear",),
        min_effect=1.0,
    ),
    Mutation(
        name="forest-limit-not-checked",
        defect_class="premature stop",
        provenance="the declared limit is not checked before enumerating",
        target=f"{FOREST}:fold_derivations_many",
        rewrites=(
            Rewrite(
                old="if total > limit:\n        raise ForestRefused(",
                new="if False:\n        raise ForestRefused(",
            ),
        ),
        rejected_by=("forest/limit",),
        min_effect=1.0,
    ),
)
"""Every wrong version of packed forests and the checks that reject it."""
