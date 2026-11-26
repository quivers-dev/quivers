"""Rule presentations: checking, the reference interpretation, adapters."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import didactic.api as dx
import pytest
import torch

from quivers.qiec.checking import KernelError
from quivers.qiec.contracts import (
    Acyclic,
    LinearCyclic,
    MonotonePolynomial,
    PresentationId,
    check_complexity,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.module import validate_module
from quivers.qiec.rules import (
    PRESENTATION_WORK,
    PremiseValue,
    closed_values,
    CyclicDerivations,
    Derivation,
    HostInterpretation,
    ItemTable,
    Judgment,
    JudgmentDecl,
    Premise,
    ReferenceInterpretation,
    Rule,
    RulePresentation,
    SideCondition,
    presentation_of,
)
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import IndexVariable
from quivers.structural.encoder import Encoder, PerOpFn
from quivers.structural.signature import (
    Binder,
    BinderArgSpec,
    BinderVarSpec,
    Constructor,
    Signature,
    Sort,
)
from tests.evidence import presentations as fixture
from tests.evidence.oracles import derivations as oracle


def reference() -> ReferenceInterpretation:
    """The reference interpretation of the attachment sentence.

    Returns
    -------
    ReferenceInterpretation
        The oracle of the presentation.
    """
    return ReferenceInterpretation(fixture.PRESENTATION, fixture.OCCURRENCES)


def oracle_trees() -> list[oracle.Tree]:
    """The derivations of the goal by the independent oracle.

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


def test_presentation_lowers_to_a_checked_module() -> None:
    """Require the lowered families and constructors to pass the QIEC checker."""
    module = fixture.PRESENTATION.to_module()
    registry = validate_module(module)
    assert set(registry.families) == {fixture.SPAN}
    families, constructors = fixture.PRESENTATION.declarations()
    assert [constructor.name for constructor in constructors] == [
        *(row[0] for row in fixture.BINARY),
        "word",
    ]
    assert [field.name for field in constructors[0].fields] == ["premise0", "premise1"]
    assert families[0].constructors == tuple(item.id for item in constructors)


def test_the_qiec_checker_rejects_an_ill_sorted_conclusion() -> None:
    """Require a category where a position belongs to fail in the QIEC checker."""
    rule = fixture.PRESENTATION.rules[0]
    wrong = rule.with_(
        conclusion=Judgment(
            family=fixture.SPAN,
            arguments=(
                fixture.position("i"),
                fixture.position("i"),
                fixture.position("j"),
            ),
        )
    )
    with pytest.raises(dx.ValidationError, match="result index"):
        fixture.PRESENTATION.with_(rules=(wrong, *fixture.PRESENTATION.rules[1:]))
    with pytest.raises(KernelError):
        families, constructors = fixture.PRESENTATION.declarations()
        module = fixture.PRESENTATION.to_module().with_(
            constructors=(
                constructors[0].with_(result_indices=wrong.conclusion.arguments),
                *constructors[1:],
            )
        )
        validate_module(module)


@pytest.mark.parametrize(
    "change",
    [
        "unknown-family",
        "wrong-arity",
        "unbound-conclusion",
        "undeclared-variable",
        "unknown-condition",
        "condition-of-unbound",
        "duplicate-name",
    ],
)
def test_malformed_presentations_are_refused(change: str) -> None:
    """Require every malformed rule to be refused at construction.

    Parameters
    ----------
    change : str
        The defect.
    """
    rule = fixture.PRESENTATION.rules[0]
    other = FamilyId.derive("other")
    rules = {
        "unknown-family": rule.with_(conclusion=Judgment(family=other)),
        "wrong-arity": rule.with_(
            conclusion=Judgment(family=fixture.SPAN, arguments=(fixture.category("S"),))
        ),
        "unbound-conclusion": rule.with_(
            telescope=(*rule.telescope, IndexBinder(name="m", sort=NAT)),
            conclusion=fixture.span("S", "i", "m"),
        ),
        "undeclared-variable": rule.with_(conclusion=fixture.span("S", "i", "z")),
        "unknown-condition": rule.with_(
            side_condition=SideCondition(
                computation=fixture.BEFORE.id.derive("x"), arguments=("i", "j")
            )
        ),
        "condition-of-unbound": rule.with_(
            telescope=(*rule.telescope, IndexBinder(name="m", sort=NAT)),
            side_condition=SideCondition(
                computation=fixture.BEFORE.id, arguments=("i", "m")
            ),
        ),
        "duplicate-name": fixture.PRESENTATION.rules[1].with_(name=rule.name),
    }
    replaced = (rules[change], *fixture.PRESENTATION.rules[1:])
    with pytest.raises(dx.ValidationError):
        fixture.PRESENTATION.with_(rules=replaced)


def test_reference_derivations_match_the_oracle() -> None:
    """Require the same trees as the independent enumeration."""
    table = reference()
    found = table.derivations(table.item(fixture.SPAN, fixture.GOAL))
    assert len(found) == 4
    assert Counter(map(repr, (fixture.shape(item) for item in found))) == Counter(
        map(repr, oracle_trees())
    )


def test_equal_inputs_are_two_occurrences() -> None:
    """Require two occurrences of one item to give two leaves."""
    table = reference()
    item = table.item(fixture.SPAN, ("NP", 4, 5))
    leaves = table.derivations(item)
    assert len(leaves) == 2
    assert {leaf.occurrence for leaf in leaves} == {
        fixture.occurrence_id("w4"),
        fixture.occurrence_id("w5"),
    }


def test_weight_and_nonlinear_fold_match_the_oracle() -> None:
    """Require the weight and a nonlinear observer of a fold to match."""
    table = reference()
    item = table.item(fixture.SPAN, fixture.GOAL)
    weights = fixture.oracle_weights()
    trees = oracle_trees()
    total = table.observe(
        item, fixture.node_weight, lambda d: Fraction(1), lambda v: v, Fraction(0)
    )
    assert total == oracle.total(trees, weights, Fraction(0), Fraction(1))

    def algebra(name: str, children: tuple[Fraction, ...]) -> Fraction:
        """A fold whose rules are not symmetric in their premises.

        Parameters
        ----------
        name : str
            The rule or occurrence.
        children : tuple of Fraction
            The premise values.

        Returns
        -------
        Fraction
            ``2 left - right + 1`` for a rule, the weight for a leaf.
        """
        if not children:
            return weights[name]
        left, right = children
        return 2 * left - right + 1

    def operation(
        node: Derivation, values: tuple[PremiseValue[Fraction], ...]
    ) -> Fraction:
        """The same fold as an operation of the presentation.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple of PremiseValue
            The premise values.

        Returns
        -------
        Fraction
            The value.
        """
        if node.occurrence is not None:
            return fixture.LEAF_WEIGHTS[fixture.NAMES[node.occurrence]]
        left, right = closed_values(values)
        return 2 * left - right + 1

    interpretation = HostInterpretation[Fraction](
        presentation=fixture.PRESENTATION,
        operations={
            item.id: operation
            for item in (*fixture.PRESENTATION.rules, *fixture.PRESENTATION.axioms)
        },
    )
    squared = table.observe(
        item, fixture.node_weight, interpretation.fold, lambda v: v * v, Fraction(0)
    )
    assert squared == oracle.nonlinear_fold(trees, weights, algebra, lambda v: v * v)


def test_one_presentation_parses_means_and_encodes() -> None:
    """Require a weight, logical forms and an encoder from one presentation."""
    table = reference()
    item = table.item(fixture.SPAN, fixture.GOAL)
    rules = fixture.PRESENTATION.rules

    def logical(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """A bracketing of the words.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple of str
            The premise forms.

        Returns
        -------
        str
            The form.
        """
        if node.occurrence is not None:
            word = node.values[0]
            assert isinstance(word, str)
            return word
        left, right = closed_values(values)
        return f"({left} {right})"

    meaning = HostInterpretation[str](
        presentation=fixture.PRESENTATION,
        operations={
            rule.id: logical for rule in (*rules, *fixture.PRESENTATION.axioms)
        },
    )
    forms = Counter(meaning.fold(d) for d in table.derivations(item))
    assert forms == Counter(
        {
            "(she (saw (stars (with telescopes))))": 2,
            "(she ((saw stars) (with telescopes)))": 2,
        }
    )
    left = torch.tensor([0.5, -1.0], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([2.0, 0.25], dtype=torch.float64)

    def encode(
        node: Derivation, values: tuple[PremiseValue[torch.Tensor], ...]
    ) -> torch.Tensor:
        """A vector per node: a fixed vector at leaves, a linear form above.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple of torch.Tensor
            The premise vectors.

        Returns
        -------
        torch.Tensor
            The vector.
        """
        if node.occurrence is not None:
            return torch.ones(2, dtype=torch.float64)
        first, second = closed_values(values)
        return left * first + right * second

    encoder = HostInterpretation[torch.Tensor](
        presentation=fixture.PRESENTATION,
        operations={rule.id: encode for rule in (*rules, *fixture.PRESENTATION.axioms)},
    )
    vectors = [encoder.fold(d) for d in table.derivations(item)]
    assert all(vector.requires_grad for vector in vectors)
    assert len({tuple(vector.tolist()) for vector in vectors}) == 2


def test_items_are_interned() -> None:
    """Require one object per distinct item."""
    table = ItemTable()
    first = table.intern(fixture.SPAN, ("NP", 0, 1))
    assert table.intern(fixture.SPAN, ("NP", 0, 1)) is first
    assert table.intern(fixture.SPAN, ("NP", 0, 2)) is not first
    assert len(table) == 2


def test_analysis_gives_components_joins_and_a_recursion_contract() -> None:
    """Require the dependency structure an executor plans from."""
    analysis = fixture.PRESENTATION.analysis
    assert analysis.max_arity == 2
    assert analysis.rules == 5
    assert all(join == ("k",) for _, join in analysis.join_variables)
    assert analysis.recursion() == MonotonePolynomial()
    assert fixture.PRESENTATION.analysis is analysis
    by_premise = fixture.PRESENTATION.rules_by_premise[fixture.SPAN]
    assert [(rule.name, position) for rule, position in by_premise][:2] == [
        ("s", 0),
        ("s", 1),
    ]


def _chain(kind: str) -> RulePresentation:
    """A presentation of two forms whose recursion is chosen.

    Parameters
    ----------
    kind : str
        ``"acyclic"``, ``"linear"`` or ``"nonlinear"``.

    Returns
    -------
    RulePresentation
        The presentation.
    """
    a, b = FamilyId.derive("a"), FamilyId.derive("b")
    rules = [
        Rule(
            id=ConstructorId.derive("ab"),
            name="ab",
            premises=(Premise(judgment=Judgment(family=a)),),
            conclusion=Judgment(family=b),
        )
    ]
    if kind != "acyclic":
        premises = (Premise(judgment=Judgment(family=b)),)
        if kind == "nonlinear":
            premises = premises * 2
        rules.append(
            Rule(
                id=ConstructorId.derive("bb"),
                name="bb",
                premises=premises,
                conclusion=Judgment(family=b),
            )
        )
    return RulePresentation(
        id=PresentationId.derive(kind),
        name=kind,
        judgments=(JudgmentDecl(id=a, name="a"), JudgmentDecl(id=b, name="b")),
        rules=tuple(rules),
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("acyclic", Acyclic()),
        ("linear", LinearCyclic()),
        ("nonlinear", MonotonePolynomial()),
    ],
)
def test_recursion_contract_of_the_family_graph(kind: str, expected: object) -> None:
    """Require the weakest contract the family graph guarantees.

    Parameters
    ----------
    kind : str
        The shape.
    expected : object
        The contract.
    """
    assert _chain(kind).analysis.recursion() == expected


def test_a_cycle_needs_a_height_bound() -> None:
    """Require a refusal above a cycle and a finite list under a height bound."""
    presentation = _chain("linear")
    slot_family = FamilyId.derive("a")
    seed = Rule(
        id=ConstructorId.derive("seed"),
        name="seed",
        conclusion=Judgment(family=slot_family),
    )
    presentation = presentation.with_(rules=(*presentation.rules, seed))
    table = ReferenceInterpretation(presentation, ())
    item = table.item(FamilyId.derive("b"), ())
    with pytest.raises(CyclicDerivations):
        table.derivations(item)
    assert [d.size() for d in table.derivations(item, height=3)] == [2, 3]


def test_round_trip_keeps_rules_indices_and_identities() -> None:
    """Require the presentation and its derivations to survive the codec."""
    decoded = loads(dumps(fixture.PRESENTATION))
    assert decoded == fixture.PRESENTATION
    assert isinstance(decoded, RulePresentation)
    assert [rule.weight for rule in decoded.rules] == [
        rule.weight for rule in fixture.PRESENTATION.rules
    ]
    table = reference()
    derivation = table.derivations(table.item(fixture.SPAN, fixture.GOAL))[0]
    assert loads(dumps(derivation)) == derivation


def test_presentation_work_claim_is_complete() -> None:
    """Require the cost claim of presentation work to pass its checker."""
    check_complexity(PRESENTATION_WORK)


def _lambda_signature() -> Signature:
    """A signature with a binder.

    Returns
    -------
    Signature
        Leaves, applications and abstractions.
    """
    return Signature(
        name="Lam",
        sorts_t=(Sort(name="T", kind="object"),),
        constructors_t=(
            Constructor(name="Leaf", codomain="T"),
            Constructor(name="App", domain=("T", "T"), codomain="T"),
        ),
        binders_t=(
            Binder(
                name="Abs",
                binds=(BinderVarSpec(var="x", sort="T"),),
                scoped=(BinderArgSpec(arg="body", sort="T"),),
                codomain="T",
            ),
        ),
    )


def test_a_signature_is_a_presentation_of_rules_without_side_conditions() -> None:
    """Require constructors and binders as rules, built once per signature."""
    signature = _lambda_signature()
    presentation = presentation_of(signature)
    assert presentation_of(signature) is presentation
    assert [rule.name for rule in presentation.rules] == ["Leaf", "App", "Abs"]
    assert all(rule.side_condition is None for rule in presentation.rules)
    abstraction = presentation.rules[2]
    assert len(abstraction.premises[0].hypotheses) == 1
    validate_module(presentation.to_module())


def test_signature_tables_are_built_once() -> None:
    """Require one table per signature, shared by every lookup."""
    signature = _lambda_signature()
    assert signature.constructors is signature.constructors
    assert signature.codomain_of("Abs") == "T"
    assert signature.domain_of("App") == ("T", "T")
    assert signature.is_binder("Abs") and not signature.is_binder("App")
    encoder = Encoder(
        name="size",
        signature=signature,
        sort_dims={"T": 1},
        op_fns={
            "Leaf": PerOpFn(op="Leaf", mode="plain", args=(), fn=lambda: torch.ones(1)),
            "App": PerOpFn(
                op="App", mode="plain", args=("f", "a"), fn=lambda f, a: f + a + 1
            ),
        },
        var_init_fns={},
        data_embedders={},
    )
    assert encoder.signature.constructors is signature.constructors


def test_judgment_arguments_must_be_refinable_indices() -> None:
    """Require a judgment form's arguments to be refinable index binders."""
    with pytest.raises(dx.ValidationError):
        JudgmentDecl(
            id=FamilyId.derive("x"),
            name="x",
            indices=(IndexBinder(name="i", sort=NAT),),
        )
    assert IndexVariable(name="i", sort=NAT).name == "i"
