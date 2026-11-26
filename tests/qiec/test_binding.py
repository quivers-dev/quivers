"""Contexts, renaming, substitution and proof substitution."""

from __future__ import annotations

import didactic.api as dx
import pytest

from quivers.core.laws import ClaimSubject, Law, LawStatus
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.rules import (
    AFFINE,
    CARTESIAN,
    LINEAR,
    ORDERED,
    RELEVANT,
    ContextStructure,
    Derivation,
    Equation,
    EquationFails,
    Hole,
    HostInterpretation,
    Item,
    OpenDerivation,
    PremiseValue,
    ReferenceInterpretation,
    RuleError,
    Scoped,
    StructureError,
    check_derivation,
    check_equation,
    check_structure,
    compose_derivations,
    identity_derivation,
    presentation_of,
    rename_hypotheses,
    required_structure,
    term_derivation,
)
from quivers.qiec.serialization import dumps, loads
from quivers.structural.signature import (
    Substitution,
    SubstitutionError,
    free_indices,
    substitute,
)
from tests.evidence import presentations as fixture

A = Item(family=fixture.FIRST)
B = Item(family=fixture.SECOND)
PAIR = Item(family=fixture.PAIR)


def pair(first: int, second: int, rule: str = "pair") -> OpenDerivation:
    """A pair of two hypotheses.

    Parameters
    ----------
    first, second : int
        The de Bruijn indices of the premises.
    rule : str
        ``"pair"`` over ``(a, b)`` or ``"twice"`` over ``(a, a)``.

    Returns
    -------
    OpenDerivation
        The derivation.
    """
    context = (A, B) if rule == "pair" else (A, A)
    return OpenDerivation(
        context=context,
        body=Derivation(
            constructor=ConstructorId.derive("pairs", rule),
            item=PAIR,
            premises=(
                Hole(index=first, item=context[1 - first]),
                Hole(index=second, item=context[1 - second]),
            ),
        ),
    )


def test_substitution_does_not_capture() -> None:
    """Require a free variable substituted under a binder to stay free."""
    term = fixture.abs_(fixture.app(fixture.var(0), fixture.var(1)))
    sigma = Substitution(replacements=(fixture.var(0),), shift=1)
    found = substitute(term, sigma, fixture.LAMBDA)
    assert found == fixture.abs_(fixture.app(fixture.var(0), fixture.var(1)))
    swapped = Substitution.renaming((1, 0), 2)
    assert substitute(
        fixture.app(fixture.var(0), fixture.var(1)), swapped, fixture.LAMBDA
    ) == (fixture.app(fixture.var(1), fixture.var(0)))
    assert free_indices(term, fixture.LAMBDA) == frozenset({0})
    assert free_indices(fixture.abs_(fixture.var(0)), fixture.LAMBDA) == frozenset()


def test_weakening_shifts_free_variables_only() -> None:
    """Require a weakening to move free variables and keep bound ones."""
    term = fixture.abs_(fixture.app(fixture.var(0), fixture.var(3)))
    weakened = substitute(term, Substitution.weakening(2), fixture.LAMBDA)
    assert weakened == fixture.abs_(fixture.app(fixture.var(0), fixture.var(5)))


def test_substitution_refuses_undeclared_operations() -> None:
    """Require an operation outside the signature to be refused."""
    with pytest.raises(SubstitutionError):
        substitute(fixture.from_tuple(("Other",)), Substitution(), fixture.LAMBDA)
    with pytest.raises(dx.ValidationError):
        Substitution(shift=-1)


@pytest.mark.parametrize(
    ("structure", "admitted"),
    [
        (ORDERED, {"ordered"}),
        (LINEAR, {"ordered", "swapped"}),
        (AFFINE, {"ordered", "swapped", "dropped"}),
        (RELEVANT, {"ordered", "swapped", "doubled"}),
        (CARTESIAN, {"ordered", "swapped", "dropped", "doubled"}),
    ],
)
def test_structures_admit_exactly_their_rules(
    structure: ContextStructure, admitted: set[str]
) -> None:
    """Require each structure to admit the uses its rules license.

    Parameters
    ----------
    structure : ContextStructure
        The structure.
    admitted : set of str
        The uses it admits.
    """
    presentation = fixture.ORDERED_PAIRS.with_(structure=structure)
    uses = {
        "ordered": pair(1, 0, "twice"),
        "swapped": pair(0, 1, "twice"),
        "dropped": rename_hypotheses(
            presentation, pair(1, 0, "twice"), (0, 2), (A, A, A)
        ),
        "doubled": OpenDerivation(
            context=(A,),
            body=Derivation(
                constructor=ConstructorId.derive("pairs", "twice"),
                item=PAIR,
                premises=(Hole(index=0, item=A), Hole(index=0, item=A)),
            ),
        ),
    }
    for name, derivation in uses.items():
        if name in admitted:
            check_structure(presentation, derivation)
        else:
            with pytest.raises(StructureError):
                check_structure(presentation, derivation)


def test_required_structure_names_each_rule() -> None:
    """Require the least structure of each use."""
    presentation = fixture.ORDERED_PAIRS
    assert required_structure(presentation, pair(1, 0)) == ORDERED
    assert required_structure(presentation, pair(0, 1, "twice")).exchange
    assert required_structure(presentation, pair(0, 0, "twice")) == ContextStructure(
        weakening=True, contraction=True
    )


def test_composition_laws() -> None:
    """Require units and associativity of plugging derivations."""
    presentation = fixture.ORDERED_PAIRS
    f = pair(1, 0)
    assert (
        compose_derivations(
            presentation, f, (identity_derivation(A), identity_derivation(B))
        )
        == f
    )
    assert compose_derivations(presentation, identity_derivation(PAIR), (f,)) == f
    with pytest.raises(RuleError):
        compose_derivations(presentation, f, (identity_derivation(A),))
    with pytest.raises(RuleError):
        compose_derivations(
            presentation, f, (identity_derivation(B), identity_derivation(A))
        )


def test_rename_checks_items() -> None:
    """Require a renaming to keep each hypothesis's item."""
    with pytest.raises(RuleError):
        rename_hypotheses(fixture.ORDERED_PAIRS, pair(1, 0), (1, 0), (A, B))


def _printer() -> HostInterpretation[str]:
    """A printer of lambda terms whose variables print their values.

    Returns
    -------
    HostInterpretation of str
        The printer.
    """
    lam = presentation_of(fixture.LAMBDA)
    ids = {rule.name: rule.id for rule in lam.rules}
    counter = iter(range(1000))

    def constant(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """A constant.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            No values.

        Returns
        -------
        str
            Its name.
        """
        return str(node.values[0])

    def application(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """An application.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            Function and argument.

        Returns
        -------
        str
            The application.
        """
        return f"({values[0]} {values[1]})"

    def abstraction(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """An abstraction with a fresh name.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            The body as a function of the variable.

        Returns
        -------
        str
            The abstraction.
        """
        body = values[0]
        assert isinstance(body, Scoped)
        name = f"v{next(counter)}"
        return f"(fun {name}. {body((name,))})"

    return HostInterpretation[str](
        presentation=lam,
        operations={
            ids["Const"]: constant,
            ids["App"]: application,
            ids["Abs"]: abstraction,
        },
    )


def test_binders_receive_their_scope() -> None:
    """Require a scoped premise to be a function of its hypothesis."""
    term = fixture.abs_(fixture.app(fixture.var(0), fixture.var(1)))
    derivation = term_derivation(fixture.LAMBDA, term, ("T",))
    printed = _printer().fold_open(derivation, ("y",))
    assert printed == "(fun v0. (v0 y))"
    with pytest.raises(RuleError):
        _printer().fold_open(derivation, ())


def test_derivations_are_checked_by_qiec() -> None:
    """Require valid derivations to pass and a tampered one to be refused."""
    lam = presentation_of(fixture.LAMBDA)
    for term in fixture.TERMS:
        check_derivation(lam, term_derivation(fixture.LAMBDA, term, ("T", "T", "T")))
    table = ReferenceInterpretation(fixture.PRESENTATION, fixture.OCCURRENCES)
    goal = table.item(fixture.SPAN, fixture.GOAL)
    closed = OpenDerivation(body=table.derivations(goal)[0])
    check_derivation(fixture.PRESENTATION, closed)
    body = closed.body
    assert isinstance(body, Derivation)
    wrong = body.with_(item=table.item(fixture.SPAN, ("S", 0, 4)))
    with pytest.raises(RuleError):
        check_derivation(fixture.PRESENTATION, OpenDerivation(body=wrong))
    with pytest.raises(RuleError):
        check_derivation(
            fixture.ORDERED_PAIRS,
            OpenDerivation(context=(A, B), body=Hole(index=0, item=A)),
        )


def test_equations_are_tested_not_proved() -> None:
    """Require a passing test to give a tested claim and a failing one to refute."""
    presentation = fixture.ORDERED_PAIRS
    law = Law(name="pair-swap", statement="pair(x, y) = pair(y, x) on twice")
    equation = Equation(law=law, left=pair(1, 0, "twice"), right=pair(0, 1, "twice"))
    ids = {rule.name: rule.id for rule in presentation.rules}

    def joined(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """Concatenate, which is not commutative.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            The two strings.

        Returns
        -------
        str
            Their concatenation.
        """
        return f"{values[0]}{values[1]}"

    def counted(node: Derivation, values: tuple[PremiseValue[str], ...]) -> str:
        """Sort the parts, which is commutative.

        Parameters
        ----------
        node : Derivation
            The node.
        values : tuple
            The two strings.

        Returns
        -------
        str
            Their sorted concatenation.
        """
        return "".join(sorted(str(value) for value in values))

    subject = ClaimSubject(kind="interpretation", name="sorted")
    for operation, holds in ((counted, True), (joined, False)):
        interpretation = HostInterpretation[str](
            presentation=presentation,
            operations={ids["pair"]: operation, ids["twice"]: operation},
        )
        environments = [("x", "y"), ("y", "y")]
        if holds:
            found = check_equation(
                interpretation,
                equation,
                environments,
                lambda a, b: a == b,
                subject=subject,
                check="tests/qiec/test_binding.py",
                domain="two strings",
            )
            assert found.status is LawStatus.TESTED
            assert not found.established
        else:
            with pytest.raises(EquationFails):
                check_equation(
                    interpretation,
                    equation,
                    environments,
                    lambda a, b: a == b,
                    subject=subject,
                    check="tests/qiec/test_binding.py",
                    domain="two strings",
                )
    with pytest.raises(dx.ValidationError):
        Equation(law=law, left=pair(1, 0), right=pair(1, 0, "twice"))


def test_open_records_round_trip() -> None:
    """Require open derivations, equations and structures to survive the codec."""
    law = Law(name="identity", statement="f = f")
    equation = Equation(law=law, left=pair(1, 0), right=pair(1, 0))
    presentation = fixture.ORDERED_PAIRS.with_(structure=LINEAR, equations=(equation,))
    for record in (pair(0, 1), equation, LINEAR, presentation):
        assert loads(dumps(record)) == record
