"""GADT branch refinement and conservative coverage interfaces."""

from __future__ import annotations

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx

from enum import Enum

from quivers.qiec.declarations import ConstructorDecl, FamilyDecl
from quivers.qiec.effects import EMPTY_ROW, EffectRow
from quivers.qiec.evidence import BranchGiven
from quivers.qiec.identifiers import (
    ConstructorId,
    EqualityId,
    StaticScopeId,
    StaticVariableId,
)
from quivers.qiec.kinds import (
    EFFECT,
    EffectBinder,
    IndexBinder,
    IndexSort,
    Kind,
    TypeBinder,
)
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_telescope,
    substitute_static,
)
from quivers.qiec.types import (
    EffectVariable,
    EffectRef,
    EqualityType,
    IndexConstructor,
    IndexLiteral,
    IndexVariable,
    StaticArgument,
    TypeApplication,
    TypeVariable,
)


class CoverageStatus(str, Enum):
    """Whether a set of branch patterns covers a family."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class Reachability(str, Enum):
    """Whether a branch can be entered, as far as its equalities decide."""

    REACHABLE = "reachable"
    IMPOSSIBLE = "impossible"
    UNKNOWN = "unknown"


class BranchPattern(QiecModel):
    """The shape of one case branch, as coverage sees it.

    Parameters
    ----------
    constructor
        The constructor the branch matches, or ``None`` for a wildcard.
    guarded
        Whether the branch carries a guard that may reject a matching value,
        which keeps it from counting toward completeness.
    """

    constructor: ConstructorId | None = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    guarded: bool = False


class CoverageResult(QiecModel):
    """The outcome of checking branch patterns against a family.

    Parameters
    ----------
    status
        Whether the patterns cover the family.
    missing
        Constructors no pattern matches.
    duplicates
        Constructors more than one pattern matches.
    """

    status: CoverageStatus = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    missing: tuple[ConstructorId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    duplicates: tuple[ConstructorId, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )


class BranchRefinement(QiecModel):
    """Rigid skolems and local equalities available inside one branch.

    Parameters
    ----------
    skolems
        The names of the branch's rigid static variables.
    givens
        The equalities the branch's constructor makes available.
    reachability
        Whether the givens are jointly satisfiable, as far as the unifier
        decides.
    """

    skolems: tuple[str, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    givens: tuple[BranchGiven, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    reachability: Reachability = dx.field(
        default=Reachability.UNKNOWN, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _binder_classifier(
    binder: TypeBinder | IndexBinder | EffectBinder,
) -> Kind | IndexSort:
    """The kind or sort that classifies what a binder ranges over.

    Parameters
    ----------
    binder : TypeBinder or IndexBinder or EffectBinder
        The binder to classify.

    Returns
    -------
    Kind or IndexSort
        A type binder's kind, an index binder's sort, or the effect kind
        for an effect binder, which ranges over a single classifier.
    """
    if isinstance(binder, TypeBinder):
        return binder.kind
    if isinstance(binder, IndexBinder):
        return binder.sort
    return EFFECT


def _closed_distinct(left: StaticArgument, right: StaticArgument) -> bool:
    """Recognize only contradictions justified by free constructors.

    Parameters
    ----------
    left : StaticArgument
        One side of the equation.
    right : StaticArgument
        The other side.

    Returns
    -------
    bool
        True only when the two cannot be equal because distinct
        constructors are distinct. False covers both "provably equal" and
        "not known", since a conservative solver must not turn ignorance
        into a contradiction: a branch wrongly certified impossible would
        be dropped from coverage and from effect inference.
    """
    if isinstance(left, IndexLiteral) and isinstance(right, IndexLiteral):
        return left.sort == right.sort and left.value != right.value
    if isinstance(left, IndexConstructor) and isinstance(right, IndexConstructor):
        return left.sort == right.sort and left.name != right.name
    if isinstance(left, IndexLiteral) and isinstance(right, IndexConstructor):
        return (
            isinstance(left.value, str)
            and left.sort == right.sort
            and left.value != right.name
        )
    if isinstance(left, IndexConstructor) and isinstance(right, IndexLiteral):
        return _closed_distinct(right, left)
    if isinstance(left, TypeApplication) and isinstance(right, TypeApplication):
        return left.constructor.id != right.constructor.id
    if isinstance(left, EffectRef) and isinstance(right, EffectRef):
        return left.id != right.id
    return False


def check_coverage(
    family: FamilyDecl,
    patterns: tuple[BranchPattern, ...],
    *,
    certified_impossible: frozenset[ConstructorId] = frozenset(),
) -> CoverageResult:
    """Check constructor coverage without pretending opaque cases are total.

    Parameters
    ----------
    family : FamilyDecl
        The family being matched on.
    patterns : tuple[BranchPattern, ...]
        The branch patterns written, in source order.
    certified_impossible : frozenset[ConstructorId]
        Constructors an equality solver has proven cannot occur, which
        may therefore be omitted without making the match incomplete.

    Returns
    -------
    CoverageResult
        The status, the constructors still missing, and any matched
        twice. The status is `UNKNOWN` rather than `COMPLETE` whenever
        the family is open, a pattern is a wildcard or guarded, or a
        pattern names a constructor the family does not declare, since
        none of those can be decided here.
    """
    known = tuple(
        pattern.constructor for pattern in patterns if pattern.constructor is not None
    )
    seen: set[ConstructorId] = set()
    duplicates: list[ConstructorId] = []
    for constructor in known:
        if constructor in seen and constructor not in duplicates:
            duplicates.append(constructor)
        seen.add(constructor)

    missing = tuple(
        constructor
        for constructor in family.constructors
        if constructor not in seen and constructor not in certified_impossible
    )
    unresolved = (
        not family.closed
        or any(pattern.constructor is None or pattern.guarded for pattern in patterns)
        or any(constructor not in family.constructors for constructor in known)
    )
    if unresolved:
        status = CoverageStatus.UNKNOWN
    elif missing or duplicates:
        status = CoverageStatus.INCOMPLETE
    else:
        status = CoverageStatus.COMPLETE
    return CoverageResult(status=status, missing=missing, duplicates=tuple(duplicates))


def refine_branch(
    family: FamilyDecl,
    constructor: ConstructorDecl,
    scrutinee: TypeApplication,
    constructor_arguments: tuple[StaticArgument, ...],
) -> BranchRefinement:
    """Introduce constructor skolems and index equalities for a branch.

    The equality solver is intentionally conservative.  Syntactically equal
    indices are reachable; contradictory closed literals are impossible; all
    other equations remain ``UNKNOWN`` and must still contribute branch
    effects and coverage.

    Parameters
    ----------
    family : FamilyDecl
        The family being matched on.
    constructor : ConstructorDecl
        The constructor this branch matches.
    scrutinee : TypeApplication
        The applied family type being scrutinised, supplying the
        parameters and the indices to be refined.
    constructor_arguments : tuple[StaticArgument, ...]
        Arguments instantiating the constructor's local telescope,
        normally the rigid skolems from `constructor_skolems`.

    Returns
    -------
    BranchRefinement
        The skolems in scope, the equalities the match makes available,
        and whether the branch is reachable.

    Raises
    ------
    TypeError
        If the constructor belongs to another family, if the scrutinee
        does not inhabit the matched family, or if the scrutinee supplies
        the wrong number of arguments.
    """
    if constructor.family != family.id:
        raise TypeError(f"constructor {constructor.name!r} belongs to another family")
    if scrutinee.constructor != family.type_constructor:
        raise TypeError("scrutinee does not inhabit the matched family")

    parameter_count = len(family.parameters)
    expected_count = parameter_count + len(family.indices)
    if len(scrutinee.arguments) != expected_count:
        raise TypeError(
            f"family {family.name!r} expects {expected_count} arguments, "
            f"got {len(scrutinee.arguments)}"
        )
    parameter_substitution = instantiate_telescope(
        family.parameters,
        scrutinee.arguments[:parameter_count],
    )
    local_substitution = instantiate_telescope(
        constructor.telescope,
        constructor_arguments,
    )
    substitution = StaticSubstitution(
        types=(*parameter_substitution.types, *local_substitution.types),
        indices=(*parameter_substitution.indices, *local_substitution.indices),
        effects=(*parameter_substitution.effects, *local_substitution.effects),
    )

    actual_indices = scrutinee.arguments[parameter_count:]
    if len(constructor.result_indices) != len(actual_indices):
        raise TypeError(
            f"constructor {constructor.name!r} returns the wrong number of indices"
        )

    givens: list[BranchGiven] = []
    states: list[Reachability] = []
    for position, (binder, actual, result) in enumerate(
        zip(family.indices, actual_indices, constructor.result_indices, strict=True)
    ):
        result = substitute_static(result, substitution)
        equality = EqualityType(
            kind=_binder_classifier(binder), left=actual, right=result
        )
        equality_id = EqualityId.derive(constructor.id, position, equality)
        givens.append(BranchGiven(id=equality_id, equality=equality))
        if actual == result:
            states.append(Reachability.REACHABLE)
        elif _closed_distinct(actual, result):
            states.append(Reachability.IMPOSSIBLE)
        else:
            states.append(Reachability.UNKNOWN)

    if any(state is Reachability.IMPOSSIBLE for state in states):
        reachability = Reachability.IMPOSSIBLE
    elif all(state is Reachability.REACHABLE for state in states):
        reachability = Reachability.REACHABLE
    else:
        reachability = Reachability.UNKNOWN
    return BranchRefinement(
        skolems=tuple(binder.name for binder in constructor.telescope),
        givens=tuple(givens),
        reachability=reachability,
    )


def constructor_skolems(
    constructor: ConstructorDecl,
    scope: StaticScopeId,
) -> tuple[StaticArgument, ...]:
    """Create rigid static variables for a constructor's local telescope.

    Parameters
    ----------
    constructor : ConstructorDecl
        The constructor whose telescope to instantiate.
    scope : StaticScopeId
        Identity of the branch scope. It enters each variable's identity,
        so skolems from two branches are never confused and cannot be
        mistaken for one another when a type escapes.

    Returns
    -------
    tuple[StaticArgument, ...]
        One rigid variable per binder, in telescope order.
    """
    arguments: list[StaticArgument] = []
    for position, binder in enumerate(constructor.telescope):
        identity = StaticVariableId.derive(
            scope,
            constructor.id,
            position,
            binder.tag,
        )
        if isinstance(binder, TypeBinder):
            arguments.append(
                TypeVariable(name=binder.name, kind=binder.kind, identity=identity)
            )
        elif isinstance(binder, IndexBinder):
            arguments.append(
                IndexVariable(name=binder.name, sort=binder.sort, identity=identity)
            )
        elif isinstance(binder, EffectBinder):
            arguments.append(EffectVariable(name=binder.name, identity=identity))
    return tuple(arguments)


def check_indexed_coverage(
    family: FamilyDecl,
    constructors: tuple[ConstructorDecl, ...],
    scrutinee: TypeApplication,
    patterns: tuple[BranchPattern, ...],
) -> CoverageResult:
    """Coverage with omissions justified by closed-index contradiction.

    Only constructors that the small equality solver certifies as impossible
    may be omitted.  An unknown result remains a required case.

    Parameters
    ----------
    family : FamilyDecl
        The family being matched on.
    constructors : tuple[ConstructorDecl, ...]
        Declarations of the family's constructors, needed to refine each
        one against the scrutinee.
    scrutinee : TypeApplication
        The applied family type being matched.
    patterns : tuple[BranchPattern, ...]
        The branch patterns written.

    Returns
    -------
    CoverageResult
        Coverage judged against the constructors that remain possible.

    Raises
    ------
    TypeError
        If a constructor belongs to another family, or the scrutinee does
        not inhabit the matched family.
    """
    impossible: set[ConstructorId] = set()
    for constructor in constructors:
        scope = StaticScopeId.derive(
            "coverage",
            family.id,
            constructor.id,
            scrutinee,
        )
        refinement = refine_branch(
            family,
            constructor,
            scrutinee,
            constructor_skolems(constructor, scope),
        )
        if refinement.reachability is Reachability.IMPOSSIBLE:
            impossible.add(constructor.id)
    return check_coverage(
        family,
        patterns,
        certified_impossible=frozenset(impossible),
    )


def join_branch_rows(
    branches: tuple[tuple[Reachability, EffectRow], ...],
) -> EffectRow:
    """Union effects of every possibly reachable branch.

    ``UNKNOWN`` is intentionally treated as reachable.  Dropping it would make
    an incomplete equality solver unsound for effect inference.

    Parameters
    ----------
    branches : tuple[tuple[Reachability, EffectRow], ...]
        Each branch's reachability and the row its body performs.

    Returns
    -------
    EffectRow
        The union over every branch not certified impossible. Only a
        proven contradiction removes a branch's effects from the result.

    Raises
    ------
    ValueError
        If two reachable branches bind one instance to different
        interfaces, or their open tails cannot merge.
    """
    result = EMPTY_ROW
    for reachability, row in branches:
        if reachability is not Reachability.IMPOSSIBLE:
            result = result.union(row)
    return result


__all__ = [
    "BranchPattern",
    "BranchRefinement",
    "CoverageResult",
    "CoverageStatus",
    "Reachability",
    "check_coverage",
    "check_indexed_coverage",
    "constructor_skolems",
    "join_branch_rows",
    "refine_branch",
]
