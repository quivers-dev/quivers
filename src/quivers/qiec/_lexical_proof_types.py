"""Expose a lexical presentation's checked proof families by deduction name."""

from __future__ import annotations

from quivers.qiec.kinds import IndexBinder, Telescope, UserIndexSort
from quivers.qiec.rules import Judgment, RulePresentation
from quivers.qiec.types import IndexConstructor, IndexLiteral, IndexTerm, IndexVariable


def export_lexical_proofs(
    presentation: RulePresentation, deduction: str
) -> RulePresentation:
    """Give the proof declarations module-local names without changing groundings.

    The same nominal rule, axiom, judgment and weight identities remain in
    every proof. Prefixing the term sorts and proof constructors lets two
    deductions export their indexed proofs into one QIEC module even when
    their term constructors differ. Every static index occurrence is
    transported along the same sort renaming.

    Parameters
    ----------
    presentation
        The original checked lexical span presentation.
    deduction
        The name prefix for its visible proof declarations.

    Returns
    -------
    RulePresentation
        The alpha-renamed presentation at the same nominal identities.

    Raises
    ------
    ValueError
        If the presentation has several judgments, side conditions, or equations.
    """
    if presentation.conditions or presentation.equations:
        raise ValueError(
            "lexical proof queries require unconditional rules without equations"
        )
    if len(presentation.judgments) != 1:
        raise ValueError("lexical proof queries require one span judgment")
    (span,) = presentation.judgments
    sorts = {
        sort: sort.with_(name=f"{deduction}__{sort.name}")
        for sort in presentation.index_sorts
    }

    def telescope(source: Telescope) -> Telescope:
        """Transport index binders while retaining their binding names and roles.

        Parameters
        ----------
        source
            The original telescope.

        Returns
        -------
        Telescope
            The binders at the renamed sorts, in their original order.
        """
        return tuple(
            binder.with_(sort=sorts[binder.sort])
            if isinstance(binder, IndexBinder)
            and isinstance(binder.sort, UserIndexSort)
            and binder.sort in sorts
            else binder
            for binder in source
        )

    def index(term: IndexTerm) -> IndexTerm:
        """Transport a constructor tree or a variable into its renamed sort.

        Parameters
        ----------
        term
            An original static pattern.

        Returns
        -------
        IndexTerm
            The pattern at the renamed sort.
        """
        if (
            isinstance(term, IndexVariable | IndexLiteral | IndexConstructor)
            and isinstance(term.sort, UserIndexSort)
            and term.sort in sorts
        ):
            renamed = term.with_(sort=sorts[term.sort])
            if isinstance(renamed, IndexConstructor):
                return renamed.with_(
                    arguments=tuple(index(child) for child in renamed.arguments)
                )
            return renamed
        return term

    def judgment(source: Judgment) -> Judgment:
        """Transport an indexed judgment to its visible argument sorts.

        Parameters
        ----------
        source
            The original judgment application.

        Returns
        -------
        Judgment
            The same nominal judgment at the renamed static arguments.
        """
        return source.with_(arguments=tuple(index(term) for term in source.arguments))

    return presentation.with_(
        index_sorts=tuple(sorts.values()),
        judgments=(
            span.with_(name=f"{deduction}__SpanProof", indices=telescope(span.indices)),
        ),
        axioms=tuple(
            axiom.with_(
                name=f"{deduction}__{axiom.name}",
                judgment=judgment(axiom.judgment),
                telescope=telescope(axiom.telescope),
            )
            for axiom in presentation.axioms
        ),
        rules=tuple(
            rule.with_(
                name=f"{deduction}__rule_{rule.name}",
                conclusion=judgment(rule.conclusion),
                telescope=telescope(rule.telescope),
                premises=tuple(
                    premise.with_(
                        judgment=judgment(premise.judgment),
                        before=tuple(judgment(item) for item in premise.before),
                        after=tuple(judgment(item) for item in premise.after),
                    )
                    for premise in rule.premises
                ),
            )
            for rule in presentation.rules
        ),
    )
