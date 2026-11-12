"""Pre-registered library of weighted deduction systems.

Ships agenda-engine `DeductionSystem` instances for standard
specializations from the semiring-parsing and agenda-driven-deduction
literature (Shieber, Schabes, and Pereira 1995; Goodman 1999; Klein and
Manning 2001; Nederhof 2003; Eisner and Blatz 2007).

Each entry is a `DeductionSystem`: a rule set, a semiring, an axiom
injector, a goal predicate, and an agenda factory (the item algebra is
implicit in the rules' patterns, and the chart is the default hash
chart). Users invoke them as

.. code-block:: python

    from quivers.stochastic.stdlib import CCG, AB, MLTT, Datalog
    view = CCG(lexical_axioms)
    weight = view.goal_weight()

A system maps its axioms to a chart: the weight of each derivable item,
the semiring sum over the item's derivations of the product of their
weights.
"""

from __future__ import annotations

from quivers.stochastic.agenda import (
    DeductionSystem,
    InferenceRule,
    Wildcard,
    cky_agenda,
    depth_first_agenda,
    knuth_agenda,
    semi_naive_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    LOG_PROB,
    VITERBI,
)


# ---------------------------------------------------------------------------
# Categorial grammars
# ---------------------------------------------------------------------------


_I, _J, _K = Wildcard("i"), Wildcard("j"), Wildcard("k")
_X, _Y, _Z = Wildcard("X"), Wildcard("Y"), Wildcard("Z")


def _span(i: Wildcard, j: Wildcard, category: tuple | Wildcard) -> tuple:
    """A span item ``("span", i, j, category)`` over the tokens ``[i, j)``.

    Parameters
    ----------
    i, j : Wildcard
        The span's start and end positions.
    category : tuple or Wildcard
        The category pattern.

    Returns
    -------
    tuple
        The item pattern.
    """
    return ("span", i, j, category)


def _fwd(result: Wildcard, argument: Wildcard) -> tuple:
    """The forward-slash category ``result / argument``."""
    return ("slash", "/", result, argument)


def _bwd(result: Wildcard, argument: Wildcard) -> tuple:
    """The backward-slash category ``result\\argument``, seeking its argument on the left."""
    return ("slash", "\\", argument, result)


def _application_rules() -> list[InferenceRule]:
    """Forward and backward application over adjacent spans.

    ``X/Y`` over ``[i, k)`` and ``Y`` over ``[k, j)`` derive ``X`` over
    ``[i, j)``; ``Y`` over ``[i, k)`` and ``X\\Y`` over ``[k, j)`` derive
    ``X`` over ``[i, j)``. Slash categories are written result first:
    ``X/Y`` and ``X\\Y`` both yield ``X``, seeking ``Y`` on the right and on
    the left.

    Returns
    -------
    list[InferenceRule]
        The two application rules.
    """
    return [
        InferenceRule(
            name="fwd_app",
            premises=(_span(_I, _K, _fwd(_X, _Y)), _span(_K, _J, _Y)),
            conclusion=_span(_I, _J, _X),
        ),
        InferenceRule(
            name="bwd_app",
            premises=(_span(_I, _K, _Y), _span(_K, _J, _bwd(_X, _Y))),
            conclusion=_span(_I, _J, _X),
        ),
    ]


def _ccg_rules() -> list[InferenceRule]:
    """CCG rules: application and harmonic composition over adjacent spans.

    Harmonic composition: ``X/Y, Y/Z ⊢ X/Z`` (forward, B>) and
    ``Y\\Z, X\\Y ⊢ X\\Z`` (backward, B<), in result-first notation
    (Steedman 2000, *The Syntactic Process*).

    Returns
    -------
    list[InferenceRule]
        Application and harmonic composition.
    """
    return [
        *_application_rules(),
        InferenceRule(
            name="fwd_comp",
            premises=(_span(_I, _K, _fwd(_X, _Y)), _span(_K, _J, _fwd(_Y, _Z))),
            conclusion=_span(_I, _J, _fwd(_X, _Z)),
        ),
        InferenceRule(
            name="bwd_comp",
            premises=(_span(_I, _K, _bwd(_Y, _Z)), _span(_K, _J, _bwd(_X, _Y))),
            conclusion=_span(_I, _J, _bwd(_X, _Z)),
        ),
    ]


def _is_sentence(item: tuple) -> bool:
    """Whether an item is a span of the category ``S``.

    Parameters
    ----------
    item : tuple
        A chart item.

    Returns
    -------
    bool
        ``True`` for ``("span", i, j, "S")``; the system's run narrows
        it to the span of the whole input.
    """
    return (
        isinstance(item, tuple)
        and len(item) == 4
        and item[0] == "span"
        and item[3] == "S"
    )


CCG = DeductionSystem(
    rules=tuple(_ccg_rules()),
    semiring=LOG_PROB,
    axiom_injector=lambda axioms: list(axioms),
    goal=_is_sentence,
    agenda_factory=cky_agenda,
)
"""Combinatory categorial grammar: application and harmonic composition over ``("span", i, j, category)`` items, log-probability semiring, CKY agenda; the goal is ``S`` over the whole input."""


AB = DeductionSystem(
    rules=tuple(_application_rules()),
    semiring=LOG_PROB,
    axiom_injector=lambda axioms: list(axioms),
    goal=_is_sentence,
    agenda_factory=cky_agenda,
)
"""AB (Ajdukiewicz-Bar-Hillel) categorial grammar: forward and backward application only, over ``("span", i, j, category)`` items, log-probability semiring, CKY agenda; the goal is ``S`` over the whole input."""


# ---------------------------------------------------------------------------
# Type theory / proof search
# ---------------------------------------------------------------------------


def _stlc_rules() -> list[InferenceRule]:
    """Simply-typed lambda calculus, bidirectional typing.

    Items: ('synth', t, A) for synthesis, ('check', t, A) for checking.
    Rules: variable-synthesis, application-synthesis, lambda-checking,
    annotation-check-to-synth, mode-switch.
    """
    return [
        # mode-switch: Synth ⊢ Check (a synthesized type checks
        # against itself).
        InferenceRule(
            name="synth_to_check",
            premises=(("synth", Wildcard("t"), Wildcard("A")),),
            conclusion=("check", Wildcard("t"), Wildcard("A")),
        ),
        # application: synth(f, A → B), check(a, A) |- synth(f a, B)
        InferenceRule(
            name="app_synth",
            premises=(
                ("synth", Wildcard("f"), ("arrow", Wildcard("A"), Wildcard("B"))),
                ("check", Wildcard("a"), Wildcard("A")),
            ),
            conclusion=("synth", ("app", Wildcard("f"), Wildcard("a")), Wildcard("B")),
        ),
    ]


STLC = DeductionSystem(
    rules=tuple(_stlc_rules()),
    semiring=BOOLEAN,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] in ("synth", "check"),
    agenda_factory=depth_first_agenda,
)
"""Bidirectional type checking for the simply-typed lambda calculus, Boolean semiring."""


def _term_depth(t) -> int:
    """Depth of a structural term (for proof-bound enforcement)."""
    if not isinstance(t, tuple):
        return 0
    if not t:
        return 0
    return 1 + max((_term_depth(c) for c in t), default=0)


# MLTT fragment proof-depth bound: a deduction whose conclusion's
# term-depth exceeds this is rejected. Bounds infinite term-tower
# regress in the application rule (id (id (id …))) and is the
# standard recipe for cycle-breaking in agenda-driven MLTT
# elaboration (Norell 2007, §3).
_MLTT_MAX_TERM_DEPTH = 4


def _mltt_rules() -> list[InferenceRule]:
    """Martin-Löf dependent type theory (a fragment): variable,
    application, Pi-formation, Pi-introduction.

    Items: ('judges', ctx, term, type) under a context Γ.

    The application rule carries a side condition bounding the
    derived term's depth, required to terminate the agenda on
    cyclic application chains (``f (f (f x))``) where each new
    item is structurally distinct from its predecessors.
    """

    def _app_term_depth_bound(bindings) -> bool:
        f = bindings.get("f")
        a = bindings.get("a")
        return _term_depth(("app", f, a)) <= _MLTT_MAX_TERM_DEPTH

    return [
        # variable: (x : A) ∈ Γ ⊢ Γ ⊢ x : A
        # Encoded as: in_ctx(Γ, x, A) |- judges(Γ, x, A)
        InferenceRule(
            name="var",
            premises=(("in_ctx", Wildcard("G"), Wildcard("x"), Wildcard("A")),),
            conclusion=("judges", Wildcard("G"), Wildcard("x"), Wildcard("A")),
        ),
        # application: judges(Γ, f, Π x:A. B), judges(Γ, a, A) |- judges(Γ, f a, B[a/x])
        # Simplified: B[a/x] = B (assuming no substitution for fragment).
        InferenceRule(
            name="app",
            premises=(
                (
                    "judges",
                    Wildcard("G"),
                    Wildcard("f"),
                    ("pi", Wildcard("A"), Wildcard("B")),
                ),
                ("judges", Wildcard("G"), Wildcard("a"), Wildcard("A")),
            ),
            conclusion=(
                "judges",
                Wildcard("G"),
                ("app", Wildcard("f"), Wildcard("a")),
                Wildcard("B"),
            ),
            side_condition=_app_term_depth_bound,
        ),
    ]


MLTT = DeductionSystem(
    rules=tuple(_mltt_rules()),
    semiring=BOOLEAN,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "judges",
    agenda_factory=depth_first_agenda,
    max_iterations=10_000,
)
"""Bidirectional type checking for a fragment of Martin-Löf dependent type theory, Boolean semiring, with a bound on term depth."""


# ---------------------------------------------------------------------------
# Datalog / weighted logic programming
# ---------------------------------------------------------------------------


def _datalog_transitive_closure_rules() -> list[InferenceRule]:
    """Transitive closure: reach(X, Y) :- edge(X, Y); reach(X, Z) :- reach(X, Y), edge(Y, Z)."""
    return [
        # base: edge(X, Y) |- reach(X, Y)
        InferenceRule(
            name="reach_base",
            premises=(("edge", Wildcard("X"), Wildcard("Y")),),
            conclusion=("reach", Wildcard("X"), Wildcard("Y")),
        ),
        # transitive: reach(X, Y), edge(Y, Z) |- reach(X, Z)
        InferenceRule(
            name="reach_step",
            premises=(
                ("reach", Wildcard("X"), Wildcard("Y")),
                ("edge", Wildcard("Y"), Wildcard("Z")),
            ),
            conclusion=("reach", Wildcard("X"), Wildcard("Z")),
        ),
    ]


Datalog = DeductionSystem(
    rules=tuple(_datalog_transitive_closure_rules()),
    semiring=BOOLEAN,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "reach",
    agenda_factory=semi_naive_agenda,
)
"""Transitive closure of an ``edge`` relation by semi-naive Datalog evaluation, Boolean semiring."""


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _dijkstra_rules() -> list[InferenceRule]:
    """Single-source shortest path on edge-weighted graphs.

    Items: ('dist', node) → log-cost from the source.
    Rules:
      source: dist(s) = 0 (axiom).
      relax: dist(u) + edge(u, v, w) |- dist(v) at u + w.

    Weight is computed by the rule's weight_fn under the Viterbi
    semiring (max of negatives = min positive distance).
    """

    def _relax_weight(bindings, premise_weights, semiring):
        # premise_weights = (dist(u), edge_weight). Times = +.
        return semiring.times(premise_weights[0], premise_weights[1])

    return [
        InferenceRule(
            name="relax",
            premises=(
                ("dist", Wildcard("u")),
                ("edge", Wildcard("u"), Wildcard("v"), Wildcard("w")),
            ),
            conclusion=("dist", Wildcard("v")),
            weight_fn=_relax_weight,
        ),
    ]


Dijkstra = DeductionSystem(
    rules=tuple(_dijkstra_rules()),
    semiring=VITERBI,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "dist",
    agenda_factory=knuth_agenda,
)
"""Single-source shortest paths by Knuth's generalization of Dijkstra's algorithm, Viterbi semiring."""


# ---------------------------------------------------------------------------
# HMM forward / Viterbi
# ---------------------------------------------------------------------------


def _hmm_forward_rules() -> list[InferenceRule]:
    """HMM forward algorithm: α(t, s) = Σ_{s'} α(t-1, s') · trans(s', s) · emit(s, x_t).

    Encoded over arity-3 items: ('alpha', t, state).
    """

    def _step(bindings, premise_weights, semiring):
        # premise_weights = (alpha(t-1, s'), trans(s', s), emit(s, x_t))
        return semiring.times(
            semiring.times(premise_weights[0], premise_weights[1]),
            premise_weights[2],
        )

    return [
        InferenceRule(
            name="forward_step",
            premises=(
                ("alpha", Wildcard("t_prev"), Wildcard("s_prev")),
                ("trans", Wildcard("s_prev"), Wildcard("s_curr")),
                ("emit", Wildcard("s_curr"), Wildcard("t_curr")),
            ),
            conclusion=("alpha", Wildcard("t_curr"), Wildcard("s_curr")),
        ),
    ]


HMM = DeductionSystem(
    rules=tuple(_hmm_forward_rules()),
    semiring=LOG_PROB,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "alpha",
    agenda_factory=cky_agenda,
)
"""The hidden Markov model forward algorithm over ``alpha`` items, log-probability semiring."""


ViterbiHMM = DeductionSystem(
    rules=tuple(_hmm_forward_rules()),
    semiring=VITERBI,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "alpha",
    agenda_factory=cky_agenda,
)
"""The hidden Markov model Viterbi algorithm: the `HMM` rules under the Viterbi semiring."""


# ---------------------------------------------------------------------------
# Edit distance (a tropical-semiring DP over alignment items)
# ---------------------------------------------------------------------------


def _edit_distance_rules() -> list[InferenceRule]:
    """Levenshtein edit distance under the tropical (Viterbi) semiring."""
    return [
        # Match: dist(i-1, j-1) + match(i, j) |- dist(i, j)
        InferenceRule(
            name="match",
            premises=(
                ("dist", Wildcard("i_p"), Wildcard("j_p")),
                (
                    "match",
                    Wildcard("i"),
                    Wildcard("j"),
                    Wildcard("i_p"),
                    Wildcard("j_p"),
                ),
            ),
            conclusion=("dist", Wildcard("i"), Wildcard("j")),
        ),
    ]


EditDistance = DeductionSystem(
    rules=tuple(_edit_distance_rules()),
    semiring=VITERBI,
    axiom_injector=lambda axioms: list(axioms),
    goal=lambda item: isinstance(item, tuple) and item[0] == "dist",
    agenda_factory=cky_agenda,
)
"""Levenshtein edit distance over ``dist`` alignment items, Viterbi semiring."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


STDLIB_DEDUCTIONS: dict[str, DeductionSystem] = {
    "CCG": CCG,
    "AB": AB,
    "STLC": STLC,
    "MLTT": MLTT,
    "Datalog": Datalog,
    "Dijkstra": Dijkstra,
    "HMM": HMM,
    "ViterbiHMM": ViterbiHMM,
    "EditDistance": EditDistance,
}
"""Mapping from name to pre-registered deduction system.

Compilation of a ``parse(NAME)`` expression in the DSL, or a
direct call from Python, resolves the name against this dict
first, then against any user-declared deductions in the
compiler's environment.
"""


__all__ = [
    "STDLIB_DEDUCTIONS",
    "CCG",
    "AB",
    "STLC",
    "MLTT",
    "Datalog",
    "Dijkstra",
    "HMM",
    "ViterbiHMM",
    "EditDistance",
]
