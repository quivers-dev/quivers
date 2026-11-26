"""Programs in the surface language used by the checks.

A source is read through `program`, so a mutation control can replace a
source by a program that denotes another law and run the same check.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from quivers.dsl import loads
from quivers.program import Program
from quivers.stochastic.agenda import DeductionSystem

LEXICAL = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=SEMIRING, start=S, depth=1]
    atoms S, V, span, a, b
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S) #[learnable]
    rule lift : span(I, J, V) |- span(I, J, S) #[learnable]
    lexicon
        "a" : S = a
        "b" : V = b
        "a" : V = a
"""
"""An ambiguous lexical deduction with two rule weights."""

NORMALIZED = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=1, normalize=lhs]
    atoms S, V, span, a, b
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S) #[learnable]
    rule lift : span(I, J, V) |- span(I, J, S) #[learnable]
    lexicon
        "a" : S = a
        "b" : V = b
        "a" : V = a
"""
"""A grammar whose productions are normalized by left-hand side."""

CHAINED = """
object Term : FinSet 4
deduction M : Term -> Term [semiring=SEMIRING, start=S, depth=1TOLERANCE]
    atoms S, T, span, a
    rule there : span(I, J, S) |- span(I, J, T) #[learnable]
    rule back : span(I, J, T) |- span(I, J, S) #[learnable]
    lexicon
        "a" : S = a
"""
"""A cycle of two unary rules over one span."""

NONLEXICAL = """
object Term : FinSet 4
deduction W : Term -> Term [semiring=SEMIRING, start=C, depth=1]
    atoms A, B, C
    rule first : A |- B #[learnable]
    rule second : B, A |- C #[learnable]
    rule direct : A |- C #[learnable]
    rule twice : B, B |- C #[learnable]
"""
"""A deduction without a lexicon, with a repeated premise."""

BARE = """
object Term : FinSet 4
deduction I : Term -> Term [semiring=SEMIRING, start=G]
    atoms A, B, K, G, p
    rule ab : A |- B
    rule wrap : K |- p(K)
    rule keep : X, K |- X
    rule goal : B, p(K) |- G
"""
"""A deduction with a rule over every judgment form: ``keep`` keeps any item."""

CYCLE = """
object Term : FinSet 2
deduction Cycle : Term -> Term [semiring=SEMIRING, start=ATOLERANCE]
    atoms A
    rule loop : A |- A #[learnable]
"""
"""A deduction without a lexicon whose one item derives itself."""

GROUPED = """
object K : FinSet 2
object G : FinSet 2
object Obs : FinSet 4
object V : Real 1
program mixture : Obs -> V
    sample prior <- Dirichlet(concentration=1.0) [over=K]
    marginalize a : K <- Categorical(probs=prior) [over=G]
        let rate = 0.125 + 0.5 * real(a)
        observe y : Obs <- Bernoulli(probs=rate) [via=idx]
    return prior
export mixture
"""
"""One draw for each of two groups."""

UNGROUPED = """
object K : FinSet 2
object Obs : FinSet 4
object V : Real 1
program mixture : Obs -> V
    sample prior <- Dirichlet(concentration=1.0) [over=K]
    marginalize a : K <- Categorical(probs=prior) [reduction=logsumexp]
        let rate = 0.125 + 0.5 * real(a)
        observe y : Obs <- Bernoulli(probs=rate)
    return prior
export mixture
"""
"""One draw shared by every row."""

HMM = """
object State : FinSet 2
object Symbol : FinSet 3
object StateProbs : Real 2
program hmm_step(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition_rows[z_prev])
        observe symbol <- Categorical(probs=emission_rows[z])
    return z
define chain = scan(hmm_step, init=initial_row)
program hmm : Symbol -> StateProbs
    sample initial_row <- Dirichlet(concentration=1.0) [over=State]
    sample transition_rows : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission_rows : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return initial_row
export hmm
"""
"""A chain of states summed out along a sequence."""

CONJUGATE = """
object Unit : FinSet 1
object V : Real 1
program pair : Unit -> V
    sample t <- Normal(loc=0.5, scale=2.0)
    observe y <- Normal(loc=t, scale=0.5)
    return t
export pair
"""
"""A normal mean with a normal prior and one observation."""

BETA = """
object Obs : FinSet 4
object V : Real 1
program coins : Obs -> V
    sample theta <- Beta(concentration1=2.0, concentration0=3.0)
    observe y : Obs <- Bernoulli(probs=theta)
    return theta
export coins
"""
"""Coin flips with a Beta prior."""

DISCRETE = """
object Unit : FinSet 1
object V : Real 1
program latent : Unit -> V
    sample a <- Bernoulli(probs=0.25)
    let location = 2.0 * real(a)
    observe y <- Normal(loc=location, scale=1.0)
    return location
export latent
"""
"""A discrete latent that is drawn and not summed out."""

PARSED = """
object Term : FinSet 4
object Sentence : FinSet 2
object Weight : Real 1
deduction G : Term -> Term [semiring=LogProb, start=S, depth=1, normalize=lhs]
    atoms S, V, span, a, b
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S)
    rule lift : span(I, J, V) |- span(I, J, S)
    lexicon
        "a" : S = a
        "b" : V = b
        "a" : V = a
program scored : Term -> Weight
    sample p_s <- Dirichlet(concentration=[1.0, 1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    let chart = parse(G, corpus, Sentence, ("S", p_s), ("V", p_v))
    score evidence = chart.goal_weight()
    return evidence
export scored
"""
"""A grammar with a prior on its productions and two observed sentences."""

RECURSIVE = """
object Unit : FinSet 1
object V : Real 1
define triangle(n : Int) : Real !{} =
    if n <= 0 then
        return 0.0
    else
        let rest <- triangle(n - 1)
        return rest + 1.0
program recursive : Unit -> V
    sample t <- Normal(loc=0.0, scale=1.0)
    let k <- triangle(3)
    let m = t + k
    observe y <- Normal(loc=m, scale=1.0)
    return t
export recursive
"""
"""A program that calls a recursive computation."""

STRUCTURED = """
signature Trees
    sorts
        T : object [dim=2]
    constructors
        Leaf : -> T
        Node : T, T -> T
decoder Dec : Trees [depth=2]
    body |-> recursive
object Unit : FinSet 1
object V : Real 2
program drawn : Unit -> V
    sample h <- Normal(loc=[0.0, 0.0], scale=[1.0, 1.0])
    sample t <- Dec(h)
    return h
export drawn
"""
"""A program that draws a tree."""

SOURCES: dict[str, str] = {
    "lexical": LEXICAL,
    "normalized": NORMALIZED,
    "chained": CHAINED,
    "nonlexical": NONLEXICAL,
    "cycle": CYCLE,
    "bare": BARE,
    "grouped": GROUPED,
    "ungrouped": UNGROUPED,
    "hmm": HMM,
    "conjugate": CONJUGATE,
    "beta": BETA,
    "discrete": DISCRETE,
    "parsed": PARSED,
    "recursive": RECURSIVE,
    "structured": STRUCTURED,
}
"""The sources by name."""

_CACHE: dict[str, Program] = {}


def program(name: str, **values: str) -> Program:
    """Compile a source.

    Parameters
    ----------
    name : str
        The source's name.
    **values : str
        Replacements of capitalized words in the source.

    Returns
    -------
    Program
        The compiled program; one object per distinct text.
    """
    text = SOURCES[name]
    for word, value in values.items():
        text = text.replace(word, value)
    if text not in _CACHE:
        _CACHE[text] = loads(text)
    return _CACHE[text]


@contextmanager
def replaced(name: str, text: str) -> Iterator[None]:
    """Use another text for a source.

    Parameters
    ----------
    name : str
        The source's name.
    text : str
        The replacement.

    Yields
    ------
    None
        Inside, the source is replaced.
    """
    original = SOURCES[name]
    SOURCES[name] = text
    try:
        yield
    finally:
        SOURCES[name] = original


def deduction(compiled: Program, name: str) -> DeductionSystem:
    """A deduction system of a compiled program.

    Parameters
    ----------
    compiled : Program
        The program.
    name : str
        The deduction's name.

    Returns
    -------
    DeductionSystem
        The system.
    """
    found = vars(compiled)["deductions"][name]
    assert isinstance(found, DeductionSystem)
    return found
