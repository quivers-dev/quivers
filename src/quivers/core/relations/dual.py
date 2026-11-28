"""Dual decomposition with exact local maximizers over declared string supports."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Literal

from quivers.core.relations.graph import StringVariable
from quivers.core.relations.topology import Words

type _Word = tuple[int, ...]
type _WordScores = dict[_Word, Fraction]
type _Adjustment = tuple[_Potential, ...]
type _Global = tuple[tuple[str, _Word], ...]


@dataclass(slots=True)
class _Potential:
    mean: _WordScores
    local: _WordScores

    def at(self, word: _Word) -> Fraction:
        return self.mean.get(word, Fraction(0)) + self.local.get(word, Fraction(0))

    def freeze(self) -> tuple[tuple[_Word, Fraction], ...]:
        return tuple(
            (word, value)
            for word in sorted(self.mean.keys() | self.local.keys())
            if (value := self.at(word))
        )


@dataclass(frozen=True, slots=True)
class FiniteStringScore:
    """An explicitly listed support with exact additive scores.

    A key lists one full word for each named variable. Unlisted tuples
    are infeasible. The scores may be rational or integral, and may
    have either sign.
    """

    variables: tuple[str, ...]
    scores: Mapping[Words, Fraction | int]

    def __post_init__(self) -> None:
        variables = tuple(self.variables)
        if (
            not variables
            or len(set(variables)) != len(variables)
            or any(not isinstance(name, str) or not name for name in variables)
        ):
            raise ValueError(
                "a string-score factor requires distinct nonempty variable names"
            )
        scores: dict[Words, Fraction] = {}
        for words, score in self.scores.items():
            if len(words) != len(variables):
                raise ValueError("a string-score key requires one word per variable")
            if type(score) not in (Fraction, int):
                raise TypeError("a string-score factor requires exact rational scores")
            scores[tuple(tuple(word) for word in words)] = Fraction(score)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "scores", MappingProxyType(scores))


@dataclass(frozen=True, slots=True)
class StringMapCertificate:
    """Local optima and potentials at an agreeing set of maximizers.

    Entries of `local_words`, `adjusted_scores` and `potentials` follow
    factor order, with words and potentials in each factor's variable
    order. Unlisted word potentials are zero.
    """

    local_words: tuple[Words, ...]
    adjusted_scores: tuple[Fraction, ...]
    potentials: tuple[tuple[tuple[tuple[_Word, Fraction], ...], ...], ...]


@dataclass(frozen=True, slots=True)
class StringMapResult:
    """A certified optimum, infeasibility, or a bounded unsuccessful search."""

    status: Literal["optimal", "infeasible", "limit"]
    iterations: int
    assignment: _Global | None
    score: Fraction | None
    upper_bound: Fraction | None
    certificate: StringMapCertificate | None = None


def _validate_problem(
    variables: tuple[StringVariable, ...], factors: tuple[FiniteStringScore, ...]
) -> tuple[FiniteStringScore, ...]:
    named = {variable.name: variable for variable in variables}
    mentioned = {name for factor in factors for name in factor.variables}
    if len(named) != len(variables) or not mentioned or set(named) != mentioned:
        raise ValueError(
            "string-score factor scopes must cover the distinct declared variables"
        )
    reduced = []
    for factor in factors:
        scope = [named[name] for name in factor.variables]
        supported: dict[Words, Fraction] = {}
        for words, score in factor.scores.items():
            if any(
                type(symbol) is not int
                or not 0 <= symbol < variable.tape.alphabet.cardinality
                for word, variable in zip(words, scope, strict=True)
                for symbol in word
            ):
                raise ValueError(
                    "a string-score key contains a symbol outside its variable type"
                )
            if all(
                variable.observed is None or variable.observed == word
                for word, variable in zip(words, scope, strict=True)
            ):
                supported[words] = Fraction(score)
        reduced.append(FiniteStringScore(factor.variables, supported))
    return tuple(reduced)


def _decode(factor: FiniteStringScore, shifts: _Adjustment) -> tuple[Fraction, Words]:
    return max(
        (
            (
                Fraction(score)
                + sum(
                    potential.at(word)
                    for potential, word in zip(shifts, words, strict=True)
                ),
                words,
            )
            for words, score in factor.scores.items()
        ),
        key=lambda result: result[0],
    )


def _agreed_candidate(
    copies: Mapping[str, list[tuple[_Word, _Potential]]],
) -> tuple[bool, dict[str, _Word]]:
    firsts = {name: alternatives[0][0] for name, alternatives in copies.items()}
    agrees = all(
        word == firsts[name] for name, group in copies.items() for word, _ in group
    )
    return agrees, firsts


def _score(
    factors: tuple[FiniteStringScore, ...], words: Mapping[str, _Word]
) -> Fraction | None:
    total = Fraction(0)
    for factor in factors:
        score = factor.scores.get(tuple(words[name] for name in factor.variables))
        if score is None:
            return None
        total += score
    return total


def _update(
    copies: Mapping[str, list[tuple[_Word, _Potential]]], step: Fraction
) -> None:
    for group in copies.values():
        counts = Counter(word for word, _ in group)
        if len(counts) == 1:
            continue
        # Store the common centered part once, rather than per copy.
        shared = group[0][1].mean
        for word, count in counts.items():
            shared[word] = shared.get(word, Fraction(0)) + step * Fraction(
                count, len(group)
            )
        for chosen, potential in group:
            local = potential.local
            local[chosen] = local.get(chosen, Fraction(0)) - step


def dual_string_map(
    variables: tuple[StringVariable, ...],
    factors: tuple[FiniteStringScore, ...],
    *,
    max_iterations: int,
    step_size: Fraction | int = Fraction(1),
) -> StringMapResult:
    """Maximize the sum of finite-support factors with exact local bounds.

    A multiplier is attached to each copy of a variable and word. Each
    update preserves zero total multiplier for every word. Thus the sum
    of independent adjusted maxima bounds every feasible joint score
    from above. Agreement on complete strings gives a feasible solution
    with that score. Iteration exhaustion reports bounds and no certificate.

    The declared supports are scanned independently on each iteration.
    The centered update is linear in the number of variable occurrences
    and uses exact rationals. The update step is `step_size / iteration` starting at iteration one.
    """
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if type(step_size) not in (Fraction, int) or step_size <= 0:
        raise ValueError("step_size must be a positive rational")
    prepared = _validate_problem(variables, factors)
    if any(not factor.scores for factor in prepared):
        return StringMapResult("infeasible", 0, None, None, None)
    means: dict[str, _WordScores] = {variable.name: {} for variable in variables}
    shifts: tuple[_Adjustment, ...] = tuple(
        tuple(_Potential(means[name], {}) for name in factor.variables)
        for factor in prepared
    )
    upper: Fraction | None = None
    best: Fraction | None = None
    best_words: _Global | None = None
    for iteration in range(1, max_iterations + 1):
        local = tuple(
            _decode(factor, shift)
            for factor, shift in zip(prepared, shifts, strict=True)
        )
        bound = sum((score for score, _ in local), Fraction(0))
        upper = bound if upper is None else min(upper, bound)
        copies: dict[str, list[tuple[_Word, _Potential]]] = defaultdict(list)
        for factor, shift, (_, words) in zip(prepared, shifts, local, strict=True):
            for name, word, potential in zip(
                factor.variables, words, shift, strict=True
            ):
                copies[name].append((word, potential))
        agrees, proposed = _agreed_candidate(copies)
        score = _score(prepared, proposed)
        if score is not None and (best is None or best < score):
            best = score
            best_words = tuple((v.name, proposed[v.name]) for v in variables)
        if agrees:
            if score != bound:
                raise ArithmeticError(
                    "agreeing local string maximizers must attain the bound"
                )
            certificate = StringMapCertificate(
                tuple(words for _, words in local),
                tuple(value for value, _ in local),
                tuple(
                    tuple(p.freeze() for p in local_shifts) for local_shifts in shifts
                ),
            )
            assignment = tuple((v.name, proposed[v.name]) for v in variables)
            return StringMapResult(
                "optimal", iteration, assignment, score, bound, certificate
            )
        _update(copies, Fraction(step_size) / iteration)
    return StringMapResult("limit", max_iterations, best_words, best, upper)


__all__ = [
    "FiniteStringScore",
    "StringMapCertificate",
    "StringMapResult",
    "dual_string_map",
]
