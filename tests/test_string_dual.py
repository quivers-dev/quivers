"""Independent full-assignment and dual-certificate checks."""

from collections import defaultdict
from fractions import Fraction
from itertools import product
from random import Random

import pytest

from quivers.core.relations import FiniteStringScore, StringVariable, dual_string_map
from quivers.core.relations import dual
from tests.test_relation_composition import _INPUT


def _assert_certificate(variables, factors, result):
    certificate = result.certificate
    assert certificate is not None
    primal = dict(result.assignment)
    balances = defaultdict(Fraction)
    for factor, local, best, potentials in zip(
        factors,
        certificate.local_words,
        certificate.adjusted_scores,
        certificate.potentials,
        strict=True,
    ):
        assert local == tuple(primal[name] for name in factor.variables)
        local_tables = tuple(dict(entries) for entries in potentials)
        for name, entries in zip(factor.variables, potentials, strict=True):
            for word, price in entries:
                balances[name, word] += price
        bound = max(
            score
            + sum(
                table.get(word, 0)
                for table, word in zip(local_tables, words, strict=True)
            )
            for words, score in factor.scores.items()
        )
        adjusted = factor.scores[local] + sum(
            table.get(word, 0) for table, word in zip(local_tables, local, strict=True)
        )
        assert adjusted == bound == best
    assert not any(balances.values())
    assert sum(certificate.adjusted_scores) == result.score == result.upper_bound
    assert set(primal) == {variable.name for variable in variables}


def _two_factors():
    variable = (StringVariable("separator", _INPUT),)
    factors = (
        FiniteStringScore(("separator",), {((0,),): 2, ((1,),): 0}),
        FiniteStringScore(("separator",), {((0,),): 0, ((1,),): 3}),
    )
    return variable, factors


def test_dual_gap_closes_only_when_exact_local_choices_agree():
    variables, factors = _two_factors()
    limited = dual_string_map(
        variables, factors, max_iterations=1, step_size=Fraction(5, 2)
    )
    assert limited.status == "limit"
    assert limited.certificate is None
    assert limited.score == 2
    assert limited.upper_bound == 5

    result = dual_string_map(
        variables, factors, max_iterations=2, step_size=Fraction(5, 2)
    )
    assert result.status == "optimal"
    assert result.iterations == 2
    assert result.assignment == (("separator", (1,)),)
    _assert_certificate(variables, factors, result)


def test_every_dual_bound_dominates_exhaustive_joint_maximum():
    generator = Random(54)
    variables = tuple(StringVariable(name, _INPUT) for name in "xyz")
    words = ((), (0,), (1,))
    for _ in range(22):
        factors = tuple(
            FiniteStringScore(
                scope,
                {
                    pair: Fraction(generator.randrange(-8, 8), 7)
                    for pair in product(words, repeat=2)
                },
            )
            for scope in (("x", "y"), ("y", "z"), ("z", "x"))
        )
        maximum = max(
            sum(
                factor.scores[
                    tuple(assignment["xyz".index(name)] for name in factor.variables)
                ]
                for factor in factors
            )
            for assignment in product(words, repeat=3)
        )
        result = dual_string_map(
            variables, factors, max_iterations=8, step_size=Fraction(9, 5)
        )
        assert result.upper_bound is not None and result.upper_bound >= maximum
        if result.score is not None:
            assert result.score <= maximum
        if result.status == "optimal":
            _assert_certificate(variables, factors, result)
            assert result.score == maximum


def test_full_word_agreement_rejects_the_same_length_mutant(monkeypatch):
    variables, factors = _two_factors()
    agree = dual._agreed_candidate

    def mutant(copies):
        _, candidate = agree(copies)
        same_length = all(
            len(word) == len(candidate[name])
            for name, group in copies.items()
            for word, _ in group
        )
        return same_length, candidate

    monkeypatch.setattr(dual, "_agreed_candidate", mutant)
    with pytest.raises(ArithmeticError, match="must attain the bound"):
        dual_string_map(variables, factors, max_iterations=1)


def test_observations_and_infeasible_or_invalid_support_are_explicit():
    variables, factors = _two_factors()
    supported = (StringVariable("separator", _INPUT, (0,)),)
    result = dual_string_map(supported, factors, max_iterations=2)
    assert result.status == "optimal" and result.score == 2
    empty = (StringVariable("separator", _INPUT, ()),)
    assert dual_string_map(empty, factors, max_iterations=2).status == "infeasible"
    with pytest.raises(ValueError, match="outside its variable type"):
        dual_string_map(
            variables,
            (FiniteStringScore(("separator",), {((2,),): 1}),),
            max_iterations=1,
        )
