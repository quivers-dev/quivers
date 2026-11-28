"""Enumerative forward-KL and proximal-optimality tests on sparse languages."""

from dataclasses import replace
from itertools import product
from math import log

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    Arc,
    Boundary,
    NgramFamily,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    penalized_regular_ngram_projection,
)

_TAPE = StringSet(alphabet=FinSet(name="Two", cardinality=2))
_WORDS = tuple(word for n in range(3) for word in product(range(2), repeat=n))
_FEATURES = ((0,), (0, 0), (0, 1), (1,), (1, 0), (1, 1))
_POLICY = dict(max_states=200, max_work=1200, max_iterations=2500)


def _language(first=0.0, second=0.0):
    topology = RelationTopology(
        (_TAPE,),
        0,
        3,
        tuple(
            Arc(state, state + 1, (symbol,), symbol)
            for state in range(2)
            for symbol in range(2)
        ),
        (Boundary(0, 2),),
        tuple(Boundary(state, 2) for state in range(3)),
        ("a", "b", "unit"),
    )
    weights = nn.Parameter(torch.tensor([first, second, 0], dtype=torch.float64))
    return RationalRelation(topology, weights, algebra=LOG_PROB), weights


def _feature_rows(like):
    return like.new_tensor(
        [
            [
                sum(word[i : i + len(gram)] == gram for i in range(len(word)))
                for gram in _FEATURES
            ]
            for word in _WORDS
        ]
    )


def test_regular_proximal_fit_matches_dense_forward_kl_and_feature_expectations():
    base, base_weights = _language()
    model, _ = _language(1.2, -0.25)
    fit = penalized_regular_ngram_projection(
        base, model, _FEATURES, penalty=0.15, tolerance=3e-7, **_POLICY
    )
    assert fit.status == "converged"
    observed = torch.stack([fit.message.weight((word,)) for word in _WORDS])
    prior = torch.stack([base.weight((word,)) for word in _WORDS]).detach()
    raw_target = (
        prior + torch.stack([model.weight((word,)) for word in _WORDS]).detach()
    )
    target = raw_target.softmax(0)
    rows = _feature_rows(prior)
    oracle_scores = prior + rows @ fit.coefficients
    log_q = oracle_scores.log_softmax(0)
    log_p = raw_target.log_softmax(0)
    torch.testing.assert_close(observed, oracle_scores)
    torch.testing.assert_close(fit.log_normalizer, oracle_scores.logsumexp(0))
    torch.testing.assert_close(fit.log_tilt_mass, raw_target.logsumexp(0))
    torch.testing.assert_close(fit.target_moments, target @ rows)
    torch.testing.assert_close(fit.fitted_moments, log_q.exp() @ rows)
    torch.testing.assert_close(
        fit.objective_change,
        (target * (log_p - log_q)).sum()
        - (target * (log_p - prior.log_softmax(0))).sum()
        + fit.regularization,
    )
    dense = NgramFamily(_WORDS, 2).project_penalized(
        raw_target,
        base_log_weights=prior,
        penalty=0.15,
        tolerance=3e-7,
        max_iterations=2500,
    )
    assert dense.status == "converged"
    torch.testing.assert_close(dense.log_probabilities, log_q, atol=1e-5, rtol=1e-5)
    assert 0 < len(fit.active_features) < len(_FEATURES)
    assert fit.retained_states < fit.full_candidate_states
    with torch.no_grad():
        base_weights[0] += 100
    torch.testing.assert_close(fit.message.weight(((0, 0),)), observed[-4])


def test_fixed_step_matches_global_covariance_bound_and_counts_moment_evaluations():
    base, weights = _language()
    model, _ = _language(1.2, -0.25)
    rows = _feature_rows(weights)
    target_scores = weights.new_tensor(
        [sum((1.2, -0.25)[symbol] for symbol in word) for word in _WORDS]
    )
    tilt = target_scores.softmax(0)
    direction = tilt @ rows - rows.mean(0)
    # The two unigram count ranges are 2 and the four bigram ranges are
    # 1. The global denominator is (2 * 4 + 4) / 4 = 3.
    fixed = penalized_regular_ngram_projection(
        base,
        model,
        _FEATURES,
        penalty=0.0,
        tolerance=1e-10,
        optimizer="fixed_lipschitz",
        **(_POLICY | dict(max_iterations=1)),
    )
    assert fixed.optimizer == "fixed_lipschitz" and fixed.status == "limit"
    assert fixed.iterations == 1 and fixed.moment_evaluations == 3
    torch.testing.assert_close(fixed.coefficients, direction / 3)
    torch.testing.assert_close(fixed.target_moments, tilt @ rows)
    torch.testing.assert_close(
        fixed.fitted_moments, (rows @ (direction / 3)).softmax(0) @ rows
    )

    unigram_rows = rows[:, (0, 3)]
    for optimizer in ("backtracking", "fixed_lipschitz"):
        fit = penalized_regular_ngram_projection(
            base,
            model,
            _FEATURES[::3],
            penalty=0.0,
            tolerance=1e-9,
            optimizer=optimizer,
            **_POLICY,
        )
        assert fit.status == "converged" and fit.optimizer == optimizer
        assert fit.moment_evaluations >= 2 + fit.iterations
        if optimizer == "fixed_lipschitz":
            assert fit.moment_evaluations == 2 + fit.iterations
        torch.testing.assert_close(
            (unigram_rows @ fit.coefficients).softmax(0), tilt, atol=1e-7, rtol=1e-7
        )


def test_backtracking_counts_rejected_sparse_moment_trials():
    length = 8
    topology = RelationTopology(
        (_TAPE,),
        0,
        length + 1,
        tuple(Arc(i, i + 1, (0,), 0) for i in range(length)),
        (Boundary(0, 0),),
        (Boundary(0, 0), Boundary(length, 1)),
        ("unit", "odds"),
    )
    base = RationalRelation(
        topology, torch.zeros(2, dtype=torch.float64), algebra=LOG_PROB
    )
    model = RationalRelation(
        topology, base._weights().new_tensor([0.0, log(7)]), algebra=LOG_PROB
    )
    exact_target = length * 7 / 8
    for optimizer in ("backtracking", "fixed_lipschitz"):
        fit = penalized_regular_ngram_projection(
            base,
            model,
            ((0,),),
            penalty=0.0,
            tolerance=1e-10,
            optimizer=optimizer,
            **(_POLICY | dict(max_iterations=1)),
        )
        assert fit.iterations == 1 and fit.target_moments.item() == pytest.approx(
            exact_target
        )
        if optimizer == "backtracking":
            assert fit.moment_evaluations == 7
        else:
            assert fit.moment_evaluations == 3
            assert fit.coefficients.item() == pytest.approx(
                (exact_target - length / 2) / 16
            )


def test_penalty_selects_the_base_and_representation_changes_with_budget():
    base, _ = _language()
    model, _ = _language(1.2, -0.25)
    strong = penalized_regular_ngram_projection(
        base, model, _FEATURES, penalty=3.0, tolerance=3e-7, **_POLICY
    )
    assert strong.status == "converged"
    assert strong.iterations == 0
    assert strong.active_features == ()
    assert strong.retained_states == base.topology.n_states
    torch.testing.assert_close(strong.log_base_mass, strong.log_normalizer)
    limited = penalized_regular_ngram_projection(
        base,
        model,
        _FEATURES,
        penalty=0.15,
        tolerance=3e-7,
        **(_POLICY | dict(max_iterations=0)),
    )
    assert limited.status == "limit" and limited.optimality_residual > 0


def test_resource_failure_and_nonacyclic_base_are_separate():
    base, _ = _language()
    model, _ = _language()
    with pytest.raises(RelationLimitError, match="max_states"):
        penalized_regular_ngram_projection(
            base,
            model,
            _FEATURES,
            penalty=0.15,
            tolerance=3e-7,
            **(_POLICY | dict(max_states=1)),
        )
    topology = replace(base.topology, arcs=(*base.topology.arcs, Arc(2, 2, (0,), 0)))
    cyclic = RationalRelation(topology, base.module(), algebra=LOG_PROB)
    with pytest.raises(ValueError, match="acyclic productive base support"):
        penalized_regular_ngram_projection(
            cyclic, model, _FEATURES, penalty=0.15, tolerance=3e-7, **_POLICY
        )


def test_unpenalized_fit_recovers_representable_tilt_and_scale():
    base, _ = _language()
    likelihood, _ = _language(0.5, -0.25)
    with torch.inference_mode():
        fit = penalized_regular_ngram_projection(
            base, likelihood, ((0,), (1,)), penalty=0.0, tolerance=3e-7, **_POLICY
        )
    assert fit.status == "converged"
    raw_target = torch.stack([likelihood.weight((word,)) for word in _WORDS])
    fitted = torch.stack([fit.message.weight((word,)) for word in _WORDS])
    torch.testing.assert_close(
        fitted + fit.log_site_scale, raw_target, rtol=1e-5, atol=1e-5
    )
    assert fit.regularization.item() == 0


def test_representable_four_word_tilt_meets_a_tight_iteration_budget():
    base, weights = _language()
    unit = RationalRelation(
        replace(base.topology, final=(Boundary(2, 2),)), weights, algebra=LOG_PROB
    )
    words = tuple(product(range(2), repeat=2))
    topology = RelationTopology(
        (_TAPE,),
        0,
        6,
        tuple(
            arc
            for state, word in enumerate(words, 1)
            for arc in (Arc(0, state, word[:1], 0), Arc(state, 5, word[1:], state))
        ),
        (Boundary(0, 0),),
        (Boundary(5, 0),),
        ("unit", "aa", "ab", "ba", "bb"),
    )
    scores = weights.new_tensor([0.0, 0.8, -0.2, 0.4, -0.5])
    likelihood = RationalRelation(topology, scores, algebra=LOG_PROB)
    fit = penalized_regular_ngram_projection(
        unit,
        likelihood,
        _FEATURES,
        penalty=0.0,
        tolerance=1e-8,
        **(_POLICY | dict(max_iterations=256)),
    )
    assert fit.status == "converged"
    actual = torch.stack([fit.message.weight((word,)) for word in words]).softmax(0)
    torch.testing.assert_close(actual, scores[1:].softmax(0), atol=3e-8, rtol=3e-8)
