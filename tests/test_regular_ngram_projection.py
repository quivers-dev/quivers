"""Enumerative forward-KL and proximal-optimality tests on sparse languages."""

from dataclasses import replace
from itertools import product

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
