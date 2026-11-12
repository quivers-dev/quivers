"""Equality, hashing, copying, and pickling of the quivers models whose
opaque fields hold live runtime objects: each such field compares by
identity, every other field by value, and copies and pickles keep the
opaque values.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal

from quivers.analysis.scope import ScopedRef
from quivers.formulas import Formula, families, formula_from_data
from quivers.formulas._fit import BayesianFit
from quivers.monadic.algebraic import EffectSignature, Handler
from quivers.monadic.bridges import ArrowMonad, CoKleisli, Kleisli
from quivers.core.algebras import GODEL
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta


def _formula() -> Formula:
    frame = pd.DataFrame(
        {"y": [0.1, 0.4, 0.2, 0.9], "x": [1.0, 2.0, 3.0, 4.0], "g": list("aabb")}
    )
    return formula_from_data("y ~ x + (1 | g)", frame)


def test_bridges_compare_their_wrapped_object_by_identity():
    monad = FuzzyPowersetMonad(GODEL)
    assert Kleisli(monad=monad) == Kleisli(monad=monad)
    assert hash(Kleisli(monad=monad)) == hash(Kleisli(monad=monad))
    assert Kleisli(monad=monad) != Kleisli(monad=FuzzyPowersetMonad(GODEL))
    arrow = torch.nn.Linear(1, 1)
    assert ArrowMonad(arrow=arrow) == ArrowMonad(arrow=arrow)
    assert ArrowMonad(arrow=arrow) != ArrowMonad(arrow=torch.nn.Linear(1, 1))
    assert CoKleisli(comonad=arrow) == CoKleisli(comonad=arrow)
    assert CoKleisli(comonad=arrow) != CoKleisli(comonad=torch.nn.Linear(1, 1))


def test_bridges_keep_their_wrapped_object_through_copy_and_pickle():
    monad = FuzzyPowersetMonad(GODEL)
    kleisli = Kleisli(monad=monad)
    assert copy.copy(kleisli).monad is monad
    assert copy.copy(kleisli) == kleisli
    for clone in (copy.deepcopy(kleisli), pickle.loads(pickle.dumps(kleisli))):
        assert isinstance(clone.monad, FuzzyPowersetMonad)
        assert clone.monad is not monad
        assert clone != kleisli


def test_handler_compares_signature_by_value_and_clauses_by_identity():
    target = FuzzyPowersetMonad(GODEL)
    clauses: dict[str, torch.nn.Module] = {}
    left = Handler(
        signature=EffectSignature(name="E"),
        target=target,
        operation_clauses=clauses,
    )
    same = Handler(
        signature=EffectSignature(name="E"),
        target=target,
        operation_clauses=clauses,
    )
    assert left == same
    assert hash(left) == hash(same)
    assert left != same.with_(signature=EffectSignature(name="F"))
    assert left != Handler(
        signature=EffectSignature(name="E"),
        target=FuzzyPowersetMonad(GODEL),
        operation_clauses=clauses,
    )
    assert left != Handler(
        signature=EffectSignature(name="E"), target=target, operation_clauses={}
    )
    shallow = copy.copy(left)
    assert shallow.target is target
    assert shallow.operation_clauses is clauses
    restored = pickle.loads(pickle.dumps(left))
    assert isinstance(restored.target, FuzzyPowersetMonad)
    assert restored.signature == left.signature


def test_scoped_ref_compares_node_by_identity():
    node = torch.nn.Linear(1, 1)
    ref = ScopedRef(name="w", kind="sample-site", path="m::w", node=node)
    assert ref == ScopedRef(name="w", kind="sample-site", path="m::w", node=node)
    assert ref != ScopedRef(
        name="w", kind="sample-site", path="m::w", node=torch.nn.Linear(1, 1)
    )
    assert ref != ScopedRef(name="w", kind="sample-site", path="n::w", node=node)
    assert copy.copy(ref).node is node
    restored = pickle.loads(pickle.dumps(ref))
    assert isinstance(restored.node, torch.nn.Linear)
    assert restored.path == "m::w"


def test_family_meta_compares_classes_by_value():
    meta = FAMILY_META["Normal"]
    rebuilt = FamilyMeta(
        qvr_name=meta.qvr_name,
        distribution_class=meta.distribution_class,
        target_names=meta.target_names,
        arg_aliases=meta.arg_aliases,
        quivers_class=meta.quivers_class,
        structured_lowering=meta.structured_lowering,
        event_rank=meta.event_rank,
    )
    assert rebuilt == meta
    assert hash(rebuilt) == hash(meta)
    assert meta.distribution_class is Normal
    laplace = FamilyMeta(
        qvr_name=meta.qvr_name,
        distribution_class=torch.distributions.Laplace,
        target_names=meta.target_names,
        arg_aliases=meta.arg_aliases,
        quivers_class=meta.quivers_class,
        structured_lowering=meta.structured_lowering,
        event_rank=meta.event_rank,
    )
    assert rebuilt != laplace
    for clone in (
        copy.copy(meta),
        copy.deepcopy(meta),
        pickle.loads(pickle.dumps(meta)),
    ):
        assert clone == meta
        assert clone.distribution_class is Normal


def test_bayesian_fit_compares_formula_by_value_and_live_objects_by_identity():
    program = torch.nn.Linear(1, 1)
    posterior = torch.nn.Linear(1, 1)
    observations = {"y": torch.zeros(4)}

    def build(formula: Formula, **overrides: dict[str, str]) -> BayesianFit:
        return BayesianFit(
            formula=formula,
            family=families["gaussian"],
            program=program,  # type: ignore[arg-type]
            posterior=posterior,  # type: ignore[arg-type]
            observations=observations,
            priors=overrides.get("priors", {"x": "Normal(0, 1)"}),
        )

    fit = build(_formula())
    assert fit == build(_formula())
    assert hash(fit) == hash(build(_formula()))
    assert fit != build(_formula(), priors={"x": "Normal(0, 2)"})
    response = np.array([0.1, 0.4, 0.2, 1.0])
    other_formula = Formula(
        formula=fit.formula.formula,
        response_name=fit.formula.response_name,
        fixed_columns=fit.formula.fixed_columns,
        random_terms=fit.formula.random_terms,
        response_values=response,
        group_levels=fit.formula.group_levels,
        group_indices=fit.formula.group_indices,
    )
    assert fit != build(other_formula)
    assert fit != BayesianFit(
        formula=fit.formula,
        family=fit.family,
        program=torch.nn.Linear(1, 1),  # type: ignore[arg-type]
        posterior=posterior,  # type: ignore[arg-type]
        observations=observations,
        priors={"x": "Normal(0, 1)"},
    )
    shallow = copy.copy(fit)
    assert shallow == fit
    assert shallow.program is program
    assert shallow.observations is observations
    restored = pickle.loads(pickle.dumps(fit))
    assert restored.formula == fit.formula
    assert restored.priors == {"x": "Normal(0, 1)"}
    torch.testing.assert_close(restored.observations["y"], observations["y"])
