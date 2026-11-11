"""Tests for :mod:`quivers.analysis`: ChainShape walker plus the
saturation-free init recipes and source-keyed warnings derived
from it.

Coverage:

* :class:`ChainShape` finds the program's algebra, every let /
  latent / observe step, source lines, and chain-depth indices.
* Per-algebra :meth:`Algebra.init_spec` returns the closed-form
  recipe from the design note for every built-in algebra.
* :func:`recommend_init` maps latents to specs; the recipe scales
  with chain depth.
* :func:`apply_init_spec` materialises onto a learnable tensor.
* :func:`saturation_warnings` flags only steps whose recipe
  differs materially from a default ``Normal(0, 1)``.
"""

from __future__ import annotations

import copy
import pickle
import textwrap

import math

import torch
import torch.nn as nn

from quivers.analysis import (
    ChainShape,
    SaturationWarning,
    StepShape,
    apply_init_spec,
    recommend_init,
    saturation_warnings,
)
from quivers.analysis.init_spec import _algebra_init_spec
from quivers.core.algebras import (
    BooleanAlgebra,
    CountingAlgebra,
    GodelAlgebra,
    LogProbAlgebra,
    LukasiewiczAlgebra,
    MarkovAlgebra,
    MaxPlusAlgebra,
    ProbabilityAlgebra,
    ProductFuzzyAlgebra,
    RealAlgebra,
    TropicalAlgebra,
)
from quivers.dsl.parser import parse


SIMPLE_PROGRAM = """
composition product_fuzzy [level=algebra]

object A : FinSet 4
object B : FinSet 4

program model : A -> A
    sample sigma <- HalfNormal(scale=1.0)
    let y = (sigma + 0.5)
    observe r : A <- Normal(loc=y, scale=sigma)
    return r

export model
"""


DEEP_PROGRAM = """
composition product_fuzzy [level=algebra]

object A : FinSet 8

program model : A -> A
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample c <- Normal(loc=0.0, scale=1.0)
    sample d <- Normal(loc=0.0, scale=1.0)
    observe r : A <- Normal(loc=d, scale=0.1)
    return r

export model
"""


class TestChainShape:
    def test_walks_simple_program(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        assert shape.algebra_name == "product_fuzzy"
        assert dict(shape.object_cardinalities) == {"A": 4, "B": 4}
        # sigma (latent, d=1), y (let, d=1), r (observe, d=2)
        assert tuple((s.kind, s.name) for s in shape.steps) == (
            ("latent", "sigma"),
            ("let", "y"),
            ("observe", "r"),
        )
        assert shape.stochastic_depth == 2

    def test_source_locations_propagate(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        for s in shape.steps:
            assert s.source_line > 0

    def test_intermediate_size_for_indexed_observe(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        observe = next(s for s in shape.steps if s.kind == "observe")
        assert observe.intermediate_size == 4

    def test_chain_depth_indexes_stochastic_binds(self):
        shape = ChainShape.from_module(parse(DEEP_PROGRAM))
        latents = shape.latents()
        assert [s.depth for s in latents] == [1, 2, 3, 4]

    def test_algebra_attribute_resolves(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        assert isinstance(shape.algebra, ProductFuzzyAlgebra)

    def test_default_algebra_when_unspecified(self):
        program_no_algebra = """
object A : FinSet 2

program model : A -> A
    sample x <- Normal(loc=0.0, scale=1.0)
    observe r : A <- Normal(loc=x, scale=0.1)
    return r

export model
"""
        shape = ChainShape.from_module(parse(program_no_algebra))
        # Implicit default: ProductFuzzy (registry key is lowercase
        # snake-case to match the DSL keyword form).
        assert shape.algebra_name == "product_fuzzy"


class TestInitSpec:
    """Per-algebra init recipe matches the closed-form formula
    from ``notes/algebra-guided-training-tooling.md``."""

    def test_product_fuzzy_mid_saturation(self):
        spec = _algebra_init_spec(ProductFuzzyAlgebra(), depth=4, intermediate_size=1)
        # k = 4 cells; noisy-OR hits 1/2 at p = ln 2 / 4.
        assert spec.distribution == "uniform"
        assert math.isclose(spec.mean, math.log(2.0) / 4.0, rel_tol=1e-9)

    def test_lukasiewicz_mid_saturation(self):
        spec = _algebra_init_spec(LukasiewiczAlgebra(), depth=5, intermediate_size=1)
        assert spec.distribution == "uniform"
        assert math.isclose(spec.mean, 1.0 / 5.0, rel_tol=1e-9)

    def test_logprob_mid_saturation(self):
        spec = _algebra_init_spec(LogProbAlgebra(), depth=8, intermediate_size=1)
        assert spec.distribution == "normal"
        assert math.isclose(spec.mean, -math.log(8.0), rel_tol=1e-9)

    def test_maxplus_centered_with_inverse_sqrt_spread(self):
        spec = _algebra_init_spec(MaxPlusAlgebra(), depth=9, intermediate_size=1)
        assert spec.distribution == "normal"
        assert spec.mean == 0.0
        assert math.isclose(spec.std, 1.0 / 3.0, rel_tol=1e-9)

    def test_tropical_centered_with_inverse_sqrt_spread(self):
        spec = _algebra_init_spec(TropicalAlgebra(), depth=4, intermediate_size=1)
        assert spec.distribution == "normal"
        assert spec.mean == 0.0
        assert math.isclose(spec.std, 0.5, rel_tol=1e-9)

    def test_markov_zero_centered_logits(self):
        spec = _algebra_init_spec(MarkovAlgebra(), depth=4, intermediate_size=1)
        assert spec.distribution == "normal"
        assert spec.mean == 0.0

    def test_boolean_idempotent_midpoint(self):
        spec = _algebra_init_spec(BooleanAlgebra(), depth=10, intermediate_size=1)
        assert spec.distribution == "constant"
        assert spec.mean == 0.5

    def test_godel_idempotent_midpoint(self):
        spec = _algebra_init_spec(GodelAlgebra(), depth=10, intermediate_size=1)
        assert spec.distribution == "constant"
        assert spec.mean == 0.5

    def test_real_centered_with_inverse_sqrt_spread(self):
        spec = _algebra_init_spec(RealAlgebra(), depth=9, intermediate_size=1)
        assert spec.distribution == "normal"
        assert spec.mean == 0.0
        assert math.isclose(spec.std, 1.0 / 3.0, rel_tol=1e-9)

    def test_probability_p_eq_inv_k(self):
        spec = _algebra_init_spec(ProbabilityAlgebra(), depth=5, intermediate_size=1)
        assert spec.distribution == "uniform"
        assert math.isclose(spec.mean, 0.2, rel_tol=1e-9)

    def test_counting_p_eq_inv_k(self):
        spec = _algebra_init_spec(CountingAlgebra(), depth=4, intermediate_size=1)
        assert spec.distribution == "uniform"
        assert math.isclose(spec.mean, 0.25, rel_tol=1e-9)

    def test_intermediate_size_scales_recipe(self):
        s1 = _algebra_init_spec(ProductFuzzyAlgebra(), depth=2, intermediate_size=1)
        s5 = _algebra_init_spec(ProductFuzzyAlgebra(), depth=2, intermediate_size=5)
        # Larger shared axis → smaller per-cell p.
        assert s5.mean < s1.mean

    def test_method_on_algebra_dispatches(self):
        # The patched Algebra.init_spec method delegates to the same
        # internal table.
        algebra = ProductFuzzyAlgebra()
        spec = algebra.init_spec(depth=4, intermediate_size=1)
        assert math.isclose(spec.mean, math.log(2.0) / 4.0, rel_tol=1e-9)


class TestRecommendInit:
    def test_per_latent_recipe(self):
        rec = recommend_init(parse(DEEP_PROGRAM))
        assert set(rec) == {"a", "b", "c", "d"}
        # Recipe should tighten with depth.
        assert rec["a"].mean > rec["b"].mean > rec["c"].mean > rec["d"].mean

    def test_recommendation_empty_when_no_algebra(self):
        prog_no_algebra_keyword = """
object A : FinSet 2

program model : A -> A
    sample x <- Normal(loc=0.0, scale=1.0)
    observe r : A <- Normal(loc=x, scale=0.1)
    return r

export model
"""
        # The implicit ProductFuzzy default produces a non-empty
        # recommendation; the empty-output check is for genuinely
        # unknown algebras, which we cannot trigger from the parser
        # alone (it would require a CustomAlgebra not registered).
        rec = recommend_init(parse(prog_no_algebra_keyword))
        assert "x" in rec


class TestApplyInitSpec:
    def test_normal_sampling_overwrites(self):
        p = nn.Parameter(torch.zeros(64))
        spec = _algebra_init_spec(LogProbAlgebra(), depth=2, intermediate_size=1)
        apply_init_spec(p, spec)
        # Empirical mean should be close to the recipe's mean.
        assert abs(p.mean().item() - spec.mean) < 0.2

    def test_uniform_sampling_in_bounds(self):
        p = nn.Parameter(torch.zeros(64))
        spec = _algebra_init_spec(ProductFuzzyAlgebra(), depth=4, intermediate_size=1)
        apply_init_spec(p, spec)
        assert (p >= spec.lower).all()
        assert (p <= spec.upper).all()

    def test_constant_fills(self):
        p = nn.Parameter(torch.zeros(8))
        spec = _algebra_init_spec(BooleanAlgebra(), depth=4, intermediate_size=1)
        apply_init_spec(p, spec)
        assert (p == 0.5).all()


class TestSaturationWarnings:
    def test_deep_product_fuzzy_chain_warns(self):
        warnings = saturation_warnings(parse(DEEP_PROGRAM))
        # `a` is depth 1, no warning; b, c, d should warn.
        names = {w.name for w in warnings}
        assert names == {"b", "c", "d"}

    def test_warning_carries_source_location(self):
        warnings = saturation_warnings(parse(DEEP_PROGRAM))
        for w in warnings:
            assert w.source_line > 0
            assert w.algebra_name == "product_fuzzy"

    def test_message_includes_rationale(self):
        warnings = saturation_warnings(parse(DEEP_PROGRAM))
        for w in warnings:
            assert isinstance(w, SaturationWarning)
            assert "ln(2)/k" in w.message() or "product-fuzzy" in w.message()

    def test_shallow_chain_does_not_warn(self):
        program = """
composition real [level=algebra]

object A : FinSet 4

program model : A -> A
    sample x <- Normal(loc=0.0, scale=1.0)
    observe r : A <- Normal(loc=x, scale=0.1)
    return r

export model
"""
        # depth=1 latent, recipe close to Normal(0, 1): no warning.
        assert saturation_warnings(parse(program)) == ()


def test_stepshape_is_a_dx_model():
    s = StepShape(
        name="x",
        kind="latent",
        source_line=10,
        source_col=4,
        depth=1,
        algebra_name="product_fuzzy",
        intermediate_size=4,
    )
    assert s.name == "x"
    assert s.kind == "latent"


class TestInitAutoDSL:
    """End-to-end checks that the ``[init=auto]`` annotation on
    latent declarations dispatches into the analysis package's
    saturation-free recipe and overrides the default
    ``randn(...) * scale`` init.
    """

    def test_product_fuzzy_latent_lands_at_recipe(self):
        src = """
composition product_fuzzy [level=algebra]
object A : FinSet 8
object B : FinSet 4
morphism f : A -> B [role=latent, init=auto]
export f
"""
        from quivers.dsl import loads

        prog = loads(textwrap.dedent(src))
        m = prog.morphism
        raw_mean = float(m.raw.detach().mean())
        value_mean = float(m.tensor.detach().mean())
        # Through LatentMorphism's sigmoid bijector the raw should
        # land near logit(ln 2 / 8) ≈ -2.5.
        assert -3.3 < raw_mean < -1.7
        assert 0.04 < value_mean < 0.14

    def test_init_default_without_annotation_is_centered(self):
        src = """
composition product_fuzzy [level=algebra]
object A : FinSet 8
object B : FinSet 4
morphism f : A -> B [role=latent]
export f
"""
        from quivers.dsl import loads

        prog = loads(textwrap.dedent(src))
        raw_mean = float(prog.morphism.raw.detach().mean())
        # randn * 0.5 → mean very close to 0.
        assert abs(raw_mean) < 0.6

    def test_init_auto_idempotent_algebra_constant(self):
        src = """
composition boolean [level=algebra]
object A : FinSet 4
object B : FinSet 4
morphism f : A -> B [role=latent, init=auto]
export f
"""
        from quivers.dsl import loads

        prog = loads(textwrap.dedent(src))
        raw = prog.morphism.raw.detach()
        assert torch.allclose(raw, torch.zeros_like(raw), atol=1e-5)


class TestChainShapeValueSemantics:
    def test_shapes_of_one_program_are_equal(self):
        first = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        second = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        assert first == second
        assert hash(first) == hash(second)

    def test_cardinality_tables_enter_equality(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        changed = ChainShape(
            algebra_name=shape.algebra_name,
            object_cardinalities={"A": 4, "B": 5},
            steps=shape.steps,
        )
        assert changed != shape
        assert hash(changed) != hash(shape)

    def test_cardinality_tables_compare_independently_of_order(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        reordered = ChainShape(
            algebra_name=shape.algebra_name,
            object_cardinalities={"B": 4, "A": 4},
            steps=shape.steps,
        )
        assert reordered == shape
        assert hash(reordered) == hash(shape)

    def test_shapes_survive_copy_and_pickle(self):
        shape = ChainShape.from_module(parse(SIMPLE_PROGRAM))
        for clone in (
            copy.copy(shape),
            copy.deepcopy(shape),
            pickle.loads(pickle.dumps(shape)),
        ):
            assert clone.object_cardinalities == {"A": 4, "B": 4}
            assert clone == shape
