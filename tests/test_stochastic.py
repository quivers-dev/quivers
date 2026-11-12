"""Tests for stochastic morphisms, Markov kernels, and the Giry monad."""

import torch
import pytest
from quivers.core.objects import FinSet, ProductSet
from quivers.core.morphisms import observed
from quivers.stochastic import (
    MARKOV,
    StochasticMorphism,
    CategoricalMorphism,
    DiscretizedNormal,
    DiscretizedLogitNormal,
    DiscretizedBeta,
    DiscretizedTruncatedNormal,
    condition,
    mix,
    factor,
    normalize,
    prob,
    marginal_prob,
    expectation,
    stochastic,
)
from quivers.stochastic.giry import GiryMonad, FinStoch
from quivers.program import Program


def _assert_row_stochastic(t: torch.Tensor, n_dom_dims: int, atol: float = 1e-05):
    """Assert that codomain fibers sum to 1."""
    cod_dims = tuple(range(n_dom_dims, t.ndim))
    row_sums = t.sum(dim=cod_dims)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=atol, rtol=0.0)


class TestMarkovAlgebra:
    def test_name(self):
        """Algebra reports its name."""
        assert MARKOV.name == "Markov"

    def test_tensor_op(self):
        """Tensor product is pointwise multiplication."""
        a = torch.tensor([0.5, 0.3])
        b = torch.tensor([0.4, 0.6])
        result = MARKOV.tensor_op(a, b)
        expected = torch.tensor([0.2, 0.18])
        torch.testing.assert_close(result, expected)

    def test_join_is_sum(self):
        """Join operation is summation."""
        t = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
        result = MARKOV.join(t, dim=1)
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_identity_tensor(self):
        """Identity is Kronecker delta."""
        ident = MARKOV.identity_tensor((3,))
        expected = torch.eye(3)
        torch.testing.assert_close(ident, expected)

    def test_compose_stochastic_matrices(self):
        """Composition of stochastic matrices is matrix multiplication."""
        m = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        n = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        result = MARKOV.compose(m, n, n_contract=1)
        expected = m @ n
        torch.testing.assert_close(result, expected, atol=1e-06, rtol=0.0)

    def test_compose_preserves_stochasticity(self):
        """Composition of row-stochastic matrices is row-stochastic."""
        m = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
        n = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        result = MARKOV.compose(m, n, n_contract=1)
        row_sums = result.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones(2), atol=1e-06, rtol=0.0)

    def test_compose_identity_left(self):
        """Composing with identity on the left is a no-op."""
        m = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        ident = torch.eye(2)
        result = MARKOV.compose(ident, m, n_contract=1)
        torch.testing.assert_close(result, m, atol=1e-06, rtol=0.0)

    def test_compose_identity_right(self):
        """Composing with identity on the right is a no-op."""
        m = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        ident = torch.eye(2)
        result = MARKOV.compose(m, ident, n_contract=1)
        torch.testing.assert_close(result, m, atol=1e-06, rtol=0.0)

    def test_compose_associativity(self):
        """Composition is associative."""
        a = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
        b = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        c = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        ab_c = MARKOV.compose(MARKOV.compose(a, b, 1), c, 1)
        a_bc = MARKOV.compose(a, MARKOV.compose(b, c, 1), 1)
        torch.testing.assert_close(ab_c, a_bc, atol=1e-05, rtol=0.0)


class TestStochasticMorphism:
    def test_row_stochastic(self):
        """StochasticMorphism produces row-stochastic tensors."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = StochasticMorphism(A, B)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_shape(self):
        """Output shape is (domain, codomain)."""
        A = FinSet(name="A", cardinality=5)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        assert f.tensor.shape == torch.Size([5, 3])

    def test_learnable_parameters(self):
        """StochasticMorphism has learnable parameters."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        prog = Program(f)
        params = list(prog.parameters())
        assert len(params) > 0
        assert all((p.requires_grad for p in params))

    def test_gradient_flow(self):
        """Gradients flow through softmax parameterization."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        prog = Program(f)
        out = prog()
        loss = out.sum()
        loss.backward()
        for p in prog.parameters():
            assert p.grad is not None

    def test_composition_is_stochastic(self):
        """Composing two stochastic morphisms gives a stochastic result."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        C = FinSet(name="C", cardinality=2)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(B, C)
        h = f >> g
        _assert_row_stochastic(h.tensor, n_dom_dims=1)

    def test_temperature(self):
        """Low temperature produces sharper distributions."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=5)
        f_warm = StochasticMorphism(A, B, temperature=10.0)
        f_cold = StochasticMorphism(A, B, temperature=0.01)
        with torch.no_grad():
            f_cold._module.logits.copy_(f_warm._module.logits)
        assert f_cold.tensor.max() > f_warm.tensor.max()

    def test_categorical_alias(self):
        """CategoricalMorphism is an alias for StochasticMorphism."""
        assert CategoricalMorphism is StochasticMorphism

    def test_stochastic_factory(self):
        """Factory function creates StochasticMorphism."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = stochastic(A, B)
        assert isinstance(f, StochasticMorphism)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_product_domain(self):
        """Stochastic morphism with product domain."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        AB = ProductSet(components=(A, B))
        C = FinSet(name="C", cardinality=4)
        f = StochasticMorphism(AB, C)
        assert f.tensor.shape == torch.Size([2, 3, 4])
        _assert_row_stochastic(f.tensor, n_dom_dims=2)


class TestDiscretizedNormal:
    def test_row_stochastic(self):
        """DiscretizedNormal produces row-stochastic tensors."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="response", cardinality=7)
        f = DiscretizedNormal(A, B, low=0.0, high=1.0)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_shape(self):
        """Output shape matches domain × codomain."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=10)
        f = DiscretizedNormal(A, B, low=-3.0, high=3.0)
        assert f.tensor.shape == torch.Size([2, 10])

    def test_learnable(self):
        """Parameters mu and log_sigma are learnable."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=7)
        f = DiscretizedNormal(A, B)
        prog = Program(f)
        params = list(prog.parameters())
        assert len(params) == 2

    def test_gradient_flow(self):
        """Gradients flow through discretized normal."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=7)
        f = DiscretizedNormal(A, B)
        prog = Program(f)
        loss = prog().sum()
        loss.backward()
        for p in prog.parameters():
            assert p.grad is not None

    def test_peaked_at_mu(self):
        """When σ is small, distribution peaks near μ."""
        A = FinSet(name="A", cardinality=1)
        B = FinSet(name="B", cardinality=11)
        f = DiscretizedNormal(A, B, low=0.0, high=1.0)
        with torch.no_grad():
            f._module.mu.fill_(0.5)
            f._module.log_sigma.fill_(-5.0)
        t = f.tensor.squeeze()
        assert t.argmax().item() == 5


class TestDiscretizedLogitNormal:
    def test_row_stochastic(self):
        """DiscretizedLogitNormal produces row-stochastic tensors."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=9)
        f = DiscretizedLogitNormal(A, B)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_learnable(self):
        """Has learnable mu and log_sigma."""
        A = FinSet(name="A", cardinality=1)
        B = FinSet(name="B", cardinality=7)
        f = DiscretizedLogitNormal(A, B)
        prog = Program(f)
        assert len(list(prog.parameters())) == 2


class TestDiscretizedBeta:
    def test_row_stochastic(self):
        """DiscretizedBeta produces row-stochastic tensors."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=9)
        f = DiscretizedBeta(A, B)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_learnable(self):
        """Has learnable log_alpha and log_beta."""
        A = FinSet(name="A", cardinality=1)
        B = FinSet(name="B", cardinality=7)
        f = DiscretizedBeta(A, B)
        prog = Program(f)
        assert len(list(prog.parameters())) == 2

    def test_uniform_for_alpha_beta_one(self):
        """Beta(1, 1) should be approximately uniform."""
        A = FinSet(name="A", cardinality=1)
        B = FinSet(name="B", cardinality=9)
        f = DiscretizedBeta(A, B)
        with torch.no_grad():
            f._module.log_alpha.fill_(0.0)
            f._module.log_beta.fill_(0.0)
        t = f.tensor.squeeze()
        torch.testing.assert_close(t, torch.full_like(t, 1.0 / 9), atol=0.02, rtol=0.0)


class TestDiscretizedTruncatedNormal:
    def test_row_stochastic(self):
        """DiscretizedTruncatedNormal produces row-stochastic tensors."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=7)
        f = DiscretizedTruncatedNormal(A, B, low=0.0, high=1.0)
        _assert_row_stochastic(f.tensor, n_dom_dims=1)

    def test_shape(self):
        """Output shape is correct."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=11)
        f = DiscretizedTruncatedNormal(A, B, low=-1.0, high=1.0)
        assert f.tensor.shape == torch.Size([3, 11])


class TestCondition:
    def test_condition_preserves_stochasticity(self):
        """Conditioning preserves row-stochasticity."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = StochasticMorphism(A, B)
        e = torch.tensor([1.0, 0.5, 0.0, 1.0])
        g = condition(f, e)
        _assert_row_stochastic(g.tensor, n_dom_dims=1)

    def test_condition_zeros_out(self):
        """Zero evidence eliminates that codomain element."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        e = torch.tensor([1.0, 0.0, 1.0])
        g = condition(f, e)
        assert (g.tensor[:, 1] < 1e-06).all()

    def test_condition_uniform_evidence(self):
        """Uniform evidence does not change the distribution."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        e = torch.ones(3)
        g = condition(f, e)
        torch.testing.assert_close(g.tensor, f.tensor, atol=1e-05, rtol=0.0)

    def test_condition_gradient_flow(self):
        """Gradients flow through conditioning."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        e = torch.tensor([1.0, 0.0, 1.0])
        g = condition(f, e)
        prog = Program(g)
        loss = prog().sum()
        loss.backward()
        for p in prog.parameters():
            assert p.grad is not None


class TestMix:
    def test_mix_preserves_stochasticity(self):
        """Mixture of stochastic morphisms is stochastic."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g)
        _assert_row_stochastic(h.tensor, n_dom_dims=1)

    def test_mix_weight_range(self):
        """Mixing weight is in (0, 1)."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g)
        w = h.weight
        assert 0.0 < w.item() < 1.0

    def test_mix_default_equal(self):
        """Default mixing weight is 0.5."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g)
        torch.testing.assert_close(h.weight, torch.tensor(0.5), atol=1e-06, rtol=0.0)

    def test_mix_type_mismatch(self):
        """Cannot mix morphisms with different types."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        C = FinSet(name="C", cardinality=4)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, C)
        with pytest.raises(TypeError, match="cannot mix"):
            mix(f, g)

    def test_mix_learnable(self):
        """Mixing weight is learnable by default."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g)
        prog = Program(h)
        loss = prog().sum()
        loss.backward()
        params = list(prog.parameters())
        assert len(params) == 3

    def test_mix_fixed_weight(self):
        """Non-learnable mixing weight is fixed."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g, learnable=False)
        prog = Program(h)
        params = list(prog.parameters())
        assert len(params) == 2

    def test_mix_interpolation(self):
        """Mixture interpolates between two morphisms."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(A, B)
        h = mix(f, g, init_logit=10.0)
        torch.testing.assert_close(h.tensor, f.tensor, atol=0.001, rtol=0.0)
        h2 = mix(f, g, init_logit=-10.0)
        torch.testing.assert_close(h2.tensor, g.tensor, atol=0.001, rtol=0.0)


class TestFactor:
    def test_factor_scales_pointwise(self):
        """Factor scales by pointwise weights."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        w = torch.tensor([2.0, 1.0, 0.5])
        g = factor(f, w)
        expected = f.tensor * w.unsqueeze(0)
        torch.testing.assert_close(g.tensor, expected, atol=1e-06, rtol=0.0)

    def test_factor_unnormalized(self):
        """Factor result is not row-stochastic."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        w = torch.tensor([2.0, 2.0, 2.0])
        g = factor(f, w)
        row_sums = g.tensor.sum(dim=-1)
        torch.testing.assert_close(
            row_sums, torch.full((2,), 2.0), atol=1e-05, rtol=0.0
        )


class TestNormalize:
    def test_normalize_makes_stochastic(self):
        """Normalize makes an unnormalized morphism row-stochastic."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        w = torch.tensor([2.0, 1.0, 0.5])
        g = normalize(factor(f, w))
        _assert_row_stochastic(g.tensor, n_dom_dims=1)

    def test_normalize_idempotent(self):
        """Normalizing an already-stochastic morphism changes nothing."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        g = normalize(f)
        torch.testing.assert_close(g.tensor, f.tensor, atol=1e-05, rtol=0.0)


class TestProb:
    def test_prob_indices(self):
        """prob extracts correct values from the tensor."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = StochasticMorphism(A, B)
        t = f.tensor
        dom_idx = torch.tensor([0, 1, 2])
        cod_idx = torch.tensor([0, 1, 2])
        result = prob(f, dom_idx, cod_idx)
        expected = torch.stack([t[0, 0], t[1, 1], t[2, 2]]).detach()
        torch.testing.assert_close(result, expected)


class TestMarginalProb:
    def test_marginal_sums_to_one(self):
        """Marginal probability over all codomain elements sums to ~1."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        f = StochasticMorphism(A, B)
        all_cod = torch.arange(4)
        result = marginal_prob(f, all_cod)
        torch.testing.assert_close(
            result.sum(), torch.tensor(1.0), atol=1e-05, rtol=0.0
        )


class TestExpectation:
    def test_expectation_identity(self):
        """Expectation of constant function is the constant."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        values = torch.ones(3) * 5.0
        result = expectation(f, values)
        torch.testing.assert_close(result, torch.full((2,), 5.0), atol=1e-05, rtol=0.0)

    def test_expectation_weighted_mean(self):
        """Expectation computes weighted mean."""
        FinSet(name="A", cardinality=1)
        B = FinSet(name="B", cardinality=3)
        data = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
        f = observed(FinSet(name="A", cardinality=1), B, data, algebra=MARKOV)
        values = torch.tensor([1.0, 2.0, 3.0])
        result = expectation(f, values)
        torch.testing.assert_close(result, torch.tensor([2.0]), atol=1e-05, rtol=0.0)


class TestGiryMonad:
    def test_unit_is_delta(self):
        """Giry unit is the Kronecker delta."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=3)
        eta = G.unit(A)
        expected = torch.eye(3)
        torch.testing.assert_close(eta.tensor, expected)

    def test_multiply_is_identity(self):
        """Giry multiplication is the identity."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=3)
        mu = G.multiply(A)
        expected = torch.eye(3)
        torch.testing.assert_close(mu.tensor, expected)

    def test_kleisli_compose(self):
        """Kleisli composition is matrix multiplication."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=2)
        C = FinSet(name="C", cardinality=2)
        f_data = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        g_data = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        f = observed(A, B, f_data, algebra=MARKOV)
        g = observed(B, C, g_data, algebra=MARKOV)
        h = G.kleisli_compose(f, g)
        expected = f_data @ g_data
        torch.testing.assert_close(h.tensor, expected, atol=1e-05, rtol=0.0)

    def test_left_unit_law(self):
        """η >=> f = f (left unit law)."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f_data = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        f = observed(A, B, f_data, algebra=MARKOV)
        eta = G.unit(A)
        result = G.kleisli_compose(eta, f)
        torch.testing.assert_close(result.tensor, f_data, atol=1e-05, rtol=0.0)

    def test_right_unit_law(self):
        """f >=> η = f (right unit law)."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f_data = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        f = observed(A, B, f_data, algebra=MARKOV)
        eta = G.unit(B)
        result = G.kleisli_compose(f, eta)
        torch.testing.assert_close(result.tensor, f_data, atol=1e-05, rtol=0.0)

    def test_associativity(self):
        """(f >=> g) >=> h = f >=> (g >=> h) (associativity)."""
        G = GiryMonad()
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=2)
        C = FinSet(name="C", cardinality=2)
        D = FinSet(name="D", cardinality=2)
        f_data = torch.tensor([[0.6, 0.4], [0.3, 0.7]])
        g_data = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        h_data = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        f = observed(A, B, f_data, algebra=MARKOV)
        g = observed(B, C, g_data, algebra=MARKOV)
        h = observed(C, D, h_data, algebra=MARKOV)
        fg_h = G.kleisli_compose(G.kleisli_compose(f, g), h)
        f_gh = G.kleisli_compose(f, G.kleisli_compose(g, h))
        torch.testing.assert_close(fg_h.tensor, f_gh.tensor, atol=1e-05, rtol=0.0)


class TestFinStoch:
    def test_identity(self):
        """FinStoch identity is the Kronecker delta."""
        cat = FinStoch()
        A = FinSet(name="A", cardinality=3)
        ident = cat.identity(A)
        torch.testing.assert_close(ident.tensor, torch.eye(3))

    def test_compose(self):
        """FinStoch compose is matrix multiplication."""
        cat = FinStoch()
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=2)
        C = FinSet(name="C", cardinality=2)
        f_data = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        g_data = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        f = observed(A, B, f_data, algebra=MARKOV)
        g = observed(B, C, g_data, algebra=MARKOV)
        h = cat.compose(f, g)
        expected = f_data @ g_data
        torch.testing.assert_close(h.tensor, expected, atol=1e-05, rtol=0.0)


class TestStochasticIntegration:
    def test_full_pipeline(self):
        """End-to-end: stochastic morphism → condition → Program → train."""
        A = FinSet(name="entity", cardinality=5)
        B = FinSet(name="response", cardinality=7)
        f = DiscretizedNormal(A, B, low=0.0, high=1.0)
        e = torch.tensor([0.0, 0.1, 0.5, 1.0, 0.5, 0.1, 0.0])
        g = condition(f, e)
        prog = Program(g)
        out = prog()
        assert out.shape == torch.Size([5, 7])
        _assert_row_stochastic(out, n_dom_dims=1)
        target_idx = torch.tensor([0, 1, 2, 3, 4])
        response_idx = torch.tensor([3, 3, 3, 3, 3])
        loss = prog.nll_loss(target_idx, response_idx)
        loss.backward()
        for p in prog.parameters():
            assert p.grad is not None

    def test_mixture_model(self):
        """PDS-style mixture: Disj(p, f, g) as mix(f, g)."""
        World = FinSet(name="world", cardinality=1)
        Response = FinSet(name="response", cardinality=7)
        f = DiscretizedTruncatedNormal(World, Response, low=0.0, high=1.0)
        g = DiscretizedTruncatedNormal(World, Response, low=0.0, high=1.0)
        with torch.no_grad():
            f._module.mu.fill_(0.8)
            f._module.log_sigma.fill_(-1.0)
            g._module.mu.fill_(0.3)
            g._module.log_sigma.fill_(-1.0)
        h = mix(f, g)
        prog = Program(h)
        out = prog()
        assert out.shape == torch.Size([1, 7])
        _assert_row_stochastic(out, n_dom_dims=1)
        params = list(prog.parameters())
        assert len(params) == 5

    def test_composition_chain_stochastic(self):
        """Chain of stochastic morphisms stays row-stochastic."""
        A = FinSet(name="A", cardinality=3)
        B = FinSet(name="B", cardinality=4)
        C = FinSet(name="C", cardinality=5)
        D = FinSet(name="D", cardinality=2)
        f = StochasticMorphism(A, B)
        g = StochasticMorphism(B, C)
        h = StochasticMorphism(C, D)
        chain = f >> g >> h
        _assert_row_stochastic(chain.tensor, n_dom_dims=1)

    def test_factor_then_normalize(self):
        """Factor + normalize is equivalent to condition."""
        A = FinSet(name="A", cardinality=2)
        B = FinSet(name="B", cardinality=3)
        f = StochasticMorphism(A, B)
        w = torch.tensor([1.0, 0.5, 2.0])
        conditioned = condition(f, w)
        factored_normalized = normalize(factor(f, w))
        torch.testing.assert_close(
            conditioned.tensor, factored_normalized.tensor, atol=1e-05, rtol=0.0
        )

    def test_dsl_with_markov_algebra(self):
        """DSL supports the markov algebra."""
        from quivers.dsl import loads

        prog = loads(
            "composition markov [level=algebra]\n"
            "object X : FinSet 3\n"
            "morphism h : X -> X [role=observed] ~ identity(X)\n"
            "export h\n"
        )
        expected = torch.eye(3)
        torch.testing.assert_close(prog(), expected)
