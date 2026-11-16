"""Count-vector laws agree with independent probabilities and event contracts."""

import numpy as np
import pytest
import torch
from scipy import stats

from quivers.continuous._multinomial import (
    Multinomial,
    DirichletMultinomial,
)
from quivers.continuous._closed_probability import (
    ClosedBinomial,
    _beta_binomial_log_mass,
)
from quivers.continuous._zero_inflated_count import BetaBinomial


@pytest.mark.parametrize("rows,categories", [(2, 3), (3, 3), (4, 2), (2, 1)])
@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_named_batch_and_event_contracts_match_scipy(rows, categories, cls):
    parameters = torch.arange(1, rows * categories + 1, dtype=torch.double).reshape(
        rows, categories
    )
    n = torch.arange(rows, dtype=torch.int64) * 3
    points = torch.zeros((rows, categories), dtype=torch.int64)
    points[:, 0] = n
    kwargs = (
        {"probs": parameters} if cls is Multinomial else {"concentration": parameters}
    )
    d = cls(n, **kwargs)
    expected = [
        (
            stats.multinomial.logpmf(
                point.numpy(), int(total), p.numpy() / p.sum().item()
            )
            if cls is Multinomial
            else stats.dirichlet_multinomial.logpmf(
                point.numpy(), p.numpy(), int(total)
            )
        )
        for point, total, p in zip(points, n, parameters, strict=True)
    ]
    # SciPy currently refuses n0 for DM, whose exact law is delta([0,...,0]).
    expected[0] = 0.0
    torch.testing.assert_close(
        d.log_prob(points),
        torch.as_tensor(np.asarray(expected), dtype=torch.double),
        atol=2e-12,
        rtol=2e-12,
    )
    assert d.batch_shape == (rows,) and d.event_shape == (categories,)
    assert d.sample((2, 3)).shape == (2, 3, rows, categories)
    assert d.expand((2, rows)).log_prob(points).shape == (2, rows)


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_broadcasting_uses_only_vector_batch_axes(cls):
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
    kwargs = {"probs": vector} if cls is Multinomial else {"concentration": vector}
    d = cls(torch.tensor([[0], [3]]), **kwargs)
    assert d.batch_shape == (2, 1) and d.event_shape == (3,)
    batch = vector.expand(2, 4, 3)
    kwargs = {"probs": batch} if cls is Multinomial else {"concentration": batch}
    d = cls(torch.tensor([[0], [3]]), **kwargs)
    assert d.batch_shape == (2, 4)
    assert d.sample((5,)).shape == (5, 2, 4, 3)


@pytest.mark.parametrize("n", [0, 1, 10, 1000001, 33554433, 2**31 - 1])
def test_two_categories_reduce_to_binomial_and_beta_binomial(n):
    k = n // 3
    point = torch.tensor([k, n - k])
    got = Multinomial(n, probs=[0.3, 0.7]).log_prob(point)
    want = ClosedBinomial(n, probs=torch.tensor(0.3, dtype=torch.double)).log_prob(
        torch.tensor(k)
    )
    torch.testing.assert_close(got, want, atol=2e-12, rtol=2e-12)
    got = DirichletMultinomial(n, concentration=[2.0, 3.0]).log_prob(point)
    want = BetaBinomial(n, 2.0, 3.0).log_prob(torch.tensor(k))
    torch.testing.assert_close(got, want, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_analytic_density_gradients_and_forward_second_ad(cls):
    n = torch.tensor([3, 5])
    y = torch.tensor([[1, 1, 1], [1, 2, 2]])
    vector = torch.tensor([0.7, 1.2, 2.3], dtype=torch.double, requires_grad=True)

    def actual(value):
        kw = {"probs": value} if cls is Multinomial else {"concentration": value}
        return cls(n, **kw).log_prob(y)

    def oracle(value):
        counts = y.double()
        combination = torch.lgamma(n.double() + 1) - torch.lgamma(counts + 1).sum(-1)
        if cls is Multinomial:
            return combination + (counts * (value / value.sum()).log()).sum(-1)
        return (
            combination
            + torch.lgamma(value.sum())
            - torch.lgamma(value.sum() + n)
            + (torch.lgamma(value + counts) - torch.lgamma(value)).sum(-1)
        )

    torch.testing.assert_close(actual(vector), oracle(vector), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.func.jacrev(actual)(vector), torch.func.jacrev(oracle)(vector)
    )
    torch.testing.assert_close(
        torch.func.jacfwd(actual)(vector), torch.func.jacfwd(oracle)(vector)
    )
    assert torch.autograd.gradcheck(actual, (vector,))
    assert torch.autograd.gradgradcheck(actual, (vector,))


def test_logit_density_keeps_extreme_and_structural_zero_scores_and_gradients():
    logits = torch.tensor(
        [0.0, -1000.0, -torch.inf], dtype=torch.double, requires_grad=True
    )
    d = Multinomial(3, logits=logits)
    score = d.log_prob(torch.tensor([2, 1, 0]))
    assert score.item() == pytest.approx(-1000 + np.log(3))
    torch.testing.assert_close(
        torch.autograd.grad(score, logits)[0],
        torch.tensor([-1.0, 1.0, 0.0], dtype=torch.double),
    )
    assert d.log_prob(torch.tensor([2, 0, 1])) == -torch.inf
    logits = torch.tensor(
        [0.0, 0.0, -torch.inf], dtype=torch.double, requires_grad=True
    )
    score = Multinomial(5, logits=logits).log_prob(torch.tensor([2, 3, 0]))
    torch.testing.assert_close(
        torch.autograd.grad(score, logits)[0],
        torch.tensor([-0.5, 0.5, 0.0], dtype=torch.double),
    )


def test_closed_zero_probabilities_have_correct_finite_cell_gradients():
    p = torch.tensor([0.0, 2.0, 3.0], dtype=torch.double, requires_grad=True)
    d = Multinomial(5, probs=p)
    score = d.log_prob(torch.tensor([0, 2, 3]))
    torch.testing.assert_close(
        torch.autograd.grad(score, p)[0],
        torch.tensor([-1.0, 0.0, 0.0], dtype=torch.double),
    )
    assert d.log_prob(torch.tensor([1, 1, 3])) == -torch.inf
    assert not d.sample((100,))[:, 0].any()


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_zero_trials_skip_singular_parameter_derivatives_and_sampling(cls):
    vector = torch.tensor([1e-320] * 3, dtype=torch.double, requires_grad=True)
    kwargs = {"probs": vector} if cls is Multinomial else {"concentration": vector}
    d = cls(0, **kwargs)
    score = d.log_prob(torch.tensor([0, 0, 0])) + d.mean.sum() + d.variance.sum()
    assert score == 0
    torch.testing.assert_close(
        torch.autograd.grad(score, vector)[0], torch.zeros_like(vector)
    )
    assert not d.sample((10,)).any()


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_off_support_events_are_impossible_without_aliasing_axes(cls):
    kw = (
        {"probs": [0.2, 0.3, 0.5]}
        if cls is Multinomial
        else {"concentration": [2.0, 3.0, 5.0]}
    )
    d = cls(3, **kw)
    for point in [
        [-1, 1, 3],
        [0.5, 0.5, 2],
        [0, 0, 0],
        [2, 1, 1],
        [float("nan"), 1, 2],
    ]:
        assert d.log_prob(torch.tensor(point)) == -torch.inf
    with pytest.raises(ValueError, match="category axis"):
        d.log_prob(torch.zeros(3, 2))


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
@pytest.mark.parametrize(
    "count",
    [2**53 + 1, torch.tensor(2**53 + 1), [2**53 + 1], float(2**53 + 2), -1, 1.5],
)
def test_trial_domain_and_exact_capacity_are_checked_before_cast(cls, count):
    kw = {"probs": [0.3, 0.7]} if cls is Multinomial else {"concentration": [2.0, 3.0]}
    with pytest.raises(ValueError):
        cls(count, **kw)
    assert cls(33554433.0, **kw).total_count.item() == 33554433


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_event_counts_above_exact_integer_capacity_are_refused(cls):
    kw = {"probs": [0.3, 0.7]} if cls is Multinomial else {"concentration": [2.0, 3.0]}
    d = cls(1, **kw)
    with pytest.raises(ValueError, match="2\\*\\*53"):
        d.log_prob(torch.tensor([2**53 + 1, 0]))


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_sampler_moments_covariance_and_exact_trial_totals(cls):
    n = torch.tensor([0, 5, 13])
    vector = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 1.0, 2.0]], dtype=torch.double
    )
    kwargs = {"probs": vector} if cls is Multinomial else {"concentration": vector}
    d = cls(n, **kwargs)
    torch.manual_seed(314)
    values = d.sample((50000,))
    assert values.dtype == torch.int64
    assert torch.equal(values.sum(-1), n.expand(50000, 3))
    means = values.double().mean(0)
    se = (d.variance / len(values)).sqrt()
    assert ((means - d.mean).abs() <= 6 * se + 1e-3).all()
    torch.testing.assert_close(
        values.double().var(0), d.variance, atol=0.025, rtol=0.035
    )
    p = vector / vector.sum(-1, keepdim=True)
    covariance = -n.unsqueeze(-1) * p[:, 0:1] * p[:, 1:2]
    if cls is DirichletMultinomial:
        phi = vector.sum(-1, keepdim=True)
        covariance = covariance * (n.unsqueeze(-1) + phi) / (1 + phi)
    actual = (
        (values[:, :, 0].double() - means[:, 0])
        * (values[:, :, 1].double() - means[:, 1])
    ).mean(0)
    torch.testing.assert_close(actual, covariance.squeeze(-1), atol=0.045, rtol=0.06)


@pytest.mark.parametrize("alpha", [1e-6, 1e-100, 1e-320])
def test_small_concentration_sampling_does_not_turn_into_a_uniform_multinomial(alpha):
    d = DirichletMultinomial(10, concentration=[alpha, 2 * alpha, 3 * alpha])
    torch.manual_seed(71)
    draws = d.sample((30000,))
    assert (draws.amax(-1) == 10).double().mean() > 0.999
    torch.testing.assert_close(
        draws.double().mean(0),
        torch.tensor([10 / 6, 20 / 6, 30 / 6], dtype=torch.double),
        atol=0.10,
        rtol=0.02,
    )


@pytest.mark.parametrize(
    "n,k,expected",
    [(10, 5, -1.40204271808803003744), (100, 50, -2.53087640397710743272)],
)
def test_extreme_concentration_beta_factor_uses_high_precision_oracle(n, k, expected):
    # Independent 90-digit log-gamma calculation, rounded to Float64.
    actual = DirichletMultinomial(n, concentration=[1e16, 1e16]).log_prob(
        torch.tensor([k, n - k])
    )
    assert actual.item() == pytest.approx(expected, abs=3e-12)
    assert actual <= 0


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_large_count_scores_against_90_digit_log_gamma_oracles(cls):
    n = 2**53
    point = torch.tensor([3002399751580330, 3002399751580330, 3002399751580332])
    d = cls(
        n,
        **(
            {"probs": [1.0, 1.0, 1.0]}
            if cls is Multinomial
            else {"concentration": [1.0, 1.0, 1.0]}
        ),
    )
    expected = (
        -36.92675920308428286369 if cls is Multinomial else -72.78045395879425782188
    )
    assert d.log_prob(point).item() == pytest.approx(expected, abs=2e-11)
    assert d.log_prob(point) <= 0


def test_beta_factor_explicit_high_order_gradients_and_large_parameter_limit():
    value = torch.tensor([1.2, 0.7], dtype=torch.double, requires_grad=True)

    def score(z):
        return _beta_binomial_log_mass(torch.tensor(2.0), torch.tensor(5.0), z[0], z[1])

    assert torch.autograd.gradcheck(score, (value,))
    assert torch.autograd.gradgradcheck(score, (value,))
    torch.testing.assert_close(
        torch.func.jacfwd(score)(value), torch.func.jacrev(score)(value)
    )


@pytest.mark.parametrize(
    "n,k,expected",
    [(10, 5, -1.40204271808803003744), (100, 50, -2.53087640397710743272)],
)
def test_existing_beta_binomial_uses_shared_stable_factor_and_moments(n, k, expected):
    from quivers.transpile.family_meta import FAMILY_META
    from quivers.continuous.inline import _BetaBinomial

    for cls in (
        BetaBinomial,
        _BetaBinomial,
        FAMILY_META["BetaBinomial"].distribution_class,
    ):
        d = cls(n, 1e16, 1e16)
        assert d.log_prob(torch.tensor(k)).item() == pytest.approx(expected, abs=3e-12)
        assert d.mean.item() == n / 2
        assert d.variance.item() == pytest.approx(n / 4, abs=2e-12)


@pytest.mark.parametrize("alpha", [1e-6, 1e-100, 1e-320])
def test_existing_beta_binomial_uses_underflow_safe_compound_sampling(alpha):
    torch.manual_seed(513)
    d = BetaBinomial(10, alpha, 2 * alpha)
    values = d.sample((20000,))
    assert ((values == 0) | (values == 10)).double().mean() > 0.999
    assert values.double().mean().item() == pytest.approx(10 / 3, abs=0.12)


def test_existing_beta_binomial_zero_trials_have_no_singular_gradients():
    alpha = torch.tensor(1e-320, dtype=torch.double, requires_grad=True)
    d = BetaBinomial(0, alpha, alpha)
    objective = d.log_prob(torch.tensor(0)) + d.mean + d.variance
    assert objective == 0
    assert torch.autograd.grad(objective, alpha)[0] == 0
    assert not d.sample((100,)).any()


@pytest.mark.parametrize("family", ["Multinomial", "DirichletMultinomial"])
@pytest.mark.parametrize("rows,categories", [(2, 3), (3, 3), (4, 2)])
def test_checked_source_vector_events_preserve_heterogeneous_trial_batches(
    family, rows, categories
):
    from quivers.modeling import Model
    from quivers.dsl.parser import parse
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.qiec.program_runtime import run_program

    parameter = "probs" if family == "Multinomial" else "concentration"
    source = f"""object Row : FinSet {rows}
object Category : FinSet {categories}
program model(n:Tensor[Int]([Row]), p:Tensor[Real]([Row,Category]), y:Tensor[Int]([Row,Category])):Row->Row
    observe y : Row <- {family}(total_count=n,{parameter}=p) [over=Category]
    return y
export model
"""
    n = torch.arange(rows, dtype=torch.int64) * 2
    p = (
        torch.arange(1, rows * categories + 1, dtype=torch.double).reshape(
            rows, categories
        )
        / 3
    ).requires_grad_()
    y = torch.zeros(rows, categories, dtype=torch.int64)
    y[:, 0] = n
    bound = Model(source).bind({"n": n, "p": p, "y": y})
    actual = bound.trace({}).log_joint.sum()
    points = y.double()
    expected = torch.lgamma(n.double() + 1) - (torch.lgamma(points + 1)).sum(-1)
    if family == "Multinomial":
        expected = expected + (points * (p / p.sum(-1, keepdim=True)).log()).sum(-1)
    else:
        expected = (
            expected
            + torch.lgamma(p.sum(-1))
            - torch.lgamma(p.sum(-1) + n)
            + (torch.lgamma(p + points) - torch.lgamma(p)).sum(-1)
        )
    torch.testing.assert_close(actual, expected.sum(), atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual, p, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), p)[0],
        atol=2e-11,
        rtol=2e-11,
    )
    checked = lower_qvr_to_qiec(parse(source))
    host = {
        "n": tuple(n.tolist()),
        "p": tuple(tuple(v) for v in p.tolist()),
        "y": tuple(tuple(v) for v in y.tolist()),
    }
    reference = run_program(checked, "model", data=host, sites={})
    assert reference.log_joint == pytest.approx(expected.sum().item(), abs=2e-11)


@pytest.mark.parametrize("family", ["Multinomial", "DirichletMultinomial"])
def test_conditional_count_vector_preserves_event_reduction_and_gradients(family):
    from quivers.continuous import (
        ConditionalMultinomial,
        ConditionalDirichletMultinomial,
        Euclidean,
    )

    cls = (
        ConditionalMultinomial
        if family == "Multinomial"
        else ConditionalDirichletMultinomial
    )
    model = cls(
        Euclidean(name="Input", dim=2),
        Euclidean(name="Category", dim=3),
        torch.tensor([0, 3, 5]),
    )
    x = torch.tensor([[0.2, 0.3], [0.1, 0.4], [0.3, 0.1]])
    y = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 1, 2]])
    score = model.log_prob(x, y)
    assert score.shape == (3,) and score[0] == 0
    score.sum().backward()
    gradients = [v.grad for v in model.parameters() if v.requires_grad]
    assert gradients and all(
        v is not None and torch.isfinite(v).all() for v in gradients
    )
    values = model.sample(x, (7,))
    assert values.shape == (7, 3, 3)
    assert torch.equal(values.sum(-1), torch.tensor([0, 3, 5]).expand(7, 3))
    with pytest.raises(NotImplementedError, match="discrete"):
        model.rsample(x)


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_native_compound_sampler_refuses_invalid_binomial_results(monkeypatch, cls):
    def bad_count(count, probability):
        return count + 1

    monkeypatch.setattr(torch, "binomial", bad_count)
    kw = (
        {"probs": [0.2, 0.3, 0.5]}
        if cls is Multinomial
        else {"concentration": [2, 3, 5]}
    )
    with pytest.raises(RuntimeError, match="above remaining trials"):
        cls(5, **kw).sample((2,))


@pytest.mark.parametrize("cls", [Multinomial, DirichletMultinomial])
def test_float32_learned_heads_keep_original_gradient_dtype(cls):
    vector = torch.tensor([0.7, 1.3, 2.0], dtype=torch.float32, requires_grad=True)
    kw = {"probs": vector} if cls is Multinomial else {"concentration": vector}
    score = cls(7, **kw).log_prob(torch.tensor([1, 4, 2]))
    assert score.dtype == torch.float64
    grad = torch.autograd.grad(score, vector)[0]
    assert grad.dtype == vector.dtype and torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "family,parameter",
    [("Multinomial", "logits"), ("DirichletMultinomial", "concentration")],
)
def test_checked_weighted_vector_events_reduce_categories_before_row_weights(
    family, parameter
):
    from quivers.modeling import Model
    from quivers.dsl.parser import parse
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.qiec.program_runtime import run_program

    source = f"""object Row : FinSet 2
object Category : FinSet 3
program model(n:Tensor[Int]([Row]), p:Tensor[Real]([Row,Category]), y:Tensor[Int]([Row,Category]), w:Tensor[Real]([Row])):Row->Row
    observe y : Row <- {family}(total_count=n,{parameter}=p) [over=Category,weights=w]
    return y
export model
"""
    n = torch.tensor([3, 5])
    p = torch.tensor(
        [[0.7, 1.3, 2.0], [1.1, 0.6, 1.4]], dtype=torch.double, requires_grad=True
    )
    y = torch.tensor([[1, 1, 1], [2, 1, 2]])
    w = torch.tensor([0.0, 2.0], dtype=torch.double)
    cls = Multinomial if family == "Multinomial" else DirichletMultinomial
    raw = cls(n, **{parameter: p}).log_prob(y)
    actual = (
        Model(source).bind({"n": n, "p": p, "y": y, "w": w}).trace({}).log_joint.sum()
    )
    expected = (w * raw).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0]
    )

    def host(value):
        return tuple(host(item) for item in value) if isinstance(value, list) else value

    data = {k: host(v.tolist()) for k, v in {"n": n, "p": p, "y": y, "w": w}.items()}
    ref = run_program(lower_qvr_to_qiec(parse(source)), "model", data=data, sites={})
    assert ref.log_joint == pytest.approx(expected.item(), abs=2e-11)


@pytest.mark.parametrize(
    "family", ["Multinomial", "DirichletMultinomial", "Categorical", "Dirichlet"]
)
def test_literal_vector_real_parameters_survive_integer_program_domain(family):
    from quivers.modeling import Model
    from quivers.dsl.parser import parse
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.qiec.program_runtime import run_program

    vector = "[0.2,0.3,0.5]"
    parameter = (
        "concentration" if family in ("Dirichlet", "DirichletMultinomial") else "probs"
    )
    counts = (
        "total_count=5," if family in ("Multinomial", "DirichletMultinomial") else ""
    )
    event = "" if family == "Categorical" else ",Category"
    axes = "" if family == "Categorical" else " [over=Category]"
    element = "Real" if family == "Dirichlet" else "Int"
    source = f"""object Row : FinSet 2
object Category : FinSet 3
program model(y:Tensor[{element}]([Row{event}])):Row->Row
    observe y:Row <- {family}({counts}{parameter}={vector}){axes}
    return y
export model
"""
    y = (
        torch.tensor([0, 2])
        if family == "Categorical"
        else torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]], dtype=torch.double)
        if family == "Dirichlet"
        else torch.tensor([[1, 2, 2], [2, 1, 2]])
    )
    actual = Model(source).bind({"y": y}).trace({}).log_joint.sum()
    p = np.array([0.2, 0.3, 0.5])
    expected = (
        stats.multinomial.logpmf(y.numpy(), 5, p).sum()
        if family == "Multinomial"
        else stats.dirichlet_multinomial.logpmf(y.numpy(), p, 5).sum()
        if family == "DirichletMultinomial"
        else np.log(p[y.numpy()]).sum()
        if family == "Categorical"
        else stats.dirichlet.logpdf(y.numpy().T, p).sum()
    )
    assert actual.item() == pytest.approx(expected, abs=2e-6)

    def host(value):
        return tuple(host(v) for v in value) if isinstance(value, list) else value

    ref = run_program(
        lower_qvr_to_qiec(parse(source)),
        "model",
        data={"y": host(y.tolist())},
        sites={},
    )
    assert ref.log_joint == pytest.approx(expected, abs=2e-10)


def test_dominant_category_does_not_round_its_nonzero_variance_away():
    n = 2**53
    # Independent formula retains both positive weights, so no subtraction
    # of a probability rounded to one enters the variance oracle.
    p = 1e16 / (1e16 + 1)
    q = 1 / (1e16 + 1)
    multinomial = Multinomial(n, probs=[1e16, 1.0])
    expected = n * p * q
    assert multinomial.variance[0].item() == pytest.approx(expected, rel=2e-15)
    beta = BetaBinomial(n, 1e16, 1.0)
    dm = DirichletMultinomial(n, concentration=[1e16, 1.0])
    expected *= (n + 1e16 + 1) / (1e16 + 2)
    assert beta.variance.item() == pytest.approx(expected, rel=2e-15)
    torch.testing.assert_close(
        dm.variance, torch.tensor([expected, expected], dtype=torch.double)
    )


@pytest.mark.parametrize(
    "family,parameter",
    [("Multinomial", "probs"), ("DirichletMultinomial", "concentration")],
)
def test_count_row_extent_is_inferred_from_typed_runtime_parameters(family, parameter):
    from quivers.modeling import Model

    source = f"""object Category : FinSet 3
program model[n:Nat](trials:Tensor[Int]([n]), p:Tensor[Real]([n,Category]), y:Tensor[Int]([n,Category])):n->n
    observe y:n <- {family}(total_count=trials,{parameter}=p) [over=Category]
    return y
export model
"""
    n = torch.tensor([0, 3, 5])
    p = torch.tensor([[0.2, 0.3, 0.5]] * 3, dtype=torch.double)
    y = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 1, 2]])
    bound = Model(source).bind({"trials": n, "p": p, "y": y})
    cls = Multinomial if family == "Multinomial" else DirichletMultinomial
    expected = cls(n, **{parameter: p}).log_prob(y).sum()
    torch.testing.assert_close(bound.trace({}).log_joint.sum(), expected)


@pytest.mark.parametrize(
    "family,parameter",
    [("Multinomial", "probs"), ("DirichletMultinomial", "concentration")],
)
@pytest.mark.parametrize("literal", [False, True])
def test_checked_source_count_capacity_is_checked_before_parameter_packing(
    family, parameter, literal
):
    from quivers.modeling import Model

    declared = "" if literal else "trials:Tensor[Int]([Row]), "
    trial = "9007199254740993" if literal else "trials"
    source = f"""object Row : FinSet 2
object Category : FinSet 3
program model({declared}y:Tensor[Int]([Row,Category])):Row->Row
    observe y:Row <- {family}(total_count={trial},{parameter}=[0.2,0.3,0.5]) [over=Category]
    return y
export model
"""
    data = {"y": torch.tensor([[0, 0, 0], [0, 0, 0]])}
    if not literal:
        data["trials"] = torch.tensor([2**53 + 1, 2**53 + 1])
    with pytest.raises(ValueError, match="2\\*\\*53"):
        Model(source).bind(data).trace({})


@pytest.mark.parametrize("family", ["Multinomial", "DirichletMultinomial"])
def test_conditional_fixed_trials_stay_exact_when_module_is_cast_to_float32(family):
    from quivers.continuous import (
        ConditionalMultinomial,
        ConditionalDirichletMultinomial,
        Euclidean,
    )

    cls = (
        ConditionalMultinomial
        if family == "Multinomial"
        else ConditionalDirichletMultinomial
    )
    n = 33554433
    model = cls(
        Euclidean(name="Input", dim=2), Euclidean(name="Category", dim=3), n
    ).float()
    assert model.total_count.dtype == torch.int64
    assert model.total_count.item() == n
    x = torch.tensor([[0.2, 0.3]])
    y = torch.tensor([[11184811, 11184811, 11184811]])
    assert torch.isfinite(model.log_prob(x, y)).all()
    assert torch.equal(model.sample(x, (3,)).sum(-1), torch.full((3, 1), n))


@pytest.mark.parametrize("family", ["Multinomial", "DirichletMultinomial"])
@pytest.mark.parametrize(
    "trials",
    [-1, 1.5, True, torch.tensor([-1, 3]), torch.tensor([1.5, 3.0]), 2**53 + 1],
)
def test_conditional_fixed_trials_are_validated_before_integer_storage(family, trials):
    from quivers.continuous import (
        ConditionalMultinomial,
        ConditionalDirichletMultinomial,
        Euclidean,
    )

    cls = (
        ConditionalMultinomial
        if family == "Multinomial"
        else ConditionalDirichletMultinomial
    )
    with pytest.raises(ValueError):
        cls(Euclidean(name="Input", dim=2), Euclidean(name="Category", dim=3), trials)


@pytest.mark.parametrize(
    "value",
    [
        float(2**53 + 2),
        torch.tensor(2**53 + 2, dtype=torch.float64),
        [2**53 + 1],
        [0.5, 2**53 + 1],
        [[0], [2**53 + 1]],
        np.array([2**53 + 1], dtype=np.int64),
        np.float64(2**53 + 2),
    ],
)
@pytest.mark.parametrize(
    "family",
    ["Binomial", "BetaBinomial", "ZeroInflatedBinomial", "ZeroInflatedBetaBinomial"],
)
def test_all_existing_finite_trial_kernels_check_capacity_before_inference(
    family, value
):
    from quivers.continuous._zero_inflated_count import (
        ZeroInflatedBinomial,
        ZeroInflatedBetaBinomial,
        BetaBinomial,
    )

    with pytest.raises(ValueError, match="2\\*\\*53"):
        if family == "Binomial":
            ClosedBinomial(value, probs=torch.tensor(0.3))
        elif family == "BetaBinomial":
            BetaBinomial(value, 2.0, 3.0)
        elif family == "ZeroInflatedBinomial":
            ZeroInflatedBinomial(0.2, value, 0.3)
        else:
            ZeroInflatedBetaBinomial(0.2, value, 2.0, 3.0)


@pytest.mark.parametrize(
    "value",
    [
        float(2**53),
        torch.tensor(2**53, dtype=torch.float64),
        [2**53],
        [0.5, 2**53],
        np.array([2**53], dtype=np.int64),
    ],
)
def test_shared_source_guard_accepts_exact_capacity_without_coercion(value):
    from quivers.continuous._closed_probability import _check_trial_count_source

    # The source guard checks capacity. Each family separately enforces
    # integral trials; a mixed sequence still must be inspected before cast.
    _check_trial_count_source(value)


def test_negative_binomial_continuous_shape_is_not_restricted_as_finite_trials():
    from quivers.continuous._closed_probability import ClosedNegativeBinomial

    d = ClosedNegativeBinomial(
        torch.tensor(1e16, dtype=torch.double),
        probs=torch.tensor(1e-12, dtype=torch.double),
    )
    assert torch.isfinite(d.log_prob(torch.tensor(10000)))
