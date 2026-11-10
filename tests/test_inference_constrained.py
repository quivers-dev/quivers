"""Tests for constrained-support variational guides, data-dict host
passing, and inline Dirichlet priors.

* :class:`quivers.inference.AutoNormalGuide` and
  :class:`quivers.inference.AutoDeltaGuide` respect each latent's
  prior support: samples are drawn in unconstrained Normal space and
  pushed through :func:`biject_to(support)` so they land inside the
  prior's support (``[0, ∞)`` for ``HalfNormal`` / ``Gamma`` /
  ``Exponential``, ``(0, 1)`` for ``Beta`` / ``LogitNormal``, the
  simplex for ``Dirichlet``, an arbitrary interval for ``Uniform``).
* :func:`quivers.inference.condition` exposes every key in its
  ``data`` dict that does not match a declared sample / observe site
  as a deterministic value in the program's runtime environment,
  visible to ``let``-expression gather (the canonical per-row
  hierarchical-regression idiom).
* Inline ``Dirichlet(concentration)`` is supported as a prior; scalar
  concentration broadcasts to a symmetric Dirichlet on the codomain's
  simplex.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import ELBO, SVI, AutoDeltaGuide, AutoNormalGuide
from quivers.inference.conditioning import condition
from quivers.inference.trace import trace


# ---------------------------------------------------------------------------
# Constrained-support guides.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dist_spec", "predicate"),
    [
        ("HalfNormal(scale=1.0)", lambda v: (v >= 0).all()),
        ("HalfCauchy(scale=1.0)", lambda v: (v >= 0).all()),
        ("Exponential(rate=1.0)", lambda v: (v > 0).all()),
        ("Gamma(concentration=2.0, rate=1.0)", lambda v: (v > 0).all()),
        ("LogNormal(loc=0.0, scale=1.0)", lambda v: (v > 0).all()),
        (
            "Beta(concentration1=2.0, concentration0=5.0)",
            lambda v: ((v > 0) & (v < 1)).all(),
        ),
        ("Uniform(low=0.0, high=1.0)", lambda v: ((v > 0) & (v < 1)).all()),
        ("Uniform(low=-3.0, high=5.0)", lambda v: ((v > -3.0) & (v < 5.0)).all()),
        ("LogitNormal(mu=0.0, sigma=1.0)", lambda v: ((v > 0) & (v < 1)).all()),
    ],
)
def test_autonormal_guide_samples_in_support(dist_spec: str, predicate) -> None:
    """The Normal-then-bijector pipeline produces samples in every
    family's declared support, so the prior's ``log_prob`` evaluates
    without ``ValueError: Expected value … to be within the support``.
    """
    src = (
        "object N : FinSet 4\n"
        "program p : N -> N\n"
        f"    sample v <- {dist_spec}\n"
        "    return v\n"
        "export p\n"
    )
    prog = loads(textwrap.dedent(src))
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(8, 1))
    assert predicate(samples["v"]), (
        f"samples for {dist_spec} not in support: {samples['v']}"
    )

    lp = guide.log_prob(torch.zeros(8, 1), samples)
    assert lp.shape == (8,)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize(
    ("dist_spec", "predicate"),
    [
        ("HalfNormal(scale=1.0)", lambda v: (v >= 0).all()),
        (
            "Beta(concentration1=2.0, concentration0=5.0)",
            lambda v: ((v > 0) & (v < 1)).all(),
        ),
        ("Uniform(low=0.0, high=1.0)", lambda v: ((v > 0) & (v < 1)).all()),
    ],
)
def test_autodelta_guide_samples_in_support(dist_spec: str, predicate) -> None:
    """``AutoDeltaGuide`` likewise transforms its point estimate
    through the support bijector so it lies inside the prior's
    support at all times during optimisation."""
    src = (
        "object N : FinSet 4\n"
        "program p : N -> N\n"
        f"    sample v <- {dist_spec}\n"
        "    return v\n"
        "export p\n"
    )
    prog = loads(textwrap.dedent(src))
    guide = AutoDeltaGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(8, 1))
    assert predicate(samples["v"])


def test_autonormal_gradient_flows_through_bijector() -> None:
    """The constrained ELBO contribution remains differentiable end-
    to-end: log_prob at constrained values produces gradients that
    flow back to the guide's loc/log_scale parameters via the
    Jacobian correction."""
    prog = loads(
        "object N : FinSet 1\n"
        "program p : N -> N\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    return sigma\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(4, 1))
    lp = guide.log_prob(torch.zeros(4, 1), samples).sum()
    lp.backward()
    loc = guide.loc_sigma
    assert loc.grad is not None
    assert torch.isfinite(loc.grad).all()


# ---------------------------------------------------------------------------
# Host-data passing through ``condition``.
# ---------------------------------------------------------------------------


def test_condition_data_dict_visible_to_let_expression() -> None:
    """A key in the ``condition`` data dict that does not match a
    declared sample / observe site is exposed as a deterministic
    value to ``let``-expression evaluation. This unlocks per-row
    covariate / index passing for hierarchical regression."""
    prog = loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=1.0)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    return mu\n"
        "export p\n"
    )
    model = prog.morphism
    subj_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    cond = condition(model, {"subj_idx": subj_idx})
    tr = cond.trace(torch.zeros(12, 1))

    # by_subj is a batch-invariant plate latent of shape (|Subj|,).
    # The gather along its only axis with subj_idx of shape (12,)
    # produces mu of shape (12,) — exactly the per-row predictor
    # the hierarchical-regression idiom needs.
    by_subj = tr.sites["by_subj"].value
    mu = tr.sites["mu"].value
    assert by_subj.shape == (4,)
    assert mu.shape == (12,)
    assert torch.allclose(mu, by_subj[subj_idx])


def test_hierarchical_regression_svi_step_runs() -> None:
    """End-to-end SVI step against an observed Bernoulli kernel over
    a per-row plate-gather predictor. The guide-supplied plate
    latent must have the same shape as the model's :class:`PlateDraw`
    output ``(|Subj|,)`` so the ``let mu = by_subj[subj_idx]``
    gather composes when ELBO substitutes the guide sample into the
    model's log-joint env. A regression test for the original
    crossed-random-effects target use case."""
    torch.manual_seed(0)
    prog = loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=1.0)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(probs=mu)\n"
        "    return mu\n"
        "export p\n"
    )
    model = prog.morphism
    guide = AutoNormalGuide(model, observed_names={"r"})

    # Guide-side and model-side plate-latent shapes must agree.
    assert guide.rsample(torch.zeros(1, 1))["by_subj"].shape == (4,)

    obs = {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }
    opt = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=5e-2)
    svi = SVI(model, guide, opt, ELBO())

    # One step must complete without the IndexError that occurs when
    # guide latents are shaped against the program-input batch axis.
    loss0 = float(svi.step(torch.zeros(1, 1), obs))
    assert torch.isfinite(torch.tensor(loss0))

    # Loss should descend with all-zero responses (model wants
    # by_subj's loc to go strongly negative so sigmoid(by_subj) → 0
    # matches the data).
    losses = [float(svi.step(torch.zeros(1, 1), obs)) for _ in range(200)]
    assert losses[-1] < loss0
    loc_final = guide.loc_by_subj.detach()
    assert (loc_final < 0).all(), (
        f"loc_by_subj should be driven negative by all-zero responses, got {loc_final}"
    )


def test_hierarchical_regression_observation_kernel_composes() -> None:
    """End-to-end shape check for the canonical crossed-random-
    effects idiom: a per-subject prior draw, gathered per response
    row, fed as the parameter of an observed Bernoulli plate. The
    trace must complete (no shape-broadcast failure inside the
    observation kernel) and the clamped observation's log_prob must
    be finite and (response_plate,)-shaped."""
    prog = loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=1.0)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(probs=mu)\n"
        "    return mu\n"
        "export p\n"
    )
    subj_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    r_obs = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    cond = condition(prog.morphism, {"subj_idx": subj_idx, "r": r_obs})
    tr = cond.trace(torch.zeros(12, 1))

    assert tr.sites["by_subj"].value.shape == (4,)
    assert tr.sites["mu"].value.shape == (12,)
    assert tr.sites["r"].value.shape == (12,)
    assert torch.allclose(tr.sites["r"].value, r_obs)
    assert torch.isfinite(tr.sites["r"].log_prob).all()


def test_condition_data_dict_alongside_observations() -> None:
    """Host data and observations share the same ``data`` channel;
    the trace clamps observed sample sites and exposes the rest as
    pre-populated environment entries."""
    prog = loads(
        "object Subj : FinSet 3\n"
        "object Resp : FinSet 6\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=1.0)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    observe r : Resp <- Normal(loc=mu, scale=1.0)\n"
        "    return r\n"
        "export p\n"
    )
    model = prog.morphism
    subj_idx = torch.tensor([0, 1, 2, 0, 1, 2])
    r = torch.zeros(6)
    cond = condition(model, {"subj_idx": subj_idx, "r": r})
    tr = cond.trace(torch.zeros(6, 1))

    # `r` is clamped as an observed sample site.
    assert torch.allclose(tr.sites["r"].value, r)
    # `subj_idx` populated env, and the gather fired without error.
    assert "mu" in tr.sites
    assert tr.sites["mu"].value.shape == (6,)


# ---------------------------------------------------------------------------
# Inline Dirichlet.
# ---------------------------------------------------------------------------


def test_inline_dirichlet_scalar_concentration() -> None:
    """``Dirichlet(alpha)`` with a scalar concentration is a symmetric
    Dirichlet on the simplex of the declared codomain's dimension."""
    prog = loads(
        "object Cat : FinSet 3\n"
        "program p : Cat -> Cat\n"
        "    sample pc <- Dirichlet(concentration=1.0)\n"
        "    return pc\n"
        "export p\n"
    )
    tr = trace(prog.morphism, torch.zeros(5, 1))
    pc = tr.sites["pc"].value
    assert pc.shape == (5, 3)
    assert torch.allclose(pc.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (pc >= 0).all()


def test_inline_dirichlet_vector_concentration() -> None:
    """``Dirichlet([α_1, …, α_K])`` with a per-component
    concentration vector is accepted. The parser flattens the
    bracketed sequence into K positional literal floats; the
    inline-distribution call site re-bundles them into a list
    before invoking ``make_fixed_dirichlet``, and the codomain-
    inference path reads the simplex dimension from the
    literal-count rather than from the program's declared
    codomain."""
    prog = loads(
        "object Item : FinSet 8\n"
        "program p : Item -> Item\n"
        "    sample pc : Item <- Dirichlet(concentration=[1.0, 2.0, 3.0])\n"
        "    return pc\n"
        "export p\n"
    )
    tr = trace(prog.morphism, torch.zeros(2, 1))
    # `pc : Item` is a plate over Item of 3-simplex Dirichlet
    # draws; the result has shape (|Item|, 3) regardless of the
    # program input's leading batch axis.
    assert tr.sites["pc"].value.shape == (8, 3)


def test_inline_dirichlet_under_autonormal_guide() -> None:
    """Dirichlet's simplex support is realised in the guide via
    ``biject_to(simplex) = StickBreakingTransform()``; guide samples
    lie on the simplex and ``log_prob`` evaluates."""
    prog = loads(
        "object Cat : FinSet 4\n"
        "program p : Cat -> Cat\n"
        "    sample pc <- Dirichlet(concentration=2.0)\n"
        "    return pc\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(6, 1))
    assert samples["pc"].shape == (6, 4)
    assert torch.allclose(samples["pc"].sum(dim=-1), torch.ones(6), atol=1e-4)
    assert (samples["pc"] >= 0).all()

    lp = guide.log_prob(torch.zeros(6, 1), samples)
    assert lp.shape == (6,)
    assert torch.isfinite(lp).all()


def test_inline_dirichlet_score_under_prior() -> None:
    """The end-to-end claim: a Dirichlet sample drawn by the guide
    can be scored under the model's prior without the ``Expected
    value to be within the support`` failure that motivated the
    constrained-support guide fix."""
    prog = loads(
        "object Cat : FinSet 3\n"
        "program p : Cat -> Cat\n"
        "    sample pc <- Dirichlet(concentration=1.0)\n"
        "    return pc\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    z = guide.rsample(torch.zeros(4, 1))
    tr = trace(prog.morphism, torch.zeros(4, 1), observations={"pc": z["pc"]})
    site = tr.sites["pc"]
    assert site.is_observed
    assert torch.isfinite(site.log_prob).all()
