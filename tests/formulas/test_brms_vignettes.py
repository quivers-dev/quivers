"""The models and post-processing calls of the brms vignettes, in brms's spelling.

Each test writes the calls of one brms vignette (distributional regression,
monotonic effects, multivariate models, nonlinear models, phylogenetic
models, missing values, families, custom families, and the multilevel and
overview papers' models) as the vignette writes them: the same formulas,
``bf``/``lf``/``set_rescor`` combinations, family names, prior selectors,
and post-processing calls. The data are small simulated frames with the
vignette's column names, and the fits run a few SVI steps (a few NUTS draws
where draws are pooled), so the tests establish that every call is accepted
and returns its result with the names brms reports. Numerical agreement
with brms is pinned by the oracle tests of each feature, for instance
``test_brms_spelling.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.distributions import Beta, Binomial, Distribution, constraints

from quivers.diagnostics.hypothesis import HypothesisSummary
from quivers.family_catalog import FamilyCatalog
from quivers.formulas import (
    AuxParam,
    ConstantPrior,
    HypothesisTable,
    PriorSpec,
    bf,
    brmsfamily,
    custom_family,
    fit,
    fit_multiple,
    lf,
    links,
    loo,
    set_rescor,
)
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import INT

SVI = {"method": "svi", "num_steps": 3}
NUTS = {
    "num_warmup": 8,
    "num_samples": 10,
    "num_chains": 2,
    "control": {"max_treedepth": 4},
}


def _table(result: HypothesisSummary | HypothesisTable) -> HypothesisTable:
    """Narrow a hypothesis result to brms's table.

    Parameters
    ----------
    result : HypothesisSummary or HypothesisTable
        The value of a ``hypothesis`` method.

    Returns
    -------
    HypothesisTable
        The table of a brms hypothesis string.
    """
    assert isinstance(result, HypothesisTable)
    return result


def _names(model: object, prefix: str) -> list[str]:
    """List a fit's brms variable names with a prefix.

    Parameters
    ----------
    model : object
        A fitted model with ``variables()``.
    prefix : str
        Leading text of the names kept.

    Returns
    -------
    list[str]
        The matching names, in reported order.
    """
    variables = getattr(model, "variables")()
    return [name for name in variables if name.startswith(prefix)]


def test_distreg_vignette() -> None:
    """Distributional models: ``sigma ~``, ``zi ~`` with ``family=``, smooths."""
    rng = np.random.default_rng(1)
    n = 40
    dat1 = pd.DataFrame({"group": np.repeat(["treat", "placebo"], n // 2)})
    dat1["symptom_post"] = np.where(
        dat1.group == "treat", rng.normal(1, 2, n), rng.normal(0, 1, n)
    )
    fit1 = fit(
        bf("symptom_post ~ group", "sigma ~ group"), data=dat1, family="gaussian", **SVI
    )
    assert type(fit1.summary()).__name__ == "FitSummary"
    assert {"b_sigma_Intercept", "b_sigma_grouptreat"} <= set(fit1.variables())
    assert fit1.conditional_effects() is not None
    hyp = _table(
        fit1.hypothesis(
            "exp(sigma_Intercept + sigma_grouptreat) > exp(sigma_Intercept)"
        )
    )
    assert hyp.rows[0].hypothesis == (
        "(exp(sigma_Intercept+sigma_grouptreat))-(exp(sigma_Intercept)) > 0"
    )

    zinb = pd.DataFrame(
        {
            "persons": rng.integers(1, 5, n),
            "child": rng.integers(0, 3, n),
            "camper": rng.integers(0, 2, n),
        }
    )
    zinb["count"] = rng.poisson(np.exp(0.2 + 0.3 * zinb.persons - 0.5 * zinb.child)) * (
        rng.random(n) > 0.3
    )
    fit_zinb1 = fit(
        "count ~ persons + child + camper",
        data=zinb,
        family="zero_inflated_poisson",
        **SVI,
    )
    assert "zi" in fit_zinb1.variables()
    fit_zinb2 = fit(
        bf("count ~ persons + child + camper", "zi ~ child"),
        data=zinb,
        family=brmsfamily("zero_inflated_poisson"),
        **SVI,
    )
    assert {"b_zi_Intercept", "b_zi_child"} <= set(fit_zinb2.variables())
    assert fit_zinb2.conditional_effects() is not None
    assert type(fit_zinb2.summary()).__name__ == "FitSummary"

    x = rng.random((60, 3))
    dat_smooth = pd.DataFrame(
        {
            "x0": x[:, 0],
            "x1": x[:, 1],
            "x2": x[:, 2],
            "fac": np.array(list("abcd"))[np.arange(60) % 4],
        }
    )
    dat_smooth["y"] = (
        np.sin(4 * dat_smooth.x1)
        + dat_smooth.x2**2
        + rng.normal(0, 0.3 + dat_smooth.x0, 60)
    )
    fit_smooth1 = fit(
        bf("y ~ s(x1) + s(x2) + (1|fac)", "sigma ~ s(x0) + (1|fac)"),
        data=dat_smooth,
        family="gaussian",
        **SVI,
    )
    assert type(fit_smooth1.summary()).__name__ == "FitSummary"
    assert fit_smooth1.conditional_effects() is not None
    assert fit_smooth1.conditional_smooths() is not None


def test_monotonic_vignette() -> None:
    """Monotonic effects, the ordered-factor model, ``loo`` of three models."""
    rng = np.random.default_rng(2)
    n = 60
    income_options = ["below_20", "20_to_40", "40_to_100", "greater_100"]
    code = rng.integers(0, 4, n)
    dat = pd.DataFrame(
        {
            "income": pd.Categorical(
                np.array(income_options)[code], categories=income_options, ordered=True
            ),
            "income_num": code + 1.0,
            "age": rng.normal(40, 10, n),
            "city": np.array(list("abcde"))[np.arange(n) % 5],
        }
    )
    dat["ls"] = np.array([30, 60, 70, 75])[code] + rng.normal(0, 7, n)
    fit1 = fit("ls ~ mo(income)", data=dat, **SVI)
    assert type(fit1.summary()).__name__ == "FitSummary"
    assert "bsp_moincome" in fit1.variables()
    assert fit1.conditional_effects() is not None
    assert fit1.plot(variable="simo", regex=True, backend="none") is not None
    fit2 = fit("ls ~ income_num", data=dat, **SVI)
    fit3 = fit("ls ~ income", data=dat, **SVI)
    assert _names(fit3, "b_") == [
        "b_Intercept",
        "b_income.L",
        "b_income.Q",
        "b_income.C",
    ]
    compared = loo(fit1, fit2, fit3)
    assert compared.diffs is not None and len(compared.diffs.rows) == 3

    prior4 = PriorSpec(
        distribution="Dirichlet([2.0, 1.0, 1.0])", class_="simo", coef="moincome1"
    )
    fit4 = fit("ls ~ mo(income)", data=dat, priors=prior4, sample_prior="yes", **SVI)
    assert type(fit4.summary()).__name__ == "FitSummary"
    fit5 = fit("ls ~ mo(income)*age", data=dat, **SVI)
    assert "bsp_moincome:age" in fit5.variables()
    assert fit5.conditional_effects("income:age") is not None
    fit6 = fit("ls ~ mo(income)*age + (mo(income) | city)", data=dat, **SVI)
    assert type(fit6.summary()).__name__ == "FitSummary"


def test_multivariate_vignette() -> None:
    """``mvbind``, ``bf + bf + set_rescor``, ``lf`` and ``+ family``."""
    rng = np.random.default_rng(3)
    n = 60
    data = pd.DataFrame(
        {
            "sex": np.array(["Fem", "Male", "UNK"])[np.arange(n) % 3],
            "hatchdate": rng.normal(0, 1, n),
            "fosternest": np.array(list("abcdef"))[np.arange(n) % 6],
            "dam": np.array(list("uvwxy"))[np.arange(n) % 5],
        }
    )
    data["tarsus"] = rng.normal(0, 1, n) + 0.5 * (data.sex == "Male")
    data["back"] = rng.normal(0, 1, n) - 0.2 * data.hatchdate
    bform1 = bf(
        "mvbind(tarsus, back) ~ sex + hatchdate + (1|p|fosternest) + (1|q|dam)"
    ) + set_rescor(True)
    fit1 = fit(bform1, data=data, **SVI)
    fit1 = fit1.add_criterion("loo")
    assert type(fit1.summary()).__name__ == "FitSummary"
    assert "rescor__tarsus__back" in fit1.variables()
    assert "cor_fosternest__tarsus_Intercept__back_Intercept" in fit1.variables()
    assert fit1.pp_check(resp="tarsus") is not None
    assert set(fit1.bayes_r2()) == {"R2tarsus", "R2back"}

    bf_tarsus = bf("tarsus ~ sex + (1|p|fosternest) + (1|q|dam)")
    bf_back = bf("back ~ hatchdate + (1|p|fosternest) + (1|q|dam)")
    fit2 = fit(bf_tarsus + bf_back + set_rescor(True), data=data, **SVI)
    compared = loo(fit1, fit2)
    assert compared.diffs is not None and len(compared.diffs.rows) == 2

    bf_tarsus = (
        bf("tarsus ~ sex + (1|p|fosternest) + (1|q|dam)")
        + lf("sigma ~ 0 + sex")
        + brmsfamily("skew_normal")
    )
    bf_back = bf("back ~ s(hatchdate) + (1|p|fosternest) + (1|q|dam)") + brmsfamily(
        "gaussian"
    )
    fit3 = fit(bf_tarsus + bf_back + set_rescor(False), data=data, **SVI)
    assert type(fit3.summary()).__name__ == "FitSummary"
    assert "b_sigma_tarsus_sexMale" in fit3.variables()
    assert fit3.conditional_effects("hatchdate", resp="back") is not None
    assert fit3.add_criterion("loo") is not None


def test_nonlinear_vignette() -> None:
    """``b1 + b2 ~ 1``, ``nlpar`` priors, ``^``, ``inv_logit``, bounded priors."""
    rng = np.random.default_rng(4)
    n = 40
    x = rng.normal(size=n)
    dat1 = pd.DataFrame({"x": x, "y": rng.normal(2 * np.exp(0.75 * x), 1)})
    prior1 = PriorSpec(distribution="Normal(1, 2)", nlpar="b1") + PriorSpec(
        distribution="Normal(0, 2)", nlpar="b2"
    )
    fit1 = fit(
        bf("y ~ b1 * exp(b2 * x)", "b1 + b2 ~ 1", nl=True),
        data=dat1,
        priors=prior1,
        **SVI,
    )
    assert _names(fit1, "b_") == ["b_b1_Intercept", "b_b2_Intercept"]
    assert type(fit1.summary()).__name__ == "FitSummary"
    assert fit1.conditional_effects() is not None
    assert fit1.pp_check() is not None
    fit2 = fit("y ~ x", data=dat1, **SVI)
    assert loo(fit1, fit2).diffs is not None

    ay = np.repeat(np.arange(1991, 1997), 5)
    dev = np.tile(np.array([6.0, 18.0, 30.0, 42.0, 54.0]), 6)
    loss = pd.DataFrame({"AY": ay.astype(str), "dev": dev})
    loss["cum"] = 5000 * (1 - np.exp(-((dev / 45) ** 1.2))) + rng.normal(0, 100, 30)
    fit_loss = fit(
        bf(
            "cum ~ ult * (1 - exp(-(dev/theta)^omega))",
            "ult ~ 1 + (1|AY)",
            "omega ~ 1",
            "theta ~ 1",
            nl=True,
        ),
        data=loss,
        family="gaussian",
        priors=PriorSpec(distribution="Normal(5000, 1000)", nlpar="ult")
        + PriorSpec(distribution="Normal(1, 2)", nlpar="omega", lower=0.0)
        + PriorSpec(distribution="Normal(45, 10)", nlpar="theta", lower=0.0),
        **SVI,
    )
    assert type(fit_loss.summary()).__name__ == "FitSummary"
    assert fit_loss.conditional_effects() is not None
    conditions = pd.DataFrame({"AY": sorted(set(loss.AY))})
    assert (
        fit_loss.conditional_effects(
            conditions=conditions, re_formula=None, method="predict"
        )
        is not None
    )

    ability = rng.normal(size=n)
    dat_ir = pd.DataFrame(
        {
            "ability": ability,
            "answer": (rng.random(n) < 0.33 + 0.67 / (1 + np.exp(-2 * ability))).astype(
                int
            ),
        }
    )
    fit_ir1 = fit("answer ~ ability", data=dat_ir, family="bernoulli", **SVI)
    fit_ir2 = fit(
        bf("answer ~ 0.33 + 0.67 * inv_logit(eta)", "eta ~ ability", nl=True),
        data=dat_ir,
        family=brmsfamily("bernoulli", "identity"),
        priors=PriorSpec(distribution="Normal(0, 5)", nlpar="eta"),
        **SVI,
    )
    assert fit_ir2.conditional_effects() is not None
    assert loo(fit_ir1, fit_ir2).diffs is not None
    fit_ir3 = fit(
        bf(
            "answer ~ guess + (1 - guess) * inv_logit(eta)",
            "eta ~ 0 + ability",
            "guess ~ 1",
            nl=True,
        ),
        data=dat_ir,
        family=brmsfamily("bernoulli", "identity"),
        priors=PriorSpec(distribution="Normal(0, 5)", nlpar="eta")
        + PriorSpec(distribution="Beta(1, 1)", nlpar="guess", lower=0.0, upper=1.0),
        **SVI,
    )
    guess = fit_ir3.variables()["b_guess_Intercept"]
    assert float(guess.min()) > 0 and float(guess.max()) < 1


def test_phylogenetics_vignette() -> None:
    """``gr(phylo, cov = A)``, phylogenetic signal, ``se()`` of an expression."""
    rng = np.random.default_rng(5)
    species = [f"sp_{index}" for index in range(8)]
    root = rng.normal(size=(8, 8))
    covariance = root @ root.T + 8 * np.eye(8)
    scale = np.sqrt(np.diag(covariance))
    A = pd.DataFrame(
        covariance / np.outer(scale, scale), index=species, columns=species
    )
    data = pd.DataFrame(
        {
            "phylo": species * 3,
            "species": species * 3,
            "cofactor": rng.normal(10, 2, 24),
        }
    )
    data["obs"] = [f"o{index}" for index in range(24)]
    data["phen"] = 0.5 * data.cofactor + rng.normal(0, 1, 24)
    data["spec_mean_cf"] = data.groupby("species").cofactor.transform("mean")
    data["within_spec_cf"] = data.cofactor - data.spec_mean_cf
    data["Zr"] = rng.normal(0.2, 0.2, 24)
    data["N"] = rng.integers(10, 50, 24)
    data["phen_pois"] = rng.poisson(3, 24)
    priors = (
        PriorSpec(distribution="Normal(0, 10)", class_="b")
        + PriorSpec(distribution="Normal(0, 50)", class_="Intercept")
        + PriorSpec(distribution="StudentT(3, 0, 20)", class_="sd")
        + PriorSpec(distribution="StudentT(3, 0, 20)", class_="sigma")
    )
    model_simple = fit(
        "phen ~ cofactor + (1|gr(phylo, cov = A))",
        data=data,
        family="gaussian",
        data2={"A": A},
        priors=priors,
        **SVI,
    )
    assert type(model_simple.summary()).__name__ == "FitSummary"
    assert model_simple.conditional_effects() is not None
    hyp = _table(
        model_simple.hypothesis(
            "sd_phylo__Intercept^2 / (sd_phylo__Intercept^2 + sigma^2) = 0", class_=None
        )
    )
    assert 0 < hyp.rows[0].estimate < 1

    model_repeat1 = fit(
        "phen ~ spec_mean_cf + (1|gr(phylo, cov = A)) + (1|species)",
        data=data,
        family="gaussian",
        data2={"A": A},
        priors=priors,
        sample_prior="yes",
        **SVI,
    )
    hyp = _table(
        model_repeat1.hypothesis(
            "sd_phylo__Intercept^2 / "
            "(sd_phylo__Intercept^2 + sd_species__Intercept^2 + sigma^2) = 0",
            class_=None,
        )
    )
    assert np.isfinite(hyp.rows[0].evid_ratio)
    model_repeat2 = model_repeat1.update(data, formula="~ . + within_spec_cf")
    assert "b_within_spec_cf" in model_repeat2.variables()

    model_fisher = fit(
        "Zr | se(sqrt(1 / (N - 3))) ~ 1 + (1|gr(phylo, cov = A)) + (1|obs)",
        data=data,
        family="gaussian",
        data2={"A": A},
        priors=PriorSpec(distribution="Normal(0, 10)", class_="Intercept")
        + PriorSpec(distribution="StudentT(3, 0, 10)", class_="sd"),
        **SVI,
    )
    assert type(model_fisher.summary()).__name__ == "FitSummary"

    model_pois = fit(
        "phen_pois ~ cofactor + (1|gr(phylo, cov = A)) + (1|obs)",
        data=data,
        family=brmsfamily("poisson", "log"),
        data2={"A": A},
        **SVI,
    )
    model_normal = fit(
        "phen_pois ~ cofactor + (1|gr(phylo, cov = A))",
        data=data,
        family="gaussian",
        data2={"A": A},
        **SVI,
    )
    assert model_pois.pp_check() is not None and model_normal.pp_check() is not None
    assert loo(model_pois, model_normal).diffs is not None
    assert model_pois.conditional_effects() is not None


def test_missings_vignette() -> None:
    """``brm_multiple`` with summaries on the combined fit, and ``mi()`` models."""
    rng = np.random.default_rng(6)
    n = 30
    nhanes = pd.DataFrame(
        {"age": rng.integers(1, 4, n).astype(float), "chl": rng.normal(190, 30, n)}
    )
    nhanes["bmi"] = 20 + 0.03 * nhanes.chl + rng.normal(0, 2, n)
    missing = nhanes.copy()
    missing.loc[::5, "chl"] = np.nan
    missing.loc[1::7, "bmi"] = np.nan
    imputed = [
        missing.fillna({"chl": 190.0 + index, "bmi": 26 + 0.1 * index})
        for index in range(3)
    ]
    fit_imp1 = fit_multiple("bmi ~ age*chl", data=imputed, **NUTS)
    assert type(fit_imp1.summary()).__name__ == "FitSummary"
    assert fit_imp1.nchains() == 6
    assert len(fit_imp1.rhats()) == 3
    assert fit_imp1.conditional_effects("age:chl") is not None
    assert fit_imp1.plot(variable="^b", regex=True, backend="none") is not None

    bform = (
        bf("bmi | mi() ~ age * mi(chl)") + bf("chl | mi() ~ age") + set_rescor(False)
    )
    fit_imp2 = fit(bform, data=missing, **SVI)
    assert type(fit_imp2.summary()).__name__ == "FitSummary"
    assert "bsp_bmi_michl" in fit_imp2.variables()
    assert fit_imp2.conditional_effects("age:chl", resp="bmi") is not None

    missing["se"] = 5.0
    bform = (
        bf("bmi | mi() ~ age * mi(chl)") + bf("chl | mi(se) ~ age") + set_rescor(False)
    )
    assert fit(bform, data=missing, **SVI).nobs() is not None


@pytest.mark.parametrize(
    ("family", "response", "registered"),
    [
        ("gaussian", "y", "gaussian"),
        ("student", "y", "student_t"),
        ("skew_normal", "y", "skew_normal"),
        ("exgaussian", "pos", "exgaussian"),
        ("lognormal", "pos", "lognormal"),
        ("Gamma", "pos", "gamma"),
        ("weibull", "pos", "weibull"),
        ("exponential", "pos", "exponential"),
        ("inverse.gaussian", "pos", "inverse_gaussian"),
        ("frechet", "pos", "frechet"),
        ("Beta", "unit", "beta"),
        ("von_mises", "angle", "von_mises"),
        ("poisson", "count", "poisson"),
        ("negbinomial", "count", "negative_binomial"),
        ("geometric", "count", "geometric"),
        ("zero_inflated_poisson", "count", "zero_inflated_poisson"),
        ("zero_inflated_negbinomial", "count", "zero_inflated_negbinomial"),
        ("hurdle_poisson", "count", "hurdle_poisson"),
        ("hurdle_negbinomial", "count", "hurdle_negbinomial"),
        ("hurdle_gamma", "hurdled", "hurdle_gamma"),
        ("hurdle_lognormal", "hurdled", "hurdle_lognormal"),
        ("bernoulli", "binary", "bernoulli"),
        ("cumulative", "rating", "cumulative"),
        ("sratio", "rating", "sratio"),
        ("cratio", "rating", "cratio"),
        ("acat", "rating", "acat"),
        ("categorical", "rating", "categorical"),
    ],
)
def test_families_vignette(family: str, response: str, registered: str) -> None:
    """Each family of the families vignette fits and predicts by its brms name."""
    rng = np.random.default_rng(7)
    n = 24
    data = pd.DataFrame({"x": np.sin(np.arange(n) * 0.7)})
    data["y"] = 0.5 * data.x + rng.normal(0, 0.5, n)
    data["pos"] = np.exp(0.3 * data.y)
    data["unit"] = 1 / (1 + np.exp(-data.y))
    data["angle"] = np.clip(data.y, -3, 3)
    data["count"] = rng.poisson(2.0, n) * (np.arange(n) % 3 != 0)
    data["hurdled"] = data.pos * (np.arange(n) % 4 != 0)
    data["binary"] = (data.y > 0).astype(int)
    data["rating"] = np.arange(n) * 5 % 4 + 1
    model = fit(f"{response} ~ x", data=data, family=family, **SVI)
    assert model.family.name == registered
    assert model.posterior_predict(num_samples=2).shape[:3] == (1, 2, n)
    # The Student-t mean needs nu above one and a circular mean a convention,
    # so those two families are checked on the linear predictor.
    predict = (
        model.posterior_linpred
        if family in {"student", "von_mises"}
        else model.posterior_epred
    )
    assert predict(new_data=data.iloc[:3], num_samples=2).shape[2] == 3


class BetaBinomial2(Distribution):
    """The beta-binomial law of the custom families vignette.

    Parameters
    ----------
    mu : torch.Tensor
        Success probability mean, in (0, 1).
    phi : torch.Tensor
        Positive precision.
    T : torch.Tensor
        Number of trials.
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    support = constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, mu: torch.Tensor, phi: torch.Tensor, T: torch.Tensor) -> None:
        """Broadcast the parameters to one batch shape.

        Parameters
        ----------
        mu : torch.Tensor
            Mean.
        phi : torch.Tensor
            Precision.
        T : torch.Tensor
            Trials.
        """
        self.mu, self.phi, self.T = torch.broadcast_tensors(mu, phi, T)
        super().__init__(self.mu.shape, validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean count.

        Returns
        -------
        torch.Tensor
            ``mu * T``.
        """
        return self.mu * self.T

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the beta-binomial log mass.

        Parameters
        ----------
        value : torch.Tensor
            Counts.

        Returns
        -------
        torch.Tensor
            Log probability mass.
        """
        a, b = self.mu * self.phi, (1 - self.mu) * self.phi
        trials, y = self.T.to(self.mu.dtype), value.to(self.mu.dtype)
        return (
            torch.lgamma(trials + 1)
            - torch.lgamma(y + 1)
            - torch.lgamma(trials - y + 1)
            + torch.lgamma(y + a)
            + torch.lgamma(trials - y + b)
            - torch.lgamma(trials + a + b)
            + torch.lgamma(a + b)
            - torch.lgamma(a)
            - torch.lgamma(b)
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw counts through the beta mixture of binomials.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Integer counts.
        """
        p = Beta(self.mu * self.phi, (1 - self.mu) * self.phi).sample(sample_shape)
        return Binomial(self.T.to(p.dtype).expand(p.shape), p).sample().to(torch.int64)


def test_customfamilies_vignette() -> None:
    """The binomial model, the custom beta-binomial, and their comparison."""
    rng = np.random.default_rng(8)
    n = 40
    cbpp = pd.DataFrame(
        {
            "period": np.array(list("1234"))[np.arange(n) % 4],
            "herd": np.array(list("abcdefgh"))[np.arange(n) % 8],
            "size": rng.integers(5, 20, n),
        }
    )
    cbpp["incidence"] = rng.binomial(cbpp["size"], 0.15)
    fit1 = fit(
        "incidence | trials(size) ~ period + (1|herd)",
        data=cbpp,
        family=brmsfamily("binomial"),
        **SVI,
    )
    semantic = DistributionFamily(
        name="beta_binomial2",
        parameters=(
            FamilyParameter(name="mu", constraint="unit_interval", rank=0),
            FamilyParameter(name="phi", constraint="positive", rank=0),
            FamilyParameter(name="T", constraint="integer", rank=0),
        ),
        support="integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
    )
    catalog = FamilyCatalog().with_family(
        semantic,
        lambda p: BetaBinomial2(p["mu"], p["phi"], p["T"]),
        support=BetaBinomial2.support,
    )
    beta_binomial2 = custom_family(
        "beta_binomial2",
        catalog=catalog,
        location="mu",
        link="logit",
        auxiliary=(AuxParam(name="phi", prior="Gamma(2.0, 0.1)", link=links["log"]),),
        data={"T": "size"},
    )
    fit2 = fit("incidence ~ period + (1|herd)", data=cbpp, family=beta_binomial2, **SVI)
    assert type(fit2.summary()).__name__ == "FitSummary"
    assert loo(fit1, fit2).diffs is not None
    assert fit2.pp_check() is not None
    assert fit2.posterior_epred(num_samples=3).shape == (1, 3, n)
    assert fit2.posterior_predict(new_data=cbpp.iloc[:4], num_samples=3).shape == (
        1,
        3,
        4,
    )
    assert fit2.conditional_effects(conditions=pd.DataFrame({"size": [1]})) is not None


def test_multilevel_and_overview_models() -> None:
    """Ordinal models on ratings 1..4, ``cs()``, group hypotheses, ``update``."""
    rng = np.random.default_rng(9)
    n = 48
    inhaler = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(n // 2), 2).astype(str),
            "treat": np.tile([0.5, -0.5], n // 2),
            "period": np.tile([-0.5, 0.5], n // 2),
            "carry": rng.choice([-1.0, 0.0, 1.0], n),
            "rating": rng.integers(1, 5, n),
        }
    )
    fit_ordinal = fit(
        "rating ~ treat + period + carry + (1|subject)",
        data=inhaler,
        family=brmsfamily("cumulative"),
        **SVI,
    )
    assert fit_ordinal.category_labels == (1, 2, 3, 4)
    assert type(fit_ordinal.summary()).__name__ == "FitSummary"
    assert fit_ordinal.conditional_effects("treat", categorical=True) is not None
    hyp = _table(fit_ordinal.hypothesis("Intercept > 0", class_="sd", group="subject"))
    assert hyp.rows[0].hypothesis == "(Intercept) > 0" and hyp.rows[0].post_prob == 1.0
    fit_sratio = fit(
        "rating ~ period + carry + cs(treat) + (1|subject)",
        data=inhaler,
        family=brmsfamily("sratio", "cloglog"),
        **SVI,
    )
    assert type(fit_sratio.summary()).__name__ == "FitSummary"
    fit_acat = fit(
        "rating ~ period + carry + cs(treat) + (1|subject)",
        data=inhaler,
        family="acat",
        **SVI,
    )
    assert fit_acat.nobs() == n
    fit_equidistant = fit(
        "rating ~ treat + period + carry",
        data=inhaler,
        family=brmsfamily("cumulative", threshold="equidistant"),
        **SVI,
    )
    assert fit_equidistant.nobs() == n

    epilepsy = pd.DataFrame(
        {
            "zAge": rng.normal(size=n),
            "zBase": rng.normal(size=n),
            "Trt": np.tile([0, 1], n // 2),
            "patient": np.repeat(np.arange(n // 4), 4).astype(str),
        }
    )
    epilepsy["obs"] = np.arange(n).astype(str)
    epilepsy["count"] = rng.poisson(4, n)
    fit_poisson = fit(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family=brmsfamily("poisson"),
        priors=PriorSpec(distribution="Normal(0, 1)", class_="b")
        + PriorSpec(
            distribution=ConstantPrior(value=1.0), class_="sd", group="patient"
        ),
        **SVI,
    )
    scale = fit_poisson.variables()["sd_patient__Intercept"]
    assert float(scale.min()) == float(scale.max()) == 1.0
    fit_obs = fit_poisson.update(formula=". ~ . + (1|obs)")
    assert "sd_obs__Intercept" in fit_obs.variables()
    assert loo(fit_poisson, fit_obs).diffs is not None
    fit_negbinomial = fit(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="negbinomial",
        **SVI,
    )
    assert fit_negbinomial.family.name == "negative_binomial"

    rent = pd.DataFrame(
        {
            "area": rng.uniform(20, 160, 60),
            "yearc": rng.uniform(1920, 2000, 60),
            "district": np.array(list("abcdef"))[np.arange(60) % 6],
        }
    )
    rent["rentsqm"] = 7 + 0.01 * (rent.yearc - 1950) + rng.normal(0, 1, 60)
    fit_rent = fit(
        bf(
            "rentsqm ~ t2(area, yearc) + (1|ID1|district)",
            "sigma ~ t2(area, yearc) + (1|ID1|district)",
        ),
        data=rent,
        **SVI,
    )
    assert "cor_district__Intercept__sigma_Intercept" in fit_rent.variables()
    assert fit_rent.conditional_smooths() is not None
