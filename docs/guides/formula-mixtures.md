# Mixtures, category formulas, and count denominators

This guide covers three parts of the brms response interface
[@burkner2017brms]: finite mixtures of response families, separate
linear predictors for the categories of a categorical or compositional
response, and the `rate()` and `subset()` response terms. Each compiles to
one QVR program through the ordinary
[`fit`](../api/formulas/fit.md) entry point, so SVI, HMC, NUTS,
prediction, refits, cross-validation, and saved artifacts work as they do
for any other formula.

## Finite mixtures

In a [finite mixture](https://en.wikipedia.org/wiki/Mixture_model) each
observation comes from one of `K` component distributions, and the
likelihood sums the weighted component densities
[@mclachlan2000finite]:

$$
p(y_i) = \sum_{k=1}^{K} \theta_{ik} \, p_k(y_i \mid \mu_{ik}, \ldots).
$$

[`mixture`](../api/formulas/index.md#quivers.formulas.mixture) builds the
family from its components, as brms `mixture()` does. Components may be
different families with the same support (all real or all integer).
The distributional parameters of component `k` carry the suffix `k`:
`mixture("gaussian", "student_t")` has `mu1`, `sigma1`, `mu2`, `sigma2`,
`nu2`, and the mixing proportions `theta1` and `theta2`.

| brms | QVR |
| --- | --- |
| `family = mixture(gaussian, student)` | `family=mixture("gaussian", "student_t")` |
| `mixture(gaussian, nmix = 3)` | `mixture("gaussian", nmix=3)` or `family="mixture", mixture_components=3` |
| `mixture(..., order = "none")` | `mixture(..., order="none")` |
| `bf(y ~ 1, mu1 ~ x, mu2 ~ z)` | `distributional={"mu1": "~ x", "mu2": "~ z"}` |
| `bf(y ~ x, theta2 ~ z)` | `distributional={"theta2": "~ z"}` |
| `bf(y ~ x, theta1 = 1, theta2 = 3)` | `fixed_parameters={"theta1": 1.0, "theta2": 3.0}` |
| `bf(y ~ x, sigma2 = "sigma1")` | `mixture(..., equal={"sigma2": "sigma1"})` |
| `prior(normal(5, 2), Intercept, dpar = mu2)` | `PriorSpec(distribution="Normal(5, 2)", class_="Intercept", dpar="mu2")` |
| `prior(dirichlet(2), theta)` | `PriorSpec(distribution="Dirichlet(concentration=[2.0, 2.0])", class_="theta")` |
| `pp_mixture(fit, summary = FALSE)` | `fitted.pp_mixture()` |

A parameter equated to a sampled one, such as `sigma2` above, appears in
`draws()` and `variables()` under its own name with the draws of its source,
because brms declares it as a transformed parameter. A parameter equated to
a predicted one varies by row and is not among the draws.

The main formula predicts every `mu<k>` without a formula of its own, and
each component receives its own coefficients and group effects. A
`mu<k> ~ ...` formula replaces the main formula for that component.
Mixing proportions take one of three forms:

* by default, one simplex `theta` with a
  [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior
  (`Dirichlet(1, ..., 1)`), as in brms,
* predicted by formulas for all but one proportion, as brms requires. The
  proportion without a formula is the reference with linear predictor
  zero, and the proportions are the
  [softmax](https://en.wikipedia.org/wiki/Softmax_function) of the `K`
  linear predictors, row by row,
* fixed for every component, then divided by their sum.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import fit, mixture

rng = np.random.default_rng(1)
rows = 60
low = rng.random(rows) < 0.6
data = pd.DataFrame({"x": rng.normal(size=rows), "z": rng.integers(0, 2, rows)})
data["y"] = np.where(low, rng.normal(0.5 * data.x, 1.0), rng.normal(5.0, 1.0))

two_normals = fit(
    "y ~ x",
    data=data,
    family=mixture("gaussian", "gaussian"),
    distributional={"theta2": "~ z"},
    method="svi",
    num_steps=200,
    seed=2,
)
membership = two_normals.pp_mixture(num_samples=20)
assert membership.shape == (1, 20, rows, 2)
locations = two_normals.posterior_linpred(num_samples=20)
assert locations.shape == (1, 20, rows, 2)
expected = two_normals.posterior_epred(new_data=data.head(3), num_samples=20)
assert expected.shape == (1, 20, 3)
```

### Identifying the components

Relabelling the components of a mixture leaves its likelihood unchanged,
so an unconstrained posterior has `K!` symmetric modes and draws can
switch labels [@stephens2000labels]. With `order="mu"` (the default when
all components share one family, as in brms) the location intercepts are
ordered: the first intercept keeps its prior, and each later intercept is
the previous one plus a positive gap,

$$
\alpha_1 < \alpha_2 < \cdots < \alpha_K, \qquad
\alpha_k = \alpha_1 + \sum_{j<k} \delta_j, \quad \delta_j > 0 .
$$

The gaps are sampled on the log scale by the same positive transform as
ordinal threshold spacings, so the mirrored labellings are outside the
parameter space rather than penalized. Two choices differ from brms, and
both follow from conventions this package uses throughout:

* brms places its intercept prior on each ordered intercept. QVR places
  the intercept prior on the first and a `HalfNormal(5)` prior on each gap
  (class `mixture_intercept_spacing`), because every latent coordinate
  carries a proper generative prior. A prior on the intercept of
  `mu2` and later components is thus not available under ordering.
* brms orders the intercepts of its centered design. QVR orders the
  intercept that the program samples: the centered one under
  `default_priors="brms"` (where intercepts are centered), and the
  uncentered one otherwise.

Ordering requires an intercept in every location formula. Mixtures of
different families default to `order="none"`.

`fitted.draws()` reports the derived intercepts of `mu2, ..., muK` beside
the sampled sites, so `fixed_effects(parameter="mu2")` and the coefficient
tables read them as ordinary coefficients.

### Censoring, truncation, and prediction

`cens()` scores a censored row by the weighted component probabilities of
its interval. Under `trunc()` each component is normalized on the
truncation interval before mixing, as in brms, so the density is
`sum_k theta_k p_k(y) / P_k(lb <= Y <= ub)`. Predictions draw a component
and then a value from that component's truncated law.

`posterior_epred` returns `sum_k theta_k E_k[Y]`, `posterior_predict` draws
a component and then a response, and `log_likelihood` returns the
pointwise log of the weighted sum. `posterior_linpred(parameter="mu")`
returns every component location on a last axis, and `parameter="mu2"`
selects one. `pp_mixture` returns the posterior probability that each row
belongs to each component, `theta_k p_k(y) / sum_j theta_j p_j(y)`, with
draws on the leading axes as brms returns with `summary = FALSE`.

Mixtures of Gaussian components compile to the native
[`MixtureNormal`](../api/continuous/families.md) kernel, and other mixtures
to the general
[`Mixture`](../api/continuous/measure.md#quivers.continuous.measure.Mixture)
of component laws. A mixture can be one response of
a [multivariate model](formula-multivariate.md).

A group identifier shared by several component formulas, `(1 | p | g)`
in the main formula for instance, correlates those group effects in one
block across the components, as in brms. Any family with a scalar
response event can be a component, including the continuous hurdle and
zero-inflated families, which brms excludes. Categorical, simplex,
Wiener, Cox, and custom families are refused as components.

## Mixtures of ordinal families

The cumulative, stopping-ratio (`sratio`), continuation-ratio (`cratio`),
adjacent-category (`acat`), and hurdle cumulative families mix with one
another but not with other families, as in brms. Each component is the
ordinary ordinal model of its family, with its own thresholds,
discrimination (`disc1 ~ x`), and category-specific `cs()` terms. The
response is an ordered factor or integer codes (`1..K` as brms codes them,
or `0..K-1`), and
`posterior_epred` returns the mixed category probabilities on a last
category axis.

`order` selects how the thresholds are identified. The default for ordinal
components is `order="none"`, where every component has its own threshold
vector. `order="mu"` gives every component one shared threshold vector,
which is ordered when any component is cumulative. The components then
differ only through their predictors and discriminations. Shared thresholds
require flexible thresholds in every component and location formulas that
all keep or all drop the intercept.

<!-- python: run -->
```python
rating_rng = np.random.default_rng(7)
ratings = pd.DataFrame(
    {"x": rating_rng.normal(size=80), "z": rating_rng.normal(size=80)}
)
ratings["y"] = pd.Categorical(
    rating_rng.integers(1, 5, 80), categories=[1, 2, 3, 4], ordered=True
)

ordinal = fit(
    "y ~ x",
    data=ratings,
    family=mixture("cumulative", "sratio"),
    distributional={"mu2": "~ z"},
    method="svi",
    num_steps=50,
    seed=5,
)
probabilities = ordinal.posterior_epred(new_data=ratings.head(3), num_samples=4)
assert probabilities.shape == (1, 4, 3, 4)
assert ordinal.pp_mixture(num_samples=4).shape == (1, 4, 80, 2)

shared = fit(
    "y ~ x",
    data=ratings,
    family=mixture("sratio", "cumulative", order="mu"),
    distributional={"mu2": "~ z"},
    method="svi",
    num_steps=50,
    seed=5,
)
assert "dpar_mu1_cutpoint_offset" not in shared.draws()
```

Each ordinal component keeps the quivers threshold coordinates of its
family: a location intercept and thresholds centered at zero, where brms
samples the thresholds alone. The two describe the same likelihood, with
brms's threshold `k` equal to the centered threshold minus the intercept,
but the default priors differ as they do for a single ordinal family.
Shared thresholds are not available with equidistant thresholds (as in
brms) or hurdle components. [Group-specific thresholds](#group-specific-thresholds)
(`thres(gr = g)`) give every component its own vector per group.

## Category-specific formulas

A categorical, multinomial, Dirichlet, Dirichlet-multinomial, or
logistic-normal response has one location parameter per non-reference
category, named `mu` followed by the category label, as in brms: levels
`A`, `B`, `C` with reference `A` give `muB` and `muC`. The main formula
predicts every category without a formula of its own. A formula for one
category gives it its own coefficients and group effects, and the
reference category's linear predictor is zero.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import PriorSpec, fit

rng = np.random.default_rng(3)
rows = 60
choices = pd.DataFrame({"x": rng.normal(size=rows), "z": rng.normal(size=rows)})
choices["y"] = pd.Categorical(rng.choice(["A", "B", "C"], rows))

per_category = fit(
    "y ~ x",
    data=choices,
    family="categorical",
    distributional={"muC": "~ z"},
    priors=PriorSpec(distribution="Normal(0.0, 1.0)", class_="b", dpar="muC"),
    method="svi",
    num_steps=50,
    seed=4,
)
assert set(per_category.fixed_effects(parameter="muC", num_samples=5)) == {
    "Intercept",
    "z",
}
probabilities = per_category.posterior_epred(new_data=choices.head(4), num_samples=5)
assert probabilities.shape == (1, 5, 4, 3)
```

The main formula and every per-category formula accept every term of an
ordinary predictor. Each free category has its own coefficients, smooths,
monotonic simplexes, Gaussian process hyperparameters, spatial fields, and
group standard deviations (`simo_muB_mok1`, `lscale_muC_gpz`,
`sdcar_muB`, `sd_mmg1g2__muB_Intercept` in brms's names). The group
coefficients of different categories are independent, and the coefficients
of one category are correlated as the formula states. A term with an
identifier, such as `(1 + x | id | g)`, correlates the coefficients of every
category, as in brms. A factor with `dist = "student"` shares one
degrees-of-freedom parameter across the categories that use it.

A prior with `dpar="muB"` governs the parameters of one category and leaves
the others at their own priors.

A category may also have a nonlinear expression of its own, given as a
[`NonlinearExpression`](../api/formulas/fit.md#quivers.formulas.NonlinearExpression) under its name. With a nonlinear main formula, every
category without a formula or expression of its own reads the main
expression.

<!-- python: run -->
```python
term_rng = np.random.default_rng(8)
choices["dose"] = term_rng.integers(1, 5, rows)
choices["rater1"] = term_rng.choice(["p", "q", "r"], rows)
choices["rater2"] = term_rng.choice(["p", "q", "r"], rows)

special = fit(
    "y ~ mo(dose) + (1 | mm(rater1, rater2))",
    data=choices,
    family="categorical",
    distributional={"muC": "~ gp(z, k = 5, c = 5/4)"},
    method="svi",
    num_steps=50,
    seed=4,
)
assert {"simo_muB_modose1[1]", "sd_mmrater1rater2__muB_Intercept", "lscale_muC_gpz"} <= set(
    special.variables()
)
assert special.posterior_epred(new_data=choices.head(4), num_samples=5).shape == (
    1,
    5,
    4,
    3,
)
```

Measurement-error `me()` terms are not available in a per-category or other
distributional formula, and `mi()` terms there compile through a
[joint formula](formula-missing-data.md). The shared logistic-normal scale
formula `sigma ~ ...` gives every scale its own coefficients for each of its
terms.

`Family.with_reference("C")` (brms `refcat`) selects the reference, and
the free categories are then `muA` and `muB`. When every category has the
same formula, the model equals the shared emission with one coefficient
vector over the free categories. Coefficient names follow brms
(`b_muC_z`).

### Category-specific scales

The [logistic-normal](https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization)
family models the log ratios of each non-reference component to the
reference as correlated Normal variables. Each non-reference component has
its own scale, named `sigma` followed by the component label, as in brms
[@burkner2017brms]: components `a`, `b`, `c` with reference `a` give
`sigmab` and `sigmac`, the standard deviations of `log(b / a)` and
`log(c / a)`. A formula for one scale predicts it through the log link with
its own coefficients and group effects. A scale without a formula stays one
positive parameter, and `fixed_parameters={"sigmac": 0.5}` holds it constant.

<!-- python: run -->
```python
shares = rng.dirichlet([2.0, 3.0, 4.0], size=rows)
parts = pd.DataFrame(
    {"a": shares[:, 0], "b": shares[:, 1], "c": shares[:, 2], "x": choices["x"]}
)

scaled = fit(
    "cbind(a, b, c) ~ x",
    data=parts,
    family="logistic_normal",
    distributional={"sigmab": "~ x"},
    priors=PriorSpec(distribution="HalfNormal(1.0)", class_="sigmac"),
    method="svi",
    num_steps=50,
    seed=4,
)
assert {"b_sigmab_Intercept", "b_sigmab_x", "sigmac"} <= set(scaled.variables())
spread = scaled.posterior_linpred(
    parameter="sigmab", transform=True, new_data=parts.head(4), num_samples=5
)
assert spread.shape == (1, 5, 4) and bool((spread > 0).all())
```

A scale formula is an ordinary scalar predictor, so it accepts the terms a
per-category location accepts. Priors address it with `dpar="sigmab"`, and
the scalar scale has the class `sigmac`. The reference component has no log
ratio and thus no scale. `sigma ~ ...` is the quivers spelling that predicts
every scale with one formula, which brms does not have, and it cannot be
combined with a `sigma<c>` formula or fixed value.

## Count denominators, thresholds, and row subsets

`y | rate(denom) ~ x` models a count over `denom` units of exposure for
the poisson, negative_binomial, negbinomial2, and geometric families, and
for mixtures whose components all belong to them. Any other family,
including a categorical or vector response, refuses `rate()` with a
`ValueError`.
The mean becomes `mu * denom` for every link, and the negative binomial
shape is multiplied by `denom` as well, so a count over `denom` units is
the sum of `denom` independent unit counts. The geometric family thus
becomes the negative binomial with shape `denom`, as in brms.
`posterior_epred` includes the denominator, while `posterior_linpred` and
`mu` do not. New rows require the denominator column. Denominators must be
positive (brms admits zero, which gives a degenerate zero mean).

`subset()` accepts a Boolean column or numeric 0/1 indicators, as brms
reads them as logical values.

`y | thres(4) ~ x` declares four thresholds (five categories) for an
ordinal family even when some categories are not observed, and the older
`cat(5)` declares the five categories directly. Integer responses are then
codes `0, ..., 4`, and a factor response must have exactly five levels.
For the hurdle cumulative family, code zero is the hurdle and lies outside
the thresholds, as in brms: `thres(4)` then declares the codes `0, ..., 5`.
The custom-family data arguments `vreal()` and `vint()` correspond to
`custom_family(..., data={"native": "column"})`.

### Group-specific thresholds

`y | thres(gr = g) ~ x` gives each level of `g` its own threshold vector
with no pooling between groups, as brms does. The vectors may differ in
length, which describes items rated on scales with different numbers of
categories. A group's number of thresholds is its largest response code,
or the value of `n` in `thres(n, gr = g)`, where `n` is a number or a
column that is constant within each group. `thresholds_by=` is the
partially pooled alternative, with one common number of thresholds.

<!-- python: run -->
```python
items = pd.DataFrame(
    {
        "rating": [0, 1, 2, 3, 1, 2, 0, 3, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2, 0, 1],
        "x": np.linspace(-1.5, 1.5, 20),
        "item": ["long"] * 10 + ["short"] * 10,
    }
)
grouped = fit(
    "rating | thres(gr = item) ~ x",
    data=items,
    family="cumulative",
    method="svi",
    num_steps=50,
    seed=6,
)
names = set(grouped.variables(num_samples=4))
assert {"b_Intercept[long,3]", "b_Intercept[short,2]", "b_x"} <= names
assert "b_Intercept[short,3]" not in names
probabilities = grouped.posterior_epred(num_samples=4)
assert probabilities.shape == (1, 4, 20, 4)
assert bool((probabilities[..., 10:, 3] == 0).all())
```

The cumulative, `sratio`, `cratio`, `acat`, and hurdle cumulative families
take threshold groups. `variables()` reports brms's thresholds
`b_Intercept[<group>,<k>]`, and a multivariate model reports
`b_<resp>_Intercept[<group>,<k>]` for a response with threshold groups and
`b_<resp>_Intercept[<k>]` for a cumulative response with one vector.
The sampled coordinates are the quivers threshold coordinates in every
group: a shape with mean zero and a location. With a formula intercept the
first group's location is zero and the other locations
(`threshold_location`) are contrasts with it. The map to brms's vectors is
one to one, but the default priors differ, as they do for a single
threshold vector. `posterior_epred` returns one column per category of the
longest scale, with probability zero for the categories a group lacks.
New rows must belong to a fitted group, because a group's thresholds are
estimated from its own rows alone. Category-specific effects are refused
with threshold groups, as in brms, and so are equidistant thresholds.

With the hurdle cumulative family a group's number of thresholds is its
largest response code minus one, since code zero is the hurdle, and
`posterior_epred` has the hurdle in its first column. In a mixture of
ordinal families every component has its own vector for every group, and
`order="mu"` gives the components one shared vector per group.

<!-- python: run -->
```python
from quivers.formulas import mixture

visits = items.assign(rating=items["rating"] + 1)
visits.loc[[2, 7, 13, 18], "rating"] = 0
hurdle = fit(
    "rating | thres(gr = item) ~ x",
    data=visits,
    family="hurdle_cumulative",
    method="svi",
    num_steps=50,
    seed=6,
)
assert {"hu", "b_Intercept[long,3]", "b_Intercept[short,2]"} <= set(
    hurdle.variables(num_samples=4)
)
assert hurdle.posterior_epred(num_samples=4).shape == (1, 4, 20, 5)

mixed = fit(
    "rating | thres(gr = item) ~ x",
    data=items,
    family=mixture("cumulative", "acat"),
    method="svi",
    num_steps=50,
    seed=6,
)
mixed_probabilities = mixed.posterior_epred(num_samples=4)
assert bool((mixed_probabilities[..., 10:, 3] == 0).all())
assert mixed.pp_mixture(num_samples=4).shape == (1, 4, 20, 2)
```

A mixture keeps the quivers threshold coordinates of each component under
their native names (`dpar_mu1_cutpoint_log_spacing`,
`dpar_mu1_threshold_location`), as an ordinal mixture without groups does.
brms fits this model but its `log_lik` and `posterior_epred` fail for it.
The components of a mixture with threshold groups must all be hurdle
cumulative or all lack the hurdle, because a group's number of thresholds is
read from one coding of the response. brms accepts the mixed case.

[`update_adterms`](../api/formulas/index.md#quivers.formulas.update_adterms)
edits the addition terms of a formula as brms `update_adterms` does:
`update_adterms("y | weights(w) ~ x", "~ trials(10)")` returns
`"y | weights(w) + trials(10) ~ x"`, replacing a term of the same kind,
and `action="replace"` drops the existing terms first.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import fit

rng = np.random.default_rng(5)
exposure = pd.DataFrame({"x": rng.normal(size=40), "hours": rng.uniform(1.0, 4.0, 40)})
exposure["events"] = rng.poisson(2.0 * exposure.hours)
exposure["keep"] = (np.arange(40) % 4 != 0).astype(int)

counts = fit(
    "events | rate(hours) + subset(keep) ~ x",
    data=exposure,
    family="negative_binomial",
    method="svi",
    num_steps=30,
    seed=6,
)
assert counts.posterior_epred(new_data=exposure.head(3), num_samples=4).shape == (1, 4, 3)
```
