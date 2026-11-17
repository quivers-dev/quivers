# Grouping terms

A group-level term `(lhs | group)` gives each level of a grouping factor its
own coefficients for the columns of `lhs`. The coefficients of one block share
a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
with a standard deviation per coefficient and, for more than one coefficient,
an LKJ correlation matrix [@burkner2017brms]. This guide covers brms's syntax
for the right side of the bar: `gr()`, interactions and nesting of grouping
factors, and multi-membership terms. The
[formula guide](analysis-data-and-formulas.md#correlated-group-effects)
covers the left side, double bars, and shared identifiers.

## Options of `gr()`

`gr(g, cor, id, cov, by, dist, pw)` names the grouping factor `g` together
with options of its block. `(lhs | gr(g))` is `(lhs | g)`.

| Option | Meaning | Bar notation |
|---|---|---|
| `cor = FALSE` | independent coefficients | `(lhs \|\| g)` |
| `id = "i"` | one correlation block for every term with this `id` and factor | `(lhs \|i\| g)` |
| `cov = M` | known covariance between the levels of `g`, read from `data2` | `group_covariances` |
| `by = f` | separate standard deviations and correlations per level of `f` | none |
| `dist = "student"` | multivariate Student-t coefficients | none |
| `pw = w` | per-level weight on the prior of the coefficients | none |

`gr(g, cov = M)` takes `M` from the `data2` argument of `fit`, `get_prior`,
and `formula_to_qvr`. `M` must be a pandas `DataFrame` whose row and column
labels name the levels of `g`, and it may include levels absent from the
training rows. As in brms, the coefficients of level $j$ become
$\sigma \, (L z)_j$ with $L$ the Cholesky factor of `M` and standard-normal
$z$, the model a
[`GroupCovariance`](../api/formulas/fit.md) for `g` defines. The fit retains
the covariance, so new levels covered by `M` are predicted from their
conditional distribution given the fitted levels.

`gr(g, by = f)` gives every level of the factor `f` its own standard
deviation for each coefficient and, in a correlated block, its own LKJ
correlation matrix. Each level of `g` must lie within one level of `f`,
and otherwise the formula is refused with brms's message. A new level of `g`
at prediction takes the level of `f` named in its fresh rows, which must be a
fitted level.

`gr(g, dist = "student")` follows brms. It draws degrees of freedom
$\nu \sim \operatorname{Gamma}(2, 0.1)$ restricted to $\nu > 1$, of
class `df`, and for each level $j$ a mixing variable
$u_j \sim \text{inverse-}\chi^2(\nu)$, and multiplies the level's Gaussian
coefficients by $\sqrt{\nu u_j}$. The coefficients of a level then follow a
[multivariate Student-t distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution)
with $\nu$ degrees of freedom, which is less sensitive to outlying levels
than the normal [@burkner2017brms]. The mixing variables are level
coordinates, so new levels draw them with the coefficients under every
new-level policy.

`gr(g, pw = w)` weights the standard-normal prior of level $j$'s
innovations by $w_j$, as brms adds $w_j \log \phi(z_j)$ to the log
density. `w` must be finite, nonnegative, and constant within each level.
Weights below one let a level's coefficients move farther from zero, and a
weight of zero removes its prior. With weights the model density is no
longer a normalized prior times a likelihood, so independent prior draws
refuse it, as for any model with a log-density factor.

These options use brms's noncentered innovations and are refused with
`reparameterize="centered"` and with a known level covariance. In a
categorical or other per-category response model each free category takes
its own group coefficients with these options (see
[category-specific formulas](formula-mixtures.md#category-specific-formulas)).
Every group-level term of one factor must use the same options.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import fit, get_prior

rows = np.arange(48)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "clinic": np.array(list("abcdef"))[rows % 6],
    }
)
data["region"] = np.where(data["clinic"].isin(list("abc")), "north", "south")
data["visits"] = np.where(data["clinic"].isin(list("ab")), 2.0, 1.0)
data["y"] = 0.4 * data["x"] + 0.1 * (rows % 5)
formula = "y ~ x + (1 + x | gr(clinic, by = region, dist = 'student', pw = visits))"
priors = get_prior(formula, data=data)
assert {"df_clinic", "cor_clinic_by1", "cor_clinic_by2"} <= priors.keys()
model = fit(formula, data=data, method="svi", num_steps=20, seed=1)
fresh = pd.DataFrame({"x": [0.0, 1.0], "clinic": ["a", "g"], "region": ["north", "south"]})
draws = model.posterior_epred(new_data=fresh, num_samples=10, seed=2, allow_new_levels=True)
assert draws.shape == (1, 10, 2)
```

## Interactions and nesting

`(1 | g1:g2)` uses the combinations of `g1` and `g2` as levels. As in brms,
a combination is labeled by pasting the levels with an underscore, so the
levels of `a` and `u` form `a_u`, and only combinations that occur in the
training rows are levels. `(1 | g1/g2)` nests `g2` within `g1` and expands to
`(1 | g1) + (1 | g1:g2)`, one block per level of the hierarchy. Prediction
builds the combinations from the source columns of the fresh rows, so an
unseen combination is a new level.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import fit

rows = np.arange(36)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "school": np.array(["a", "b", "c"])[rows % 3],
        "class": np.array(["u", "v", "w"])[(rows // 3) % 3],
    }
)
data["y"] = 0.4 * data["x"] + 0.1 * (rows % 5)
labels = ["u", "v", "w"]
similarity = pd.DataFrame(
    [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]], index=labels, columns=labels
)
model = fit(
    "y ~ x + (1 | school/class) + (1 + x | gr(class, cov = K, cor = FALSE))",
    data=data,
    data2={"K": similarity},
    method="svi",
    num_steps=20,
    seed=1,
)
assert set(model.formula.group_levels) == {"school", "school:class", "class"}
fresh = pd.DataFrame({"x": [0.0, 1.0], "school": ["a", "d"], "class": ["u", "w"]})
draws = model.posterior_epred(
    new_data=fresh, num_samples=10, seed=2, allow_new_levels=True
)
assert draws.shape == (1, 10, 2)
```

A coefficient appears once per grouping factor. `(1 | g) + (x | g)` repeats
the intercept of `g` in two blocks and is refused, as brms refuses it with
"Duplicated group-level effects are not allowed".

## Multi-membership terms

A row can belong to several levels of one grouping factor at once, for
instance a pupil who attended two schools. `mm(g1, ..., gK)` declares such a
multi-membership factor [@browne2001multiple]. Its levels are the union of
the levels of the member columns `g1` to `gK`, and a group-level coefficient
$r$ enters row $n$ as

$$
\sum_{k=1}^{K} W_{kn} \, r_{J_{kn}} \, Z_{kn},
$$

where $J_{kn}$ is the level in column `gk`, $W_{kn}$ its weight, and
$Z_{kn}$ the row's covariate for that member. By default every member has
weight $1/K$. `weights = cbind(w1, ..., wK)` takes the weights from columns,
normalized to sum to one in each row unless `scale = FALSE`, which also
permits negative weights. A covariate `x` enters every member with the same
value. `mmc(x1, ..., xK)` gives each member its own covariate, as in
`(1 + mmc(x1, x2) | mm(g1, g2))`. `mm()` also takes `cor` and `id` with
their `gr()` meaning.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import fit

rows = np.arange(40)
data = pd.DataFrame(
    {
        "first": np.array(["a", "b", "c", "d"])[rows % 4],
        "second": np.array(["b", "c", "d", "e"])[(rows // 4) % 4],
        "years1": 1.0 + rows % 3,
        "years2": 1.0 + rows % 2,
        "x1": np.cos(rows),
        "x2": np.sin(rows),
    }
)
data["y"] = 0.3 * data["x1"] + 0.05 * (rows % 7)
model = fit(
    "y ~ x1 + (1 + mmc(x1, x2) | mm(first, second, weights = cbind(years1, years2)))",
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
)
assert model.formula.group_levels["mmfirstsecond"] == ("a", "b", "c", "d", "e")
fresh = data.iloc[:3].assign(second=["e", "z", "a"])
draws = model.posterior_epred(
    new_data=fresh, num_samples=10, seed=2, allow_new_levels=True
)
assert draws.shape == (1, 10, 3)
```

The factor is named by brms's label, `mm` followed by the member names
(`mmfirstsecond`), which `PriorSpec(class_="sd", group=...)` selects. The
coefficients use the ordinary Gaussian group layout, correlation blocks,
centered and noncentered coordinates, and new-level policies. Each member
gathers the shared coefficient vector at its own level, so scoring and
prediction cost O(n K) per coefficient for n rows, without a rows-by-levels
matrix. Fresh rows are coded on one prediction axis that covers every
member, and a level unseen in training is a new level for every member that
names it. Member columns may not be missing. Multi-membership factors take
scalar group coefficients only: ordinal category-specific coefficients,
grouped thresholds, and `cov` on a multi-membership factor are refused. A
categorical or other per-category response gives each free category its own
scalar multi-membership coefficients (see
[category-specific formulas](formula-mixtures.md#category-specific-formulas)).
