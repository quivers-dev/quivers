# Structured predictor terms

A penalized smooth has coupled coefficient coordinates. Its penalty identifies
which directions receive Gaussian regularization and which require ordinary
population priors. QVR retains the penalized directions as a
[`GaussianDesignTerm`](../api/formulas/formula.md), separately from
[`FixedColumn`](../api/formulas/formula.md) values, and emits them through the
existing formula compiler, inference methods, and prediction evaluator.

## Smooth terms

`s()` and `t2()` follow brms, which takes them from mgcv
[@burkner2017brms; @wood2017generalized]. brms constructs each smooth with
mgcv's `smoothCon`, absorbs the sum-to-zero
[identifiability](https://en.wikipedia.org/wiki/Identifiability) constraint,
reparameterizes the penalty to a diagonal, and splits the basis into
unpenalized columns and penalized columns, as in mgcv's mixed-model
representation [@wood2017generalized; @pedersen2019hierarchical]. QVR ports
that pipeline in NumPy. Its design matrices equal the `Xs` and `Zs` matrices
that brms passes to Stan, column by column up to sign.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import PriorSpec, fit

rows = np.arange(1, 41)
data = pd.DataFrame(
    {
        "x": (rows * 0.618034) % 1,
        "z": (rows * 0.414214) % 1,
        "g": np.array(["a", "b"])[rows % 2],
    }
)
data["y"] = np.sin(3 * data["x"]) + data["z"] ** 2
model = fit(
    "y ~ s(x) + t2(x, z, k=4) + s(z, by=g) + g",
    data=data,
    family="gaussian",
    method="svi",
    num_steps=20,
    seed=1,
    priors=PriorSpec(distribution="HalfNormal(0.7)", class_="sds"),
)
means = model.posterior_epred(
    new_data=pd.DataFrame({"x": [0.2, 0.8], "z": [0.5, 0.1], "g": ["a", "b"]}),
    num_samples=10,
    seed=3,
)
assert means.shape == (1, 10, 2)
```

### Bases

`s(x)` uses mgcv's default
[thin-plate regression spline](https://en.wikipedia.org/wiki/Thin_plate_spline)
[@wood2003thin]: the leading eigenvectors of the full thin-plate spline at the
unique covariate values, constrained orthogonal to the polynomials of the
penalty null space, followed by those polynomials. `s(x, z)` is the
isotropic multivariate version. `bs=` selects the other bases brms users rely
on.

| `bs` | Basis | Default `k` | Penalty null space |
|---|---|---|---|
| `tp` | thin-plate regression spline | null dimension plus 8, 27, or 100 | polynomials of degree below `m` |
| `ts` | `tp` with shrunk null space | as `tp` | none |
| `cr` | cubic regression spline, knots at quantiles | 10 | linear functions |
| `cs` | `cr` with shrunk null space | 10 | none |
| `cc` | cyclic cubic regression spline | 10 | constants |
| `ps` | P-spline, coefficient differences | 10 | polynomials of degree below `m[2]` |
| `bs` | B-spline, integrated squared derivative | 10 | polynomials of degree below `m[2]` |
| `re` | random effect, identity penalty | number of levels | none |
| `gp` | low-rank Gaussian process smooth | dimension plus 11, 31, or 101 | constant and linear functions, or the constant alone when stationary |
| `ds` | Duchon spline | null dimension plus 10, 30, or 100 | polynomials of degree below `m[1]` |
| `mrf` | Markov random field over the levels of a factor, penalty from `xt` | number of regions | null space of the penalty (the constant for a connected neighbourhood graph) |
| `sos` | spline on the sphere over latitude and longitude | 50 | constants, and for `m = -1` also the three coordinates of the embedded sphere |
| `fs` | one smooth per level of a factor, thin-plate unless `xt` names another base | as the base | none: every null column is penalized |

`k` sets the basis dimension and `m` the penalty order with mgcv's meaning,
including `m = c(m1, m2)` for `ps` and `bs` and `m = c(m, 0)` for a
thin-plate spline without its null space. `fx = TRUE` keeps the basis as
unpenalized population columns. A thin-plate smooth over more than 2000
unique covariate combinations is refused, because mgcv then subsamples knots
with R's random number generator. Give `knots` for such a term, or use
`bs = 'cr'` or `bs = 'ps'`.

`s(x, bs = 'gp')` is mgcv's Gaussian process smooth
[@kammann2003geoadditive; @wood2017generalized], a spline with a fixed
correlation function and not the `gp()` term, whose hyperparameters are
sampled. Its knots are the unique covariate rows, and its columns are the
correlations between rows and knots reduced to the leading eigenvectors of
the knot correlation matrix. `m = c(type, range, power)` selects the
correlation function with mgcv's meaning: type 1 is spherical, 2 the power
exponential with the given power in (0, 2], and 3, 4, and 5 are Matérn with
smoothness 1.5 (the default), 2.5, and 3.5. The range defaults to the largest
distance between knots. A negative type makes the smooth stationary, with
only the constant unpenalized. More than 2000 unique covariate rows are
refused, as for a thin-plate smooth, and so are user knots.

<!-- python: run -->
```python
matern = fit(
    "y ~ s(x, bs = 'gp', k = 6, m = c(-4, 0.5))",
    data=data,
    method="svi",
    num_steps=50,
    seed=3,
)
assert matern.posterior_linpred(new_data=data.head(3), num_samples=5).shape == (
    1,
    5,
    3,
)
```

`s(x, g, bs = 'fs')` is mgcv's factor-smooth interaction
[@wood2017generalized]: one copy of a thin-plate smooth of `x` for every
level of the factor `g`, in the manner of a random curve per level. The base
penalty is made diagonal, and the copies share one smoothing standard
deviation for the penalized range space and one more for each polynomial of
the null space, so the term has no unpenalized columns and no centering
constraint. Exactly one covariate is a factor, and `k` and `m` belong to the
base. A fresh row whose factor level was not fitted is refused, as in brms.

`xt` names a base other than the thin plate, as `xt = 'cr'` or
`xt = list(bs = 'cr')`. The bases `tp`, `cr`, `cc`, `ps`, `bs`, and `gp` are
accepted: each has one penalty whose null space is the polynomials below its
penalty order. The shrinkage bases `ts` and `cs` have a penalty of full rank,
for which mgcv's construction fails, and are refused. `fx = TRUE` is refused
as well, because brms fails on an unpenalized `fs` term.

A `by` variable multiplies the term as it does for every other smooth: a
numeric `by` scales each row, and a factor `by` gives every level its own
copy of the term with its own smoothing standard deviations.

<!-- python: run -->
```python
curves = pd.DataFrame({"x": np.tile(np.linspace(0.0, 1.0, 15), 4)})
curves["subject"] = np.repeat(["a", "b", "c", "d"], 15)
curves["y"] = np.sin(3 * curves["x"]) + np.repeat([0.0, 0.3, -0.2, 0.5], 15)

by_subject = fit(
    "y ~ s(x, subject, bs = 'fs', k = 5, m = 1)",
    data=curves,
    method="svi",
    num_steps=50,
    seed=3,
)
scales = [name for name in by_subject.draws() if name.endswith("_scale")]
assert len(scales) == 2
fresh_curves = pd.DataFrame({"x": [0.25, 0.75], "subject": ["b", "d"]})
assert by_subject.posterior_linpred(new_data=fresh_curves, num_samples=5).shape == (
    1,
    5,
    2,
)

curves["task"] = np.tile(np.repeat(["easy", "hard"], [8, 7]), 4)
by_task = fit(
    "y ~ s(x, subject, bs = 'fs', k = 5, xt = list(bs = 'cr'), by = task)",
    data=curves,
    method="svi",
    num_steps=50,
    seed=3,
)
assert "sds_sxsubjecttaskeasy_1" in by_task.variables(num_samples=5)
```

`s(region, bs = 'mrf', xt = list(penalty = P))` is mgcv's
[Markov random field](https://en.wikipedia.org/wiki/Markov_random_field)
smooth [@rue2005gaussian; @wood2017generalized]: one effect per level of the
factor `region`, with the Gaussian prior whose precision is proportional to
the penalty matrix `P`. `xt = list(nb = A)` gives a neighbourhood instead,
from which mgcv builds the penalty: the diagonal entry of a region is its
number of listed neighbours and the entry of each neighbour is minus one,
the [graph Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix).

brms evaluates `xt` in R's global environment. In QVR, `P` and `A` are
entries of `data2` named in `xt`. A penalty is a data frame labeled by region
on both axes, in any order, or an unlabeled array in the order of the
factor's levels. A neighbourhood is such a matrix whose row for a region is
nonzero at the neighbours listed for it. mgcv's boundary polygons (`polys`)
are not read.

`k` below the number of regions keeps the `k` smoothest coordinates of
mgcv's reparameterization. A region without observations keeps its effect
when its factor level is kept, with `drop_unused_levels=False` on a
categorical column, and fresh rows can then predict it. A fresh row whose
region is not in the fitted structure is refused, as in brms.

<!-- python: run -->
```python
regions = ["north", "east", "south", "west", "centre"]
adjacent = pd.DataFrame(0.0, index=regions, columns=regions)
for a, b in [
    ("north", "east"),
    ("east", "south"),
    ("south", "west"),
    ("west", "north"),
    ("centre", "north"),
    ("centre", "south"),
]:
    adjacent.loc[a, b] = adjacent.loc[b, a] = 1.0
areas = pd.DataFrame({"region": np.tile(regions, 6)})
areas["y"] = np.tile([0.4, 0.1, -0.3, 0.0, 0.2], 6) + np.linspace(-0.2, 0.2, 30)

spatial = fit(
    "y ~ s(region, bs = 'mrf', xt = list(nb = adjacent))",
    data=areas,
    data2={"adjacent": adjacent},
    method="svi",
    num_steps=50,
    seed=3,
)
assert "sds_sregion_1" in spatial.variables(num_samples=5)
assert spatial.posterior_linpred(new_data=areas.head(5), num_samples=5).shape == (
    1,
    5,
    5,
)
```

`s(x, z, bs = 'ds')` is mgcv's Duchon spline [@duchon1977splines;
@wood2017generalized], which generalizes the thin-plate spline. Its penalty
integrates the squared derivatives of order `m[1]` after weighting their
[Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) by the
frequency norm to the power `2 * m[2]`. `m = c(m, s)` thus gives the
derivative order `m` and the weight exponent `s`, and the default
`m = c(2, 0)` is the thin-plate penalty. In `d` dimensions `s` is a multiple
of one half with `-d/2 < s < d/2`, and `m + s > d/2` makes the smooth
continuous. The kernel is the distance to the power `2m + 2s - d`, times its
logarithm when that power is even. The null space holds the polynomials of
degree below `m`. For instance, `m = c(1, 0.5)` over two covariates
penalizes first derivatives and leaves only the constant unpenalized, where
a thin-plate spline over two covariates must leave the linear functions
unpenalized as well.

The knots of a Duchon spline are the unique covariate rows, or the `knots`
described below. With more knots than `k`, the columns are the kernel
between rows and knots reduced to the `k` leading eigenvectors of the knot
kernel matrix. Otherwise there is one kernel column per knot. `by`, `fx`,
and use as a `t2()` margin work as for the other bases.

<!-- python: run -->
```python
site = np.arange(40)
plane = pd.DataFrame(
    {"x": (site * 0.618034) % 1, "z": 2 * ((site * 0.414214) % 1) - 0.5}
)
plane["y"] = np.sin(3 * plane["x"]) + plane["z"] ** 2

duchon = fit(
    "y ~ s(x, z, bs = 'ds', k = 10, m = c(1, 0.5))",
    data=plane,
    method="svi",
    num_steps=50,
    seed=3,
)
names = duchon.variables(num_samples=5)
assert "sds_sxz_1" in names and not any(name.startswith("bs_") for name in names)
corners = pd.DataFrame({"x": [0.0, 1.0], "z": [-0.5, 1.5]})
assert duchon.posterior_linpred(new_data=corners, num_samples=5).shape == (1, 5, 2)
```

`s(lat, lon, bs = 'sos')` is mgcv's spline on the
[sphere](https://en.wikipedia.org/wiki/Sphere)
[@wahba1981spline; @wood2017generalized]. Its first covariate is latitude and
its second is longitude, both in degrees. The kernel of the basis depends on
two locations only through the
[great-circle distance](https://en.wikipedia.org/wiki/Great-circle_distance)
between them, so the smooth is continuous across the 180th meridian and at
the poles. Its knots are the unique locations, and its columns are the kernel
between rows and knots reduced to the `k` leading eigenvectors of the knot
kernel matrix, as for a thin-plate regression spline.

`m` selects the penalty with mgcv's meaning. `m = 0`, the default, is the
second-order spline on the sphere, whose penalty integrates the squared
[Laplace-Beltrami operator](https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator)
of the function. `m` from 1 to 4 is a pseudospline, whose kernel has a closed
form and whose penalty is equivalent to one of order `(m + 2) / 2`
[@wahba1981spline]. `m = -1` and `m = -2` embed the sphere in three
dimensions and use the kernel of a spline that penalizes second or first
derivatives there [@duchon1977splines]. As in mgcv, `m` is rounded, a value
above 4 is read as 4, and a value below -2 is read as -1.

`knots` with one entry for each of the two covariates gives the knot
locations. `by` and `fx` work as for other bases. A latitude outside
[-90, 90] or a longitude outside [-180, 360] is refused, in the training
data, the knots, and fresh rows. More than 2000 unique locations without
`knots` are refused, as for a thin-plate smooth.

<!-- python: run -->
```python
station = np.arange(1, 41)
globe = pd.DataFrame(
    {
        "lat": np.degrees(np.arcsin(2 * ((station * 0.618034) % 1) - 1)),
        "lon": ((station * 0.414214) % 1) * 360 - 180,
    }
)
globe["y"] = np.sin(np.radians(globe["lon"])) * np.cos(np.radians(globe["lat"]))

sphere = fit(
    "y ~ s(lat, lon, bs = 'sos', k = 8, m = 2)",
    data=globe,
    method="svi",
    num_steps=50,
    seed=3,
)
assert "sds_slatlon_1" in sphere.variables(num_samples=5)
poles = pd.DataFrame({"lat": [90.0, -90.0, 0.0], "lon": [0.0, 0.0, 180.0]})
assert sphere.posterior_linpred(new_data=poles, num_samples=5).shape == (1, 5, 3)
```

The adaptive (`ad`) and soap film (`so`) bases have several overlapping
penalties that mgcv cannot write as random effects, so brms refuses them and
so does QVR.

mgcv's `sp` argument, which fixes smoothing parameters in `gam`, is accepted
and has no effect, as in brms: brms estimates the smoothing standard
deviations, and its Stan program and data are the same with and without
`sp`. The argument remains part of the term's label.

`fit(..., knots={"x": [...]})` is brms's `knots` argument: knot values keyed
by covariate, which mgcv's constructors read as follows. `cr` and `cs` take
exactly `k` knots in place of the quantiles. `cc` takes `k` knots, or two end
points between which the knots are spread through the data. `ps` and `bs`
take two end points that replace the data range (and must include the data),
or the full knot sequence, `k + m[1] + 2` knots for `ps` and `k + m[1] + 1`
for `bs`. `bs` also takes four knots, the outer two bounding the evaluation
range and the inner two the interior knots. Knots must increase strictly. A
`re` basis reads none, as in mgcv. An `sos` basis takes the knot latitudes
and longitudes, one entry per covariate and of equal length.

A `tp`, `ts`, or `ds` basis takes knot locations in place of the unique
covariate rows, with one entry of equal length for each covariate of the
smooth. At least `k` knots are needed. With exactly `k` distinct knots the
basis has one function per knot, and with more it is the eigenbasis of the
knots, so knots also make a smooth over more than 2000 unique rows
available. Each margin of a `t2()` term reads the knots of its own
covariates in the same way. Knots of a `gp`, `fs`, or `mrf` basis are
refused with an error. The
basis keeps its knots, so prediction, refits, and artifacts use them, and
joint formulas take `knots` once for every response, as `brm` does.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import fit

rows = np.arange(40)
data = pd.DataFrame({"x": (rows * 0.618034) % 1})
data["y"] = np.sin(3 * data["x"]) + 0.1 * np.cos(7 * rows)

knotted = fit(
    "y ~ s(x, bs = 'cr', k = 5)", data=data, method="svi", num_steps=50,
    knots={"x": [0.0, 0.2, 0.5, 0.8, 1.0]}, seed=1,
)
grid = pd.DataFrame({"x": [0.1, 0.5, 0.9]})
assert knotted.posterior_epred(new_data=grid, num_samples=4).shape == (1, 4, 3)
```

### By variables, identifiers, and tensor products

`s(x, by = w)` multiplies the smooth by the numeric column `w`. A non-constant
`w` removes the sum-to-zero constraint, as in mgcv. `s(x, by = g)` with a
factor `g` builds one smooth per level, each with its own smoothing scale. As
in brms, it does not add the main effect of `g`, so the formula above adds `g`
itself. An ordered factor builds no smooth for its first level.

`t2(x, z)` is mgcv's tensor product with separate penalties
[@wood2013straightforward]. Each margin is reparameterized so that its
penalty is the identity on its range. Products of marginal range and null
columns then form blocks, each with its own scale. `k`, `bs`, `m`, and `d`
take one value per margin, `full` and `ord` keep mgcv's meaning, and `by`
works as for `s()`. `te()` and `ti()` raise the brms error naming `t2`.

`id` follows mgcv: smooths sharing an `id` share one smoothing scale, and they
must use the same basis, dimension, and order. brms ignores `id`, giving each
smooth its own scale, so this is a deliberate difference.

### Priors and parameterization

The unpenalized columns become ordinary population coefficients of class `b`,
named after the term (`s(x)_1`). Each penalized block multiplies independent
standard-Normal innovations by a positive scale of class `sds`, with the
authored term as its `coef`. Its default is the configured
`random_scale_prior` (`HalfNormal(1.0)` unless changed), and a typed
`PriorSpec(class_='sds')` replaces it through the common prior catalogue, as
`get_prior` lists. This default differs from brms on purpose. brms gives
`sds`, like `sd`, a half Student-t prior with three degrees of freedom and
scale $\max(2.5, s)$, where $s$ is the median absolute deviation of the
response on the link scale rounded to one decimal, for identity, log,
inverse, and square-root links. QVR uses one configured prior for every
group and smoothing scale, so `sds` and `sd` stay on the same default, and
that default does not depend on the response values. The brms default is
available as `PriorSpec(distribution="HalfStudentT(3.0, s)", class_="sds")`
with $s$ computed as above.

Smooths, monotonic effects, and Gaussian processes enter every predictor:
location, distributional, and nonlinear-parameter formulas, and the
locations of mixture components. A categorical or other component-valued
response keeps a separate block and scale per free response component. In
an ordinal model, `cs(s(x))`, `cs(mo(x))`, and `cs(gp(x))` give one effect
per transition between adjacent categories. An `offset()` term combines
with each of these terms.

### Prediction

Prediction evaluates fresh rows through the retained basis: the thin-plate
eigenvector map and unique training covariates, the cubic and B-spline knots,
the absorbed constraint, and every reparameterization. Nothing is refitted.
Evaluating a thin-plate basis at `n` fresh rows costs O(n u d) for `u` unique
training covariate rows in `d` dimensions. The other bases cost O(n k).
`cr` and `ps` extrapolate linearly beyond the training range, and `cc` wraps
values into its period.

### Where QVR differs from brms

Several choices of mgcv and brms are not reproducible or not consistent, so
QVR fixes them and says so here.

1. mgcv takes some coordinates from the orthonormal basis that the
   [LAPACK](https://en.wikipedia.org/wiki/LAPACK) eigensolver returns for a
   repeated zero eigenvalue of a penalty. Where that basis is determined, for
   a null space spanned by coordinate axes or for the two-dimensional
   marginal null spaces of `t2`, QVR returns it, and the matrices equal
   brms's. Elsewhere it depends on rounding, for instance the unpenalized
   columns of `s(x, m = 3)` or `s(x, z, v)`. QVR then uses the principal axes
   of the null-space design columns. These columns span the same space as
   brms's, so the model agrees under brms's default flat prior on them.
   `bs = 'cs'` shrinks its two null directions by different factors in an
   order that likewise depends on rounding in mgcv, with or without user
   knots.
   An `fs` smooth is affected more deeply. mgcv reads its null-space columns
   from the same kind of rounding-dependent basis, and each column has its
   own smoothing standard deviation, so the rotation changes the prior and
   not only the coordinates. QVR orthonormalizes the null-space polynomials
   in increasing degree (the constant, then the linear function, and so on).
   The function space and the penalized range columns equal mgcv's. The
   null-space columns equal mgcv's after one rotation, and every block is
   scaled by one common factor, because mgcv's penalty scaling reads the row
   sums of the rotated columns. With `m = 1` the null space is the constant
   alone and the design equals brms's up to column signs.
   brms's Stan data also ignore the `by` variable of an `fs` smooth: mgcv
   multiplies the smooth's own design by `by`, which is what `mgcv::gam`
   fits, and the conversion to random effects that brms calls reads an
   unmultiplied copy. A factor `by` thus gives brms identical blocks for
   every level. QVR applies `by` as mgcv's own design does.
2. brms predicts a `t2` smooth on fresh rows through mgcv's prediction
   constraint, which differs from the constraint absorbed in its training
   design, so brms's fresh-row design does not reproduce its own training
   rows. QVR uses the training constraint for both.
3. `id` shares smoothing scales, as described above.
4. A low-rank `mrf` basis keeps the eigenvectors of the smallest eigenvalues
   of the rescaled penalty. When the cut falls between two equal eigenvalues,
   as for `k = 4` on a ring of six regions, the kept space depends on
   rounding in mgcv, and QVR refuses that `k`. The `xt` objects of an `mrf`
   smooth come from `data2`, where brms reads R's global environment.
5. An `sos` smooth checks that its covariates are a latitude and a longitude
   in degrees. mgcv evaluates the kernel at any pair of numbers. Knots given
   for one of its two covariates alone, and more knots than rows, are
   refused, where mgcv ignores them. Its `xt` entries `max.knots` and `seed`
   are refused, because they select a random knot subsample drawn with R's
   generator. With `m = -1`, knots that lie on one circle do not identify
   the null space and are refused.
6. mgcv replaces an `m[2]` of a `ds` smooth that violates Duchon's
   conditions and warns. QVR refuses it and names a valid value. Thus
   `s(x, z, bs = 'ds', m = 1)`, which mgcv fits as `m = c(1, 0.5)`, must be
   written with both values. Knots of a `tp`, `ts`, or `ds` smooth that
   mgcv ignores (fewer knots than `k`, more knots than rows, or knots for
   some covariates of the smooth alone) are refused, and so are the `xt`
   entries `max.knots` and `seed` of a `ds` smooth.

## Monotonic effects

`mo(x)` models the effect of an ordinal predictor as monotonic without
assuming equal distances between its categories [@burkner2020modelling].
`x` must be an integer column or an ordered
[categorical](https://pandas.pydata.org/docs/user_guide/categorical.html)
column. With categories coded $0, \ldots, D$, the term contributes

$$
b \, D \sum_{i=1}^{x_n} \zeta_i
$$

to row $n$. The coefficient $b$ is the expected average difference between
adjacent categories. The vector $\zeta$ lies on the
[probability simplex](https://en.wikipedia.org/wiki/Simplex#Probability)
and gives the share of the total effect $bD$ that falls between each pair of
adjacent categories. Its default prior is the flat
[Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
$\operatorname{Dirichlet}(1, \ldots, 1)$, as in brms.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import PriorSpec, fit, get_prior

rows = np.arange(36)
data = pd.DataFrame(
    {
        "income": pd.Categorical(
            np.array(["low", "middle", "high", "top"])[rows % 4],
            categories=["low", "middle", "high", "top"],
            ordered=True,
        ),
        "age": 20 + rows % 5,
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = np.array([0.0, 1.0, 1.4, 1.5])[rows % 4] + 0.1 * (rows % 5)
assert "simo_moincome1" in get_prior("y ~ mo(income)", data=data)
model = fit(
    "y ~ mo(income) + mo(age) + (1 + mo(income) | g)",
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
    priors=PriorSpec(distribution="Dirichlet(2.0)", class_="simo", coef="moage1"),
)
grid = pd.DataFrame(
    {
        "income": pd.Categorical(["low", "top"], ordered=True),
        "age": [20, 24],
        "g": ["a", "b"],
    }
)
means = model.posterior_epred(new_data=grid, num_samples=10, seed=3)
assert means.shape == (1, 10, 2)
```

An integer predictor is coded by its distance from the training minimum, and
an ordered factor by the position of its level among the levels observed in
training. Both follow brms, which drops unused factor levels before coding.
The predictor needs at least two categories. A column of other values, a
missing value, or an unordered factor is refused with brms's error message.

### Interactions, identifiers, and group-level slopes

A term with `mo()` factors is a brms special-effects term. `mo(x) * z`
expands to `mo(x)`, `z`, and `mo(x):z`, and `mo(x):z` contributes
$b \cdot \operatorname{mo}(x) \cdot z$ with its own coefficient and its own
simplex. When the other components of a term evaluate to several columns,
for instance a factor in `mo(x):f`, each column becomes a separate term with
its own coefficient and simplex, as brms does. `mo(x1):mo(x2)` multiplies two
monotonic factors, each with its own simplex.

`mo(x, id = "i")` shares one simplex among every `mo()` factor with the same
`id` in a formula, which keeps predictions conditionally monotonic across an
interaction [@burkner2020modelling]. Factors sharing an `id` need the same
number of categories.

A group-level slope `(mo(x) | g)` multiplies its group coefficient by the
population term's value, so row $n$ of group $j$ receives
$(b + r_j) \, D \sum_{i \le x_n} \zeta_i$ with the population simplex. The
same term must appear among the population effects. brms reads a group-level
`mo()` term the same way, but it silently drops the slope when the population
term is absent, and QVR refuses that formula instead. Group-level `mo()`
slopes use the ordinary correlation blocks, double bars, shared identifiers,
and new-level policies.

### Priors and prediction

The coefficient $b$ has class `b`, with the authored term as its `coef`
(`mo(income)`), and the site `bsp_<term>`. Its default is the configured
`fixed_prior` scaled by the root mean square of the category codes (times the
covariate of an interaction), the column the term equals when the simplex is
uniform. Each simplex has class `simo` and the site `simo_<label>`. Its
`coef` is brms's label, the term label without parentheses and spaces
followed by the factor's position in the term (`moincome1`), or the `id` of a
shared simplex. A `simo` prior must be a Dirichlet distribution. A scalar
concentration, such as `Dirichlet(2.0)`, is recycled to the simplex
dimension, as brms recycles `dirichlet(2)`.

Fresh rows are coded with the training minimum or the training levels.
Values outside the training range, values that are not whole numbers, and
unseen levels are refused, as brms's `validate_newdata` refuses them. Scoring
and prediction cost O(n D) per factor for n rows, through the product of the
retained row-by-step matrix of cumulative indicators with the simplex.

brms labels the coefficient of `mo(income)` as `moincome`. QVR keeps the
authored term label for class `b` and `sd` selectors, as for every other
population and group coefficient, and uses brms's labels for class `simo`.

## Gaussian process terms

`gp(x)` adds a function $f(x)$ to a linear predictor and gives $f$ a
[Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior with
mean zero and a stationary covariance kernel [@rasmussen2006gp]. Without `k`
(or with `k = NA`) the GP is exact, as in brms (see
[Exact Gaussian processes](#exact-gaussian-processes)). With `k = m`, QVR fits
brms's Hilbert-space approximation of that prior
[@solin2020hilbert; @riutort2023practical]. The kernel is expanded in the
eigenfunctions of the
[Laplace operator](https://en.wikipedia.org/wiki/Laplace_operator) on a box
$[-L, L]$ that contains the centered covariates:

$$
f(x) \approx \sum_{j=1}^{m} \sqrt{S\big(\sqrt{\lambda_j}\big)}\,
\phi_j(x)\, z_j,
\qquad z_j \sim \mathcal N(0, 1),
$$

where $\phi_j(x) = L^{-1/2} \sin\big(\sqrt{\lambda_j}\,(x + L)\big)$,
$\sqrt{\lambda_j} = j\pi / (2L)$, and $S$ is the kernel's
[spectral density](https://en.wikipedia.org/wiki/Spectral_density). The
basis $\phi_j(x_n)$ depends only on the data, so the term is a Gaussian design
block like a smooth, and its scales $\sqrt{S}$ are expressions of the sampled
marginal standard deviation `sdgp` and length scale `lscale`.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import PriorSpec, conditional_smooths, fit, get_prior

rows = np.arange(1, 41)
data = pd.DataFrame(
    {
        "x": (rows * 0.618034) % 1,
        "z": (rows * 0.414214) % 1,
        "g": np.array(["a", "b"])[rows % 2],
    }
)
data["y"] = np.sin(6 * data["x"]) + data["z"]
formula = "y ~ gp(x, by = g, k = 8) + gp(x, z, k = 4, cov = 'matern32', iso = FALSE)"
priors = get_prior(formula, data=data)
assert priors["lscale_gpxga"].startswith("InverseGamma(")
model = fit(
    formula,
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
    priors=PriorSpec(distribution="HalfNormal(1.0)", class_="sdgp"),
)
fresh = pd.DataFrame({"x": [0.1, 0.5, 0.9], "z": [0.2, 0.4, 0.8], "g": ["a", "b", "a"]})
means = model.posterior_epred(new_data=fresh, num_samples=10, seed=3)
effect = conditional_smooths(model, "gp(x, by=g, k=8)", fresh, num_samples=10, seed=3)
assert means.shape == effect.shape == (1, 10, 3)
```

### Exact Gaussian processes

Without `k`, the term evaluates $f$ at the $u$ distinct scaled inputs
$x_1, \dots, x_u$ of its rows in brms's non-centered form:

$$
f = \operatorname{chol}\big(K + 10^{-12} I\big)\, z,
\qquad K_{ij} = k(x_i, x_j;\ \sigma, \ell),
\qquad z \sim \mathcal N(0, I_u),
$$

where $\operatorname{chol}$ is the lower
[Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition),
written in the compiled program with the QVR builtin `cholesky`, and the
diagonal $10^{-12}$ is the one brms's Stan functions add. Each row adds $f$
at its input times its `by` value. The kernels are brms's: with
$r = \lVert x - x' \rVert$,
$k_\infty = \sigma^2 \exp(-r^2 / 2\ell^2)$,
$k_{1/2} = \sigma^2 \exp(-r/\ell)$,
$k_{3/2} = \sigma^2 (1 + \sqrt3\, r/\ell) \exp(-\sqrt3\, r/\ell)$, and
$k_{5/2} = \sigma^2 (1 + \sqrt5\, r/\ell + 5r^2/3\ell^2)
\exp(-\sqrt5\, r/\ell)$. With `iso = FALSE`, $k$ is $\sigma^2$ times the
product of the one-dimensional kernels of each covariate with its own
$\ell_d$. `sdgp`, `lscale`, their labels, and their default priors are those
of the approximation, and the innovations are the site `gp_<label>_z`
(brms's `zgp`).

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import conditional_smooths, fit

rows = np.arange(1, 13)
data = pd.DataFrame({"x": np.round((rows * 0.618034) % 1, 1)})
data["y"] = np.sin(6 * data["x"])
model = fit("y ~ gp(x, cov = 'matern52')", data=data, method="svi", num_steps=20, seed=1)
fresh = pd.DataFrame({"x": [0.05, 0.4, 0.95]})
draws = model.posterior_linpred(new_data=fresh, num_samples=10, seed=3)
effect = conditional_smooths(model, "gp(x, cov='matern52')", fresh, num_samples=10, seed=3)
assert draws.shape == effect.shape == (1, 10, 3)
```

Evaluating the density costs O(u^3) for the factor and O(n u) for the rows.
Prediction follows brms's `.predictor_gp_new`. Given each posterior draw of
$f$ at the training inputs $X$ and of $\sigma$ and $\ell$, $f$ at the $m$
distinct fresh inputs $X_*$ is drawn jointly from its conditional Gaussian
law, with mean $K_*^\top K^{-1} f$ and covariance
$K_{**} - K_*^\top K^{-1} K_* + 10^{-8} I$, where $K_* = k(X, X_*)$ and the
$10^{-8}$ diagonal is brms's prediction nugget. That costs
O(u^3 + u^2 m + u m^2 + m^3) per draw, and nothing is refitted. A fresh
input equal to a training input reproduces its fitted value up to the nugget's
standard deviation of $10^{-4}$. Training rows predicted without `new_data`
replay $f$ exactly. A factorization that fails at a prediction raises brms's
error that the covariance matrix is not positive definite. brms sets no limit
on $u$, and neither does QVR.

### Arguments

Every argument of brms's `gp()` is accepted.

- `k` is the number of basis functions per covariate, so a term with $D$
  covariates has $m = k^D$ coordinates, ordered as R's `expand.grid` orders
  them. Without `k`, or with `k = NA`, the GP is exact.
- `c` is the boundary factor, one value or one per covariate, with default
  $5/4$. As in brms, $L_d = c_d \max(1, r)$, where $r$ is the range of all
  centered (and scaled) covariate values together. The exact GP ignores
  `c`, as brms does.
- `cov` names the kernel: `exp_quad` (the default), `matern32`, `matern52`,
  or `exponential` (alias `matern12`).
- `iso = FALSE` gives each covariate of a multivariate term its own length
  scale. With one covariate it has no effect.
- `scale = TRUE` (the default) divides the covariates by their largest
  pairwise distance, so the largest distance is one. The divisor and the
  training means of the scaled covariates are retained, and fresh rows are
  scaled and centered with them.
- `gr = TRUE` (the default) computes the centering means over the distinct
  covariate rows, as brms's grouping of repeated inputs does. With
  `gr = FALSE` the means are taken over every row. The approximation has
  one coordinate per basis function either way, so `gr` changes nothing
  else. For the exact GP, `gr = TRUE` gives rows with equal covariates one
  latent value, in order of first appearance (brms's `Jgp`), and
  `gr = FALSE` gives every row its own, so repeated inputs share only the
  kernel's correlation.
- `by` multiplies the function by a numeric variable, or fits one GP per
  level of a factor. With `cmc = FALSE` a factor gives one GP on every row
  plus one per treatment contrast. Each of these GPs has its own scaling,
  centering, boundary, `sdgp`, and `lscale`.

Several `gp()` terms of one covariate, such as
`gp(x, k = 4) + gp(x, k = 3)`, are separate processes, each with its own
`sdgp` and `lscale`. As in brms, the later terms take repaired labels
(`sdgp_gpx__1`, `lscale_gpx__1`).

R's `TRUE`, `FALSE`, and `NA` are recognized in formulas, so brms formulas
can be written unchanged.

### Spectral densities and priors

For $D$ covariates with length scales $\ell_d$ (all equal when `iso = TRUE`)
and frequency $\omega$, the densities are brms's:

$$
S_\infty(\omega) = \sigma^2 (2\pi)^{D/2} \prod_d \ell_d
\exp\Big(-\tfrac12 \sum_d \ell_d^2 \omega_d^2\Big),
\qquad
S_\nu(\omega) = \sigma^2 C_{\nu,D} \prod_d \ell_d
\Big(2\nu + \sum_d \ell_d^2 \omega_d^2\Big)^{-(\nu + D/2)},
$$

with $C_{\nu,D} = 2^D \pi^{D/2}\, \Gamma(\nu + D/2)\,(2\nu)^\nu / \Gamma(\nu)$
for the [Matérn kernels](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
with $\nu = 1/2, 3/2, 5/2$ [@rasmussen2006gp]. The compiled program computes
$\exp(\tfrac12 \log S)$ in place of $\sqrt S$, which is equal and keeps
finite gradients where $S$ underflows.

`sdgp` has class `sdgp` and `lscale` class `lscale`, with brms's labels as
`coef`: `gp` followed by the covariate names, the `by` variable, and the
level or contrast (`gpxga`), and for `iso = FALSE` the covariate name
(`gpxzx`). The sites are `sdgp_<label>` and `lscale_<label>`. Both are
sampled, never fixed. Their defaults are brms's. `sdgp` gets a half Student-t
prior with three degrees of freedom and scale $\max(2.5, s)$, where $s$ is the
median absolute deviation of the response on the link scale rounded to one
decimal, for identity, log, inverse, square-root, and inverse-square links.
`lscale` gets an
[inverse-gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
prior that puts one percent of its mass below the smallest positive distance
between training covariate rows and one percent above the largest, solved
numerically and rounded to six decimals as brms does. If that solve fails,
the default is a half-Normal prior with scale $0.5$, as in brms.

### Prediction and cost

For the approximation, scoring costs O(n m) for n rows and m basis functions. Prediction at r fresh
rows evaluates the retained basis at them, which costs O(r m D), and the
innovations, `sdgp`, and `lscale` of each draw give their coefficients.
Nothing is refitted. Fresh rows of a factor `by` must use training levels.
`conditional_smooths` returns one GP term's contribution at explicit rows.

### Where QVR differs from brms

1. For `iso = FALSE` with a Matérn kernel, brms's approximation uses the
   density above with per-covariate length scales, while its exact GP
   multiplies one-dimensional kernels. QVR follows brms in both.
2. For a factor `by`, brms's R function `predictor_gp` assigns each sub-GP's
   values to its rows rather than adding them, so with `cmc = FALSE` its
   predictions keep only the last contrast on rows that several contrasts
   share. QVR adds every sub-GP, as brms's Stan model does when fitting.
3. brms's prediction methods accept a larger nugget `nug` for new inputs.
   QVR uses brms's default $10^{-8}$ and has no such argument.
4. A matrix factorization reads every entry of its matrix at once, which the
   entrywise transpile lowering cannot express, so a program with an exact
   GP is refused by every transpile target.
5. A categorical or other per-category response gives every free category
   its own `gp()` hyperparameters and coordinates, as brms does (see
   [category-specific formulas](formula-mixtures.md#category-specific-formulas)).
