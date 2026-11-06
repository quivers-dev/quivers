# PyMC

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{PyMC}$.

PyMC first builds a graph of random variables and deterministic expressions;
a compiled log-density function then evaluates that graph at supplied values.
The renderer must preserve which variables are observed, which remain
unknown, and which coordinates belong to one joint event. Axis names help
identify dimensions, but the distribution's batch structure determines the
independence assumptions.

## Semantics

PyMC’s [linear-regression model specification](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#model-specification) shows how priors, deterministic expressions, and observed variables enter a model context. The [first QVR model](../../tutorials/qvr/01-first-model.md#qvr-vs-pymc-vs-numpyro) presents the same regression in QVR, PyMC, and NumPyro.

PyMC assembles a joint density from distributions and additional log-density
factors, using the graphical-model perspective [@koller2009probabilistic].
The implementation uses the
[`pymc.Model`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html)
context: each `pymc.<Distribution>(name, ..., dims=..., observed=...)`
constructor inside the `with pymc.Model()` scope registers a
random variable; the joint log-density is computed by
[`pymc.Model.compile_logp`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html#pymc.Model.compile_logp).
The reference is Salvatier, Wiecki, and Fonnesbeck
[@salvatier2016pymc3] for PyMC3 and the
[PyMC v5 documentation](https://www.pymc.io/projects/docs/en/stable/)
for the current API.

## Unconstrained-space change of variables

PyMC applies per-distribution unconstrained-space transforms
during inference (e.g.
[`pymc.distributions.transforms.log`](https://www.pymc.io/projects/docs/en/stable/api/distributions/transforms.html)
for `<lower=0>`-style constraints) and adds the Jacobian
automatically via the same change-of-variables principle as Stan.
The constrained-space density that `compile_logp` returns at a
constrained-space point is the QVR reference; no model-side
emit-time Jacobian is needed. $\Psi_{\mathsf{PyMC}} = \mathrm{id}$
at the constrained-space level.

## Family parameterizations

PyMC families use the
[`pymc.distributions.*`](https://www.pymc.io/projects/docs/en/stable/api/distributions.html)
classes with PyMC-specific keyword arguments. The QVR ↔ PyMC arg
mapping requires non-trivial aliases:

| QVR arg | PyMC arg |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `a` |
| `concentration1` (Beta) | `alpha` |
| `concentration0` (Beta) | `beta` |
| `probs` | `p` |
| `total_count` (Binomial) | `n` |
| `df` (StudentT) | `nu` |

These live in
`FAMILY_META[F].arg_aliases["pymc"]`
as the shared source of target argument names. The renamings are syntactic; the density
$f_{\mathsf{PyMC}}(v \mid \text{renamed args})$ equals
$f_{\mathrm{QVR}}(v \mid \text{original args})$ term-by-term, so
$c_{F, \mathsf{PyMC}} = 0$ for every family with native PyMC
support.

## Per-construct emit

**Sample / observe.** `<name> = pymc.<Family>(<name>, **args,
dims=(<plate axes>), observed=<obs>)` inside the
`with pymc.Model() as model:` scope. The `dims=` declaration
carries the batch-axis names; PyMC's `coords` (in the
`pymc.Model(coords=...)` constructor) declares each axis with its
size.

**Plate.** `dims=` names axes and `coords` supplies their labels or
sizes. Independence comes from the distribution's batch shape and
parameter broadcasting. For instance, a scalar Normal with batch size ten
represents ten independent scalar events, while an MvNormal with event size
ten represents one joint event whose covariance determines dependence.

**Marginalize.** Supported finite blocks are enumerated and their integrated
log density is added through a `pymc.Potential`. Grouped row contributions
are accumulated before the latent-axis reduction, eliminating the
marginalized coordinate.

**Score / let / return.**
[`pymc.Potential("name", expr)`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Potential.html)
adds a score directly to the joint log density;
deterministic assignment for let; the function returns the
`pymc.Model` instance.

## Acceptance

* **Tier 1 structural.** Every emit has a `def build_model(...)`
  with a `with pymc.Model(coords=...) as model:` body containing
  one `pymc.<Family>(...)` call per IR sample / observe step plus
  `return model`.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** Python's AST parser accepts the emitted source.
* **Tier 3 numeric equivalence.** `pymc.Model.compile_logp()` is
  compared with the QVR reference on the selected fixture grids.
