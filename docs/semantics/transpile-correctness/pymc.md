# PyMC

A PyMC model is a graph of random variables and deterministic expressions, built inside a
`with pymc.Model()` context, from which PyMC compiles functions that evaluate the joint log density
[@salvatier2016pymc3]. The renderer must keep which variables are observed, which are latent, and which
coordinates belong to one joint event ([Transpilation correctness](index.md)). PyMC's documentation is at
[pymc.io](https://www.pymc.io/projects/docs/en/stable/).

## How PyMC evaluates a log density

Each constructor `pymc.<Distribution>(name, ..., dims=..., observed=...)` in the model's context adds a
random variable, and [`pymc.Potential`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Potential.html)
adds a term to the log density. The joint log density is the sum of the log densities of the random
variables and the potentials. The test probe sums
[`pymc.logp`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.logp.html) over every free and
observed variable, at the free variables' constrained values and the observed data.

## Change of variables

PyMC samples a constrained variable in unconstrained coordinates, through a transform such as
[`log`](https://www.pymc.io/projects/docs/en/stable/api/distributions/transforms.html) for a positive
variable, and [`Model.compile_logp`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html#pymc.Model.compile_logp)
expects transformed values and adds the Jacobian. The probe does not use `compile_logp`: it evaluates
`pymc.logp` at constrained values, without a Jacobian, which is the convention of the QVR density, so
$\Psi_{\mathsf{PyMC}} = \mathrm{id}$ for the comparison.

## Family parameterizations

PyMC's [distributions](https://www.pymc.io/projects/docs/en/stable/api/distributions.html) use their own
argument names:

| QVR argument | PyMC argument |
|---|---|
| `loc` | `mu` |
| `scale` | `sigma` |
| `concentration` (Dirichlet) | `a` |
| `concentration1` (Beta) | `alpha` |
| `concentration0` (Beta) | `beta` |
| `probs` | `p` |
| `total_count` (Binomial) | `n` |
| `df` (StudentT) | `nu` |

They are recorded in `FAMILY_META[F].arg_aliases["pymc"]`. The renaming changes no value, so
$c_{F, \mathsf{PyMC}} = 0$ for every family PyMC has.

## Emitted constructs

**`sample` and `observe`.** `<name> = pymc.<Family>("<name>", **args, dims=(<plate axes>), observed=<obs>)`
inside `with pymc.Model(coords=...) as model:`. `dims` names the batch axes, and `coords` gives each axis
its labels.

**Plates.** The independence of the draws comes from the distribution's batch shape and the broadcasting
of its arguments. A scalar Normal with batch size ten is ten independent scalar events, and an `MvNormal`
with event size ten is one joint event whose covariance couples its coordinates.

**`marginalize`.** A supported block is enumerated, and its summed-out log density is added with a
`pymc.Potential`. The rows of each group are summed before the reduction.

**`score`, `let`, `return`.** `pymc.Potential("name", expr)` for a score, an assignment for a `let`, and
`pymc.Deterministic("<name>_value", ...)` for each returned value. The emitted `build_model` returns the
model.

## Evidence

- **Level 1.** Every emitted module has a `def build_model(...)` with a `with pymc.Model(coords=...) as model:`
  body, one `pymc.<Family>(...)` call for each draw and observation, and `return model`, and calling the
  stages directly or composed gives the same output.
- **Level 2.** Python's `ast` parser accepts the emitted source.
- **Level 3.** The summed `pymc.logp` is compared with the QVR density on the selected fixture grids.
