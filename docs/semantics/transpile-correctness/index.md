# Transpilation Correctness: Contract and Evidence

`quivers.transpile.transpile(module, target=...)` supports eleven registered targets: BUGS, Church, Edward2, Gen, JAGS, NumPyro, PyMC, Pyro, Stan, Turing, and WebPPL. This chapter states the density-preservation contract and describes its structural, syntax, and numerical tests.

The practical question is whether the emitted program assigns the same
relative weights to the same model choices and observations. The checks
below evaluate both programs at supplied parameter values. This comparison
works across systems with different random-number generators and inference algorithms.

The target pages use three pieces of notation. $\pi_{F,T}$ maps the
parameters of family $F$ into target $T$'s parameter convention, such as
scale to precision. $c_{F,T}$ denotes a parameter-independent log-density
difference when one is present. $\Psi_T$ maps unconstrained inference
coordinates to constrained model values when discussing a change of
variables. A scale-to-precision conversion belongs to $\pi_{F,T}$; representing a
positive variable as the exponential of an unconstrained one belongs to $\Psi_T$.

## 1. Pipeline

The production path is:

$$
\mathrm{Module}
\xrightarrow{\text{target-specific composite expansion}}
\mathrm{Module}
\xrightarrow{\mathrm{Lower}}
\mathrm{IRProgram}
\xrightarrow{\mathrm{Renderer}_{T}}
\mathrm{Schema}_{T}
\xrightarrow{\mathrm{emit\_pretty}}
\mathrm{bytes}.
$$

`Lower` resolves program steps into target-neutral IR nodes. A renderer builds a panproto schema for one target grammar, and `emit_pretty` serializes that schema. The target registry in `quivers.transpile._RENDERERS` also associates each backend with a construct-support tier.

The pipeline is deterministic on the fixtures covered by `tests/transpile/test_lens_laws.py`. Those tests establish renderer determinism, equality between direct and composed pipeline invocation, and a parse/re-emit fixed point. Despite the test filename, the checked equalities concern this one-way compilation pipeline.

## 2. Correctness criterion

Adding the same constant to every log weight leaves a normalized posterior
unchanged. Suppose the reference log densities at three points are
$(-3,-2,-5)$ and the target returns $(-1,0,-3)$. All differences are 2, so
exponentiating multiplies every weight by the same $e^2$, which cancels on
normalization. Differences $(2,2,3)$ would instead change their relative
weights. The test measures this variation in log-density differences.

For a supported QVR fixture $M$, target $T$, parameter point $\theta$, and observed data $y$, the numeric suite compares

$$
\ell_{\mathrm{QVR}}(\theta;y)
\quad\text{and}\quad
\ell_T(\theta;y).
$$

The comparison admits a target-dependent additive constant because probabilistic libraries may omit normalizers that do not depend on $\theta$. The helper thus checks that

$$
\ell_T(\theta_i;y)-\ell_{\mathrm{QVR}}(\theta_i;y)
$$

is approximately constant over the selected point set. That is, the spread of the differences should be small.

Each result is specific to the tested model, observations, and parameter grid. To investigate a discrepancy, inspect the per-point differences: a constant shift suggests an omitted normalizer, while a changing difference suggests a parameterization, support, or factor-reduction error.

The cancellation of a parameter-independent factor is the same normalization
principle used in Bayesian posterior calculations [@gelman2013bayesian].
Both scores must also use the same support and retained latent coordinates.
A density evaluated at a hidden value and a density summed over that value
are different functions.

## 3. Support is explicit

Correctness is conditional on support. The frontend calls `unsupported_for(...)`, family lowering consults `FAMILY_META` in `src/quivers/transpile/family_meta.py`, and renderers may raise `UnsupportedConstruct` for target-specific gaps. The resulting diagnostic names the unsupported family or construct.

The family and construct matrices are executable inventories. They should be consulted instead of inferring support from a backend's native library. A target may have a native distribution that QVR does not yet map, or QVR may implement a distribution through an equivalent target expression rather than a same-named constructor.

## 4. Distribution parameterization

Stan’s [changes-of-variables example](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#changes-of-variables) derives a transformed density and its Jacobian. The [Stan target page](stan.md#unconstrained-space-change-of-variables) applies that correction to the coordinates used by the repository’s probe.

A scale-to-precision conversion changes a constructor argument:
$\tau=1/\sigma^2$ describes the same Normal family at a different parameter
coordinate. A change of the sampled variable is different. If a positive
variable is represented as $\theta=\exp(u)$, equal-length intervals in $u$
do not have equal lengths in $\theta$. The Jacobian compensates for this
stretching: the density in $u$ includes $\log|d\theta/du|=u$ in addition
to the density evaluated at $\theta=\exp(u)$.

For each supported family, the renderer must preserve:

1. argument meaning and order;
2. event and batch shape;
3. support constraints;
4. any Jacobian term introduced by a change of variables;
5. observation and plate reduction conventions.

`FAMILY_META[F].target_names` and target argument aliases record part of this mapping. Helpers may implement the rest, including scale/rate conversions, truncation, or matrix reconstruction. The per-target pages document important cases, but the source and numeric tests remain authoritative for current behavior.

## 5. Plates, marginalization, and `via`

The [plate definition](../programs.md#24-indexed-bind-plate) gives the independent product, while [grouped marginalization](../programs.md#27-grouped-marginalize-with-multi-observe-fibration) specifies how several observations share one hidden choice. Stan’s [mixture calculation](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#summing-out-the-responsibility-parameter) works through the corresponding finite sum.

An indexed `sample` or `observe` denotes repeated sites over a finite axis. A renderer may express that repetition through vectorization, a plate context, or an explicit loop. Correctness requires the same product of per-site factors, with event dimensions excluded from the plate reduction.

Scoped finite `marginalize` must preserve the integrated density. Renderers
can enumerate support values and add their reduced log weight, or use an
equivalent collapsed distribution. Keeping the hidden value as an ordinary
sample site instead would describe a joint distribution on a larger space,
not the requested marginal density. BUGS supports a categorical-mixture
collapse and refuses other blocks; JAGS can construct the weighted finite
sum and score it through its zero trick. Edward2 uses a mixture distribution
where applicable and a scored reduction for supported grouped cases.
A grouped block first sums each atom's row log likelihoods within the group
selected by `via`, then reduces the atom axis. Reducing rows independently
would describe one hidden choice per row.

The `via=` form groups or reindexes observations through a finite map. Correctness requires that each observation contribute to the factor for its mapped group exactly once. `tests/transpile/test_via_fibration_numeric.py` checks selected cases numerically.

## 6. Evidence tiers

The test files use the following practical tiers:

| Tier | Test | What it checks |
|---|---|---|
| 1 | `test_structural.py`, `test_lens_laws.py` | emitted schema shape, renderer determinism, pipeline composition, re-emission |
| 2 | `test_external_syntax.py` | acceptance by a target parser/compiler when available |
| 3 | `test_numeric_equivalence.py` | log-density comparison up to an additive constant in target runtimes |
| 4 | `test_construct_matrix.py` | construct-by-backend compatibility |
| 5 | `test_family_matrix.py` | distribution-family support matrix |
| 6 | `test_composition_fixtures.py` | larger composition fixtures across targets |

Additional numeric suites cover closed-form marginals, scoped marginalization, `via` grouping, and documentation-gallery programs.

The main Tier-3 grid currently uses seven composition fixtures. It selects five values per parameter axis and caps the Cartesian grid at sixteen points per fixture. Expected unsupported cells are asserted explicitly.

## 7. Runtime coverage

Docker images provide pinned runtimes for Stan, NumPyro, Pyro, PyMC, Edward2, Turing, Gen, WebPPL, JAGS, and the BUGS-syntax route through JAGS. Their build and probe contract is documented in `tests/transpile/docker/README.md`.

Church is absent from the shared Docker matrix because the repository does not provision a maintained Church runtime image with the same programmable joint-density interface. Church-specific audit tests cover selected emissions separately.

External-syntax checks are also target-specific:

- Stan uses `stanc --info -`;
- Python targets use Python's AST parser;
- Julia targets use `Meta.parse`;
- WebPPL output is checked as JavaScript with `node --check`;
- BUGS and JAGS are compiled through JAGS;
- Church has no canonical external parser in the shared suite.

These commands validate the named parser's accepted syntax. WebPPL execution additionally requires its primitives and compiler transformations; checking a specific BUGS dialect requires that implementation.

## 8. Per-target notes

- [Stan](stan.md)
- [NumPyro](numpyro.md)
- [Pyro](pyro.md)
- [PyMC](pymc.md)
- [Edward2](edward2.md)
- [Turing](turing.md)
- [Gen](gen.md)
- [Church](church.md)
- [WebPPL](webppl.md)
- [BUGS](bugs.md)
- [JAGS](jags.md)

Each page describes emitted structure, parameter conversions, known gaps, and the evidence exercised for that target.

## 9. Extending coverage

New family or construct support should add structural, syntax, and numerical tests together. Choose parameter points that exercise support boundaries and the converted arguments; include both scalar and plated cases when the family supports them. A continuous change of variables also needs a Jacobian check.

Gradient comparisons and posterior-inference diagnostics require separate tests. For Church and WebPPL, complete language validation would additionally require running the emitted program through the intended language implementation.
