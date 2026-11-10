# NumPyro

This page discharges the per-target obligations of
[Transpilation correctness](index.md) for $\mathsf{T} = \mathrm{NumPyro}$.

A NumPyro model is a Python function with named random-choice sites. A
trace records which sites were reached and the values used there. For a
density comparison, the probe supplies values for the latent sites and the
observations, then sums their log-density contributions. Repeating this
evaluation at the test grid points checks how the emitted model weights
different latent configurations.

## Semantics

NumPyro's denotational semantics is the trace semantics of
Goodman et al. [@goodman2008church] /
Wingate, Stuhlmüller, Goodman
[@wingate2011lightweight]
implemented via the effect-handler architecture of Phan, Pradhan,
and Jankowiak
[@phan2019numpyro]. A program
is a JAX-traced Python function whose [`numpyro.sample` invocations](https://num.pyro.ai/en/stable/getting_started.html#a-simple-example-8-schools)
register sites in a trace. The log-density probe is
[`numpyro.infer.util.log_density`](https://num.pyro.ai/en/stable/utilities.html#log-density);
NumPyro returns the joint log-density of the trace, including the
[observation likelihoods](../programs.md#22-observe).

## Unconstrained-space change of variables

NumPyro applies the same change-of-variables principle as Stan but
with reparameterization handled by per-site
[`TransformReparam`](https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html)
or auto-transformations registered in
[`AutoNormal`](https://num.pyro.ai/en/stable/autoguide.html#autonormal).
For QVR-emitted code at the level the renderer produces, no
explicit reparameterization is requested: $\Psi_{\mathsf{NumPyro}}
= \mathrm{id}$ and $\log|\det J| \equiv 0$. The inference layer
may add reparameterizations at runtime, but those transformations must be accounted for only when a probe uses
transformed coordinates. The comparison here uses the model-side density
at the supplied constrained values.

## Family parameterizations

NumPyro provides a JAX implementation whose distribution interface
follows the design of `torch.distributions`, as described in its
[distribution reference](https://num.pyro.ai/en/stable/distributions.html). The mapping is identity for every QVR family that has
a corresponding [`numpyro.distributions.*`](https://num.pyro.ai/en/stable/distributions.html)
class:

* `Normal(μ, σ)` ↔ `Normal(loc=μ, scale=σ)`
* `Dirichlet(α)` ↔ `Dirichlet(concentration=α)` (with `jnp.full((K,), α)` broadcast when α is scalar)
* `Categorical(probs=p)` ↔ `Categorical(probs=p)`
* `Bernoulli(p)` ↔ `Bernoulli(probs=p)`
* `LogitNormal` ↔ `LogitNormal(loc=μ, scale=σ)`

In each case $\pi_{F, \mathsf{NumPyro}}$ is the identity and
$c_{F, \mathsf{NumPyro}} = 0$.

## Per-construct emit

**Sample / observe.** `numpyro.sample("x", <dist>)` for latents;
`numpyro.sample("y", <dist>, obs=y_data)` for observations. Each
contributes the documented per-site log-density to the
[`log_density`](https://num.pyro.ai/en/stable/utilities.html#log-density)
probe.

**Plate.** Nested `with numpyro.plate(name, B):` contexts per the
documented [`plate`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.plate)
primitive. The semantics is
exactly the conditionally-independent product measure of $B$
i.i.d. draws ([plate discussion](index.md#5-plates-marginalization-and-via)).

**Marginalize.** The renderer enumerates the finite support, accumulates each
atom's prior and scoped log likelihood, and adds the `logsumexp` reduction
through `numpyro.factor`. Grouped scopes accumulate rows by `via` before
reducing the latent axis. No sample site remains for the marginalized value.

**Score / let / return.** `numpyro.factor("name", expr)` for
score; deterministic assignment for let; native `return` for the
return clause.

## Acceptance

* **Tier 1 pipeline composition.** Structural checks cover the
  emitted `model` function and the `Lower >> NumPyroRenderer >>
  EmitPretty(python)` pipeline.
* **Tier 2 external syntax.** Python's AST parser accepts the
  generated modules in the external-validation matrix.
* **Tier 3 numeric equivalence.** For selected fixtures,
  `numpyro.infer.util.log_density` is compared with the QVR
  reference on the shared finite grid and tolerance described in
  [the test contract](index.md#6-evidence-tiers).
