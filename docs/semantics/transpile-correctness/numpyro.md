# NumPyro

A NumPyro model is a Python function, traced by JAX, whose calls `numpyro.sample` register named sites,
and whose handlers record those sites in an execution trace [@phan2019numpyro]. To compare densities, the
test probe gives values for the latent sites and the observations and sums the log densities the trace
records ([Transpilation correctness](index.md)).

## How NumPyro evaluates a log density

NumPyro evaluates a model by running it under effect handlers, which record each site's distribution and
value in a trace [@phan2019numpyro], following the lightweight implementation of probabilistic programs by
execution traces [@wingate2011lightweight]. The probe is
[`numpyro.infer.util.log_density`](https://num.pyro.ai/en/stable/utilities.html#log-density), which returns
the sum of the log densities of the trace's sites, the
[observations](../programs.md#22-observe) included.

## Change of variables

The renderer requests no reparameterization, so the emitted model is in constrained coordinates:
$\Psi_{\mathsf{NumPyro}} = \mathrm{id}$ and the log Jacobian is $0$. NumPyro's inference may transform
coordinates, through
[`TransformReparam`](https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html) or an automatic
guide such as [`AutoNormal`](https://num.pyro.ai/en/stable/autoguide.html#autonormal), and a probe in those
coordinates would need their Jacobian. The comparison here uses the model's density at constrained values.

## Family parameterizations

NumPyro's [distributions](https://num.pyro.ai/en/stable/distributions.html) follow the interface of
`torch.distributions` in JAX, and the map is the identity for every QVR family with a NumPyro class:

- `Normal(μ, σ)` to `Normal(loc=μ, scale=σ)`;
- `Dirichlet(α)` to `Dirichlet(concentration=α)`, with `jnp.full((K,), α)` for a scalar α;
- `Categorical(probs=p)` to `Categorical(probs=p)`;
- `Bernoulli(p)` to `Bernoulli(probs=p)`, and `Bernoulli(logits=l)` to `Bernoulli(logits=l)`;
- `LogitNormal` to `LogitNormal(loc=μ, scale=σ)`.

In each case $\pi_{F, \mathsf{NumPyro}}$ is the identity and $c_{F, \mathsf{NumPyro}} = 0$.

## Emitted constructs

**`sample` and `observe`.** `numpyro.sample("x", <dist>)` for a latent variable and
`numpyro.sample("y", <dist>, obs=y_data)` for an observation. Each adds its site's log density to the
value of `log_density`.

**Plates.** Nested `with numpyro.plate(name, B):` contexts
([`plate`](https://num.pyro.ai/en/stable/primitives.html#numpyro.primitives.plate)), which state that the
$B$ draws are conditionally independent, so their density is the product
([plates](index.md#5-plates-marginalization-and-via)).

**`marginalize`.** The renderer computes, for each value of the latent variable, the log prior of the value
and the log likelihood of the scope, stacks them, reduces them with `logsumexp`, and adds the result with
`numpyro.factor`. A grouped block sums the rows of each group, by `via`, before the reduction. No sample site
remains for the variable summed out.

**`score`, `let`, `return`.** `numpyro.factor("name", expr)` for a score, an assignment for a `let`, and
Python's `return`.

## Evidence

- **Level 1.** Structural tests check the emitted `model` function and the pipeline `Lower`,
  `NumPyroRenderer`, and `EmitPretty(python)`.
- **Level 2.** Python's `ast` parser accepts the modules of the syntax matrix.
- **Level 3.** On selected fixtures, `numpyro.infer.util.log_density` is compared with the QVR density on
  the shared grid and tolerance ([evidence](index.md#6-evidence-tiers)).
