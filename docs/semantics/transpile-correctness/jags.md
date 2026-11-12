# JAGS

JAGS (Just Another Gibbs Sampler) reads models in the syntax of BUGS, with extensions, and fits them by Gibbs
sampling [@plummer2003jags]. A JAGS model is a directed graphical model, whose joint distribution factors
over its stochastic nodes, and the data given with it decide which nodes are observed. A `marginalize` block
of QVR asks for the finite sum $\sum_z p(z) p(y \mid z)$, and the renderer computes that sum and adds its
logarithm to the joint density ([Transpilation correctness](index.md)).

## How the test evaluates a log density

The runtime is JAGS through its Python binding [`pyjags`](https://github.com/michaelnowotny/pyjags). The
probe compiles the model with the data and with the latent variables fixed as initial values, reads JAGS's
`deviance` monitor before any sampling, and takes $-\mathrm{deviance}/2$ as the log density. The renderer
writes the subset of BUGS that JAGS accepts, and its conversions of parameters and its zeros trick are those
of the [BUGS page](bugs.md).

## Change of variables

As for BUGS, $\Psi_{\mathsf{JAGS}} = \mathrm{id}$.

## Family parameterizations {#family-parameterizations}

JAGS shares the Normal family with a precision of BUGS (`dnorm(μ, τ)` with $\tau = 1/\sigma^2$), the precision
conventions of `dlnorm`, `dt`, and `dmnorm`, and the calculation that gives $c_{F, \mathsf{JAGS}} = 0$ for every
family. The [BUGS page](bugs.md#family-parameterizations) has the full table, and JAGS differs in these
entries:

| QVR family | JAGS | Note |
|---|---|---|
| `Dirichlet(α)` | `ddirich(α)` | BUGS writes `ddirch` |
| `Gamma(α, β)` | `dgamma(α, β)` | as in BUGS |
| `Bernoulli(p)` | `dbern(p)` | as in BUGS |
| `Bernoulli(logits=l)` | refused as `family:Bernoulli:logits` | `dbern` takes no log odds, so write `probs` |
| `Categorical(probs=p)` | `dcat(p)` at the code `y_code <- y + 1`, which the `data` block binds, and a drawn class is read back as `code - 1` | $y \mapsto y + 1$ |
| `Normal(μ, σ)` | `dnorm(μ, 1/σ²)` | precision |
| `FisherSnedecor(d_1, d_2)` | `df(n, m)` | BUGS has no mapping |

JAGS also has the functions `pow` and `inprod`, and further distributions in its modules, described in the
[JAGS user manual](https://sourceforge.net/projects/mcmc-jags/files/Manuals/).

## Emitted constructs

Draws, observations, and plates are written as in [BUGS](bugs.md#per-construct-emit), with the names and
arguments of JAGS. Unlike BUGS, JAGS has a `data { ... }` block, which transforms the data before the model,
so it can bind the shifted class codes of observed categorical values. The weighted sum that a `marginalize`
block must compute is defined on [Programs as kernels](../programs.md#26-marginalize).

**`marginalize`.** The renderer builds the weighted sum over the values of the latent variable and scores its
logarithm with the zeros trick [@lunn2012bugs]. A grouped block first sums the rows that share a latent
variable. The supported forms of scope and family are checked one by one, and they are more than the
categorical mixtures BUGS accepts.

**`score`, `let`.** A score uses the zeros trick, as in BUGS, and a `let` is a deterministic node `<-`. The
Poisson rate of the zeros trick must be nonnegative, so the constant it carries must exceed the score on the
whole domain of the parameters.

## Evidence

- **Level 1.** The emitted models have the same shape as for BUGS, and calling the stages directly or composed
  gives the same output.
- **Level 2.** The test writes a JAGS command file, asks JAGS to load the model, and rejects any error output of
  the compiler.
- **Level 3.** The `pyjags` probe evaluates selected emitted densities under the shared tolerance.
