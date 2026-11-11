"""Semantic mutation catalogue for the equivalence sensitivity suite.

This module records source rewrites that make a passing
``(example, backend)`` cell denote a different measure. The grid is
small and explicit.

Every rewrite is a literal-text substitution with an exact
occurrence count:

1. The mutant is a **valid program in the target language**, so the
   backend's own probe evaluates it and the harness compares two
   real log-density vectors rather than an exception.
2. The mutation is anchored to renderer output. If the anchor moves,
   the test reports the missing substitution.

The defect classes are drawn from failures this codebase actually
shipped (a family's shape and scale arguments transposed; a sampling
statement dropping the data-only summands of a log-density; a
marginalize block lowered to a live draw; a continuous leaf cast
through an integer dtype; a gather reading one index off) plus the
standard perturbations of a probabilistic program: a prior term
removed, a parameter-dependent normalizer removed, a parameter
scaled, a nearby family substituted.

Each [`Mutation`][tests.transpile._mutations.Mutation] records a
`min_spread` measured on the current point set. This floor detects a
loss of sensitivity even while the mutation remains above the
equivalence tolerance.
"""

from __future__ import annotations

import didactic.api as dx


class SourceRewrite(dx.Model):
    """One literal-text substitution against an emitted program.

    `occurrences` is the exact number of times `old` must appear. An
    anchor that also matches another prior or a likelihood would
    change more of the program than the catalogue declares.
    """

    old: str
    new: str
    occurrences: int = 1


class BackendMutant(dx.Model):
    """The rewrite sequence that realises one mutation on one backend.

    A mutation's meaning is fixed (drop the prior term, transpose the
    family arguments); its spelling is per-target, because the
    backends disagree about whether a log-density is accumulated into
    an explicit `target` accumulator, declared as a sampling
    statement, or built as a distribution object.
    """

    backend: str
    rewrites: tuple[SourceRewrite, ...]


class Mutation(dx.Model):
    """One defect class, applied to one gallery example.

    Attributes
    ----------
    name
        Unique catalogue key; also the pytest parameter id.
    defect_class
        Short slug naming what kind of measure error the mutant
        carries.
    provenance
        Where the defect class comes from: a failure this codebase
        shipped, or a standard perturbation.
    example
        Stem of the `docs/examples/source/<stem>.qvr` gallery
        example the mutation is anchored to.
    min_spread
        Lower bound on the constant-spread deviation the mutant must
        produce, in nats. Set to roughly half the smallest spread
        measured across the mutation's backends when the catalogue
        was built.
    mutants
        One entry per backend the mutation is defined for.
    """

    name: str
    defect_class: str
    provenance: str
    example: str
    min_spread: float
    mutants: tuple[BackendMutant, ...]


class AcceptedRewrite(dx.Model):
    """A rewrite the equivalence check must **not** reject.

    Theorem 4.1 quotients by a point-independent additive constant,
    so a mutant that shifts the log-density by a constant denotes the
    same measure and has to pass. Without these the reject tests
    would be satisfied by a check that fails on everything.
    """

    name: str
    example: str
    backend: str
    rationale: str
    rewrites: tuple[SourceRewrite, ...]


class BlindSpot(dx.Model):
    """A rewrite the equivalence check provably does **not** reject.

    Each entry is a defect the constant-spread contract cannot see,
    pinned here as a measured fact rather than left as an assumption.
    The sensitivity suite asserts these stay invisible, so the day
    one of them starts failing is the day the check grew a capability
    (or the harness grew an unintended coupling) and the registry is
    stale.
    """

    name: str
    example: str
    backend: str
    why_invisible: str
    rewrites: tuple[SourceRewrite, ...]


def apply_rewrites(
    source: str, rewrites: tuple[SourceRewrite, ...], *, context: str
) -> str:
    """Apply `rewrites` in order, asserting each anchor's occurrence
    count.

    Raises
    ------
    AssertionError
        When an anchor appears a number of times other than its
        pinned `occurrences`. The message carries the anchor and the
        observed count: the emitted program moved under the
        catalogue, and the fix is to re-derive the anchor from the
        current emit, never to loosen the count.
    """
    out = source
    for rewrite in rewrites:
        found = out.count(rewrite.old)
        if found != rewrite.occurrences:
            raise AssertionError(
                f"{context}: mutation anchor {rewrite.old!r} appears "
                f"{found} time(s) in the emitted program but the "
                f"catalogue pins {rewrite.occurrences}. The renderer's "
                f"emit moved under `tests/transpile/_mutations.py`; "
                f"re-derive the anchor from the current emit. Do not "
                f"relax the count: an anchor that matches more of the "
                f"program than the catalogue claims mutates a defect "
                f"class other than the one under test, and an anchor "
                f"that matches nothing reports a mutant the check "
                f"never saw."
            )
        out = out.replace(rewrite.old, rewrite.new)
    if out == source:
        raise AssertionError(
            f"{context}: the rewrite sequence left the emitted program "
            f"byte-identical, so the 'mutant' denotes the same measure "
            f"as the original and its rejection would be untestable."
        )
    return out


# ---------------------------------------------------------------------
# The catalogue.
#
# Backend spellings below are anchored to the current renderer emit.
# Read the emit before editing an entry:
#
#     transpile(parse(Path(f"docs/examples/source/{ex}.qvr").read_text()),
#               target=backend)
# ---------------------------------------------------------------------


_AR1_PRIOR_FLATTENED = Mutation(
    name="prior_term_flattened",
    defect_class="prior-term-dropped",
    provenance=(
        "standard perturbation: the prior's dependence on its latent "
        "is removed. A flat prior over a range the point set never "
        "leaves contributes a constant, which is exactly what a "
        "dropped prior term contributes under Theorem 4.1's quotient."
    ),
    example="ar1",
    min_spread=0.003,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="target +=normal_lpdf(alpha | 0,5);",
                    new="target +=uniform_lpdf(alpha | -1000,1000);",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="numpyro.distributions.Normal(loc=0,scale=5)",
                    new="numpyro.distributions.Uniform(low=-1000,high=1000)",
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="pyro.distributions.Normal(0,5)",
                    new="pyro.distributions.Uniform(-1000.,1000.)",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("alpha" ,mu=0,sigma=5)',
                    new='pymc.Uniform("alpha" ,lower=-1000,upper=1000)',
                ),
            ),
        ),
        BackendMutant(
            backend="edward2",
            rewrites=(
                SourceRewrite(
                    old='edward2.Normal(loc=0,scale=5,name="alpha" )',
                    new='edward2.Uniform(low=-1000.,high=1000.,name="alpha" )',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="alpha ~ Normal(0, 5)",
                    new="alpha ~ Uniform(-1000, 1000)",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="@trace(normal(0, 5) , :alpha)",
                    new="@trace(uniform(-1000, 1000) , :alpha)",
                ),
            ),
        ),
        BackendMutant(
            backend="webppl",
            rewrites=(
                SourceRewrite(
                    old="Gaussian({\n    mu:0,sigma:5\n  })",
                    new="Uniform({a:-1000,b:1000})",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="alpha ~ dnorm(0,1/(5*5))",
                    new="alpha ~ dunif(-1000,1000)",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="alpha ~ dnorm(0,1/(5*5))",
                    new="alpha ~ dunif(-1000,1000)",
                ),
            ),
        ),
    ),
)


_AR1_SCALE_INFLATED = Mutation(
    name="likelihood_scale_inflated",
    defect_class="parameter-scaled-by-a-constant",
    provenance=(
        "standard perturbation: the observation scale enters the "
        "likelihood multiplied by 1.3, so the density is a valid "
        "Normal density for the wrong dispersion."
    ),
    example="ar1",
    min_spread=20.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="normal_lpdf(y[m_Step] | mu[m_Step],sigma)",
                    new="normal_lpdf(y[m_Step] | mu[m_Step],1.3*sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(loc=mu,scale=sigma),obs=y",
                    new="Normal(loc=mu,scale=1.3*sigma),obs=y",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("y" ,mu=mu,sigma=sigma',
                    new='pymc.Normal("y" ,mu=mu,sigma=1.3*sigma',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Normal.(mu, sigma)",
                    new="Normal.(mu, 1.3 * sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new="dnorm(mu[m_Step],1/(1.3*sigma*1.3*sigma))",
                ),
            ),
        ),
    ),
)


_AR1_FAMILY_SUBSTITUTED = Mutation(
    name="likelihood_family_substituted",
    defect_class="nearby-family-substituted",
    provenance=(
        "standard perturbation: the Normal observation family is "
        "replaced by a nearby heavy-tailed one (StudentT / Laplace / "
        "Cauchy / double-exponential, whichever the target spells "
        "natively). Location and scale are unchanged, so only the "
        "shape of the density moves."
    ),
    example="ar1",
    min_spread=20.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="normal_lpdf(y[m_Step] | mu[m_Step],sigma)",
                    new="student_t_lpdf(y[m_Step] | 4,mu[m_Step],sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="numpyro.distributions.Normal(loc=mu,scale=sigma)",
                    new="numpyro.distributions.StudentT(df=4,loc=mu,scale=sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="pyro.distributions.Normal(mu,sigma)",
                    new="pyro.distributions.StudentT(4,mu,sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("y" ,mu=mu,sigma=sigma',
                    new='pymc.StudentT("y" ,nu=4,mu=mu,sigma=sigma',
                ),
            ),
        ),
        BackendMutant(
            backend="edward2",
            rewrites=(
                SourceRewrite(
                    old='edward2.Normal(loc=mu,scale=sigma,value=y,name="y" )',
                    new='edward2.StudentT(df=4.,loc=mu,scale=sigma,value=y,name="y" )',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Normal.(mu, sigma)",
                    new="Laplace.(mu, sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="normal(mu[m_Step], sigma)",
                    new="cauchy(mu[m_Step], sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="webppl",
            rewrites=(
                SourceRewrite(
                    old="Gaussian({\n      mu:mu[n],sigma:sigma\n    })",
                    new="Cauchy({location:mu[n],scale:sigma})",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new="ddexp(mu[m_Step],1/(sigma*sigma))",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new="ddexp(mu[m_Step],1/(sigma*sigma))",
                ),
            ),
        ),
    ),
)


_AR1_PLATE_INDEX_ROTATED = Mutation(
    name="plate_index_rotated_by_one",
    defect_class="index-base-shifted",
    provenance=(
        "shipped defect class: a plate subscript read one position "
        "off. Each observation is scored against its neighbour's "
        "location, which is a permutation of the same numbers and "
        "therefore invisible to any check that compares sums rather "
        "than per-point densities."
    ),
    example="ar1",
    min_spread=12.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="normal_lpdf(y[m_Step] | mu[m_Step],sigma)",
                    new="normal_lpdf(y[m_Step] | mu[(m_Step % 64)+1],sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(loc=mu,scale=sigma),obs=y",
                    new="Normal(loc=jnp.roll(mu,1),scale=sigma),obs=y",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("y" ,mu=mu,sigma=sigma',
                    new=(
                        'pymc.Normal("y" ,'
                        "mu=pymc.math.concatenate([mu[-1:],mu[:-1]]),sigma=sigma"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Normal.(mu, sigma)",
                    new="Normal.(circshift(mu, 1), sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new=("dnorm(mu[m_Step+1-64*equals(m_Step,64)],1/(sigma*sigma))"),
                ),
            ),
        ),
    ),
)


_AR1_LOCATION_TRUNCATED = Mutation(
    name="location_parameter_truncated_to_integer",
    defect_class="continuous-value-truncated-to-integer",
    provenance=(
        "shipped defect class: a continuous leaf reached a backend "
        "through an integer dtype and was truncated, making the "
        "density piecewise constant in integer buckets. Here the "
        "observation's location parameter takes the truncation."
    ),
    example="ar1",
    min_spread=23.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="normal_lpdf(y[m_Step] | mu[m_Step],sigma)",
                    new="normal_lpdf(y[m_Step] | floor(mu[m_Step]),sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(loc=mu,scale=sigma)",
                    new="Normal(loc=jnp.floor(mu),scale=sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(mu,sigma),obs=y",
                    new="Normal(torch.floor(mu),sigma),obs=y",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("y" ,mu=mu,sigma=sigma',
                    new='pymc.Normal("y" ,mu=pymc.math.floor(mu),sigma=sigma',
                ),
            ),
        ),
        BackendMutant(
            backend="edward2",
            rewrites=(
                SourceRewrite(
                    old='edward2.Normal(loc=mu,scale=sigma,value=y,name="y" )',
                    new=(
                        'edward2.Normal(loc=tf.math.floor(mu),scale=sigma,value=y,name="y" )'
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Normal.(mu, sigma)",
                    new="Normal.(floor.(mu), sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="normal(mu[m_Step], sigma)",
                    new="normal(floor(mu[m_Step]), sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="webppl",
            rewrites=(
                SourceRewrite(
                    old="mu:mu[n],sigma:sigma",
                    new="mu:Math.floor(mu[n]),sigma:sigma",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new="dnorm(trunc(mu[m_Step]),1/(sigma*sigma))",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new="dnorm(trunc(mu[m_Step]),1/(sigma*sigma))",
                ),
            ),
        ),
    ),
)


_AR1_OBSERVATION_TERM_DROPPED = Mutation(
    name="one_observation_data_dependence_dropped",
    defect_class="data-dependent-term-dropped",
    provenance=(
        "shipped defect class: a log-density that drops summands "
        "carrying data. The final observation's scale is inflated to "
        "1e6, which flattens that summand's dependence on its datum "
        "to below float64 resolution while leaving the other 63 "
        "untouched. This mutant is detectable only because the point "
        "set moves the observed data: against a frozen data section "
        "the dropped summand contributes a perfectly constant offset "
        "and the check passes."
    ),
    example="ar1",
    min_spread=0.9,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="normal_lpdf(y[m_Step] | mu[m_Step],sigma)",
                    new=(
                        "normal_lpdf(y[m_Step] | mu[m_Step],m_Step == 64 ? 1e6 : sigma)"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(loc=mu,scale=sigma)",
                    new=(
                        "Normal(loc=mu,scale=jnp.concatenate("
                        "[sigma*jnp.ones(63),jnp.array([1e6])]))"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="Normal(mu,sigma),obs=y",
                    new=(
                        "Normal(mu,torch.cat([sigma*torch.ones("
                        "63,dtype=torch.float64),torch.tensor("
                        "[1e6],dtype=torch.float64)])),obs=y"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Normal("y" ,mu=mu,sigma=sigma',
                    new=(
                        'pymc.Normal("y" ,mu=mu,sigma=pymc.math.concatenate('
                        "[sigma*np.ones(63),np.array([1e6])])"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="edward2",
            rewrites=(
                SourceRewrite(
                    old='edward2.Normal(loc=mu,scale=sigma,value=y,name="y" )',
                    new=(
                        "edward2.Normal(loc=mu,scale=tf.concat("
                        "[sigma*tf.ones([63]),tf.constant([1e6])],axis=0),"
                        'value=y,name="y" )'
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Normal.(mu, sigma)",
                    new="Normal.(mu, vcat(fill(sigma, 63), 1e6))",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="normal(mu[m_Step], sigma)",
                    new="normal(mu[m_Step], m_Step == 64 ? 1e6 : sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="webppl",
            rewrites=(
                SourceRewrite(
                    old="mu:mu[n],sigma:sigma",
                    new="mu:mu[n],sigma:(n === 63 ? 1e6 : sigma)",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new=(
                        "dnorm(mu[m_Step],1/(sigma*sigma)"
                        "*pow(1.0E-12,equals(m_Step,64)))"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="dnorm(mu[m_Step],1/(sigma*sigma))",
                    new=(
                        "dnorm(mu[m_Step],1/(sigma*sigma)"
                        "*pow(1.0E-12,equals(m_Step,64)))"
                    ),
                ),
            ),
        ),
    ),
)


_AR1_PARAMETER_NORMALIZER_DROPPED = Mutation(
    name="parameter_dependent_normalizer_dropped",
    defect_class="parameter-dependent-normalizer-dropped",
    provenance=(
        "standard perturbation: the Normal likelihood keeps its "
        "quadratic kernel and loses the `-log(sigma)` normalizer. The "
        "dropped term depends on a latent, so the offset drifts as "
        "the latents move even though the data-dependence is intact. "
        "Expressed only on the targets whose surface carries an "
        "explicit log-weight accumulator."
    ),
    example="ar1",
    min_spread=18.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="target +=normal_lpdf(y[m_Step] | mu[m_Step],sigma);",
                    new="target += -0.5*square((y[m_Step]-mu[m_Step])/sigma);",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        '  with numpyro.plate("Step" ,64) :\n'
                        '    numpyro.sample("y" ,numpyro.distributions.'
                        "Normal(loc=mu,scale=sigma),obs=y)"
                    ),
                    new='  numpyro.factor("y" ,-0.5*jnp.sum(((y-mu)/sigma)**2))',
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        '  with pyro.plate("Step" ,64) :\n'
                        '    pyro.sample("y" ,pyro.distributions.'
                        "Normal(mu,sigma),obs=y)"
                    ),
                    new='  pyro.factor("y" ,(-0.5*((y-mu)/sigma)**2).sum())',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="  y ~ product_distribution(Normal.(mu, sigma) )",
                    new=("  Turing.@addlogprob! -0.5 * sum(((y .- mu) ./ sigma) .^ 2)"),
                ),
            ),
        ),
    ),
)


_GAMMA_ARGUMENTS_TRANSPOSED = Mutation(
    name="family_arguments_transposed",
    defect_class="family-arguments-transposed",
    provenance=(
        "shipped defect: a two-parameter positive family received its "
        "shape where its scale or rate belonged and vice versa. Both "
        "arguments of the Gamma likelihood are positive, so the mutant "
        "is a perfectly well-formed Gamma density for the wrong "
        "parameterisation and nothing short of a numeric comparison "
        "distinguishes it."
    ),
    example="gamma_regression",
    min_spread=400.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="gamma_lpdf(y[m_Resp] | nu[m_Resp],rate[m_Resp])",
                    new="gamma_lpdf(y[m_Resp] | rate[m_Resp],nu[m_Resp])",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="Gamma(concentration=nu,rate=rate),obs=y",
                    new="Gamma(concentration=rate,rate=nu),obs=y",
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="Gamma(nu,rate),obs=y",
                    new="Gamma(rate,nu),obs=y",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Gamma("y" ,alpha=nu,beta=rate',
                    new='pymc.Gamma("y" ,alpha=rate,beta=nu',
                ),
            ),
        ),
        BackendMutant(
            backend="edward2",
            rewrites=(
                SourceRewrite(
                    old='edward2.Gamma(concentration=nu,rate=rate,value=y,name="y" )',
                    new='edward2.Gamma(concentration=rate,rate=nu,value=y,name="y" )',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="Gamma.(nu, inv(rate) )",
                    new="Gamma.(rate, inv(nu) )",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="gamma(nu[m_Resp], inv(rate[m_Resp]) )",
                    new="gamma(rate[m_Resp], inv(nu[m_Resp]) )",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dgamma(nu[m_Resp],rate[m_Resp])",
                    new="dgamma(rate[m_Resp],nu[m_Resp])",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="dgamma(nu[m_Resp],rate[m_Resp])",
                    new="dgamma(rate[m_Resp],nu[m_Resp])",
                ),
            ),
        ),
    ),
)


_NEGBIN_FAMILY_SUBSTITUTED = Mutation(
    name="count_family_substituted",
    defect_class="nearby-family-substituted",
    provenance=(
        "standard perturbation, count-data flavour: the "
        "NegativeBinomial observation is replaced by the Poisson with "
        "the same mean. The two agree in first moment and differ only "
        "in dispersion, which is the substitution a renderer makes "
        "when a target lacks the over-dispersed family."
    ),
    example="negbin_regression",
    min_spread=107.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old=(
                        "neg_binomial_2_lpmf(y[m_Resp] | disp[m_Resp]*"
                        "probs[m_Resp]/(1-probs[m_Resp]),disp[m_Resp])"
                    ),
                    new=(
                        "poisson_lpmf(y[m_Resp] | disp[m_Resp]*"
                        "probs[m_Resp]/(1-probs[m_Resp]))"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        "NegativeBinomial2(mean=(disp*probs)/(1-probs),"
                        "concentration=disp)"
                    ),
                    new="Poisson(rate=(disp*probs)/(1-probs))",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.NegativeBinomial("y" ,n=disp,p=(1-probs)',
                    new='pymc.Poisson("y" ,mu=(disp*probs)/(1-probs)',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="NegativeBinomial.(disp, 1 .- probs)",
                    new="Poisson.(disp .* probs ./ (1 .- probs))",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="neg_binom(disp[m_Resp], 1 - probs[m_Resp])",
                    new=("poisson(disp[m_Resp] * probs[m_Resp] / (1 - probs[m_Resp]))"),
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="dnegbin(1-probs[m_Resp],disp[m_Resp])",
                    new="dpois(disp[m_Resp]*probs[m_Resp]/(1-probs[m_Resp]))",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="dnegbin(1-probs[m_Resp],disp[m_Resp])",
                    new="dpois(disp[m_Resp]*probs[m_Resp]/(1-probs[m_Resp]))",
                ),
            ),
        ),
    ),
)


_NEGBIN_GATHER_SHIFTED = Mutation(
    name="gather_index_base_shifted",
    defect_class="index-base-shifted",
    provenance=(
        "shipped defect: a gather read its index one base off. The "
        "mutant rotates `beta_0[out_idx]` by one group, so every "
        "response is scored against a neighbouring group's intercept. "
        "This is the 0-based / 1-based confusion that separates the "
        "array conventions of the targets."
    ),
    example="negbin_regression",
    min_spread=96.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="b0[m_Resp] = beta_0[out_idx[m_Resp]+1];",
                    new="b0[m_Resp] = beta_0[((out_idx[m_Resp]+1) % 3)+1];",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="b0 = beta_0[out_idx]",
                    new="b0 = beta_0[(out_idx+1) % 3]",
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old="b0 = beta_0[out_idx]",
                    new="b0 = beta_0[(out_idx+1) % 3]",
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="b0 = @. beta_0[out_idx + 1]",
                    new="b0 = @. beta_0[mod(out_idx + 1, 3) + 1]",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="b0 = @. beta_0[out_idx + 1]",
                    new="b0 = @. beta_0[mod(out_idx + 1, 3) + 1]",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="b0[m_Resp] <- beta_0[out_idx[m_Resp]+1]",
                    new=(
                        "b0[m_Resp] <- beta_0[out_idx[m_Resp]+2-3*"
                        "equals(out_idx[m_Resp],2)]"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="b0[m_Resp] <- beta_0[out_idx[m_Resp]+1]",
                    new=(
                        "b0[m_Resp] <- beta_0[out_idx[m_Resp]+2-3*"
                        "equals(out_idx[m_Resp],2)]"
                    ),
                ),
            ),
        ),
    ),
)


_CHANGEPOINT_DATA_ONLY_NORMALIZER_DROPPED = Mutation(
    name="data_only_normalizer_dropped",
    defect_class="data-dependent-term-dropped",
    provenance=(
        "shipped defect: Stan's `~` sampling operator discards every "
        "summand of a log-density that carries no parameter, so a "
        "Poisson likelihood loses `-log(y!)`. The mutant reproduces "
        "that exactly, on Stan through the operator itself and "
        "elsewhere by scoring the unnormalised Poisson kernel. The "
        "resulting offset is a function of the data alone: it is "
        "perfectly constant as the latents move and moves only when "
        "the observations do, which makes it the sharpest available "
        "probe of the point set's data axis."
    ),
    example="changepoint",
    min_spread=5.9,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="target +=poisson_lpmf(y[m_Step] | rate[m_Step]);",
                    new="y[m_Step] ~ poisson(rate[m_Step]);",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        '  with numpyro.plate("Step" ,64) :\n'
                        '    numpyro.sample("y" ,numpyro.distributions.'
                        "Poisson(rate=rate),obs=y)"
                    ),
                    new='  numpyro.factor("y" ,jnp.sum(y*jnp.log(rate)-rate))',
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        '  with pyro.plate("Step" ,64) :\n'
                        '    pyro.sample("y" ,pyro.distributions.'
                        "Poisson(rate),obs=y)"
                    ),
                    new='  pyro.factor("y" ,(y*torch.log(rate)-rate).sum())',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="  y ~ product_distribution(Poisson.(rate) )",
                    new="  Turing.@addlogprob! sum(y .* log.(rate) .- rate)",
                ),
            ),
        ),
    ),
)


_CHANGEPOINT_PRIOR_TRANSPOSED = Mutation(
    name="prior_arguments_transposed",
    defect_class="family-arguments-transposed",
    provenance=(
        "shipped defect class, prior flavour: a Gamma prior's shape "
        "and rate arguments swap places. `Gamma(2, 1)` and `Gamma(1, "
        "2)` are both proper densities on the positive reals, so the "
        "mutant is well-formed everywhere the point set visits and "
        "only the latent-dependence of the prior term moves."
    ),
    example="changepoint",
    min_spread=0.37,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="gamma_lpdf(rate_before | 2,1)",
                    new="gamma_lpdf(rate_before | 1,2)",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old=(
                        '"rate_before" ,numpyro.distributions.'
                        "Gamma(concentration=2,rate=1)"
                    ),
                    new=(
                        '"rate_before" ,numpyro.distributions.'
                        "Gamma(concentration=1,rate=2)"
                    ),
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old='"rate_before" ,pyro.distributions.Gamma(2,1)',
                    new='"rate_before" ,pyro.distributions.Gamma(1,2)',
                ),
            ),
        ),
        BackendMutant(
            backend="pymc",
            rewrites=(
                SourceRewrite(
                    old='pymc.Gamma("rate_before" ,alpha=2,beta=1)',
                    new='pymc.Gamma("rate_before" ,alpha=1,beta=2)',
                ),
            ),
        ),
        BackendMutant(
            backend="turing",
            rewrites=(
                SourceRewrite(
                    old="rate_before ~ Gamma(2, inv(1) )",
                    new="rate_before ~ Gamma(1, inv(2) )",
                ),
            ),
        ),
        BackendMutant(
            backend="gen",
            rewrites=(
                SourceRewrite(
                    old="@trace(gamma(2, inv(1) ) , :rate_before)",
                    new="@trace(gamma(1, inv(2) ) , :rate_before)",
                ),
            ),
        ),
        BackendMutant(
            backend="jags",
            rewrites=(
                SourceRewrite(
                    old="rate_before ~ dgamma(2,1)",
                    new="rate_before ~ dgamma(1,2)",
                ),
            ),
        ),
        BackendMutant(
            backend="bugs",
            rewrites=(
                SourceRewrite(
                    old="rate_before ~ dgamma(2,1)",
                    new="rate_before ~ dgamma(1,2)",
                ),
            ),
        ),
    ),
)


CATALOGUE: tuple[Mutation, ...] = (
    _AR1_PRIOR_FLATTENED,
    _AR1_SCALE_INFLATED,
    _AR1_FAMILY_SUBSTITUTED,
    _AR1_PLATE_INDEX_ROTATED,
    _AR1_LOCATION_TRUNCATED,
    _AR1_OBSERVATION_TERM_DROPPED,
    _AR1_PARAMETER_NORMALIZER_DROPPED,
    _GAMMA_ARGUMENTS_TRANSPOSED,
    _NEGBIN_FAMILY_SUBSTITUTED,
    _NEGBIN_GATHER_SHIFTED,
    _CHANGEPOINT_DATA_ONLY_NORMALIZER_DROPPED,
    _CHANGEPOINT_PRIOR_TRANSPOSED,
)


# ---------------------------------------------------------------------
# The marginalize leg.
#
# No gallery example whose cells are live carries a `marginalize`
# block, so the "integrated marginal replaced by a live draw" defect
# needs its own fixture. This is the Normal-mixture marginalize
# fixture: the Stan renderer enumerates the discrete class and sums
# with `log_sum_exp`, and the mutant keeps a single component's
# contribution, which is what a lowering that draws the latent
# instead of integrating it computes.
# ---------------------------------------------------------------------

MARGINALIZE_SOURCE = """object Cls : FinSet 2
object Obs : FinSet 20
program normal_mix : Obs -> Obs
    sample probs <- Dirichlet(concentration=1.0) [over=Cls]
    sample mu_low <- Normal(loc=-2.0, scale=1.0)
    sample mu_diff <- HalfNormal(scale=1.0)
    let mu = factor c : Cls in mu_low + c * mu_diff
    let sigma = 0.5
    marginalize cls : Cls <- Categorical(probs=probs) [over=Obs, reduction=logsumexp]
        observe y : Obs <- Normal(loc=mu[cls], scale=sigma) [via=idx]
    return probs
export normal_mix
"""


class MixturePoint(dx.Model):
    """One evaluation point for the marginalize fixture.

    The fixture is not a gallery example, so it has no synthetic-data
    block and no
    [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]
    perturbation to inherit; its point set is written out here. Both
    sections move: `probs` / `mu_low` / `mu_diff` carry the latents
    and `response` carries the observed data, so a mutant whose error
    is latent-only and one whose error is data-only both surface.
    """

    probs: tuple[float, ...]
    mu_low: float
    mu_diff: float
    response: tuple[float, ...]


_MIXTURE_RESPONSE = (
    -2.0,
    -1.5,
    -2.5,
    -1.8,
    -2.2,
    -1.0,
    -2.7,
    1.5,
    2.0,
    2.5,
    1.8,
    2.2,
    1.0,
    -2.0,
    -1.7,
    2.1,
    1.9,
    -2.4,
    2.3,
    -1.2,
)
"""Ground-truth responses: a left cluster near `mu_low` and a right
cluster near `mu_low + mu_diff`, so both mixture components carry real
mass and the `log_sum_exp` over classes is not dominated by one term."""


def _shifted(shift: float) -> tuple[float, ...]:
    """`_MIXTURE_RESPONSE` displaced by a per-position multiple of
    `shift`, so a data perturbation moves every observation by a
    different amount rather than translating the whole vector (a pure
    translation would leave a location-family likelihood's spread
    misleadingly small)."""
    return tuple(
        value + shift * (1 + index % 3) for index, value in enumerate(_MIXTURE_RESPONSE)
    )


MARGINALIZE_POINTS: tuple[MixturePoint, ...] = (
    MixturePoint(
        probs=(0.5, 0.5),
        mu_low=-2.0,
        mu_diff=4.0,
        response=_shifted(0.0),
    ),
    MixturePoint(
        probs=(0.3, 0.7),
        mu_low=-1.5,
        mu_diff=3.0,
        response=_shifted(0.0),
    ),
    MixturePoint(
        probs=(0.7, 0.3),
        mu_low=-2.5,
        mu_diff=5.0,
        response=_shifted(0.0),
    ),
    MixturePoint(
        probs=(0.5, 0.5),
        mu_low=-2.0,
        mu_diff=4.0,
        response=_shifted(0.31),
    ),
    MixturePoint(
        probs=(0.3, 0.7),
        mu_low=-1.5,
        mu_diff=3.0,
        response=_shifted(-0.27),
    ),
    MixturePoint(
        probs=(0.5, 0.5),
        mu_low=-2.0,
        mu_diff=4.0,
        response=_shifted(0.55),
    ),
)
"""Six points: the first three move the latents at fixed data, the
last three move the data at latent settings the first three already
visited, so the two axes are separable in a failure message."""


MARGINALIZE_MUTATION = Mutation(
    name="integrated_marginal_replaced_by_draw",
    defect_class="marginal-replaced-by-draw",
    provenance=(
        "shipped defect: a `marginalize` block lowered to a live "
        "sample of the discrete latent, so the emitted program scored "
        "the joint at one class rather than the sum over classes. The "
        "mutant replaces `log_sum_exp(lps_cls[g])` with the first "
        "component's entry, which is exactly what such a lowering "
        "computes once the drawn class is clamped."
    ),
    example="normal_mix",
    min_spread=66.0,
    mutants=(
        BackendMutant(
            backend="stan",
            rewrites=(
                SourceRewrite(
                    old="target +=log_sum_exp(lps_cls[g_Obs]);",
                    new="target +=lps_cls[g_Obs,1];",
                ),
            ),
        ),
        BackendMutant(
            backend="numpyro",
            rewrites=(
                SourceRewrite(
                    old="jsp.logsumexp(__marg_cls_w+__marg_cls,axis=- 1)",
                    new="(__marg_cls_w+__marg_cls)[...,0]",
                ),
            ),
        ),
        BackendMutant(
            backend="pyro",
            rewrites=(
                SourceRewrite(
                    old="torch.logsumexp(__marg_cls_w+__marg_cls,dim=- 1)",
                    new="(__marg_cls_w+__marg_cls)[...,0]",
                ),
            ),
        ),
    ),
)


class EnumerationMarker(dx.Model):
    """A target that lowers `marginalize` to an explicit enumeration,
    paired with the token that identifies it in emitted source."""

    backend: str
    marker: str


MARGINALIZE_ENUMERATION_MARKERS: tuple[EnumerationMarker, ...] = (
    EnumerationMarker(backend="stan", marker="log_sum_exp"),
    EnumerationMarker(backend="numpyro", marker="logsumexp"),
    EnumerationMarker(backend="pyro", marker="logsumexp"),
)
"""The targets whose renderers lower `marginalize` to an explicit
enumeration, and the exhaustive list of targets the marginalize
mutation can be measured on.

Every other target is excluded for a stated reason rather than
oversight. PyMC, Turing, Gen, and Edward2 lower the block to a
sampled discrete latent, so their probes score a conditional joint
rather than the marginal and there is no correct baseline to mutate
away from. BUGS raises `UnsupportedConstruct` on the fixture's
broadcast concentration. WebPPL does emit an enumeration, but its
runtime rejects the fixture's Dirichlet concentration before any
scoring happens, so neither the original nor the mutant reaches a
log-density.
"""


# ---------------------------------------------------------------------
# Rewrites the check must accept, and rewrites it provably misses.
# ---------------------------------------------------------------------

ACCEPTED: tuple[AcceptedRewrite, ...] = (
    AcceptedRewrite(
        name="constant_offset_added",
        example="ar1",
        backend="stan",
        rationale=(
            "A point-independent constant added to the log-density "
            "denotes the same measure. Theorem 4.1 quotients by "
            "exactly this constant, so the check has to accept it; a "
            "check that rejected it would be asserting equality "
            "rather than measure equivalence and would fail on every "
            "genuine base-measure difference between targets."
        ),
        rewrites=(
            SourceRewrite(
                old="} model {",
                new="} model {\n  target += 3.7;",
            ),
        ),
    ),
    AcceptedRewrite(
        name="constant_offset_added",
        example="ar1",
        backend="numpyro",
        rationale=(
            "Same constant-offset control on a target whose "
            "log-weight arrives through a factor site rather than a "
            "`target` accumulator."
        ),
        rewrites=(
            SourceRewrite(
                old="  mu = alpha+(phi*y_prev)",
                new='  numpyro.factor("c" ,3.7)\n  mu = alpha+(phi*y_prev)',
            ),
        ),
    ),
)


BLIND_SPOTS: tuple[BlindSpot, ...] = (
    BlindSpot(
        name="support_constraint_erased",
        example="ar1",
        backend="stan",
        why_invisible=(
            "Every evaluation point is in support by construction, "
            "and the Stan probe reads the constrained-space density "
            "with `jacobian=False`, so erasing `<lower = 0>` from a "
            "scale parameter's declaration leaves the measured "
            "log-density identical at every point. The emitted "
            "program is still wrong: its sampler is free to leave the "
            "support. Catching this needs a declaration-level check, "
            "not a density comparison."
        ),
        rewrites=(
            SourceRewrite(
                old="real <lower = 0> sigma;",
                new="real sigma;",
            ),
        ),
    ),
    BlindSpot(
        name="truncation_erased",
        example="ar1",
        backend="turing",
        why_invisible=(
            "Dropping the truncation from a half-Cauchy prior changes "
            "the density by `log 2`, a point-independent constant, "
            "which Theorem 4.1's quotient absorbs. The support widens "
            "from the positive half-line to the whole line, and no "
            "point the harness evaluates sits in the difference, so "
            "the only observable trace of the change is a constant "
            "the contract is defined to ignore."
        ),
        rewrites=(
            SourceRewrite(
                old="truncated(Cauchy(0, 1) , 0, Inf)",
                new="Cauchy(0, 1)",
            ),
        ),
    ),
    BlindSpot(
        name="exported_value_negated",
        example="ar1",
        backend="stan",
        why_invisible=(
            "The program's exported value is not part of its joint "
            "density, so negating it moves nothing the check reads. "
            "Every backend shares this blind spot: the equivalence "
            "tier validates the measure a program denotes and says "
            "nothing about what the program returns."
        ),
        rewrites=(
            SourceRewrite(
                old="real <lower = -1 , upper = 1> phi_value = phi;",
                new="real <lower = -1 , upper = 1> phi_value = -phi;",
            ),
        ),
    ),
    BlindSpot(
        name="exported_value_negated",
        example="ar1",
        backend="numpyro",
        why_invisible=(
            "The same blind spot on a target whose exported value is "
            "the model function's return rather than a generated "
            "quantity."
        ),
        rewrites=(SourceRewrite(old="  return phi", new="  return -phi"),),
    ),
)


def mutation_by_name(name: str) -> Mutation:
    """Look a catalogue entry up by its `name`."""
    for mutation in (*CATALOGUE, MARGINALIZE_MUTATION):
        if mutation.name == name:
            return mutation
    raise KeyError(
        f"no mutation named {name!r} in the catalogue; available: "
        f"{[m.name for m in (*CATALOGUE, MARGINALIZE_MUTATION)]}"
    )


def rewrites_for(mutation: Mutation, backend: str) -> tuple[SourceRewrite, ...]:
    """The rewrite sequence `mutation` defines for `backend`."""
    for mutant in mutation.mutants:
        if mutant.backend == backend:
            return mutant.rewrites
    raise KeyError(
        f"mutation {mutation.name!r} defines no mutant for backend "
        f"{backend!r}; it covers "
        f"{[m.backend for m in mutation.mutants]}"
    )


def gallery_cells() -> list[tuple[str, str, str]]:
    """Every `(mutation name, example, backend)` triple in the
    gallery-anchored catalogue, in a deterministic order.

    The order is the catalogue's declaration order, then the backend
    order each mutation declares. Nothing here is sampled: a
    randomised subset would make the suite's sensitivity a different
    number on every run, and a mutation that stopped being rejected
    could hide behind a run that never selected it.
    """
    return [
        (mutation.name, mutation.example, mutant.backend)
        for mutation in CATALOGUE
        for mutant in mutation.mutants
    ]


__all__ = [
    "ACCEPTED",
    "BLIND_SPOTS",
    "CATALOGUE",
    "MARGINALIZE_ENUMERATION_MARKERS",
    "MARGINALIZE_MUTATION",
    "MARGINALIZE_POINTS",
    "MARGINALIZE_SOURCE",
    "AcceptedRewrite",
    "BackendMutant",
    "BlindSpot",
    "EnumerationMarker",
    "MixturePoint",
    "Mutation",
    "SourceRewrite",
    "apply_rewrites",
    "gallery_cells",
    "mutation_by_name",
    "rewrites_for",
]
