# Transpilation correctness

`quivers.transpile.transpile(module, target=...)` translates a module for eleven targets: BUGS,
Church, Edward2, Gen, JAGS, NumPyro, PyMC, Pyro, Stan, Turing, and WebPPL
([Transpilation architecture](../transpile-architecture.md)). A translation is correct when the
emitted program gives the same log joint density as the QVR program at every point, up to a constant
that does not depend on the point. This criterion is tested by evaluating both programs at fixed
parameter values, which works across systems with different random number generators and inference
algorithms.

The target pages use three pieces of notation. $\pi_{F,T}$ maps the parameters of the family $F$ to
the target $T$'s parameterization, such as a scale to a precision. $c_{F,T}$ is a log density
difference that does not depend on the parameters, when there is one. $\Psi_T$ maps the unconstrained
coordinates a target's inference uses to the constrained values of the model, a
[change of variables](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#changes-of-variables)
[@stanReparameterization]. A conversion from scale to precision belongs to $\pi_{F,T}$, and writing a
positive variable as the exponential of an unconstrained one belongs to $\Psi_T$.

## 1. Pipeline {#1-pipeline}

$$
\mathrm{Module}
\xrightarrow{\text{expansion of composite bindings for the target}}
\mathrm{Module}
\xrightarrow{\mathrm{Lower}}
\mathrm{IRProgram}
\xrightarrow{\mathrm{Renderer}_{T}}
\mathrm{Schema}_{T}
\xrightarrow{\mathrm{emit\_pretty}}
\mathrm{bytes}.
$$

`Lower` turns the steps of a program into IR nodes that do not depend on the target, a renderer
builds a panproto schema in one target's grammar, and `emit_pretty` prints the schema. The registry of
renderers also gives each target a level of supported constructs.

The pipeline is deterministic: rendering one module twice gives the same bytes, calling the stages
directly gives the same result as calling their composite, and parsing the emitted source and emitting
it again gives the same source, which the tests in `tests/transpile/test_lens_laws.py` check on their
fixtures.

## 2. Correctness criterion {#2-correctness-criterion}

Adding one constant to every log weight leaves a normalized posterior unchanged, so a target may omit a
normalizing constant that does not depend on the parameters. For a supported module $M$, a target $T$,
parameter points $\theta_1, \ldots, \theta_m$, and data $y$, the numeric tests compute the log joint
densities $\ell_{\mathrm{QVR}}(\theta_i ; y)$ and $\ell_T(\theta_i ; y)$, in the same coordinates, and check
that the differences

$$
\ell_T(\theta_i ; y) - \ell_{\mathrm{QVR}}(\theta_i ; y)
$$

are equal across the points to within a tolerance. A result holds for the tested model, data, and
points. A constant difference points to an omitted normalizing constant, and a difference that changes
from point to point points to a wrong parameterization, support, or reduction. Both densities must also
keep the same latent variables: a density evaluated at a value of a latent variable and a density summed
over its values are different functions.

## 3. Support is explicit

Correctness is claimed only for supported programs. The front end calls `unsupported_for(...)`,
lowering consults `FAMILY_META`, and a renderer raises `UnsupportedConstruct` for a construct its target
lacks, with a diagnostic that names the unsupported family or construct. The matrices of
[supported families and constructs](../../transpile-support.md) record which combinations are supported,
and they, not the target's own library, decide support: a target may have a distribution that QVR does
not map, and QVR may write a family as an equivalent expression instead of a constructor of the same
name.

## 4. Distribution parameterization {#4-distribution-parameterization}

A conversion from scale to precision changes an argument, $\tau = 1/\sigma^2$, and describes the same
Normal family in another parameterization. A change of the variable itself is different. If a positive
variable is written $\theta = \exp(u)$, intervals of equal length in $u$ have different lengths in $\theta$,
and the density in $u$ includes the logarithm of the
[Jacobian](stan.md#unconstrained-space-change-of-variables), $\log|\mathrm{d}\theta / \mathrm{d}u| = u$,
beside the density at $\theta = \exp(u)$ [@stanReparameterization].

For each supported family, a renderer keeps:

1. the meaning and order of the arguments;
2. the event and batch shapes;
3. the support;
4. any Jacobian a change of variables introduces;
5. the way observations and plates are reduced.

`FAMILY_META[F].target_names` and the argument aliases record part of this map, and helpers in the
renderers implement the rest, such as conversions between scale and rate, truncation, and the
reconstruction of matrices. The target pages give the main cases, and the source and the numeric tests
decide the rest.

## 5. Plates, marginalization, and `via` {#5-plates-marginalization-and-via}

A [plated `sample` or `observe`](../programs.md#24-indexed-bind-plate) is a product of factors, one for
each element of a finite set. A renderer may write it as one vectorized statement, a plate context, or a
loop, and it must produce the same product, with the event dimensions of each factor kept joint.

A `marginalize` block must keep the summed density. A renderer may enumerate the values of the latent
variable and add the reduced log weight, or use an equivalent distribution with the variable summed out.
Keeping the variable as a sample site instead describes a joint distribution on a larger space, not the
requested marginal density. BUGS collapses a categorical mixture into one `dcat` and refuses other blocks,
JAGS builds the weighted finite sum and scores it with the zeros trick [@lunn2012bugs], and Edward2 uses a
mixture distribution where the scope has that form and a scored factor otherwise. In a grouped block, the
log likelihoods of the rows of each group are summed, for each value of the latent variable, before that
variable is summed out ([Programs as kernels §2.7](../programs.md#27-grouped-marginalize-with-multi-observe-fibration)).
Summing out a value for each row would give each row its own latent variable.

`via=` names an [index map](../programs.md#27-grouped-marginalize-with-multi-observe-fibration) from
observation rows to groups. Each observation must contribute once to the factor of its group, which
`tests/transpile/test_via_index_map_numeric.py` checks on selected cases.

## 6. Evidence {#6-evidence-tiers}

| Level | Tests | What they check |
|---|---|---|
| 1 | `test_structural.py`, `test_lens_laws.py` | the shape of the emitted schema, determinism, composition of the stages, re-emission |
| 2 | `test_external_syntax.py` | acceptance by the target's parser or compiler, when it is installed |
| 3 | `test_numeric_equivalence.py` | equality of log densities up to a constant in the target's runtime |
| 4 | `test_construct_matrix.py` | which constructs each target supports |
| 5 | `test_family_matrix.py` | which families each target supports |
| 6 | `test_composition_fixtures.py` | larger composite fixtures on every target |

Further numeric tests cover closed-form marginals, `marginalize` blocks, grouping by `via`, and the
programs of the example gallery. The level 3 grid uses seven composite fixtures, five values on each
parameter axis, and at most sixteen points for each fixture, and it states the unsupported cells it
expects.

## 7. Runtimes

Docker images provide fixed versions of the runtimes for Stan, NumPyro, Pyro, PyMC, Edward2, Turing, Gen,
WebPPL, and JAGS, and BUGS syntax is run through JAGS. `tests/transpile/docker/README.md` describes how
they are built and probed. Church has no image, because no maintained Church runtime has the same
interface for evaluating a joint density, and separate tests check selected Church output.

The syntax checks for each target:

- Stan: `stanc --info -`;
- Python targets: Python's `ast` parser;
- Julia targets: `Meta.parse`;
- WebPPL: `node --check`, as JavaScript;
- BUGS and JAGS: compilation by JAGS;
- Church: no external parser, only a check that parentheses balance.

These check the syntax the named tool accepts. Running WebPPL also needs its primitives and its compiler's
transformations, and checking a particular dialect of BUGS needs that implementation.

## 8. Targets

[Stan](stan.md), [NumPyro](numpyro.md), [Pyro](pyro.md), [PyMC](pymc.md), [Edward2](edward2.md),
[Turing](turing.md), [Gen](gen.md), [Church](church.md), [WebPPL](webppl.md), [BUGS](bugs.md), and
[JAGS](jags.md): for each, how it evaluates a log density, the conversions of parameters, the emitted
constructs, and the evidence.

## 9. Extending coverage

Support for a new family or construct needs structural, syntax, and numeric tests together. The numeric
points should reach the boundaries of the support and exercise the converted arguments, with scalar and
plated cases when the family has both, and a continuous change of variables needs a check of its Jacobian.
Gradients and posterior inference need tests of their own. For Church and WebPPL, a full check of the
language would run the emitted program in the intended implementation.
