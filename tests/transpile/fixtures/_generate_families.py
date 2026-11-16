"""Generate ``families/<family>.qvr`` from the live QVR family registry.

Run as a script:
``.venv/bin/python tests/transpile/fixtures/_generate_families.py``.

The generator inspects
[`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry]
at generation time and emits one minimal `.qvr` per source-supported family. Each fixture
uses whichever of the QVR family-application surfaces actually
compiles for that family:

- **Inline-sample** (``sample x <- F(literals)``) for the 13 families
  in [`_FIXED_FACTORIES`][quivers.continuous.inline._FIXED_FACTORIES].
- **Inline-observe with variable arg** for `TruncatedNormal` (its
  inline builder requires at least one variable parameter).
- **Inline-observe with literal args** for scalar response families, so an unsupported
  discrete latent cannot mask the observed family's capability boundary.
- **Morphism kernel** (``morphism k : Obs -> Obs [role=kernel] ~ F``)
  for families that have a `Conditional<F>` class but no inline
  factory.
- **Vector / matrix families** (`Dirichlet`, `MultivariateNormal`,
  `Wishart`, `InverseWishart`, `MatrixNormal`, `LowRankMVN`, `GP`,
  `Categorical`, `RelaxedOneHotCategorical`) get hand-written
  fixtures that declare the event dimension via `object` + `over=`.

Re-running is idempotent: existing fixture content is overwritten
only when the regenerated source differs. The script reports which
families were freshly written, which were unchanged, and which have explicit
source-capability exclusions. Unaccounted registrations fail before any writes.
"""

from __future__ import annotations

import pathlib
import sys

from quivers.dsl.compiler._prelude import _get_family_registry


# Families generated with the inline-sample form. Args are scalar
# literals; values picked to put the prior in a benign region.
_INLINE_SAMPLE: dict[str, str] = {
    "Beta": "concentration1=2.0, concentration0=2.0",
    "Bernoulli": "probs=0.5",
    "Exponential": "rate=1.0",
    "Gamma": "concentration=2.0, rate=1.0",
    "HalfCauchy": "scale=1.0",
    "Horseshoe": "scale=1.5",
    "HalfNormal": "scale=1.0",
    "LogGamma": "concentration=0.5, rate=2.0",
    "LogNormal": "loc=0.0, scale=1.0",
    "LogitBeta": "concentration1=0.5, concentration0=2.0",
    "LogitNormal": "mu=0.0, sigma=1.0",
    "Normal": "loc=0.0, scale=1.0",
    "Uniform": "low=0.0, high=1.0",
}


# Observations exercise their family directly without requiring a
# discrete latent, which would independently prevent Stan compilation.
_INLINE_OBSERVE: dict[str, str] = {
    "Independent": "base=Normal(loc=[0.0, 0.0], scale=1.0), reinterpreted_batch_ndims=1",
    "BernsteinHazard": "breaks=[0.0, 1.0], controls=[[1.0]], rate=1.0",
    "ExtendedBeta": "mu=0.3, phi=4.0, kappa=0.2",
    "BetaBinomial": "total_count=4.0, concentration1=2.0, concentration0=2.0",
    "Binomial": "total_count=4.0, probs=0.5",
    "COMPoisson": "rate=2.0, dispersion=1.3",
    "Geometric": "probs=0.5",
    "HalfStudentT": "df=4.0, scale=1.0",
    "HurdlePoisson": "zero_prob=0.25, rate=2.0",
    "Logistic": "loc=0.0, scale=1.0",
    "NegativeBinomial": "total_count=2.0, probs=0.5",
    "OrderedLogistic": "predictor=0.2, cutpoints=[-1.0, 1.0]",
    "Poisson": "rate=2.0",
    "VonMises": "loc=0.0, concentration=1.0",
    "ZeroInflatedPoisson": "zero_prob=0.25, rate=2.0",
    "ZeroOneInflatedBeta": "mu=0.4, phi=3.0, zoi=0.2, coi=0.3",
    "ZeroInflatedNegativeBinomial": "zero_prob=0.25, total_count=2.0, probs=0.5",
    "ZeroInflatedBinomial": "zero_prob=0.25, total_count=4.0, probs=0.5",
    "ZeroInflatedBetaBinomial": "zero_prob=0.25, total_count=4.0, concentration1=2.0, concentration0=2.0",
    "HurdleNegativeBinomial": "zero_prob=0.25, total_count=2.0, probs=0.5",
    "SkewNormal": "loc=0.2, scale=1.1, skewness=0.7",
    "ShiftedLogNormal": "loc=0.2, scale=0.5, shift=0.1",
    "Frechet": "scale=1.3, concentration=3.0",
    "GeneralizedExtremeValue": "loc=0.2, scale=1.1, concentration=0.1",
    "DiscreteWeibull": "probs=0.4, concentration=1.3",
    "Wiener": "boundary=1.2, ndt=0.1, bias=0.4, drift=0.7",
    "ZeroInflatedAsymmetricLaplace": "zero_prob=0.2, loc=0.1, scale=0.7, quantile=0.35",
}


# Existing curated sources are literal snapshots: regeneration must not
# rewrite a valid measured fixture just to change its presentation.
_SPECIAL_SOURCES: dict[str, str] = {
    "MultivariateStudentT": """object Row : FinSet 3
object Dim : FinSet 2
program model : Row -> Row
    observe v : Row <- MultivariateStudentT(df=5.0, loc=[0.0, 1.0], scale_tril=[[1.0, 0.0], [0.5, 1.0]]) [over=Dim]
    return v
export model
""",
    "AdditiveLogRatioNormal": """object Row : FinSet 3
object Component : FinSet 3
program model : Row -> Row
    observe y : Row <- AdditiveLogRatioNormal(loc=[0.2, -0.1], scale_tril=[[0.7, 0.0], [0.2, 0.9]]) [over=Component]
    return y
export model
""",
    "Multinomial": """object Row : FinSet 3
object Category : FinSet 3
program model : Row -> Row
    observe y : Row <- Multinomial(total_count=5, probs=[0.2, 0.3, 0.5]) [over=Category]
    return y
export model
""",
    "DirichletMultinomial": """object Row : FinSet 3
object Category : FinSet 3
program model : Row -> Row
    observe y : Row <- DirichletMultinomial(total_count=5, concentration=[1.0, 2.0, 3.0]) [over=Category]
    return y
export model
""",
    "AsymmetricLaplace": """object X : Real 1
program model : X -> X
    observe y <- AsymmetricLaplace(loc=0.0, scale=1.0, quantile=0.8)
    return y
export model
""",
    "ExGaussian": """object X : Real 1
program model : X -> X
    observe y <- ExGaussian(loc=0.3, scale=0.7, rate=1.4)
    return y
export model
""",
    "InverseGaussian": """object X : Real 1
program model : X -> X
    observe y <- InverseGaussian(mu=1.3, shape=2.4)
    return y
export model
""",
    "HurdleGamma": """object Row : FinSet 3
program model : Row -> Row
    observe y : Row <- HurdleGamma(zero_prob=0.3, concentration=0.4, rate=1.7)
    return y
export model
""",
    "HurdleLogNormal": """object Row : FinSet 3
program model : Row -> Row
    observe y : Row <- HurdleLogNormal(zero_prob=0.25, loc=-0.2, scale=0.7)
    return y
export model
""",
    "LKJCholesky": """# Matrix event axes are distinct from the response plate.
object Obs : FinSet 3
object Dim : FinSet 2
program lkjcholesky_fixture : Obs -> Obs
    observe y : Obs <- LKJCholesky(concentration=2.0) [over=[Dim, Dim]]
    return y
export lkjcholesky_fixture
""",
    "LogisticNormal": """# A scalar logit produces a two-component simplex event.
object Obs : FinSet 3
object Dim : FinSet 2
program logisticnormal_fixture : Obs -> Obs
    observe y : Obs <- LogisticNormal(loc=0.0, scale=1.0) [over=Dim]
    return y
export logisticnormal_fixture
""",
    "OneHotCategorical": """# The class axis is a vector event, not an observation plate.
object Obs : FinSet 3
object Dim : FinSet 3
program onehotcategorical_fixture : Obs -> Obs
    observe y : Obs <- OneHotCategorical(probs=[0.2, 0.3, 0.5]) [over=Dim]
    return y
export onehotcategorical_fixture
""",
    "MixtureNormal": """# Retain a named component axis and normalized simplex weights.
object Obs : FinSet 3
object Comp : FinSet 2
program mixturenormal_fixture(concentration: Tensor[Real]([Comp])) : Obs -> Obs
    sample weights <- Dirichlet(concentration=concentration) [over=Comp]
    sample loc <- Normal(loc=0.0, scale=1.0) [over=Comp]
    sample scale <- HalfNormal(scale=1.0) [over=Comp]
    observe y : Obs <- MixtureNormal(weights=weights, loc=loc, scale=scale)
    return y
export mixturenormal_fixture
""",
    "Mixture": """# A canonical mixture retains its distribution-valued components.
object Obs : FinSet 3
program mixture_fixture : Obs -> Obs
    observe y : Obs <- Mixture(weights=[0.3, 0.7], components=[Normal(loc=-1.0, scale=0.5), Normal(loc=1.0, scale=0.5)])
    return y
export mixture_fixture
""",
    "Transformed": """# Pushforward is the checked program surface of canonical Transformed.
object Obs : FinSet 3
program transformed_fixture : Obs -> Obs
    observe y : Obs <- Pushforward(base=Normal(loc=0.0, scale=1.0), bijector=Exp)
    return y
export transformed_fixture
""",
}


EXCLUDED_FAMILIES: dict[str, str] = {}


# Families generated with morphism-kernel form (Real codomain).
# These have a `Conditional<F>` class but no inline factory.
_MORPHISM_KERNEL: frozenset[str] = frozenset(
    {
        "Cauchy",
        "Chi2",
        "ContinuousBernoulli",
        "FisherSnedecor",
        "Gumbel",
        "InverseGamma",
        "Kumaraswamy",
        "Laplace",
        "Pareto",
        "RelaxedBernoulli",
        "StudentT",
        "Weibull",
        "GeneralizedPareto",
    }
)


# Families requiring the observe-with-variable form.
_OBSERVE_WITH_VAR: frozenset[str] = frozenset({"TruncatedNormal"})


_GENERATED_HEADER = "# Auto-generated fixture exercising the "


def _inline_sample_source(family: str, args: str) -> str:
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Regenerate via tests/transpile/fixtures/_generate_families.py.\n"
        f"object Obs : FinSet 8\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample theta <- {family}({args})\n"
        f"    return theta\n"
        f"export {family.lower()}_fixture\n"
    )


def _inline_observe_source(family: str, args: str) -> str:
    """Generate an observed scalar-response family fixture.

    Parameters
    ----------
    family : str
        Registered scalar-response family name.
    args : str
        Benign literal arguments in canonical keyword order.

    Returns
    -------
    str
        A complete program that observes the family on response rows.
    """
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Regenerate via tests/transpile/fixtures/_generate_families.py.\n"
        f"object Obs : FinSet 8\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    observe y : Obs <- {family}({args})\n"
        "    return y\n"
        f"export {family.lower()}_fixture\n"
    )


def _observe_with_var_source(family: str) -> str:
    """For TruncatedNormal: ``observe y <- F(mu, ...)`` with a
    variable ``mu`` so the family's inline-with-variable builder
    fires."""
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Uses the inline-with-variable builder (literals-only path is\n"
        f"# unavailable for {family}).\n"
        f"object Obs : FinSet 8\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample mu <- Uniform(low=0.0, high=1.0)\n"
        f"    observe y : Obs <- {family}(mu=mu, sigma=0.2, low=0.0, high=1.0)\n"
        f"    return mu\n"
        f"export {family.lower()}_fixture\n"
    )


def _morphism_kernel_source(family: str) -> str:
    """``morphism k : Obs -> Obs [role=kernel] ~ F(args)``.

    Uses Real codomain since all morphism-kernel-only families take
    values on the full real line or its subset. Carries the
    canonical-args inline on the ``~`` clause so backends that
    require positional family args (WebPPL's `(location, scale)`
    object-literal, Stan's `<lower=0>` constraint detection) have
    them available after morphism resolution.
    """
    args = _MORPHISM_KERNEL_DEFAULT_ARGS.get(family, "")
    args_clause = f"({args})" if args else ""
    return (
        f"{_GENERATED_HEADER}{family} family.\n"
        f"# Family lacks an inline factory; declared via a morphism\n"
        f"# kernel with `~ {family}{args_clause}`.\n"
        f"object Obs : Real 4\n"
        f"morphism {family.lower()}_kernel : Obs -> Obs [role=kernel] ~ {family}{args_clause}\n"
        f"program {family.lower()}_fixture : Obs -> Obs\n"
        f"    sample x <- {family.lower()}_kernel\n"
        f"    return x\n"
        f"export {family.lower()}_fixture\n"
    )


_MORPHISM_KERNEL_DEFAULT_ARGS: dict[str, str] = {
    "Cauchy": "loc=0.0, scale=1.0",
    "Chi2": "df=3.0",
    "ContinuousBernoulli": "probs=0.5",
    "FisherSnedecor": "df1=5.0, df2=5.0",
    "Gumbel": "loc=0.0, scale=1.0",
    "InverseGamma": "concentration=3.0, rate=1.0",
    "Kumaraswamy": "concentration1=2.0, concentration0=2.0",
    "Laplace": "loc=0.0, scale=1.0",
    "Pareto": "scale=1.0, alpha=2.0",
    "RelaxedBernoulli": "temperature=0.5, probs=1.0",
    "StudentT": "df=5.0, loc=0.0, scale=1.0",
    "Weibull": "scale=1.5, concentration=1.0",
    "GeneralizedPareto": "loc=0.0, scale=1.0, concentration=0.5",
}


# Hand-written sources for vector / matrix families. Each one
# declares an event dimension via ``object`` and uses ``[over=Dim]``
# on the family application. Wishart / InverseWishart / MatrixNormal
# / LKJCholesky need TWO event axes; LowRankMVN needs an explicit
# rank parameter.
_VECTOR_SOURCES: dict[str, str] = {
    "Categorical": (
        "# Hand-written Categorical fixture (vector family).\n"
        "#\n"
        "# The observed `cls` site is what carries the family-support claim.\n"
        "# The `marginalize` block below draws on Categorical too, but a\n"
        "# marginalized latent is integrated out rather than declared, so on\n"
        "# every backend that lowers the integral to a weighted logsumexp the\n"
        "# emitted source names the weights (`log(probs)`) and never the family.\n"
        "# Naming the family there would take the *draw* lowering, which\n"
        "# denotes a measure on a strictly larger space than the QVR reference\n"
        "# integrates. The observed site keeps the matrix cell honest without\n"
        "# reaching for it.\n"
        "object Item : FinSet 8\n"
        "object Comp : FinSet 4\n"
        "program categorical_fixture : Item -> Item\n"
        "    sample probs <- Dirichlet(concentration=concentration) [over=Comp]\n"
        "    observe cls : Item <- Categorical(probs=probs)\n"
        "    marginalize z : Comp <- Categorical(probs=probs) [over=Item, reduction=logsumexp]\n"
        "        observe r : Item <- Normal(loc=0.0, scale=1.0) [via=idx]\n"
        "    return probs\n"
        "export categorical_fixture\n"
    ),
    "Dirichlet": (
        "# Hand-written Dirichlet fixture (vector family).\n"
        "object Obs : FinSet 4\n"
        "object Comp : FinSet 3\n"
        "program dirichlet_fixture : Obs -> Obs\n"
        "    sample probs <- Dirichlet(concentration=concentration) [over=Comp]\n"
        "    return probs\n"
        "export dirichlet_fixture\n"
    ),
    "GP": (
        "# Hand-written GP fixture (function-space family).\n"
        "# The native constructor supplies its default RBF covariance.\n"
        "object Idx : FinSet 8\n"
        "object Obs : Real 1\n"
        "morphism gp_kernel : Idx -> Obs [role=kernel] ~ GP\n"
        "program gp_fixture : Idx -> Obs\n"
        "    sample f <- gp_kernel\n"
        "    return f\n"
        "export gp_fixture\n"
    ),
    "InverseWishart": (
        "# Hand-written InverseWishart fixture (matrix family).\n"
        "object Row : FinSet 3\n"
        "object Col : FinSet 3\n"
        "object Obs : Real 9\n"
        "program inversewishart_fixture : Obs -> Obs\n"
        "    sample sigma <- InverseWishart(df=4.0, scale_tril=[[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]) [over=[Row, Col]]\n"
        "    return sigma\n"
        "export inversewishart_fixture\n"
    ),
    "LowRankMVN": (
        "# Hand-written LowRankMVN fixture (vector family with low-rank factor).\n"
        "object Dim : FinSet 5\n"
        "object Obs : Real 5\n"
        "program lowrankmvn_fixture : Obs -> Obs\n"
        "    sample x <- LowRankMVN(loc=[0.0, 0.0, 0.0, 0.0, 0.0], cov_factor=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]], cov_diag=[1.0, 1.0, 1.0, 1.0, 1.0]) [over=Dim]\n"
        "    return x\n"
        "export lowrankmvn_fixture\n"
    ),
    "MatrixNormal": (
        "# Hand-written MatrixNormal fixture (matrix family; Kronecker covariance).\n"
        "# ``over=[Row, Col]`` binds the two event axes to the Kronecker factors.\n"
        "object Row : FinSet 3\n"
        "object Col : FinSet 4\n"
        "morphism mn_kernel : Row * Col -> Row * Col [role=kernel, over=[Row, Col]] ~ MatrixNormal\n"
        "program matrixnormal_fixture : Row * Col -> Row * Col\n"
        "    sample m <- mn_kernel\n"
        "    return m\n"
        "export matrixnormal_fixture\n"
    ),
    "MultivariateNormal": (
        "# Hand-written MultivariateNormal fixture (vector family).\n"
        "object Dim : FinSet 4\n"
        "object Obs : Real 4\n"
        "morphism mvn_kernel : Obs -> Obs [role=kernel] ~ MultivariateNormal\n"
        "program multivariatenormal_fixture : Obs -> Obs\n"
        "    sample x <- mvn_kernel\n"
        "    return x\n"
        "export multivariatenormal_fixture\n"
    ),
    "RelaxedOneHotCategorical": (
        "# Hand-written RelaxedOneHotCategorical fixture "
        "(Gumbel-softmax over a simplex).\n"
        "object Cls : FinSet 4\n"
        "object Obs : Real 4\n"
        "program relaxedonehotcategorical_fixture : Obs -> Obs\n"
        "    sample z <- RelaxedOneHotCategorical(temperature=0.5, probs=[0.0, 0.0, 0.0, 0.0]) [over=Cls]\n"
        "    return z\n"
        "export relaxedonehotcategorical_fixture\n"
    ),
    "Wishart": (
        "# Hand-written Wishart fixture (matrix family).\n"
        "object Dim : FinSet 3\n"
        "object Obs : Real 9\n"
        "program wishart_fixture : Obs -> Obs\n"
        "    sample sigma <- Wishart(df=4.0, scale_tril=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) [over=[Dim, Dim]]\n"
        "    return sigma\n"
        "export wishart_fixture\n"
    ),
}


def fixture_sources() -> dict[str, str]:
    """Account for every registered family before generating any files.

    Returns
    -------
    dict[str, str]
        Canonical source by family, excluding only documented source gaps.

    Raises
    ------
    ValueError
        If a registration has no strategy, strategies overlap, a configured
        family disappeared, or an exclusion lacks its explanation.
    """
    registry = set(_get_family_registry())
    groups = (
        set(_INLINE_SAMPLE),
        set(_INLINE_OBSERVE),
        set(_OBSERVE_WITH_VAR),
        set(_MORPHISM_KERNEL),
        set(_VECTOR_SOURCES),
        set(_SPECIAL_SOURCES),
        set(EXCLUDED_FAMILIES),
    )
    configured: set[str] = set()
    for group in groups:
        overlap = configured & group
        if overlap:
            raise ValueError(
                f"families have overlapping fixture strategies: {sorted(overlap)}"
            )
        configured.update(group)
    if missing := registry - configured:
        raise ValueError(
            f"registered families have no fixture strategy: {sorted(missing)}"
        )
    if stale := configured - registry:
        raise ValueError(
            f"fixture strategies name unregistered families: {sorted(stale)}"
        )
    if any(not reason.strip() for reason in EXCLUDED_FAMILIES.values()):
        raise ValueError(
            "every excluded family requires a source-capability explanation"
        )

    sources = {}
    for family in sorted(registry - EXCLUDED_FAMILIES.keys()):
        if family in _INLINE_SAMPLE:
            source = _inline_sample_source(family, _INLINE_SAMPLE[family])
        elif family in _INLINE_OBSERVE:
            source = _inline_observe_source(family, _INLINE_OBSERVE[family])
        elif family in _OBSERVE_WITH_VAR:
            source = _observe_with_var_source(family)
        elif family in _MORPHISM_KERNEL:
            source = _morphism_kernel_source(family)
        elif family in _VECTOR_SOURCES:
            source = _VECTOR_SOURCES[family]
        else:
            source = _SPECIAL_SOURCES[family]
        sources[family] = source
    return sources


def main(*, output_dir: pathlib.Path | None = None) -> int:
    """Write current fixtures, preserving unchanged files and reporting exclusions.

    Parameters
    ----------
    output_dir : pathlib.Path or None
        Destination, defaulting to the repository's family fixtures.

    Returns
    -------
    int
        Zero after complete accounting, or one before any writes if a family
        lacks an explicit strategy or exclusion reason.
    """
    try:
        sources = fixture_sources()
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    out_dir = output_dir or pathlib.Path(__file__).resolve().parent / "families"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    unchanged: list[str] = []
    for family, source in sources.items():
        path = out_dir / f"{family.lower()}.qvr"
        if path.exists() and path.read_text() == source:
            unchanged.append(family)
            continue
        path.write_text(source)
        written.append(family)

    print(f"wrote {len(written)} fixture(s); {len(unchanged)} unchanged")
    for family, reason in sorted(EXCLUDED_FAMILIES.items()):
        print(f"EXCLUDED {family}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
