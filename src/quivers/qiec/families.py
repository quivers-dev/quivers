"""The semantic registry of distribution families.

One record per family says what a distribution *is*, independently of any
host library: its parameters with their constraints and ranks, the type and
event rank of a sample, its support, whether it is discrete, whether a
sample is reparameterizable, and the spelling each transpile target uses.
The compiler, the reference runtime, and every renderer read this table;
host-specific metadata such as the ``torch.distributions`` class is kept
beside it in the layers that need it and checked against it in tests, so a
family cannot mean one thing to the checker and another to a backend.

Four families form the measure algebra a program's operator forms lower
to. ``Restrict(base, low, high)`` is the restriction of a base to an
interval, a sub-probability measure whose density is the base's inside the
interval and whose mass is the base's mass there; ``Mixture(weights,
component)`` is the weighted sum of measures, with raw weights, whose mass
is the weighted sum of the components' masses and whose ``component`` is
either one sampleable plated over the mixture axis or a tensor of
sampleables, one per component; ``Normalize(base)`` rescales a measure to
a probability measure; ``PointMass(value)`` is the Dirac measure. A
``sample`` or ``observe`` step normalizes the measure it draws from, so
``Restrict`` at a step is the truncated family and a mixture of restricted
components weighs each by its mass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quivers.qiec.identifiers import DistributionId
from quivers.qiec.types import BOOL, INT, REAL, TypeExpr

type ParameterConstraint = str
"""A symbolic constraint on a parameter value, such as ``"positive"``,
``"unit_interval"``, ``"simplex"``, ``"positive_definite"``, or
``"dependent"`` when the admissible values depend on other parameters. A
compositional family's parameters use ``"sampleable"`` for a distribution
it is built from and ``"transform"`` for a named bijector chain; an atom's
``"numeric"`` value is an integer or a real, and fixes which the atom
samples."""

type SupportKind = str
"""A symbolic description of where samples live, in the same vocabulary as
the parameter constraints."""


@dataclass(frozen=True, slots=True)
class FamilyParameter:
    """One parameter of a distribution family.

    Parameters
    ----------
    name
        The parameter's name, as the source spells it by keyword.
    constraint
        The symbolic constraint on admissible values.
    rank
        The tensor rank of one parameter value: 0 for a scalar, 1 for a
        vector, 2 for a matrix.
    event_dims
        For each of the value's own dimensions, the dimension of the
        family's sample it has the extent of, or ``None`` when the
        sample's shape does not fix it; empty when no dimension is tied
        to the sample. A multivariate normal's ``scale_tril`` is
        ``(0, 0)``: a square matrix over the sample's one axis.
    """

    name: str
    constraint: ParameterConstraint
    rank: int
    event_dims: tuple[int | None, ...] = ()


@dataclass(frozen=True, slots=True)
class DistributionFamily:
    """The semantic record of one distribution family.

    Parameters
    ----------
    name
        The family's source name, such as ``"Normal"``.
    parameters
        The parameters, in the order a positional application supplies
        them. Alternative parameterizations, such as ``probs`` and
        ``logits``, are both listed; an application supplies one.
    support
        Where samples live.
    element
        The scalar type of one sample element.
    event_rank
        The rank of one sample: 0 for a scalar family, 1 for a vector
        family, 2 for a matrix family.
    event_sources
        The parameters whose trailing ``event_rank`` dimensions fix the
        sample's event shape, one per alternative parameterization (an
        application supplies at most one of them), or empty when the
        family is scalar or the shape comes from elsewhere.
    discrete
        Whether samples are discrete.
    reparameterizable
        Whether a sample can be drawn as a differentiable function of the
        parameters.
    compositional
        Whether the family is built from other distributions or
        morphisms rather than from tensor parameters alone.
    targets
        The spelling each transpile target uses for the family.
    finite_support
        The finite set of values a sample ranges over, when the family
        has one independent of its parameters.
    relaxes
        The discrete family a continuous relaxation stands in for, and
        the atoms of that family's support as the relaxation's samples
        spell them: a marginalization over the relaxation enumerates
        those atoms under the discrete family's masses, as the torch
        runtime does.
    """

    name: str
    parameters: tuple[FamilyParameter, ...]
    support: SupportKind
    element: TypeExpr
    event_rank: int
    event_sources: tuple[str, ...]
    discrete: bool
    reparameterizable: bool
    compositional: bool
    targets: Mapping[str, str] = field(default_factory=dict)
    finite_support: tuple[object, ...] | None = None
    relaxes: tuple[str, tuple[object, ...]] | None = None

    def __post_init__(self) -> None:
        """Freeze the target spellings and check the parameter names.

        Raises
        ------
        ValueError
            If two parameters share a name, an event source names no
            parameter or has a rank below the event rank, or a
            parameter ties a dimension to one the sample does not have.
        """
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"family {self.name!r} repeats a parameter name")
        for source in self.event_sources:
            if source not in names:
                raise ValueError(
                    f"family {self.name!r} names event source "
                    f"{source!r}, which is not a parameter"
                )
            if self.parameters[names.index(source)].rank < self.event_rank:
                raise ValueError(
                    f"family {self.name!r} names event source {source!r}, of "
                    f"rank below the family's event rank {self.event_rank}"
                )
        for parameter in self.parameters:
            dims = parameter.event_dims
            if dims and (
                len(dims) != parameter.rank
                or any(
                    dim is not None and not 0 <= dim < self.event_rank for dim in dims
                )
            ):
                raise ValueError(
                    f"family {self.name!r} ties parameter {parameter.name!r} "
                    f"to sample dimensions {dims!r}, which do not fit its rank "
                    f"{parameter.rank} and the event rank {self.event_rank}"
                )

    @property
    def id(self) -> DistributionId:
        """The family's stable identity.

        Returns
        -------
        DistributionId
            An identity derived from the ``builtin`` namespace and the
            name, the same in every module.
        """
        return DistributionId.derive("builtin", self.name)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """The parameter names in positional order.

        Returns
        -------
        tuple[str, ...]
            The names.
        """
        return tuple(parameter.name for parameter in self.parameters)

    def parameter(self, name: str) -> FamilyParameter:
        """Look up a parameter by name.

        Parameters
        ----------
        name : str
            The parameter's name.

        Returns
        -------
        FamilyParameter
            The parameter.

        Raises
        ------
        KeyError
            If the family has no such parameter.
        """
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"family {self.name!r} has no parameter {name!r}")

    def event_argument[V](
        self, arguments: Iterable[tuple[str, V]]
    ) -> tuple[str, V] | None:
        """The supplied argument that fixes the sample's event shape.

        Parameters
        ----------
        arguments : Iterable[tuple[str, V]]
            An application's named arguments.

        Returns
        -------
        tuple[str, V] | None
            The first argument filling one of `event_sources`, as its
            name and value, or ``None`` when the application supplies
            none of them.
        """
        for name, value in arguments:
            if name in self.event_sources:
                return (name, value)
        return None


_FAMILIES: tuple[DistributionFamily, ...] = (
    DistributionFamily(
        "Normal",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dnorm",
            "church": "gaussian",
            "edward2": "Normal",
            "gen": "normal",
            "jags": "dnorm",
            "numpyro": "Normal",
            "pymc": "Normal",
            "pyro": "Normal",
            "stan": "normal",
            "turing": "Normal",
            "webppl": "Gaussian",
        },
    ),
    DistributionFamily(
        "LogitNormal",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "LogitNormal",
            "numpyro": "LogitNormal",
            "pymc": "LogitNormal",
            "pyro": "LogitNormal",
            "stan": "logit_normal",
        },
    ),
    DistributionFamily(
        "Beta",
        (
            FamilyParameter("concentration1", "positive", 0),
            FamilyParameter("concentration0", "positive", 0),
        ),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dbeta",
            "church": "beta",
            "edward2": "Beta",
            "gen": "beta",
            "jags": "dbeta",
            "numpyro": "Beta",
            "pymc": "Beta",
            "pyro": "Beta",
            "stan": "beta",
            "turing": "Beta",
            "webppl": "Beta",
        },
    ),
    DistributionFamily(
        "TruncatedNormal",
        (
            FamilyParameter("loc", "real", 0),
            FamilyParameter("scale", "positive", 0),
            FamilyParameter("low", "real", 0),
            FamilyParameter("high", "real", 0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dnorm",
            "edward2": "TruncatedNormal",
            "gen": "truncated_normal",
            "jags": "dnorm",
            "numpyro": "TruncatedNormal",
            "pymc": "TruncatedNormal",
            "pyro": "TruncatedNormal",
            "stan": "normal",
            "turing": "truncated",
        },
    ),
    DistributionFamily(
        "Dirichlet",
        (FamilyParameter("concentration", "positive", 1, (0,)),),
        support="simplex",
        element=REAL,
        event_rank=1,
        event_sources=("concentration",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "ddirch",
            "church": "dirichlet",
            "edward2": "Dirichlet",
            "gen": "dirichlet",
            "jags": "ddirich",
            "numpyro": "Dirichlet",
            "pymc": "Dirichlet",
            "pyro": "Dirichlet",
            "stan": "dirichlet",
            "turing": "Dirichlet",
            "webppl": "Dirichlet",
        },
    ),
    DistributionFamily(
        "Cauchy",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dt",
            "church": "cauchy",
            "edward2": "Cauchy",
            "gen": "cauchy",
            "jags": "dt",
            "numpyro": "Cauchy",
            "pymc": "Cauchy",
            "pyro": "Cauchy",
            "stan": "cauchy",
            "turing": "Cauchy",
            "webppl": "Cauchy",
        },
    ),
    DistributionFamily(
        "Laplace",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "ddexp",
            "edward2": "Laplace",
            "gen": "laplace",
            "jags": "ddexp",
            "numpyro": "Laplace",
            "pymc": "Laplace",
            "pyro": "Laplace",
            "stan": "double_exponential",
            "turing": "Laplace",
            "webppl": "Laplace",
        },
    ),
    DistributionFamily(
        "Gumbel",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "Gumbel",
            "numpyro": "Gumbel",
            "pymc": "Gumbel",
            "pyro": "Gumbel",
            "stan": "gumbel",
            "turing": "Gumbel",
        },
    ),
    DistributionFamily(
        "LogNormal",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="positive",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dlnorm",
            "church": "lognormal",
            "edward2": "LogNormal",
            "gen": "lognormal",
            "jags": "dlnorm",
            "numpyro": "LogNormal",
            "pymc": "LogNormal",
            "pyro": "LogNormal",
            "stan": "lognormal",
            "turing": "LogNormal",
            "webppl": "LogNormal",
        },
    ),
    DistributionFamily(
        "StudentT",
        (
            FamilyParameter("df", "positive", 0),
            FamilyParameter("loc", "real", 0),
            FamilyParameter("scale", "positive", 0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dt",
            "church": "student-t",
            "edward2": "StudentT",
            "jags": "dt",
            "numpyro": "StudentT",
            "pymc": "StudentT",
            "pyro": "StudentT",
            "stan": "student_t",
            "turing": "TDist",
            "webppl": "StudentT",
        },
    ),
    DistributionFamily(
        "Exponential",
        (FamilyParameter("rate", "positive", 0),),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dexp",
            "church": "exponential",
            "edward2": "Exponential",
            "gen": "exponential",
            "jags": "dexp",
            "numpyro": "Exponential",
            "pymc": "Exponential",
            "pyro": "Exponential",
            "stan": "exponential",
            "turing": "Exponential",
            "webppl": "Exponential",
        },
    ),
    DistributionFamily(
        "Gamma",
        (
            FamilyParameter("concentration", "positive", 0),
            FamilyParameter("rate", "positive", 0),
        ),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dgamma",
            "church": "gamma",
            "edward2": "Gamma",
            "gen": "gamma",
            "jags": "dgamma",
            "numpyro": "Gamma",
            "pymc": "Gamma",
            "pyro": "Gamma",
            "stan": "gamma",
            "turing": "Gamma",
            "webppl": "Gamma",
        },
    ),
    DistributionFamily(
        "Chi2",
        (FamilyParameter("df", "positive", 0),),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dchisqr",
            "edward2": "Chi2",
            "jags": "dchisqr",
            "numpyro": "Chi2",
            "pymc": "ChiSquared",
            "pyro": "Chi2",
            "stan": "chi_square",
            "turing": "Chisq",
        },
    ),
    DistributionFamily(
        "HalfCauchy",
        (FamilyParameter("scale", "positive", 0),),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dt",
            "church": "cauchy",
            "edward2": "HalfCauchy",
            "gen": "cauchy",
            "jags": "dt",
            "numpyro": "HalfCauchy",
            "pymc": "HalfCauchy",
            "pyro": "HalfCauchy",
            "stan": "cauchy",
            "turing": "truncated",
            "webppl": "Cauchy",
        },
    ),
    DistributionFamily(
        "HalfNormal",
        (FamilyParameter("scale", "positive", 0),),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dnorm",
            "church": "gaussian",
            "edward2": "HalfNormal",
            "gen": "normal",
            "jags": "dnorm",
            "numpyro": "HalfNormal",
            "pymc": "HalfNormal",
            "pyro": "HalfNormal",
            "stan": "normal",
            "turing": "truncated",
            "webppl": "Gaussian",
        },
    ),
    DistributionFamily(
        "InverseGamma",
        (
            FamilyParameter("concentration", "positive", 0),
            FamilyParameter("rate", "positive", 0),
        ),
        support="positive",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "InverseGamma",
            "numpyro": "InverseGamma",
            "pymc": "InverseGamma",
            "pyro": "InverseGamma",
            "stan": "inv_gamma",
            "turing": "InverseGamma",
        },
    ),
    DistributionFamily(
        "Weibull",
        (
            FamilyParameter("scale", "positive", 0),
            FamilyParameter("concentration", "positive", 0),
        ),
        support="positive",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dweib",
            "church": "weibull",
            "edward2": "Weibull",
            "gen": "weibull",
            "jags": "dweib",
            "numpyro": "Weibull",
            "pymc": "Weibull",
            "pyro": "Weibull",
            "stan": "weibull",
            "turing": "Weibull",
            "webppl": "Weibull",
        },
    ),
    DistributionFamily(
        "Pareto",
        (
            FamilyParameter("scale", "positive", 0),
            FamilyParameter("alpha", "positive", 0),
        ),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dpar",
            "church": "pareto",
            "edward2": "Pareto",
            "jags": "dpar",
            "numpyro": "Pareto",
            "pymc": "Pareto",
            "pyro": "Pareto",
            "stan": "pareto",
            "turing": "Pareto",
        },
    ),
    DistributionFamily(
        "Kumaraswamy",
        (
            FamilyParameter("concentration1", "positive", 0),
            FamilyParameter("concentration0", "positive", 0),
        ),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "Kumaraswamy",
            "gen": "kumaraswamy",
            "numpyro": "Kumaraswamy",
            "pymc": "Kumaraswamy",
            "pyro": "Kumaraswamy",
            "stan": "kumaraswamy",
            "turing": "Kumaraswamy",
            "webppl": "Kumaraswamy",
        },
    ),
    DistributionFamily(
        "ContinuousBernoulli",
        (
            FamilyParameter("probs", "unit_interval", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "ContinuousBernoulli",
            "gen": "continuous_bernoulli",
            "numpyro": "ContinuousBernoulli",
            "pymc": "ContinuousBernoulli",
            "pyro": "ContinuousBernoulli",
            "stan": "continuous_bernoulli",
            "turing": "ContinuousBernoulli",
            "webppl": "ContinuousBernoulli",
        },
        relaxes=("Bernoulli", (0.0, 1.0)),
    ),
    DistributionFamily(
        "FisherSnedecor",
        (FamilyParameter("df1", "positive", 0), FamilyParameter("df2", "positive", 0)),
        support="positive",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "jags": "df",
            "numpyro": "FisherSnedecor",
            "pyro": "FisherSnedecor",
            "turing": "FDist",
        },
    ),
    DistributionFamily(
        "Uniform",
        (FamilyParameter("low", "real", 0), FamilyParameter("high", "real", 0)),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dunif",
            "church": "uniform",
            "edward2": "Uniform",
            "gen": "uniform",
            "jags": "dunif",
            "numpyro": "Uniform",
            "pymc": "Uniform",
            "pyro": "Uniform",
            "stan": "uniform",
            "turing": "Uniform",
            "webppl": "Uniform",
        },
    ),
    DistributionFamily(
        "MultivariateNormal",
        (
            FamilyParameter("loc", "real", 1, (0,)),
            FamilyParameter("covariance_matrix", "positive_definite", 2, (0, 0)),
            FamilyParameter("precision_matrix", "positive_definite", 2, (0, 0)),
            FamilyParameter("scale_tril", "lower_cholesky", 2, (0, 0)),
        ),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=("loc",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dmnorm",
            "church": "multivariate-gaussian",
            "edward2": "MultivariateNormalFullCovariance",
            "gen": "mvnormal",
            "jags": "dmnorm",
            "numpyro": "MultivariateNormal",
            "pymc": "MvNormal",
            "pyro": "MultivariateNormal",
            "stan": "multi_normal",
            "turing": "MvNormal",
            "webppl": "MultivariateGaussian",
        },
    ),
    DistributionFamily(
        "LowRankMVN",
        (
            FamilyParameter("loc", "real", 1, (0,)),
            FamilyParameter("cov_factor", "real", 2, (0, None)),
            FamilyParameter("cov_diag", "positive", 1, (0,)),
        ),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=("loc",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "numpyro": "LowRankMultivariateNormal",
            "pyro": "LowRankMultivariateNormal",
        },
    ),
    DistributionFamily(
        "RelaxedBernoulli",
        (
            FamilyParameter("temperature", "positive", 0),
            FamilyParameter("probs", "unit_interval", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "RelaxedBernoulli",
            "numpyro": "RelaxedBernoulli",
            "pyro": "RelaxedBernoulli",
        },
        relaxes=("Bernoulli", (0.0, 1.0)),
    ),
    DistributionFamily(
        "RelaxedOneHotCategorical",
        (
            FamilyParameter("temperature", "positive", 0),
            FamilyParameter("probs", "simplex", 1, (0,)),
            FamilyParameter("logits", "real", 1, (0,)),
        ),
        support="simplex",
        element=REAL,
        event_rank=1,
        event_sources=("probs", "logits"),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "edward2": "RelaxedOneHotCategorical",
            "pyro": "RelaxedOneHotCategorical",
        },
    ),
    DistributionFamily(
        "Wishart",
        (
            FamilyParameter("df", "positive", 0),
            FamilyParameter("covariance_matrix", "positive_definite", 2, (0, 1)),
            FamilyParameter("precision_matrix", "positive_definite", 2, (0, 1)),
            FamilyParameter("scale_tril", "lower_cholesky", 2, (0, 1)),
        ),
        support="positive_definite",
        element=REAL,
        event_rank=2,
        event_sources=("covariance_matrix", "precision_matrix", "scale_tril"),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dwish",
            "jags": "dwish",
            "numpyro": "Wishart",
            "pymc": "Wishart",
            "pyro": "Wishart",
            "stan": "wishart",
            "turing": "Wishart",
        },
    ),
    DistributionFamily(
        "InverseWishart",
        (
            FamilyParameter("df", "positive", 0),
            FamilyParameter("scale_tril", "lower_cholesky", 2, (0, 1)),
        ),
        support="positive_definite",
        element=REAL,
        event_rank=2,
        event_sources=("scale_tril",),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={
            "numpyro": "InverseWishart",
            "pyro": "InverseWishart",
            "stan": "inv_wishart",
            "turing": "InverseWishart",
        },
    ),
    DistributionFamily(
        "MatrixNormal",
        (
            FamilyParameter("loc", "real", 2, (0, 1)),
            FamilyParameter("row_covariance", "positive_definite", 2, (0, 0)),
            FamilyParameter("col_covariance", "positive_definite", 2, (1, 1)),
        ),
        support="real",
        element=REAL,
        event_rank=2,
        event_sources=("loc",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "church": "matrix-normal",
            "edward2": "MatrixNormalLinearOperator",
            "gen": "matrix_normal",
            "numpyro": "MatrixNormal",
            "pymc": "MatrixNormal",
            "pyro": "MatrixNormal",
            "stan": "matrix_normal",
            "turing": "MatrixNormal",
            "webppl": "MatrixNormal",
        },
    ),
    DistributionFamily(
        "GP",
        (
            FamilyParameter("mean", "real", 1, (0,)),
            FamilyParameter("kernel", "positive_definite", 2, (0, 0)),
        ),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=True,
        targets={
            "bugs": "dmnorm",
            "church": "multivariate-gaussian",
            "edward2": "MultivariateNormalFullCovariance",
            "gen": "mvnormal",
            "jags": "dmnorm",
            "numpyro": "MultivariateNormal",
            "pymc": "MvNormal",
            "pyro": "MultivariateNormal",
            "stan": "multi_normal",
            "turing": "MvNormal",
            "webppl": "MultivariateGaussian",
        },
    ),
    DistributionFamily(
        "Horseshoe",
        (FamilyParameter("scale", "positive", 0),),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={},
    ),
    DistributionFamily(
        "Bernoulli",
        (
            FamilyParameter("probs", "unit_interval", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="boolean",
        element=BOOL,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        finite_support=(False, True),
        targets={
            "bugs": "dbern",
            "church": "flip",
            "edward2": "Bernoulli",
            "gen": "bernoulli",
            "jags": "dbern",
            "numpyro": "Bernoulli",
            "pymc": "Bernoulli",
            "pyro": "Bernoulli",
            "stan": "bernoulli",
            "turing": "Bernoulli",
            "webppl": "Bernoulli",
        },
    ),
    DistributionFamily(
        "Categorical",
        (FamilyParameter("probs", "simplex", 1), FamilyParameter("logits", "real", 1)),
        support="dependent",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "bugs": "dcat",
            "church": "categorical",
            "edward2": "Categorical",
            "gen": "categorical",
            "jags": "dcat",
            "numpyro": "Categorical",
            "pymc": "Categorical",
            "pyro": "Categorical",
            "stan": "categorical",
            "turing": "Categorical",
            "webppl": "Categorical",
        },
    ),
    DistributionFamily(
        "Poisson",
        (FamilyParameter("rate", "nonnegative", 0),),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "bugs": "dpois",
            "church": "poisson",
            "edward2": "Poisson",
            "gen": "poisson",
            "jags": "dpois",
            "numpyro": "Poisson",
            "pymc": "Poisson",
            "pyro": "Poisson",
            "stan": "poisson",
            "turing": "Poisson",
            "webppl": "Poisson",
        },
    ),
    DistributionFamily(
        "NegativeBinomial",
        (
            FamilyParameter("total_count", "nonnegative", 0),
            FamilyParameter("probs", "unit_interval_half_open", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "bugs": "dnegbin",
            "church": "negative-binomial",
            "edward2": "NegativeBinomial",
            "gen": "neg_binom",
            "jags": "dnegbin",
            "numpyro": "NegativeBinomial2",
            "pymc": "NegativeBinomial",
            "pyro": "NegativeBinomial",
            "stan": "neg_binomial_2",
            "turing": "NegativeBinomial",
            "webppl": "NegativeBinomial",
        },
    ),
    DistributionFamily(
        "Geometric",
        (
            FamilyParameter("probs", "unit_interval", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "church": "geometric",
            "edward2": "Geometric",
            "gen": "geometric",
            "numpyro": "Geometric",
            "pymc": "Geometric",
            "pyro": "Geometric",
            "turing": "Geometric",
        },
    ),
    DistributionFamily(
        "Binomial",
        (
            FamilyParameter("total_count", "nonnegative_integer", 0),
            FamilyParameter("probs", "unit_interval", 0),
            FamilyParameter("logits", "real", 0),
        ),
        support="dependent",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "bugs": "dbin",
            "edward2": "Binomial",
            "jags": "dbin",
            "numpyro": "Binomial",
            "pymc": "Binomial",
            "pyro": "Binomial",
            "stan": "binomial",
            "turing": "Binomial",
            "webppl": "Binomial",
        },
    ),
    DistributionFamily(
        "VonMises",
        (
            FamilyParameter("loc", "real", 0),
            FamilyParameter("concentration", "positive", 0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={
            "edward2": "VonMises",
            "numpyro": "VonMises",
            "pymc": "VonMises",
            "pyro": "VonMises",
            "stan": "von_mises",
            "turing": "VonMises",
        },
    ),
    DistributionFamily(
        "LogisticNormal",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="simplex",
        element=REAL,
        event_rank=1,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "numpyro": "LogisticNormal",
            "pyro": "LogisticNormal",
            "webppl": "LogisticNormal",
        },
    ),
    DistributionFamily(
        "OneHotCategorical",
        (
            FamilyParameter("probs", "simplex", 1, (0,)),
            FamilyParameter("logits", "real", 1, (0,)),
        ),
        support="one_hot",
        element=INT,
        event_rank=1,
        event_sources=("probs", "logits"),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "edward2": "OneHotCategorical",
            "numpyro": "OneHotCategorical",
            "pyro": "OneHotCategorical",
        },
    ),
    DistributionFamily(
        "LKJCholesky",
        (FamilyParameter("concentration", "positive", 0),),
        support="corr_cholesky",
        element=REAL,
        event_rank=2,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={
            "edward2": "LKJ",
            "gen": "lkj_cholesky",
            "numpyro": "LKJCholesky",
            "pymc": "LKJCholesky",
            "pyro": "LKJCorrCholesky",
            "stan": "lkj_corr_cholesky",
            "turing": "LKJCholesky",
            "webppl": "LKJCholesky",
        },
    ),
    DistributionFamily(
        "Mixture",
        (
            FamilyParameter("weights", "simplex", 1),
            FamilyParameter("component", "sampleable", 0),
        ),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={
            "edward2": "MixtureSameFamily",
            "jags": "dpois",
            "numpyro": "MixtureSameFamily",
            "pymc": "Mixture",
            "pyro": "MixtureSameFamily",
            "turing": "MixtureModel",
            "webppl": "Mixture",
        },
    ),
    DistributionFamily(
        "MixtureNormal",
        (
            FamilyParameter("weights", "simplex", 1),
            FamilyParameter("loc", "real", 1),
            FamilyParameter("scale", "positive", 1),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={
            "edward2": "MixtureSameFamily",
            "gen": "HomogeneousMixture",
            "numpyro": "MixtureSameFamily",
            "pymc": "NormalMixture",
            "pyro": "MixtureSameFamily",
            "stan": "log_mix",
            "turing": "MixtureModel",
            "webppl": "Mixture",
        },
    ),
    DistributionFamily(
        "Independent",
        (
            FamilyParameter("base", "sampleable", 0),
            FamilyParameter("reinterpreted_batch_ndims", "nonnegative_integer", 0),
        ),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={
            "edward2": "Independent",
            "numpyro": "Independent",
            "pyro": "Independent",
        },
    ),
    DistributionFamily(
        "Transformed",
        (
            FamilyParameter("base", "sampleable", 0),
            FamilyParameter("transforms", "transform", 0),
        ),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={
            "edward2": "TransformedDistribution",
            "numpyro": "TransformedDistribution",
            "pyro": "TransformedDistribution",
        },
    ),
    DistributionFamily(
        "Truncated",
        (
            FamilyParameter("base", "sampleable", 0),
            FamilyParameter("low", "real", 0),
            FamilyParameter("high", "real", 0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={
            "numpyro": "TruncatedDistribution",
            "pymc": "Truncated",
            "pyro": "TruncatedDistribution",
            "turing": "truncated",
        },
    ),
    DistributionFamily(
        "ZeroInflatedPoisson",
        (
            FamilyParameter("zero_prob", "unit_interval", 0),
            FamilyParameter("rate", "positive", 0),
        ),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={},
    ),
    DistributionFamily(
        "HurdlePoisson",
        (
            FamilyParameter("zero_prob", "unit_interval", 0),
            FamilyParameter("rate", "positive", 0),
        ),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={},
    ),
    DistributionFamily(
        "ZeroOneInflatedBeta",
        (
            FamilyParameter("mu", "unit_interval", 0),
            FamilyParameter("phi", "positive", 0),
            FamilyParameter("zoi", "unit_interval", 0),
            FamilyParameter("coi", "unit_interval", 0),
        ),
        support="unit_interval",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={},
    ),
    DistributionFamily(
        "Restrict",
        (
            FamilyParameter("base", "sampleable", 0),
            FamilyParameter("low", "real", 0),
            FamilyParameter("high", "real", 0),
        ),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={},
    ),
    DistributionFamily(
        "Normalize",
        (FamilyParameter("base", "sampleable", 0),),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=True,
        targets={},
    ),
    DistributionFamily(
        "PointMass",
        (FamilyParameter("value", "numeric", 0),),
        support="dependent",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={},
    ),
    DistributionFamily(
        "LKJCorrelationFactor",
        (FamilyParameter("concentration", "positive", 0),),
        support="positive_definite",
        element=REAL,
        event_rank=2,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
        targets={
            "numpyro": "LKJ",
            "pymc": "LKJCorr",
            "pyro": "LKJ",
            "stan": "lkj_corr",
        },
    ),
    DistributionFamily(
        "BetaBinomial",
        (
            FamilyParameter("total_count", "nonnegative_integer", 0),
            FamilyParameter("concentration1", "positive", 0),
            FamilyParameter("concentration0", "positive", 0),
        ),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "bugs": "dbetabin",
            "edward2": "BetaBinomial",
            "gen": "beta_binomial",
            "jags": "dbetabin",
            "numpyro": "BetaBinomial",
            "pymc": "BetaBinomial",
            "pyro": "BetaBinomial",
            "stan": "beta_binomial",
            "turing": "BetaBinomial",
            "webppl": "BetaBinomial",
        },
    ),
    DistributionFamily(
        "OrderedLogistic",
        (FamilyParameter("eta", "real", 0), FamilyParameter("cutpoints", "real", 1)),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "numpyro": "OrderedLogistic",
            "pymc": "OrderedLogistic",
            "pyro": "OrderedLogistic",
            "stan": "ordered_logistic",
        },
    ),
    DistributionFamily(
        "OrderedProbit",
        (FamilyParameter("eta", "real", 0), FamilyParameter("cutpoints", "real", 1)),
        support="nonnegative_integer",
        element=INT,
        event_rank=0,
        event_sources=(),
        discrete=True,
        reparameterizable=False,
        compositional=False,
        targets={
            "numpyro": "OrderedProbit",
            "pymc": "OrderedProbit",
            "pyro": "OrderedProbit",
            "stan": "ordered_probit",
        },
    ),
    DistributionFamily(
        "Logistic",
        (FamilyParameter("loc", "real", 0), FamilyParameter("scale", "positive", 0)),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dlogis",
            "edward2": "Logistic",
            "gen": "logistic",
            "jags": "dlogis",
            "numpyro": "Logistic",
            "pymc": "Logistic",
            "pyro": "Logistic",
            "stan": "logistic",
            "turing": "Logistic",
            "webppl": "Logistic",
        },
    ),
    DistributionFamily(
        "HalfStudentT",
        (FamilyParameter("df", "positive", 0), FamilyParameter("scale", "positive", 0)),
        support="positive",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "bugs": "dt",
            "edward2": "HalfStudentT",
            "gen": "half_student_t",
            "jags": "dt",
            "numpyro": "HalfStudentT",
            "pymc": "HalfStudentT",
            "pyro": "HalfStudentT",
            "stan": "student_t",
            "turing": "HalfStudentT",
            "webppl": "HalfStudentT",
        },
    ),
)

FAMILIES: Mapping[str, DistributionFamily] = MappingProxyType(
    {item.name: item for item in _FAMILIES}
)
"""Every family by source name.

The mapping is closed: a name absent here is not a distribution family, and
no backend may invent one.
"""


def family(name: str) -> DistributionFamily:
    """Look up a distribution family by source name.

    Parameters
    ----------
    name : str
        The family's source name.

    Returns
    -------
    DistributionFamily
        The family's semantic record.

    Raises
    ------
    KeyError
        If no family has the name.
    """
    try:
        return FAMILIES[name]
    except KeyError as error:
        raise KeyError(f"unknown distribution family {name!r}") from error


__all__ = [
    "DistributionFamily",
    "FAMILIES",
    "FamilyParameter",
    "ParameterConstraint",
    "SupportKind",
    "family",
]
