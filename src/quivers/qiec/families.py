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
from types import MappingProxyType

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import validate_shape
from quivers.qiec.identifiers import DistributionId
from quivers.qiec.terms import LiteralData
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


class FamilyParameter(dx.Model):
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


class DistributionFamily(ValueModel):
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
    """

    name: str
    parameters: tuple[FamilyParameter, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    support: SupportKind
    element: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    event_rank: int
    event_sources: tuple[str, ...]
    discrete: bool
    reparameterizable: bool
    compositional: bool
    targets: Mapping[str, str] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    finite_support: tuple[LiteralData, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_family(self) -> None:
        """Validate semantic fields, parameter names, and event dimensions.

        Raises
        ------
        TypeError
            If a recursive field or target spelling has the wrong type.
        ValueError
            If two parameters share a name, an event source names no
            parameter or has a rank below the event rank, or a
            parameter ties a dimension to one the sample does not have.
        """
        validate_shape(self.parameters, tuple[FamilyParameter, ...], path="parameters")
        validate_shape(self.element, TypeExpr, path="element")
        validate_shape(
            self.finite_support, tuple[LiteralData, ...] | None, path="finite_support"
        )
        if not isinstance(self.targets, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.targets.items()
        ):
            raise TypeError("targets must map target names to family spellings")
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
        name="Normal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
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
        name="LogitNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="Beta",
        parameters=(
            FamilyParameter(name="concentration1", constraint="positive", rank=0),
            FamilyParameter(name="concentration0", constraint="positive", rank=0),
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
        name="TruncatedNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
            FamilyParameter(name="low", constraint="real", rank=0),
            FamilyParameter(name="high", constraint="real", rank=0),
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
        name="Dirichlet",
        parameters=(
            FamilyParameter(
                name="concentration", constraint="positive", rank=1, event_dims=(0,)
            ),
        ),
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
        name="Cauchy",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
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
        name="Laplace",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="Gumbel",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="LogNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="StudentT",
        parameters=(
            FamilyParameter(name="df", constraint="positive", rank=0),
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
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
        name="Exponential",
        parameters=(FamilyParameter(name="rate", constraint="positive", rank=0),),
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
        name="Gamma",
        parameters=(
            FamilyParameter(name="concentration", constraint="positive", rank=0),
            FamilyParameter(name="rate", constraint="positive", rank=0),
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
        name="LogGamma",
        parameters=(
            FamilyParameter(name="concentration", constraint="positive", rank=0),
            FamilyParameter(name="rate", constraint="positive", rank=0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "numpyro": "LogGamma",
            "pyro": "LogGamma",
            "stan": "log_gamma",
        },
    ),
    DistributionFamily(
        name="LogitBeta",
        parameters=(
            FamilyParameter(name="concentration1", constraint="positive", rank=0),
            FamilyParameter(name="concentration0", constraint="positive", rank=0),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "numpyro": "LogitBeta",
            "pyro": "LogitBeta",
            "stan": "logit_beta",
        },
    ),
    DistributionFamily(
        name="Chi2",
        parameters=(FamilyParameter(name="df", constraint="positive", rank=0),),
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
        name="HalfCauchy",
        parameters=(FamilyParameter(name="scale", constraint="positive", rank=0),),
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
        name="HalfNormal",
        parameters=(FamilyParameter(name="scale", constraint="positive", rank=0),),
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
        name="InverseGamma",
        parameters=(
            FamilyParameter(name="concentration", constraint="positive", rank=0),
            FamilyParameter(name="rate", constraint="positive", rank=0),
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
        name="Weibull",
        parameters=(
            FamilyParameter(name="scale", constraint="positive", rank=0),
            FamilyParameter(name="concentration", constraint="positive", rank=0),
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
        name="Pareto",
        parameters=(
            FamilyParameter(name="scale", constraint="positive", rank=0),
            FamilyParameter(name="alpha", constraint="positive", rank=0),
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
        name="Kumaraswamy",
        parameters=(
            FamilyParameter(name="concentration1", constraint="positive", rank=0),
            FamilyParameter(name="concentration0", constraint="positive", rank=0),
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
        name="ContinuousBernoulli",
        parameters=(
            FamilyParameter(name="probs", constraint="unit_interval", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
    ),
    DistributionFamily(
        name="FisherSnedecor",
        parameters=(
            FamilyParameter(name="df1", constraint="positive", rank=0),
            FamilyParameter(name="df2", constraint="positive", rank=0),
        ),
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
        name="Uniform",
        parameters=(
            FamilyParameter(name="low", constraint="real", rank=0),
            FamilyParameter(name="high", constraint="real", rank=0),
        ),
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
        name="MultivariateNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=1, event_dims=(0,)),
            FamilyParameter(
                name="covariance_matrix",
                constraint="positive_definite",
                rank=2,
                event_dims=(0, 0),
            ),
            FamilyParameter(
                name="precision_matrix",
                constraint="positive_definite",
                rank=2,
                event_dims=(0, 0),
            ),
            FamilyParameter(
                name="scale_tril",
                constraint="lower_cholesky",
                rank=2,
                event_dims=(0, 0),
            ),
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
        name="LowRankMVN",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=1, event_dims=(0,)),
            FamilyParameter(
                name="cov_factor", constraint="real", rank=2, event_dims=(0, None)
            ),
            FamilyParameter(
                name="cov_diag", constraint="positive", rank=1, event_dims=(0,)
            ),
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
        name="RelaxedBernoulli",
        parameters=(
            FamilyParameter(name="temperature", constraint="positive", rank=0),
            FamilyParameter(name="probs", constraint="unit_interval", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
    ),
    DistributionFamily(
        name="RelaxedOneHotCategorical",
        parameters=(
            FamilyParameter(name="temperature", constraint="positive", rank=0),
            FamilyParameter(
                name="probs", constraint="simplex", rank=1, event_dims=(0,)
            ),
            FamilyParameter(name="logits", constraint="real", rank=1, event_dims=(0,)),
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
        name="Wishart",
        parameters=(
            FamilyParameter(name="df", constraint="positive", rank=0),
            FamilyParameter(
                name="covariance_matrix",
                constraint="positive_definite",
                rank=2,
                event_dims=(0, 0),
            ),
            FamilyParameter(
                name="precision_matrix",
                constraint="positive_definite",
                rank=2,
                event_dims=(0, 0),
            ),
            FamilyParameter(
                name="scale_tril",
                constraint="lower_cholesky",
                rank=2,
                event_dims=(0, 0),
            ),
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
        name="InverseWishart",
        parameters=(
            FamilyParameter(name="df", constraint="positive", rank=0),
            FamilyParameter(
                name="scale_tril",
                constraint="lower_cholesky",
                rank=2,
                event_dims=(0, 0),
            ),
        ),
        support="positive_definite",
        element=REAL,
        event_rank=2,
        event_sources=("scale_tril",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
        targets={
            "numpyro": "InverseWishart",
            "pyro": "InverseWishart",
            "stan": "inv_wishart",
            "turing": "InverseWishart",
        },
    ),
    DistributionFamily(
        name="MatrixNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=2, event_dims=(0, 1)),
            FamilyParameter(
                name="row_covariance",
                constraint="positive_definite",
                rank=2,
                event_dims=(0, 0),
            ),
            FamilyParameter(
                name="col_covariance",
                constraint="positive_definite",
                rank=2,
                event_dims=(1, 1),
            ),
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
        name="GP",
        parameters=(
            FamilyParameter(name="mean", constraint="real", rank=1, event_dims=(0,)),
            FamilyParameter(
                name="kernel", constraint="positive_definite", rank=2, event_dims=(0, 0)
            ),
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
        name="Horseshoe",
        parameters=(FamilyParameter(name="scale", constraint="positive", rank=0),),
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
        name="Bernoulli",
        parameters=(
            FamilyParameter(name="probs", constraint="unit_interval", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
        name="Categorical",
        parameters=(
            FamilyParameter(name="probs", constraint="simplex", rank=1),
            FamilyParameter(name="logits", constraint="real", rank=1),
        ),
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
        name="Poisson",
        parameters=(FamilyParameter(name="rate", constraint="nonnegative", rank=0),),
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
        name="NegativeBinomial",
        parameters=(
            FamilyParameter(name="total_count", constraint="nonnegative", rank=0),
            FamilyParameter(name="probs", constraint="unit_interval_half_open", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
        name="Geometric",
        parameters=(
            FamilyParameter(name="probs", constraint="unit_interval", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
        name="Binomial",
        parameters=(
            FamilyParameter(
                name="total_count", constraint="nonnegative_integer", rank=0
            ),
            FamilyParameter(name="probs", constraint="unit_interval", rank=0),
            FamilyParameter(name="logits", constraint="real", rank=0),
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
        name="VonMises",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="concentration", constraint="positive", rank=0),
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
        name="LogisticNormal",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="OneHotCategorical",
        parameters=(
            FamilyParameter(
                name="probs", constraint="simplex", rank=1, event_dims=(0,)
            ),
            FamilyParameter(name="logits", constraint="real", rank=1, event_dims=(0,)),
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
        name="LKJCholesky",
        parameters=(
            FamilyParameter(name="concentration", constraint="positive", rank=0),
        ),
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
        name="Mixture",
        parameters=(
            FamilyParameter(name="weights", constraint="simplex", rank=1),
            FamilyParameter(name="component", constraint="sampleable", rank=0),
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
        name="MixtureNormal",
        parameters=(
            FamilyParameter(name="weights", constraint="simplex", rank=1),
            FamilyParameter(name="loc", constraint="real", rank=1),
            FamilyParameter(name="scale", constraint="positive", rank=1),
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
        name="Independent",
        parameters=(
            FamilyParameter(name="base", constraint="sampleable", rank=0),
            FamilyParameter(
                name="reinterpreted_batch_ndims",
                constraint="nonnegative_integer",
                rank=0,
            ),
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
        name="Transformed",
        parameters=(
            FamilyParameter(name="base", constraint="sampleable", rank=0),
            FamilyParameter(name="transforms", constraint="transform", rank=0),
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
        name="Truncated",
        parameters=(
            FamilyParameter(name="base", constraint="sampleable", rank=0),
            FamilyParameter(name="low", constraint="real", rank=0),
            FamilyParameter(name="high", constraint="real", rank=0),
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
        name="ZeroInflatedPoisson",
        parameters=(
            FamilyParameter(name="zero_prob", constraint="unit_interval", rank=0),
            FamilyParameter(name="rate", constraint="nonnegative", rank=0),
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
        name="HurdlePoisson",
        parameters=(
            FamilyParameter(name="zero_prob", constraint="unit_interval", rank=0),
            FamilyParameter(name="rate", constraint="nonnegative", rank=0),
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
        name="ZeroOneInflatedBeta",
        parameters=(
            FamilyParameter(name="mu", constraint="unit_interval_open", rank=0),
            FamilyParameter(name="phi", constraint="positive", rank=0),
            FamilyParameter(name="zoi", constraint="unit_interval", rank=0),
            FamilyParameter(name="coi", constraint="unit_interval", rank=0),
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
        name="Restrict",
        parameters=(
            FamilyParameter(name="base", constraint="sampleable", rank=0),
            FamilyParameter(name="low", constraint="real", rank=0),
            FamilyParameter(name="high", constraint="real", rank=0),
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
        name="Normalize",
        parameters=(FamilyParameter(name="base", constraint="sampleable", rank=0),),
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
        name="PointMass",
        parameters=(FamilyParameter(name="value", constraint="numeric", rank=0),),
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
        name="LKJCorrelationFactor",
        parameters=(
            FamilyParameter(name="concentration", constraint="positive", rank=0),
        ),
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
        name="BetaBinomial",
        parameters=(
            FamilyParameter(
                name="total_count", constraint="nonnegative_integer", rank=0
            ),
            FamilyParameter(name="concentration1", constraint="positive", rank=0),
            FamilyParameter(name="concentration0", constraint="positive", rank=0),
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
        name="OrderedLogistic",
        parameters=(
            FamilyParameter(name="eta", constraint="real", rank=0),
            FamilyParameter(name="cutpoints", constraint="real", rank=1),
        ),
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
        name="OrderedProbit",
        parameters=(
            FamilyParameter(name="eta", constraint="real", rank=0),
            FamilyParameter(name="cutpoints", constraint="real", rank=1),
        ),
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
        name="Logistic",
        parameters=(
            FamilyParameter(name="loc", constraint="real", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
        name="HalfStudentT",
        parameters=(
            FamilyParameter(name="df", constraint="positive", rank=0),
            FamilyParameter(name="scale", constraint="positive", rank=0),
        ),
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
