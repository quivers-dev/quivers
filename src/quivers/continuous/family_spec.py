"""Single source of truth for every parametric distribution family
the quivers stack supports.

A `FamilySpec` carries everything the three call paths need:

* the **conditional** path
  (`quivers.continuous.families._IndependentConditional`)
  consumes ``params`` to know how to transform unbounded MLP outputs
  and how to instantiate a `torch.distributions` object;
* the **fixed inline** path
  ([`quivers.continuous.inline.FixedDistribution`][quivers.continuous.inline.FixedDistribution]) consumes
  ``params`` and ``support`` to construct a distribution from
  literal compile-time floats;
* the **mixed inline** path
  ([`quivers.continuous.inline.MixedInlineDistribution`][quivers.continuous.inline.MixedInlineDistribution])
  consumes the same spec to build a distribution whose parameters
  are stacked runtime tensors.

Every family registers itself exactly once via `register`; the
inline and conditional generators read from the same record. This
is the architectural seam that keeps the family catalog from
duplicating itself across three modules.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Literal

import didactic.api as dx
import torch
import torch.distributions as D
from torch.distributions import constraints as _constraints

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.bijectors import Bijector
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.param_transforms import (
    TRANSFORM_TO_BIJECTOR,
    closed_interval_clamp,
    resolve_inline_clamp,
    resolve_transform,
)

# ---------------------------------------------------------------------------
# Parameter transforms (used by the conditional path to convert raw
# MLP outputs into constraint-respecting values).
#
# The transform vocabulary is bijector-typed: every entry in
# `TRANSFORM_TO_BIJECTOR` is a
# [`Bijector`][quivers.continuous.bijectors.Bijector] exposing
# `forward`, `inverse`, `forward_log_det_jacobian`, and
# `inverse_log_det_jacobian`. String keys are kept as a convenience
# surface (backward-compatible with existing family declarations);
# `ParamSpec.transform` accepts either a string key or a bijector
# instance directly.
# ---------------------------------------------------------------------------


# Historical alias retained so downstream imports that pull
# `_RAW_TRANSFORMS` off this module resolve to a `dict[str,
# Callable[[Tensor], Tensor]]` that routes through the bijector
# forward map.
_RAW_TRANSFORMS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    name: bij.forward for name, bij in TRANSFORM_TO_BIJECTOR.items()
}


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


type ParamKind = Literal["scalar", "vector", "integer"]
"""Per-parameter shape contract on the inline call side.

* ``"scalar"`` — broadcasts across the codomain's last axis
  (one parameter value per per-codomain dimension, as in
  ``Normal(loc, scale)``).
* ``"vector"`` — a single vector parameter for the whole sample
  (used by ``Dirichlet`` whose concentration is the full
  simplex-dimensional vector, not a per-component scalar).
* ``"integer"`` — a count-typed parameter (``total_count`` for
  ``Binomial`` / ``NegativeBinomial`` / ``Multinomial``); the
  fixed-factory accepts a Python ``int``.
"""


def _validate_transform(value: str | Bijector) -> str | Bijector:
    """Validate a `ParamSpec.transform` argument.

    String values must be registered in
    `TRANSFORM_TO_BIJECTOR`;
    `Bijector` instances pass through. Any other type raises.
    """
    if isinstance(value, Bijector):
        return value
    if isinstance(value, str):
        if value not in TRANSFORM_TO_BIJECTOR:
            raise ValueError(
                f"ParamSpec: unknown transform {value!r}. "
                f"Valid transforms: {sorted(TRANSFORM_TO_BIJECTOR)}"
            )
        return value
    raise TypeError(
        "ParamSpec: transform must be a string key or a Bijector; "
        f"got {type(value).__name__}"
    )


class ParamSpec(ValueModel):
    """Spec for a single parameter of a distribution family.

    Parameter specifications compare by identity. Their transforms retain
    live bijector references, and string transforms expose their registry
    keys through ``transform_name``. Field updates use ``with_``.

    Attributes
    ----------
    name : str
        Keyword argument expected by the underlying torch
        ``Distribution`` constructor.
    transform : str | Bijector
        Either a string key registered in
        `TRANSFORM_TO_BIJECTOR`
        or a `Bijector` instance. The bijector's `forward` is
        applied to the raw parameter tensor on the conditional
        path; its full four-primitive interface is available to
        guides and pushforwards.
    kind : ParamKind
        Per-parameter shape contract on the inline call side.
    inline_bounds : tuple[float, float] | None
        A closed interval for inline parameter validation, such as
        the nonnegative domain of a Poisson ``rate``. ``None`` uses
        the transform's domain map, which preserves positive values
        and refuses nonpositive values for a positive parameter.
    """

    name: str
    transform: str | Bijector = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    kind: ParamKind = "scalar"
    inline_bounds: tuple[float, float] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_parameter(self) -> None:
        """Validate the transform and optional interval.

        Raises
        ------
        TypeError
            If the transform or bounds have the wrong runtime type.
        ValueError
            If a string transform is absent from the registry.
        """
        _validate_transform(self.transform)
        if self.inline_bounds is not None and (
            not isinstance(self.inline_bounds, tuple)
            or len(self.inline_bounds) != 2
            or not all(isinstance(bound, (int, float)) for bound in self.inline_bounds)
        ):
            raise TypeError("inline_bounds must be a pair of numbers or None")

    def _local_key(self) -> tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]:
        """Compare parameter specifications by their own identity.

        Returns
        -------
        tuple[tuple[Hashable, ...], tuple[ValueModel, ...]]
            The instance identity with no recursively compared children.
        """
        return (id(self),), ()

    @property
    def bijector(self) -> Bijector:
        """The resolved [`Bijector`][quivers.continuous.bijectors.Bijector]
        for this parameter.

        For a string-typed `transform`, looks the key up in
        `TRANSFORM_TO_BIJECTOR`;
        for a `Bijector`-typed `transform`, returns it unchanged.
        """
        return resolve_transform(self.transform)

    @property
    def raw_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The bijector's `forward` map, exposed as a plain callable.

        For callers that treat the transform as
        `Callable[[Tensor], Tensor]`.
        """
        return self.bijector.forward

    @property
    def inline_clamp(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Domain map for explicit literal or runtime parameters.

        Valid positive values are preserved and nonpositive values
        are refused. A declared `inline_bounds` interval preserves its
        endpoints and refuses values outside that interval.
        """
        if self.inline_bounds is not None:
            low, high = self.inline_bounds
            return closed_interval_clamp(low, high)
        return resolve_inline_clamp(self.transform)

    @property
    def transform_name(self) -> str:
        """A stable string identifier for the transform.

        Returns the registered key when `transform` is a string;
        returns a synthesised name based on the bijector class for
        `Bijector`-typed transforms.
        """
        if isinstance(self.transform, str):
            return self.transform
        return f"<bijector:{type(self.transform).__name__.removesuffix('Bijector')}>"


type OutputKind = Literal[
    "independent",  # per-dim independent, e.g. Normal[d]
    "vector",  # a single vector output (Dirichlet, LogisticNormal)
    "mvn",  # multivariate normal-style joint
    "matrix",  # matrix-valued (Wishart, LKJCholesky)
    "categorical",  # discrete index over a FinSet
    "mixture",  # MixtureSameFamily wrapper
]
"""How the distribution's output relates to the codomain.

The conditional path reads ``output_kind`` to decide whether to use
the generic per-dim ``_IndependentConditional`` or a hand-written
class.  The inline path reads it to pick the matching codomain
inference rule.
"""


class FamilySpec(ValueModel):
    """Single source of truth for a distribution family.

    Runtime classes, constraints, and builder callables retain their
    identities. Parameter specifications retain their identity within
    the ordered parameter tuple.

    Used by:

    * `quivers.continuous.families._IndependentConditional`
      and the standalone ``ConditionalX`` classes for the
      learnable-parameter path;
    * [`quivers.continuous.inline.FixedDistribution`][quivers.continuous.inline.FixedDistribution] and
      [`quivers.continuous.inline.MixedInlineDistribution`][quivers.continuous.inline.MixedInlineDistribution]
      for the DSL inline call surface.
    """

    name: str
    """DSL key. Programs reference this name in ``<- F(args)`` lines."""

    dist_class: type[D.Distribution] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    """The underlying `torch.distributions` class."""

    params: tuple[ParamSpec, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    """Ordered parameter specs, matching the dist's positional / keyword API."""

    support: _constraints.Constraint = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    """Output support, used by variational guides to pick a bijector."""

    discrete: bool = False
    """True for integer / categorical outputs."""

    output_kind: OutputKind = "independent"
    """How the distribution's output relates to the codomain shape."""

    docstring: str = ""
    """Used as the generated ``ConditionalX`` class docstring."""

    # Hand-written overrides — used by families that don't fit the
    # generic generators (TruncatedNormal's bounded inline support,
    # Dirichlet's vector concentration, MultivariateNormal's
    # covariance, etc.). Populated as functions by the registering
    # module if needed.
    fixed_factory_override: Callable[..., ContinuousMorphism] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    """Override for the all-literal ``FixedDistribution`` factory."""

    mixed_builder_override: Callable[[list[torch.Tensor]], D.Distribution] | None = (
        dx.field(default=None, opaque=True, extras=OPAQUE_BY_IDENTITY)
    )
    """Override for the mixed-mode dist builder
    ``(list[Tensor]) -> Distribution``."""

    conditional_class_override: type | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    """Override for the conditional class (used by Categorical,
    Bernoulli, MultivariateNormal, etc.)."""

    @dx.model_validator()
    def _validate_runtime_fields(self) -> None:
        """Require distribution classes, parameter records, and runtime builders.

        Raises
        ------
        TypeError
            If a class, support, parameter tuple, or builder is malformed.
        """
        if not isinstance(self.dist_class, type) or not issubclass(
            self.dist_class, D.Distribution
        ):
            raise TypeError(
                "dist_class must be a torch.distributions.Distribution class"
            )
        if not isinstance(self.params, tuple) or not all(
            isinstance(param, ParamSpec) for param in self.params
        ):
            raise TypeError("params must be a tuple of ParamSpec values")
        if not isinstance(self.support, _constraints.Constraint):
            raise TypeError("support must be a torch distribution constraint")
        for builder in (self.fixed_factory_override, self.mixed_builder_override):
            if builder is not None and not callable(builder):
                raise TypeError("family builders must be callable or None")
        if self.conditional_class_override is not None and not isinstance(
            self.conditional_class_override, type
        ):
            raise TypeError("conditional_class_override must be a class or None")

    @property
    def param_names(self) -> tuple[str, ...]:
        """Return parameter names in constructor order.

        Returns
        -------
        tuple[str, ...]
            Names shared by the inline and conditional constructors.
        """
        return tuple(p.name for p in self.params)


# Global mutable registry.  Populated by `register` at import time.
FAMILY_REGISTRY: dict[str, FamilySpec] = {}
"""Every registered family specification, keyed by its DSL name."""


def register_family(spec: FamilySpec) -> FamilySpec:
    """Register a `FamilySpec` under its name; idempotent on
    re-registration of the same name to support hot-reload during
    development.
    """
    FAMILY_REGISTRY[spec.name] = spec
    return spec


def get_family_spec(name: str) -> FamilySpec | None:
    """Look up a registered family by DSL name; ``None`` if absent."""
    return FAMILY_REGISTRY.get(name)


def family_names() -> tuple[str, ...]:
    """Sorted tuple of every registered family name."""
    return tuple(sorted(FAMILY_REGISTRY))


# ---------------------------------------------------------------------------
# Default builders driven by FamilySpec.  Imported and used by
# inline.py to construct FixedDistribution / MixedInlineDistribution
# without per-family boilerplate.
# ---------------------------------------------------------------------------


def build_torch_distribution(
    spec: FamilySpec,
    params: list[torch.Tensor] | dict[str, torch.Tensor],
) -> D.Distribution:
    """Instantiate the underlying torch distribution from a list /
    dict of parameter tensors that have already been clamped to the
    family's constraints.
    """
    if isinstance(params, dict):
        kwargs = params
    else:
        kwargs = {p.name: t for p, t in zip(spec.params, params)}
    return spec.dist_class(**kwargs)


def clamp_param(spec: FamilySpec, name: str, value: torch.Tensor) -> torch.Tensor:
    """Apply the inline safety clamp for the named parameter of
    ``spec``."""
    for p in spec.params:
        if p.name == name:
            return p.inline_clamp(value)
    raise KeyError(f"family {spec.name!r} has no parameter named {name!r}")


__all__ = [
    "FAMILY_REGISTRY",
    "FamilySpec",
    "OutputKind",
    "ParamKind",
    "ParamSpec",
    "build_torch_distribution",
    "clamp_param",
    "family_names",
    "get_family_spec",
    "register_family",
]
