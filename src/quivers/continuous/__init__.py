"""Continuous morphisms: the hybrid discrete-continuous architecture.

This package extends quivers with continuous distributions, so that
morphisms between continuous measurable spaces sit alongside the finite
tensor infrastructure.

The key abstraction is `ContinuousMorphism`, which defines a conditional
distribution p(y | x) through two operations:

    log_prob(x, y)  evaluate the log density or mass
    rsample(x)      draw reparameterized samples

Composition uses ancestral sampling (exact for discrete intermediates,
Monte Carlo for continuous ones), and the ``>>`` and ``@`` operators
work across discrete and continuous morphisms.

The package re-exports every public name of its modules:

- spaces (`Euclidean`, `Simplex`, `ProductSpace`, ...) and morphisms
  (`ContinuousMorphism`, `SampledComposition`, `FanOutMorphism`, ...);
- conditional families (`ConditionalNormal`, ...), ordered and
  zero-inflated families, normalizing flows, and boundary morphisms;
- the family registry (`FamilySpec`, `register_family`), parameter
  sources, bijectors, and the compositional measure algebra;
- the program builders: [`MonadicProgram`][quivers.continuous.MonadicProgram],
  its step records (`Draw`, `Observe`, `Let`, `Score`), the inline
  distributions the compiler emits (`FixedDistribution`,
  `MixedInlineDistribution`, the `make_fixed_*` factories), and the
  plate steps (`PlateDraw`, `VectorisedObserve`).
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from quivers.continuous import (
    _ordered,
    _zip_hurdle,
    bijectors,
    boundaries,
    deterministic,
    families,
    family_spec,
    flows,
    log_space,
    measure,
    morphisms,
    ordered,
    param_source,
    param_transforms,
    plate,
    program_steps,
    programs,
    scan,
    spaces,
)
from quivers.continuous._ordered import *  # noqa: F403
from quivers.continuous._zip_hurdle import *  # noqa: F403
from quivers.continuous.bijectors import *  # noqa: F403
from quivers.continuous.boundaries import *  # noqa: F403
from quivers.continuous.deterministic import *  # noqa: F403
from quivers.continuous.families import *  # noqa: F403
from quivers.continuous.family_spec import *  # noqa: F403
from quivers.continuous.flows import *  # noqa: F403
from quivers.continuous.log_space import *  # noqa: F403
from quivers.continuous.measure import *  # noqa: F403
from quivers.continuous.morphisms import *  # noqa: F403
from quivers.continuous.ordered import *  # noqa: F403
from quivers.continuous.param_source import *  # noqa: F403
from quivers.continuous.param_transforms import *  # noqa: F403
from quivers.continuous.plate import *  # noqa: F403
from quivers.continuous.program_steps import *  # noqa: F403
from quivers.continuous.programs import *  # noqa: F403
from quivers.continuous.scan import *  # noqa: F403
from quivers.continuous.spaces import *  # noqa: F403
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.program_steps import StepArgument

if TYPE_CHECKING:
    from quivers.continuous.inline import (
        FixedDistribution,
        MixedInlineDistribution,
        get_inline_param_names,
        get_inline_parameterizations,
        make_fixed_bernoulli,
        make_fixed_beta,
        make_fixed_categorical,
        make_fixed_dirichlet,
        make_fixed_exponential,
        make_fixed_gamma,
        make_fixed_halfcauchy,
        make_fixed_halfnormal,
        make_fixed_lkj_cholesky,
        make_fixed_log_gamma,
        make_fixed_logit_beta,
        make_fixed_logitnormal,
        make_fixed_lognormal,
        make_fixed_normal,
        make_fixed_truncated_normal,
        make_fixed_uniform,
        make_inline_distribution,
        reload_inline_registry,
    )

# The inline-distribution builders. Their module reads the DSL's argument
# syntax, and the DSL compiler imports this package, so they bind on first
# access rather than at import.
_INLINE_EXPORTS = (
    "FixedDistribution",
    "MixedInlineDistribution",
    "get_inline_param_names",
    "get_inline_parameterizations",
    "make_fixed_bernoulli",
    "make_fixed_beta",
    "make_fixed_categorical",
    "make_fixed_dirichlet",
    "make_fixed_exponential",
    "make_fixed_gamma",
    "make_fixed_halfcauchy",
    "make_fixed_halfnormal",
    "make_fixed_lkj_cholesky",
    "make_fixed_log_gamma",
    "make_fixed_logit_beta",
    "make_fixed_logitnormal",
    "make_fixed_lognormal",
    "make_fixed_normal",
    "make_fixed_truncated_normal",
    "make_fixed_uniform",
    "make_inline_distribution",
    "reload_inline_registry",
)


type _InlineExport = (
    type[ContinuousMorphism]
    | Callable[
        ...,
        ContinuousMorphism
        | tuple[ContinuousMorphism, tuple[StepArgument, ...] | None]
        | tuple[str, ...]
        | tuple[tuple[str, ...], ...]
        | None,
    ]
)


def __getattr__(name: str) -> _InlineExport:
    """Bind an inline-distribution builder on first access.

    Parameters
    ----------
    name : str
        The attribute requested.

    Returns
    -------
    _InlineExport
        The builder `quivers.continuous.inline` defines under that name.

    Raises
    ------
    AttributeError
        If ``name`` is not an inline-distribution builder.
    """
    if name in _INLINE_EXPORTS:
        value: _InlineExport = getattr(
            importlib.import_module("quivers.continuous.inline"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = []
__all__ += spaces.__all__
__all__ += morphisms.__all__
__all__ += families.__all__
__all__ += ordered.__all__
__all__ += _ordered.__all__
__all__ += _zip_hurdle.__all__
__all__ += log_space.__all__
__all__ += boundaries.__all__
__all__ += flows.__all__
__all__ += scan.__all__
__all__ += deterministic.__all__
__all__ += bijectors.__all__
__all__ += measure.__all__
__all__ += family_spec.__all__
__all__ += param_source.__all__
__all__ += param_transforms.__all__
__all__ += program_steps.__all__
__all__ += programs.__all__
__all__ += [
    "FixedDistribution",
    "MixedInlineDistribution",
    "get_inline_param_names",
    "get_inline_parameterizations",
    "make_fixed_bernoulli",
    "make_fixed_beta",
    "make_fixed_categorical",
    "make_fixed_dirichlet",
    "make_fixed_exponential",
    "make_fixed_gamma",
    "make_fixed_halfcauchy",
    "make_fixed_halfnormal",
    "make_fixed_lkj_cholesky",
    "make_fixed_log_gamma",
    "make_fixed_logit_beta",
    "make_fixed_logitnormal",
    "make_fixed_lognormal",
    "make_fixed_normal",
    "make_fixed_truncated_normal",
    "make_fixed_uniform",
    "make_inline_distribution",
    "reload_inline_registry",
]
__all__ += plate.__all__
