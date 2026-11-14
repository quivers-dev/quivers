"""The deduction-system surface: model type, abstract primitives,
and the three independent operations on a deduction.

This package gathers everything a user needs for working with a
weighted deduction system in one importable path. The contents
fall into three orthogonal groups:

* **Model type** (`DeductionSystem`): the concrete agenda-based
  chart deduction that compiles from a `deduction` block in the
  QVR DSL. Re-exported from [`quivers.stochastic.agenda`][quivers.stochastic.agenda].
* **Abstract primitives** (`.primitives`):
  `Axiom`, `Deduction`, `Goal`,
  `Schedule`, `DeductiveSystem`: the protocol layer
  the agenda implementation derives from. Most users will not
  need these directly; they exist for custom-deduction subclasses
  and for the inside-algorithm framework
  ([`quivers.stochastic.inside`][quivers.stochastic.inside]).
* **Operations**: three independent surfaces over a
  ``DeductionSystem`` with no overlap in purpose:
    * `.fit`: point-estimate gradient fitting (MAP / MLE);
    * `.bayes`: Bayesian wrapping for NUTS / SVI;
    * `.sample`: exact length-conditional forward
      sampling of yields.

These three live in separate submodules because they answer
different questions (estimate parameters at a point vs. sample a
posterior vs. generate synthetic data) and so that adding more
operations in the future does not bloat any one module.
"""

from __future__ import annotations

import importlib
from types import FunctionType
from typing import TYPE_CHECKING

from quivers.stochastic.agenda import DeductionSystem
from quivers.stochastic.deduction.fit import adam_fit_deduction
from quivers.stochastic.deduction.primitives import (
    Axiom,
    Deduction,
    DeductiveSystem,
    Goal,
    Schedule,
)
from quivers.stochastic.deduction.sample import sample_corpus

if TYPE_CHECKING:
    from quivers.inference.lifts import bayesian_lift_parameters
    from quivers.stochastic.deduction.bayes import nuts_program_from_deduction

_LAZY_EXPORTS: dict[str, str] = {
    "bayesian_lift_parameters": "quivers.inference.lifts",
    "nuts_program_from_deduction": "quivers.stochastic.deduction.bayes",
}


def __getattr__(name: str) -> FunctionType:
    """Load inference operations when their public names are requested.

    The deduction primitives are independent of inference. Their package
    can initialize while the compiler is importing chart parsers.

    Parameters
    ----------
    name : str
        The requested public attribute.

    Returns
    -------
    FunctionType
        The operation from its defining module.

    Raises
    ------
    AttributeError
        If the name is not a deferred inference operation.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: FunctionType = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    # Model type
    "DeductionSystem",
    # Abstract primitives
    "Axiom",
    "Deduction",
    "DeductiveSystem",
    "Goal",
    "Schedule",
    # Operations
    "adam_fit_deduction",
    "bayesian_lift_parameters",
    "nuts_program_from_deduction",
    "sample_corpus",
]
