"""Compile-time saturation warnings.

Source-keyed warnings for latent steps whose algebra-specific initialization
recipe differs materially from the default ``Normal(0, 1)`` initialization.
The check implements the two closed-form conditions documented in the public
analysis guide and returns `SaturationWarning` objects with `source_line` and
`source_col` coordinates. It currently diagnoses:

1. **Deep chains in a bounded algebra with no user-set init.**
   ``ProductFuzzyAlgebra`` over a depth-10 chain with a default
   ``Normal(0, 1)`` initial value will saturate noisy-OR at 1 on
   every cell; the gradient through the join goes to zero.
2. **Intermediate-axis blowup with a small algebra unit.**
   ``StandardMVAlgebra`` saturates its bounded sum at 1 when the
   intermediate axis is large; the recipe is ``p ≈ 1 / k`` and the
   warning fires when no init is declared.

The downstream `apply_init_spec` path consumes the same
`ChainShape`, so once the user applies the recommendation
the warning goes away.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.analysis.chain_shape import ChainShape
from quivers.analysis.init_spec import InitSpec, _algebra_init_spec
from quivers.dsl.ast_nodes import Module


class SaturationWarning(dx.Model):
    """One saturation-risk diagnosis at a specific source location.

    Attributes
    ----------
    name : str
        Latent variable name.
    source_line, source_col : int
        Position of the latent declaration in the source.
    algebra_name : str
        Governing algebra at the latent's location.
    depth : int
        Chain depth of the latent (number of stochastic-bind steps
        from the program entry, 1-indexed).
    intermediate_size : int | None
        Inferred plate / shared-axis size.
    init_spec : InitSpec
        The recommended algebra-specific initialization for this latent.
    """

    name: str
    source_line: int = 0
    source_col: int = 0
    algebra_name: str = ""
    depth: int = 0
    intermediate_size: int | None = None
    init_spec: InitSpec | None = None

    def message(self) -> str:
        """One-line human-readable diagnosis suitable for surfacing
        at ``qvr check`` / Compiler warning level."""
        loc = f"line {self.source_line}" if self.source_line else "(unknown source)"
        spec = self.init_spec
        if spec is None:
            return (
                f"{self.name!r} at {loc}: saturation risk under "
                f"{self.algebra_name} at depth {self.depth}; no init "
                "recipe registered for this algebra"
            )
        return (
            f"{self.name!r} at {loc}: {spec.rationale}. Consider declaring "
            f"the latent with the recommended init "
            f"({spec.distribution}, mean={spec.mean:.4g}, std={spec.std:.4g})."
        )


def saturation_warnings(module: Module) -> tuple[SaturationWarning, ...]:
    """Per-latent saturation diagnoses.

    Returns a warning for every ``latent`` step in the program at
    depth ≥ 2 or with intermediate size > 1 under an algebra whose
    algebra-specific recipe differs materially from a default
    ``Normal(0, 1)`` init. The threshold for "differs materially"
    is the absolute distance between the recipe's mean and 0
    (Normal default) plus the relative distance between its std
    and 1; we surface a warning when either pulls the
    recommendation more than 20% away from the default.

    Suppressing warnings: declare the latent with an explicit init
    via the DSL surface (when available) or apply the recommended
    init programmatically via
    [`quivers.analysis.init_spec.apply_init_spec`][quivers.analysis.init_spec.apply_init_spec].

    Parameters
    ----------
    module : Module
        The parsed QVR module whose program is analysed.

    Returns
    -------
    tuple[SaturationWarning, ...]
        One warning per latent whose recommended init departs from the
        default; empty when the module declares no governing algebra.
    """
    shape = ChainShape.from_module(module)
    algebra = shape.algebra
    if algebra is None:
        return ()
    out: list[SaturationWarning] = []
    for step in shape.steps:
        if step.kind != "latent":
            continue
        size = step.intermediate_size or 1
        if step.depth < 2 and size <= 1:
            continue
        spec = _algebra_init_spec(algebra, step.depth, size)
        if not _spec_differs_from_default(spec):
            continue
        out.append(
            SaturationWarning(
                name=step.name,
                source_line=step.source_line,
                source_col=step.source_col,
                algebra_name=step.algebra_name,
                depth=step.depth,
                intermediate_size=step.intermediate_size,
                init_spec=spec,
            )
        )
    return tuple(out)


def _spec_differs_from_default(spec: InitSpec) -> bool:
    """Whether the recipe disagrees materially with a
    ``Normal(0, 1)`` default. Conservative threshold: 20% relative
    in either location or scale.
    """
    if spec.distribution == "constant":
        return abs(spec.mean) > 0.2
    if abs(spec.mean) > 0.2:
        return True
    if abs(spec.std - 1.0) > 0.2:
        return True
    return False


__all__ = ["SaturationWarning", "saturation_warnings"]
