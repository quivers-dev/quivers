"""A response law that states its own kernel.

A family whose likelihood is not one native observation of a linked location
declares a `KernelLaw`: the law builds the family's `ResponseKernel` and says
which data the kernel reads by name besides the formulas' columns. A named QVR
program is such a law (`quivers.formulas._program_kernel.ProgramKernelSpec`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import didactic.api as dx

from quivers._values import ValueModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch

    from quivers.dsl.ast_nodes import Module
    from quivers.family_catalog import FamilyCatalog
    from quivers.formulas._kernel import KernelInputs, ParameterSource, ResponseKernel
    from quivers.formulas.formula import Formula

__all__: list[str] = []

type KernelDatum = np.ndarray | pd.DataFrame | torch.Tensor | float | int
"""A value the kernel reads by name that is not indexed by the rows."""


class KernelLaw(dx.TaggedUnion, ValueModel, discriminator="kind"):
    """A response law that builds its own kernel."""

    @property
    def error_scale_argument(self) -> str | None:
        """Return the native observation argument giving its error scale.

        Returns
        -------
        str or None
            Constructor parameter that takes the residual scale, if declared.
        """
        return None

    @property
    def error_scale_binding(self) -> str | None:
        """Return an authored binding of the observation's residual scale.

        Returns
        -------
        str or None
            Binding passed directly as the residual scale, if it has one.
        """
        return None

    def kernel(
        self, inputs: KernelInputs, formula: Formula
    ) -> tuple[ResponseKernel, dict[str, ParameterSource]]:
        """State the kernel and the sources of its slots.

        Parameters
        ----------
        inputs : KernelInputs
            The family, the planned predictors, and the fixed values.
        formula : Formula
            Formula with the response and its modifiers.

        Returns
        -------
        tuple[ResponseKernel, dict[str, ParameterSource]]
            The kernel and the source of each slot a source feeds.
        """
        raise NotImplementedError

    @property
    def row_columns(self) -> tuple[str, ...]:
        """The data columns the kernel reads at each row, by name.

        Returns
        -------
        tuple[str, ...]
            Column names, which join the decision of the rows kept.
        """
        raise NotImplementedError

    def bindings(
        self,
        frame: pd.DataFrame,
        data2: Mapping[str, KernelDatum] | None,
    ) -> dict[str, torch.Tensor]:
        """Bind the data the kernel reads by name.

        Parameters
        ----------
        frame : pandas.DataFrame
            The rows of the response, in row order.
        data2 : Mapping[str, KernelDatum] or None
            Values that are not indexed by the rows.

        Returns
        -------
        dict[str, torch.Tensor]
            Values by input name.
        """
        raise NotImplementedError

    def check(self, module: Module, families: FamilyCatalog | None) -> None:
        """Check a model emitted with the law's kernel.

        Parameters
        ----------
        module : Module
            The emitted model.
        families : FamilyCatalog or None
            Compilation-local laws of the model.

        Raises
        ------
        ValueError
            If the model does not check, with the law's reason.
        """
        raise NotImplementedError
