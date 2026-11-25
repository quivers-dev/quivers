"""The units of a score, read from one run.

A score step may declare that its contribution is a sum of one log weight
per unit (see
[`UnitFactorization`][quivers.continuous.program_steps.UnitFactorization]).
`SiteUnits` is that declaration with the values of one run: it reads the
log weight of each unit, the unit of each row of a response, and, where
the rows factor, the log likelihood of each row.
"""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.program_steps import UnitFactorization


class UnfactoredObservationError(TypeError):
    """An observation's rows have no log likelihoods of their own.

    The message says why the rows do not factor, and which program has a
    log likelihood per row.
    """


class SiteUnits(ValueModel):
    """The units of one score site, at the values of one run.

    Parameters
    ----------
    factorization : UnitFactorization
        What the score step declares.
    environment : dict[str, torch.Tensor]
        The run's values the factorization reads.
    occurrence : str
        The suffix of the site's occurrence, ``"@<n>"`` for a score a scan
        reaches once per position, else empty.
    """

    factorization: UnitFactorization = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    environment: dict[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    occurrence: str = ""

    @property
    def axes(self) -> tuple[str, ...]:
        """The objects whose product indexes the units.

        Returns
        -------
        tuple[str, ...]
            The names, outermost first; empty for a score with one unit.
        """
        return self.factorization.axes

    @property
    def sizes(self) -> tuple[int, ...]:
        """The extent of each unit axis.

        Returns
        -------
        tuple[int, ...]
            The extents.
        """
        return self.factorization.sizes

    @property
    def responses(self) -> tuple[str, ...]:
        """The labels of the responses whose rows the units hold.

        Returns
        -------
        tuple[str, ...]
            The labels, in the order the program observes them.
        """
        return tuple(
            f"{item.response}{self.occurrence}" for item in self.factorization.rows
        )

    def log_weights(self) -> torch.Tensor:
        """The log weight of each unit, as the site scored it.

        Returns
        -------
        torch.Tensor
            The weights, of shape `sizes`. They sum to the score the step
            contributes, before any score transformer on the stack.
        """
        return self.factorization.weights(self.environment).reshape(self.sizes)

    def log_likelihood(self) -> torch.Tensor:
        """The log density of each unit's rows given the run's latents.

        Returns
        -------
        torch.Tensor
            The log densities, of shape `sizes`, without likelihood powers.

        Raises
        ------
        UnfactoredObservationError
            If the score's weights are not log densities.
        """
        self._require_density()
        return self.factorization.raw_weights(self.environment).reshape(self.sizes)

    def row_units(self, response: str) -> torch.Tensor:
        """The unit each row of a response belongs to.

        Parameters
        ----------
        response : str
            The response's label.

        Returns
        -------
        torch.Tensor
            ``(rows,)``, counted row-major over the unit axes.

        Raises
        ------
        KeyError
            If the score reads no such response.
        """
        for label, item in zip(self.responses, self.factorization.rows, strict=True):
            if label == response:
                return item.units(self.environment)
        raise KeyError(response)

    def row_log_likelihood(self, response: str) -> torch.Tensor:
        """The log likelihood of each row of a response, where it exists.

        A row has one when its unit holds that row and no other row of any
        response of the score. The log likelihood of the row is then the
        log weight of its unit.

        Parameters
        ----------
        response : str
            The response's label.

        Returns
        -------
        torch.Tensor
            ``(rows,)``, without likelihood powers.

        Raises
        ------
        KeyError
            If the score reads no such response.
        UnfactoredObservationError
            If a unit holds several rows, or the score's weights are not
            log densities.
        """
        self._require_density()
        rows = self.row_units(response)
        units = max(1, int(torch.tensor(self.sizes).prod()) if self.sizes else 1)
        counts = torch.zeros(units, dtype=torch.long, device=rows.device)
        for label in self.responses:
            held = self.row_units(label)
            counts = counts + torch.bincount(held, minlength=units)
        shared = counts[rows] > 1
        if bool(shared.any()):
            raise UnfactoredObservationError(
                self._refusal(response, rows, counts, int(rows[shared][0]))
            )
        return self.log_likelihood().reshape(-1)[rows]

    def _require_density(self) -> None:
        """Refuse to read a weight that is not a log density as one.

        Raises
        ------
        UnfactoredObservationError
            If the factorization states an obstacle.
        """
        obstacle = self.factorization.density_obstacle
        if obstacle is not None:
            raise UnfactoredObservationError(
                f"observations {list(self.responses)} have no log likelihood: "
                f"{obstacle}"
            )

    def _refusal(
        self, response: str, rows: torch.Tensor, counts: torch.Tensor, unit: int
    ) -> str:
        """Why a response's rows do not factor, and what would make them.

        Parameters
        ----------
        response : str
            The response's label.
        rows : torch.Tensor
            The unit of each of its rows.
        counts : torch.Tensor
            The number of rows of all the responses in each unit.
        unit : int
            A unit that holds several rows.

        Returns
        -------
        str
            The message.
        """
        latent = self.factorization.latent
        held = int(counts[unit])
        own = int((rows == unit).sum())
        others = [label for label in self.responses if label != response]
        if own == held:
            contents = f"{held} rows of {response!r}"
        else:
            contents = f"{own} of {response!r} and {held - own} of {others}"
        if self.axes:
            position = torch.unravel_index(torch.tensor(unit), self.sizes)
            where = ", ".join(
                f"{axis}={int(index)}"
                for axis, index in zip(self.axes, position, strict=True)
            )
            holder = f"the unit {where} holds {contents}, which share one {latent!r}"
        else:
            holder = f"all its rows ({contents}) share one {latent!r}"
        remedy = next(
            item.one_row_per_unit
            for label, item in zip(self.responses, self.factorization.rows, strict=True)
            if label == response
        )
        return (
            f"observation {response!r} has no log likelihood per row: {holder}, "
            "and summing it out leaves the log of a sum of products over those "
            "rows, which is not a sum of one term per row. "
            f"{remedy}. The joint score of the rows of each unit exists and can "
            "be read by unit."
        )


__all__ = ["SiteUnits", "UnfactoredObservationError"]
