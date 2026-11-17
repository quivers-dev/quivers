"""Typed references for recorded outcome events, independent of executors."""

from __future__ import annotations

from quivers._values import ValueModel


class ObservationEvent(ValueModel):
    """Describe a recorded event and the response law's sampling bounds.

    Parameters
    ----------
    status : str or int or None
        Binding or code: minus one left, zero exact, one right, two interval.
    upper : str or float or None
        Binding or second endpoint for interval censoring.
    lower_bound, upper_bound : str or float or None
        Inclusive truncation bounds; omitted bounds are infinite.
    """

    status: str | int | None = None
    upper: str | float | None = None
    lower_bound: str | float | None = None
    upper_bound: str | float | None = None

    def names(self) -> tuple[str, ...]:
        """Return runtime dependencies once in canonical field order.

        Returns
        -------
        tuple[str, ...]
            Referenced bindings, without duplicate names or numeric constants.
        """
        return tuple(
            dict.fromkeys(
                value
                for value in (
                    self.status,
                    self.upper,
                    self.lower_bound,
                    self.upper_bound,
                )
                if isinstance(value, str)
            )
        )


# Source option, metadata field, and carrier share one canonical vocabulary.
EVENT_OPTIONS: dict[str, tuple[str, str]] = {
    "cens": ("status", "Int"),
    "cens_upper": ("upper", "Real"),
    "trunc_lower": ("lower_bound", "Real"),
    "trunc_upper": ("upper_bound", "Real"),
}
