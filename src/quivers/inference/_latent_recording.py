"""Mark the forward runs that only read a program's latent structure.

An inference driver records one run of a program to learn which latent sites
it reaches and their shapes; the values drawn in that run are discarded and
replaced by the driver's initialization. A law that has no draws, such as an
improper flat prior, may answer such a run with a placeholder of the right
shape, and must refuse every other request for a draw.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_RECORDING: ContextVar[bool] = ContextVar("quivers_latent_recording", default=False)


@contextmanager
def latent_structure_recording() -> Iterator[None]:
    """Mark the enclosed forward run as a structure-only recording.

    Yields
    ------
    None
        Control returns to the caller with the mark set; it is restored on
        exit, including on error.
    """
    token = _RECORDING.set(True)
    try:
        yield
    finally:
        _RECORDING.reset(token)


def recording_latent_structure() -> bool:
    """Report whether the current run only records latent structure.

    Returns
    -------
    bool
        True inside `latent_structure_recording`, otherwise False.
    """
    return _RECORDING.get()


__all__ = ["latent_structure_recording", "recording_latent_structure"]
