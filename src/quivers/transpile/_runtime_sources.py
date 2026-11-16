"""Compose standalone target runtimes from their shared source fragments."""

from pathlib import Path


def javascript_runtime_source(path: Path) -> bytes:
    """Prepend shared scalar probability definitions to a JavaScript runtime.

    Parameters
    ----------
    path : Path
        Ordinary WebPPL or checked QIEC runtime body.

    Returns
    -------
    bytes
        Complete standalone JavaScript source for parsing and grafting.
    """
    return (
        (path.parent / "runtime_probability.js").read_bytes()
        + b"\n"
        + path.read_bytes()
    )
