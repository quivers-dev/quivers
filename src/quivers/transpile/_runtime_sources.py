"""Compose standalone target runtimes from their shared source fragments."""

from pathlib import Path


def python_runtime_source(path: Path) -> bytes:
    """Expand shared probability fragments before parsing a Python runtime.

    Parameters
    ----------
    path : Path
        Ordinary target runtime or checked distribution bridge.

    Returns
    -------
    bytes
        Standalone source with one canonical probability implementation.

    Raises
    ------
    ValueError
        If a runtime's shared-fragment marker is incomplete or repeated.
    """
    source = path.read_text()
    for target in ("pyro", "pymc", "numpyro", "edward2"):
        start = f"# qvr-shared-probability:{target}:start"
        end = f"# qvr-shared-probability:{target}:end"
        if start not in source and end not in source:
            continue
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError(f"invalid probability fragment markers in {path.name}")
        prefix, rest = source.split(start)
        _, suffix = rest.split(end)
        fragment = path.parent / f"runtime_probability_{target}.py"
        source = prefix + fragment.read_text() + suffix
    return source.encode()


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
