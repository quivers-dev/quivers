"""Jupyter kernel for QVR.

Two console-script entry points:

- ``qvr kernel install`` registers the kernelspec.
- ``qvr kernel run`` is invoked by Jupyter once a notebook starts.

The kernel reuses [`ReplSession`][quivers.cli.ReplSession] so
notebook cells behave exactly like the REPL: leading ``:`` runs a meta
command; bare cells parse first as statements (appended to the
session's module) and fall back to type-printing.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from quivers.kernel.install import KERNELSPEC, install_kernelspec

if TYPE_CHECKING:
    from quivers.kernel.quivers_kernel import QuiversKernel


def __getattr__(name: str) -> type[QuiversKernel]:
    """Resolve the kernel class when its optional runtime is requested.

    Parameters
    ----------
    name : str
        The requested public name.

    Returns
    -------
    type[QuiversKernel]
        The Jupyter kernel implementation.

    Raises
    ------
    AttributeError
        If the name is not the kernel class.
    ModuleNotFoundError
        If the Jupyter kernel dependencies are absent.
    """
    if name != "QuiversKernel":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: type[QuiversKernel] = importlib.import_module(
        "quivers.kernel.quivers_kernel"
    ).QuiversKernel
    globals()[name] = value
    return value


__all__ = ["KERNELSPEC", "QuiversKernel", "install_kernelspec"]
