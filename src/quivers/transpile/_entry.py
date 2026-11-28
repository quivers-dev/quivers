"""The `transpile` entry point and its backend table.

The table pairs each target key with its renderer, grammar, and support
tier. It lives apart from the package's `__init__` because it imports
every renderer, and the renderers reach the DSL compiler, which itself
imports `quivers.transpile` while initializing.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import Module
from quivers.transpile._api import (
    CHURCH_LIKE,
    PYTHON_DEEP,
    STAN_LIKE,
    UnsupportedConstruct,
)
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import parser_registry
from quivers.transpile._support import unsupported_for
from quivers.transpile.ir import IRProgram
from quivers.transpile.plan import Lower
from quivers.transpile.renderers import (
    BUGSRenderer,
    ChurchRenderer,
    Edward2Renderer,
    GenRenderer,
    JAGSRenderer,
    NumPyroRenderer,
    PyMCRenderer,
    PyroRenderer,
    RendererBase,
    StanRenderer,
    TuringRenderer,
    WebPPLRenderer,
)


_RENDERERS: dict[str, tuple[type[RendererBase], str, frozenset[str]]] = {
    "stan": (StanRenderer, "stan", STAN_LIKE),
    "numpyro": (NumPyroRenderer, "python", PYTHON_DEEP),
    "pyro": (PyroRenderer, "python", PYTHON_DEEP),
    "pymc": (PyMCRenderer, "python", STAN_LIKE),
    "edward2": (Edward2Renderer, "python", STAN_LIKE),
    "turing": (TuringRenderer, "julia", STAN_LIKE),
    "gen": (GenRenderer, "julia", STAN_LIKE),
    "church": (ChurchRenderer, "scheme", CHURCH_LIKE),
    "webppl": (WebPPLRenderer, "javascript", CHURCH_LIKE),
    "bugs": (BUGSRenderer, "bugs", STAN_LIKE),
    "jags": (JAGSRenderer, "jags", STAN_LIKE),
}


def transpile(module: Module, *, target: str) -> bytes:
    """Transpile a QVR module to the named ``target`` backend.

    Parameters
    ----------
    module
        The parsed [`Module`][quivers.dsl.ast_nodes.Module] AST.
    target
        A registered backend key. See
        [`available_targets`][quivers.transpile.available_targets].

    Returns
    -------
    bytes
        The transpiled source bytes.

    Raises
    ------
    UnsupportedConstruct
        If ``target`` is not registered, or if the module contains
        constructs the chosen renderer cannot lower (e.g. a
        non-finite-support marginalize on Stan).
    """
    if target not in _RENDERERS:
        raise UnsupportedConstruct(
            target,
            [f"target:unknown:{target}:{','.join(sorted(_RENDERERS))}"],
        )
    _, _, support_tier = _RENDERERS[target]
    unsupported_for(f"qvr-{target}", module, allow=support_tier)
    expanded = expand_composite_lets(module, target=target)
    ir = Lower().forward(expanded, target=target)
    return _emit_target(ir, target)


def _emit_target(ir: IRProgram, target: str) -> bytes:
    """Render and emit an already prepared plan through its registered target.

    Parameters
    ----------
    ir : IRProgram
        The target's checked program or standalone computation plan.
    target : str
        The registered backend key.

    Returns
    -------
    bytes
        Source emitted by the target grammar.
    """
    renderer_cls, grammar, _ = _RENDERERS[target]
    return bytes(parser_registry().emit_pretty(grammar, renderer_cls().render(ir)))


def available_targets() -> list[str]:
    """List every registered backend, sorted.

    Returns
    -------
    list[str]
        The keys [`transpile`][quivers.transpile.transpile] accepts as
        ``target``.
    """
    return sorted(_RENDERERS)
