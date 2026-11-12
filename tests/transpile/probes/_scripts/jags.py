"""In-container JAGS / BUGS probe.

Reads /io/source.jags (or source.bugs), compiles via pyjags, and
extracts the joint log-density via the ``deviance`` monitor at the
supplied point. Contract: joint log-density = ``-deviance / 2``.

JAGS evaluates the joint log-density at a fixed point by treating
every supplied value as observed. The probe puts both the QVR
``params`` (latent parameters) and ``data`` (observed nodes) into
the pyjags ``data`` dict; every named node in either dict becomes a
fixed observed node, so the deviance JAGS computes is the joint
log-density of the model evaluated at the requested point.

pyjags' built-in ``Model.__init__`` only auto-loads the ``dic``
module; the deviance monitor type lives in ``dic.so`` but the
``base`` and ``bugs`` modules supply the basic monitor factories
(`Trace`, `Mean`) and the standard distributions / samplers that
``dnorm`` / ``dbeta`` / ``dbern`` etc. resolve to, so the probe
loads all three explicitly before constructing the model.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. The BUGS language has no
`return`, so the renderer exposes each returned name as its own
deterministic relation ``<name>_value <- <name>`` and the probe
monitors that node through the same trace machinery the deviance
comes from. With every stochastic node supplied as data the
relation has a single determined value, so its one-iteration trace
is the exported value rather than a draw.
"""

import json
import os
import pathlib
import re

import numpy as np
import pyjags

from _reshape import (
    as_nested,
    load_export_names,
    load_tables,
    reshape_point,
)


pyjags.load_module("basemod")
pyjags.load_module("bugs")
pyjags.load_module("dic")


def _arr(value):
    return np.asarray(value)


def _monitored_export(dumped: dict, alias: str, name: str) -> list:
    """Read one monitored alias out of a `dumpMonitors` payload.

    The payload is dumped unflattened, so pyjags shapes a monitored
    node as ``(*node_dims, iterations, chains)`` with its element at
    each index where JAGS holds it; the probe runs one iteration on one
    chain, so dropping the two trailing axes recovers the node itself,
    row-major like the reference. A scalar node comes back as
    ``(1, 1, 1)`` and collapses to a bare float.
    """
    if alias not in dumped:
        msg = (
            f"jags probe: the emitted model declares no {alias!r} "
            f"relation, so it exposes nothing for the QVR program's "
            f"exported {name!r}. Monitored: {sorted(dumped)}"
        )
        raise RuntimeError(msg)
    array = np.asarray(dumped[alias])
    if array.ndim < 2:
        msg = (
            f"jags probe: monitor for {alias!r} came back with shape "
            f"{array.shape}, which carries no (iteration, chain) axes "
            f"to drop."
        )
        raise RuntimeError(msg)
    return as_nested(array[..., 0, 0])


#: A zeros-trick carrier the emitted model scores: the `score` step's
#: `zero_<name> ~ dpois(C_<name>)` relation.
#: A drawn class's decoding relation, `<name>[...] <- <name>_code[...]-1`.
_DECODED_CLASS_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?\s*<-\s*\1_code(?:\[[^\]]*\])?-1\b"
)

_CARRIER_RE = re.compile(r"\b(zero_[A-Za-z_][A-Za-z0-9_]*)\s*~\s*dpois\(")


def main() -> None:
    io = pathlib.Path("/io")
    ext = os.environ.get("FIXTURE_EXT", "jags")
    source_path = io / f"source.{ext}"
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)
    # The point payload is in QVR's zero-based coding, and the emitted
    # program reads an index covariate or a class code one higher where
    # JAGS / BUGS count from one, so the payload is bound as it is.
    source_text = source_path.read_text()
    # A `score` step is a zeros-trick relation `zero_<name> ~ dpois(C)`
    # whose carrier the BUGS emission leaves to the host's data list,
    # the language having no data block of its own; JAGS binds the
    # carrier in its `data { ... }` block, and a carrier the source
    # binds is not supplied twice.
    carriers = {
        name
        for name in _CARRIER_RE.findall(source_text)
        if f"{name} <- 0" not in source_text
    }
    # A drawn class is the one-based node `<name>_code`, which the
    # program decodes into the QVR code `<name> <- <name>_code-1`.
    decoded = {name for name in _DECODED_CLASS_RE.findall(source_text)}

    log_densities = []
    exports = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        # Combine params + data: JAGS treats every supplied node as
        # observed, so the deviance includes the joint log-density of
        # the latent parameters AND the observed data nodes.
        merged: dict[str, np.ndarray] = {}
        for k, v in reshaped.get("data", {}).items():
            merged[k] = _arr(v)
        for k, v in reshaped.get("params", {}).items():
            if k in decoded:
                # The program draws this class as the one-based node
                # `<k>_code` and decodes it, `<k> <- <k>_code - 1`; a
                # clamp fixes the drawn node, so it inverts that
                # relation.
                merged[f"{k}_code"] = _arr(v) + 1
                continue
            merged[k] = _arr(v)
        for name in carriers:
            merged[name] = _arr(0)
        model = pyjags.Model(
            file=str(source_path),
            data=merged,
            chains=1,
            adapt=0,
            threads=1,
            progress_bar=False,
        )
        # `trace` monitor records the value at each iteration; with
        # every model variable fixed as observed there is nothing for
        # the sampler to update, so the single-iteration trace value
        # is exactly the joint log-density (times -2).
        model.console.setMonitor("deviance", 1, "trace")
        for name in export_names:
            model.console.setMonitor(f"{name}_value", 1, "trace")
        model.console.update(1)
        # Unflattened, so a node of rank two or more keeps its dims; the
        # flat form lists its elements in JAGS's column-major storage.
        dumped = model.console.dumpMonitors("trace", False)
        dev_arr = dumped["deviance"]
        dev = float(np.asarray(dev_arr).flatten()[0])
        log_densities.append(-dev / 2)
        if export_names:
            exports.append(
                [
                    _monitored_export(dumped, f"{name}_value", name)
                    for name in export_names
                ]
            )

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
