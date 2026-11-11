"""In-container Pyro probe.

Reshapes each `Point` per `/io/shapes.json` (when present) so a
gallery example whose data tensor was declared with a multi-dim
shape sees `torch.tensor(value).shape` match what the model
expects.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. Pyro records a model's
return under the `_RETURN` node of the same trace the log-density
comes from, so the export is read off Pyro's own return surface
without a second evaluation.
"""

import json
import pathlib

import pyro
import torch

from _reshape import (
    export_payload,
    load_export_names,
    load_tables,
    reshape_point,
)


def _tensor(value):
    if isinstance(value, (int, float)):
        return torch.tensor(float(value))
    return torch.tensor(value, dtype=torch.float64)


def _site_tensor(value, dtype):
    # A site the caller types `int` (a drawn class code) is an integer
    # tensor, as Pyro draws it, so the model can subscript with it. A
    # site typed `float32` takes the dtype of the model's own literal
    # tensors, so a family that solves against a literal parameter
    # (Wishart's scale) sees one dtype throughout; every other site keeps
    # the float64 the data it is combined with carries.
    if dtype == "int":
        return torch.tensor(value, dtype=torch.long)
    if dtype == "float32":
        return torch.tensor(value, dtype=torch.float32)
    return _tensor(value)


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)

    ns = {"pyro": pyro, "torch": torch}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    exports = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        data_kw = {k: _tensor(v) for k, v in reshaped.get("data", {}).items()}
        param_dict = {
            k: _site_tensor(v, dtypes.get(k))
            for k, v in reshaped.get("params", {}).items()
        }
        conditioned = pyro.condition(model, data=param_dict)
        traced = pyro.poutine.trace(conditioned).get_trace(**data_kw)
        log_densities.append(float(traced.log_prob_sum()))
        if export_names:
            return_node = traced.nodes["_RETURN"]
            if "value" not in return_node:
                raise RuntimeError(
                    "pyro trace recorded no `value` on its `_RETURN` "
                    "node, so the program's exported value cannot be "
                    "read; the emitted model returns nothing."
                )
            exports.append(export_payload(export_names, return_node["value"]))

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
