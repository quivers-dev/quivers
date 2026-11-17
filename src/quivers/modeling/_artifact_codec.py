"""Closed declarative values and tensor references for fitted artifacts.

The decoder never imports a name supplied by a checkpoint. Extension factories
are caller-provided code; their names identify explicitly registered contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from quivers.continuous.programs import MonadicProgram


__all__: list[str] = []


def _module_state(module: torch.nn.Module) -> dict[str, object]:
    """Retain module tensors, alias identities, modes, and gradient ownership.

    Parameters
    ----------
    module : torch.nn.Module
        Live deterministic model or guide.

    Returns
    -------
    dict[str, object]
        Declarative state sufficient to preserve tied parameters and buffers.
    """
    aliases: dict[int, list[str]] = {}
    for name, tensor in (
        *module.named_parameters(remove_duplicate=False),
        *module.named_buffers(remove_duplicate=False),
    ):
        aliases.setdefault(id(tensor), []).append(name)
    state = module.state_dict()
    return {
        "state": dict(state),
        "metadata": getattr(state, "_metadata", {}),
        "buffers": {
            name: name.rpartition(".")[2]
            not in module.get_submodule(
                name.rpartition(".")[0]
            )._non_persistent_buffers_set
            for name, _ in module.named_buffers(remove_duplicate=False)
        },
        "training": {key: child.training for key, child in module.named_modules()},
        "buffer_gradients": {
            key: value.requires_grad
            for key, value in module.named_buffers(remove_duplicate=False)
        },
        "gradients": {
            key: value.requires_grad for key, value in module.named_parameters()
        },
        "aliases": [names for names in aliases.values() if len(names) > 1],
    }


def _restore_module(module: torch.nn.Module, state: dict) -> None:
    """Restore exact tensor carriers while retaining declared shared modules.

    Parameters
    ----------
    module : torch.nn.Module
        Rebuilt module with its original architecture.
    state : dict
        Checked module-state payload from the fitted artifact.

    Raises
    ------
    ValueError, RuntimeError
        If architectures, state names, or shapes disagree.
    """
    tensors = OrderedDict(state["state"])
    setattr(tensors, "_metadata", state["metadata"])
    module.load_state_dict(tensors, strict=True, assign=True)
    # Assignment can promote a buffer holding a Parameter into _parameters.
    # Restore its declared registration before tying saved aliases.
    for name, persistent in state["buffers"].items():
        owner_name, _, attribute = name.rpartition(".")
        owner = module.get_submodule(owner_name)
        value = getattr(owner, attribute)
        if attribute in owner._parameters:
            del owner._parameters[attribute]
        owner.register_buffer(attribute, value, persistent=persistent)
    for names in state["aliases"]:
        owner_name, _, attribute = names[0].rpartition(".")
        shared = getattr(module.get_submodule(owner_name), attribute)
        for name in names[1:]:
            owner_name, _, attribute = name.rpartition(".")
            owner = module.get_submodule(owner_name)
            if attribute in owner._buffers:
                owner._buffers[attribute] = shared
            else:
                setattr(owner, attribute, shared)
    for name, gradient in state["buffer_gradients"].items():
        owner_name, _, attribute = name.rpartition(".")
        getattr(module.get_submodule(owner_name), attribute).requires_grad_(gradient)
    parameters = dict(module.named_parameters())
    if set(parameters) != set(state["gradients"]):
        raise ValueError(
            "artifact parameter identity layout changed during reconstruction"
        )
    for name, gradient in state["gradients"].items():
        parameters[name].requires_grad_(gradient)
    children = dict(module.named_modules())
    if set(children) != set(state["training"]):
        raise ValueError("artifact module hierarchy changed during reconstruction")
    for name, training in state["training"].items():
        children[name].training = training


@dataclass(frozen=True)
class ArtifactContext:
    """Checked execution context supplied to an explicit reconstruction factory.

    Parameters
    ----------
    program : MonadicProgram
        Recompiled model with restored deterministic parameter state.
    input : torch.Tensor
        Original one-run input.
    observations : Mapping[str, torch.Tensor]
        Original checked host data and response bindings.
    """

    program: MonadicProgram
    input: torch.Tensor
    observations: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ArtifactFactory:
    """Explicit portable configuration for an external Python class.

    Parameters
    ----------
    value_type : type
        Exact class accepted by this factory; subclasses require registration.
    encode : Callable[[object], object]
        Extract declarative configuration using supported artifact values.
    decode : Callable[[object, ArtifactContext or None], object]
        Reconstruct the class from configuration. Guide factories receive the
        checked model context; ordinary option factories receive None. Factories
        must not fit, optimize, or sample. Module tensor state is stored separately.
    """

    value_type: type
    encode: Callable[[object], object]
    decode: Callable[[object, ArtifactContext | None], object]


def _builtin_types(formula: bool = False) -> dict[str, type]:
    """Return the closed built-in model constructors, never checkpoint imports.

    Parameters
    ----------
    formula : bool
        Include closed declarative formula records, importing extras only here.

    Returns
    -------
    dict[str, type]
        Stable local names for explicitly supported configuration records.
    """
    from quivers.inference import MCMCResult, PriorDraws
    from quivers.modeling.methods import MCMCMethod, VariationalMethod

    constructors = (MCMCResult, PriorDraws, MCMCMethod, VariationalMethod)
    if formula:
        from quivers.formulas.family import (
            Family,
            Link,
            AuxParam,
            OrdinalSpec,
            ResponseSpec,
        )
        from quivers.formulas._survival import BaselineSpec
        from quivers.formulas._predictor_graph import NonlinearExpression
        from quivers.formulas._priors import (
            PriorSpec,
            PriorSet,
            RegularizedHorseshoe,
            R2D2,
        )
        from quivers.formulas._group_covariance import GroupCovariance
        from quivers.formulas._custom_types import CustomFamilySpec

        constructors += (
            Family,
            Link,
            AuxParam,
            OrdinalSpec,
            ResponseSpec,
            BaselineSpec,
            NonlinearExpression,
            PriorSpec,
            PriorSet,
            RegularizedHorseshoe,
            R2D2,
            GroupCovariance,
            CustomFamilySpec,
        )
    return {cls.__name__: cls for cls in constructors}


class _Codec:
    """Encode a declarative tree with tensors held in a separate payload.

    Parameters
    ----------
    factories : Mapping[str, ArtifactFactory] or None
        Explicit extension reconstructors provided by the caller.
    tensors : dict[str, torch.Tensor] or None
        Existing checkpoint tensors when decoding, or a fresh output table.
    formula : bool
        Enable declarative formula records and dataframe reconstruction.
        Source-only artifacts do not import optional formula dependencies.
    """

    def __init__(
        self,
        factories: Mapping[str, ArtifactFactory] | None = None,
        tensors: dict[str, torch.Tensor] | None = None,
        *,
        formula: bool = False,
    ) -> None:
        self.formula = formula
        self.factories = dict(factories or {})
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(factory, ArtifactFactory)
            or not isinstance(factory.value_type, type)
            or not callable(factory.encode)
            or not callable(factory.decode)
            for name, factory in self.factories.items()
        ):
            raise TypeError(
                "artifact factories require nonempty string names and explicit class/encode/decode contracts"
            )
        self.tensors = {} if tensors is None else tensors

    def encode(self, value: object) -> object:
        """Encode one supported value without a general object fallback.

        Parameters
        ----------
        value : object
            Declarative configuration, tensor, or registered extension.

        Returns
        -------
        object
            JSON-safe tagged tree.

        Raises
        ------
        TypeError
            If an object has no explicit portable contract.
        """
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is float:
            return {"kind": "float", "value": value.hex()}
        if isinstance(value, torch.Tensor):
            key = str(len(self.tensors))
            self.tensors[key] = value.detach().clone()
            return {"kind": "tensor", "key": key, "requires_grad": value.requires_grad}
        if type(value).__module__.split(".")[0] == "numpy":
            import numpy as np

            if isinstance(value, np.generic):
                return {
                    "kind": "numpy_scalar",
                    "dtype": str(value.dtype),
                    "value": self.encode(value.item()),
                }
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufUS":
                return {
                    "kind": "numpy_array",
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "values": self.encode(value.tolist()),
                }
            raise TypeError(
                "artifact NumPy arrays require real numeric, Boolean, or string values"
            )
        for name, factory in self.factories.items():
            if value is factory.value_type:
                return {"kind": "factory_class", "name": name}
            if type(value) is factory.value_type:
                result = {
                    "kind": "factory",
                    "name": name,
                    "config": self.encode(factory.encode(value)),
                }
                from quivers.family_catalog import FamilyCatalog

                if isinstance(value, FamilyCatalog):
                    from quivers.qiec.serialization import dumps

                    result["semantic"] = dumps(value.definitions)
                if isinstance(value, nn.Module):
                    result["state"] = self.encode(_module_state(value))
                return result
        from quivers.family_catalog import FamilyCatalog

        if isinstance(value, FamilyCatalog):
            raise TypeError(
                "custom family catalogs require an explicit ArtifactFactory"
            )
        if isinstance(value, Mapping):
            return {
                "kind": "mapping",
                "items": [[self.encode(k), self.encode(v)] for k, v in value.items()],
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [self.encode(v) for v in value],
            }
        if self.formula and type(value).__module__.split(".")[0] in {
            "pandas",
            "polars",
            "pyarrow",
        }:
            from quivers.formulas._artifact import _encode_frame

            return {"kind": "dataframe", "value": self.encode(_encode_frame(value))}
        from quivers.inference import AutoNormalGuide, ELBO, HMCKernel, MCMC, NUTSKernel
        from quivers.inference.estimators import Reparameterized

        if value is AutoNormalGuide:
            return {"kind": "class", "name": "AutoNormalGuide"}
        if type(value) in _builtin_types(self.formula).values():
            return {
                "kind": "record",
                "name": type(value).__name__,
                "fields": self.encode(
                    {
                        name: getattr(value, name)
                        for name in cast(Any, type(value)).__field_specs__
                    }
                ),
            }
        if type(value) is ELBO and type(value.estimator) is Reparameterized:
            return {"kind": "elbo", "particles": value.num_particles}
        if type(value) in (NUTSKernel, HMCKernel):
            config = dict(cast(NUTSKernel | HMCKernel, value).configuration())
            config.pop("kernel")
            return {
                "kind": "kernel",
                "name": type(value).__name__,
                "config": self.encode(config),
            }
        if type(value) is MCMC:
            fields = (
                "kernel",
                "num_warmup",
                "num_samples",
                "num_chains",
                "init_strategy",
                "seed",
                "initial_values",
                "max_seconds",
                "max_evaluations",
            )
            if any(
                getattr(value, name) is not None
                for name in ("generator", "checkpoint", "callback")
            ):
                raise TypeError(
                    "MCMC callbacks, live generators, and checkpoint destinations require an explicit ArtifactFactory"
                )
            return {
                "kind": "mcmc",
                "config": self.encode({name: getattr(value, name) for name in fields}),
            }
        raise TypeError(
            f"no artifact contract for {type(value).__name__}; supply an explicit ArtifactFactory"
        )

    def decode(self, node: object, context: ArtifactContext | None = None) -> Any:
        """Decode only local allowlisted contracts and caller-provided factories.

        Parameters
        ----------
        node : object
            JSON-safe tagged value.
        context : ArtifactContext or None
            Checked context for a registered guide reconstruction.

        Returns
        -------
        object
            Reconstructed value.

        Raises
        ------
        ValueError, TypeError
            If a node, constructor, or factory contract is invalid.
        """
        if node is None or type(node) in (str, int, bool):
            return node
        if not isinstance(node, dict):
            raise ValueError("malformed artifact value")
        kind = node.get("kind")
        if kind == "float":
            return float.fromhex(node["value"])
        if kind == "tensor":
            return self.tensors[node["key"]].requires_grad_(node["requires_grad"])
        if kind in ("numpy_scalar", "numpy_array"):
            import numpy as np

            dtype = np.dtype(node["dtype"])
            if dtype.kind not in "biufUS":
                raise ValueError("unsupported artifact NumPy dtype")
            if kind == "numpy_scalar":
                return np.asarray(self.decode(node["value"]), dtype=dtype)[()]
            return np.asarray(self.decode(node["values"]), dtype=dtype).reshape(
                node["shape"]
            )
        if kind in ("list", "tuple"):
            values = [self.decode(item, context) for item in node["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind == "mapping":
            pairs = [
                (self.decode(k, context), self.decode(v, context))
                for k, v in node["items"]
            ]
            values = dict(pairs)
            if len(values) != len(pairs):
                raise ValueError("duplicate artifact mapping keys")
            return values
        if kind in ("factory", "factory_class"):
            name = node["name"]
            if name not in self.factories:
                raise ValueError(
                    f"artifact requires explicitly registered factory {name!r}"
                )
            factory = self.factories[name]
            if kind == "factory_class":
                return factory.value_type
            value = factory.decode(self.decode(node["config"], context), context)
            if type(value) is not factory.value_type:
                raise TypeError(f"factory {name!r} returned the wrong class")
            from quivers.family_catalog import FamilyCatalog

            if isinstance(value, FamilyCatalog):
                from quivers.qiec.serialization import dumps

                if dumps(value.definitions) != node.get("semantic"):
                    raise ValueError(
                        "artifact catalog factory changed its checked semantic definitions"
                    )
            elif "semantic" in node:
                raise TypeError(
                    "artifact semantic metadata requires a FamilyCatalog factory"
                )
            if "state" in node:
                if not isinstance(value, nn.Module):
                    raise TypeError("artifact module factory must return nn.Module")
                _restore_module(value, self.decode(node["state"]))
            return value
        if kind == "dataframe" and self.formula:
            from quivers.formulas._artifact import _decode_frame

            return _decode_frame(self.decode(node["value"], context))
        from quivers.inference import AutoNormalGuide, ELBO, HMCKernel, MCMC, NUTSKernel

        if kind == "class" and node["name"] == "AutoNormalGuide":
            return AutoNormalGuide
        if kind == "record" and node["name"] in _builtin_types(self.formula):
            cls = _builtin_types(self.formula)[node["name"]]
            fields = self.decode(node["fields"], context)
            if not isinstance(fields, dict) or set(fields) != set(cls.__field_specs__):
                raise ValueError(
                    "artifact record fields disagree with its supported schema"
                )
            return cls(**fields)
        if kind == "elbo":
            return ELBO(num_particles=node["particles"])
        if kind == "kernel" and node["name"] in ("NUTSKernel", "HMCKernel"):
            from quivers.inference.mcmc import AdaptationSchedule

            config = self.decode(node["config"])
            schedule = config.pop("schedule")
            config["adaptation_schedule"] = AdaptationSchedule(
                init_buffer=schedule[0],
                term_buffer=schedule[1],
                base_window=schedule[2],
            )
            return (NUTSKernel if node["name"] == "NUTSKernel" else HMCKernel)(**config)
        if kind == "mcmc":
            return MCMC(**self.decode(node["config"]))
        raise ValueError(f"unknown artifact value kind {kind!r}")
