"""Retained fitting recipes and adapters for explicit held-out partitions."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Never, SupportsIndex

import torch

from quivers.diagnostics.cross_validation import FoldPrediction, FoldSplit

type SnapshotPolicy = Literal["clone", "reference"]
"""Retain an independent data snapshot or explicitly share the caller's data."""


def _snapshot[Data](value: Data) -> Data:
    """Copy data without preserving tensor autograd graphs or shared storage."""
    memo: dict[int, object] = {}
    seen: set[int] = set()

    def tensors(item: object) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            memo[id(item)] = item.detach().clone()
        elif isinstance(item, Mapping):
            for child in item.values():
                tensors(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                tensors(child)

    tensors(value)
    return deepcopy(value, memo)


@dataclass(frozen=True, eq=False, init=False)
class RefitRecipe[Data, Method, Result]:
    """An identity-held fitting operation with explicit data ownership.

    This is live process state, not a portable model format. A recipe runs
    the supplied fitter anew; it never resumes or mutates a fitted posterior.

    Parameters
    ----------
    data : Data
        Original training data. Clone mode retains one independent copy,
        separate from posterior draws, and returns fresh copies to callers.
    method : Method
        Retained inference configuration, held by identity. Its own random
        seed remains unchanged unless the caller supplies another method.
    operation : Callable[[Data, Method, int or None], Result]
        Existing fitting adapter. The optional seed overrides preparation;
        its interpretation is part of the adapter's declared contract.
    snapshot_policy : {"clone", "reference"}
        Clone by default. Reference mode explicitly observes later mutations
        of the supplied data, and is useful when another owner freezes them.
    snapshot : Callable[[Data], Data] or None
        Optional backend-specific data copier. The default copies containers
        and arrays, detaching and cloning tensor data without autograd graphs.
    provenance : Mapping[str, str or int] or None
        Read-only descriptive reconstruction metadata, such as source/entry
        and compilation seed. Callables and modules remain live objects.

    Raises
    ------
    ValueError
        If the snapshot policy is unknown.
    """

    _data: Data
    method: Method
    _operation: Callable[[Data, Method, int | None], Result]
    snapshot_policy: SnapshotPolicy
    _snapshot: Callable[[Data], Data]
    provenance: Mapping[str, str | int]

    def __init__(
        self,
        data: Data,
        method: Method,
        operation: Callable[[Data, Method, int | None], Result],
        *,
        snapshot_policy: SnapshotPolicy = "clone",
        snapshot: Callable[[Data], Data] | None = None,
        provenance: Mapping[str, str | int] | None = None,
    ) -> None:
        if snapshot_policy not in {"clone", "reference"}:
            raise ValueError("snapshot_policy must be 'clone' or 'reference'")
        copier = _snapshot if snapshot is None else snapshot
        object.__setattr__(self, "_snapshot", copier)
        object.__setattr__(
            self, "_data", copier(data) if snapshot_policy == "clone" else data
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "_operation", operation)
        object.__setattr__(self, "snapshot_policy", snapshot_policy)
        object.__setattr__(self, "provenance", MappingProxyType(dict(provenance or {})))

    @property
    def data(self) -> Data:
        """Return isolated training data, or the explicitly shared reference.

        Returns
        -------
        Data
            A fresh snapshot in clone mode, the original object otherwise.
        """
        return (
            self._snapshot(self._data)
            if self.snapshot_policy == "clone"
            else self._data
        )

    def run(
        self,
        data: Data | None = None,
        *,
        method: Method | None = None,
        seed: int | None = None,
    ) -> Result:
        """Fit retained or replacement data through the original fitting adapter.

        Parameters
        ----------
        data : Data or None
            Replacement training data, or the retained data snapshot.
        method : Method or None
            Explicit inference override; otherwise retain the original method.
        seed : int or None
            Optional preparation override; None preserves the recipe's policy.

        Returns
        -------
        Result
            A fresh result with the fitter's original result semantics.
        """
        return self._operation(
            self.data if data is None else data,
            self.method if method is None else method,
            seed,
        )

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        """Refuse to serialize live fitting operations and inference objects.

        Parameters
        ----------
        protocol : SupportsIndex
            Requested pickle protocol.

        Raises
        ------
        TypeError
            Recipes require a separate durable reconstruction format.
        """
        raise TypeError(
            "RefitRecipe contains live fitting operations and is not a durable fitted artifact"
        )


def refit_callback[Data, Method, Result](
    recipe: RefitRecipe[Data, Method, Result],
    partition: Callable[[Data, tuple[int, ...]], Data],
    score: Callable[[Result, Data, int], Mapping[str, torch.Tensor]],
    *,
    method_factory: Callable[[Method, int], Method],
) -> Callable[[FoldSplit], FoldPrediction]:
    """Adapt one recipe to the existing cross-validation runner.

    Parameters
    ----------
    recipe : RefitRecipe
        Retained fitting adapter and original training data.
    partition : Callable[[Data, tuple[int, ...]], Data]
        Nonmutating row selection producing fresh partitions, preserving
        non-row constants and identities. The adapter snapshots original data
        once, rather than copying the full training data for every fold.
    score : Callable[[Result, Data, int], Mapping[str, torch.Tensor]]
        Existing raw held-out likelihood accessor. Its tensors retain joint
        chain/draw/test-row axes; weights are not predictive replications.
    method_factory : Callable[[Method, int], Method]
        Required explicit inference seed/configuration policy for each fold.
        Returning the original method intentionally retains its own seed.

    Returns
    -------
    Callable[[FoldSplit], FoldPrediction]
        A callback for cross_validate, with actual fitted results retained.
        Preprocessing and inference stay in the original fitting operation.
    """
    original = recipe.data

    def run(split: FoldSplit) -> FoldPrediction:
        fitted = recipe.run(
            partition(original, split.train),
            method=method_factory(recipe.method, split.seed),
            seed=split.seed,
        )
        return FoldPrediction(
            log_likelihood=score(fitted, partition(original, split.test), split.seed),
            fit=fitted,
        )

    return run


__all__ = ["SnapshotPolicy", "RefitRecipe", "refit_callback"]
