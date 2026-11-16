"""Source adapters for shared retained recipes and explicit row partitions."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from quivers.modeling.refit import RefitRecipe, SnapshotPolicy, _snapshot

if TYPE_CHECKING:
    from quivers.modeling.model import BoundModel, InferenceMethod, ModelFit
    from quivers.inference import PriorMode


def source_recipe(
    bound: BoundModel,
    method: InferenceMethod[ModelFit],
    *,
    snapshot_policy: SnapshotPolicy,
    sample_prior: PriorMode,
    num_prior_samples: int,
    prior_seed: int,
) -> RefitRecipe[Mapping[str, object], InferenceMethod[ModelFit], ModelFit]:
    """Retain an exact source recipe, reusing unchanged checked preparation."""
    model = bound.specification
    source, entry, syntax, checked = (
        model.source,
        model.entry,
        model.syntax,
        model.checked_module,
    )
    arguments = _snapshot(dict(bound.template_arguments))
    input_value = bound.input.detach().clone()

    def run(
        data: Mapping[str, object],
        chosen_method: InferenceMethod[ModelFit],
        seed: int | None,
    ) -> ModelFit:
        from quivers.modeling.model import Model, ModelFit

        # Public Model attributes can be reassigned. Such mutation must not
        # cause a retained recipe to fit a different source or reuse stale ASTs.
        unchanged = (
            model.source == source
            and model.entry == entry
            and model.syntax is syntax
            and model.checked_module is checked
        )
        specification = model if unchanged else Model(source, entry=entry)
        fitted = specification.fit(
            data,
            method=chosen_method,
            parameters=_snapshot(arguments),
            input=input_value.clone(),
            seed=bound.compilation_seed if seed is None else seed,
            sample_prior=sample_prior,
            num_prior_samples=num_prior_samples,
            prior_seed=prior_seed,
            snapshot_policy=snapshot_policy,
        )
        if not isinstance(fitted, ModelFit):
            raise TypeError("a source refit method must return ModelFit")
        return fitted

    return RefitRecipe(
        dict(bound.data),
        method,
        run,
        snapshot_policy=snapshot_policy,
        provenance={
            "source": source,
            "entry": entry,
            "compilation_seed": bound.compilation_seed,
        },
    )


def partition_rows(
    bound: BoundModel, data: Mapping[str, object], positions: tuple[int, ...], axis: str
) -> Mapping[str, object]:
    """Select checked leading row axes, preserving unrelated nominal constants."""
    parameters = bound.schema.entry.parameters
    declared = {
        parameter.name
        for parameter in parameters
        if parameter.axes and parameter.axes[0] == axis
    }
    if not declared:
        raise ValueError(f"no checked input has leading nominal row axis {axis!r}")
    nonleading = [
        parameter.name for parameter in parameters if axis in parameter.axes[1:]
    ]
    if nonleading:
        raise ValueError(
            f"row partition axis {axis!r} also occurs inside inputs {nonleading}; supply an explicit partitioner"
        )
    result = dict(data)
    for name in declared & result.keys():
        tensor = torch.as_tensor(result[name])
        index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
        result[name] = tensor.index_select(0, index)
    return result
