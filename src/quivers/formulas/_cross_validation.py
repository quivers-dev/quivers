"""Formula partitions and row attachments adapt the shared refit callback."""

from collections.abc import Callable, Mapping
import inspect
from typing import TYPE_CHECKING

import narwhals as nw
from narwhals.typing import IntoDataFrame
import torch

from quivers.diagnostics.cross_validation import FoldPrediction, FoldSplit
from quivers.modeling.refit import RefitRecipe, refit_callback
from quivers.modeling._row_attachments import (
    AttachedRows,
    RowAttachment,
    original_positions,
    validate_split_rows,
)

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def formula_refitter(
    fitted: BayesianFit,
    *,
    method_factory: Callable[[Mapping[str, object], int], Mapping[str, object]],
    partition: Callable[[IntoDataFrame, tuple[int, ...]], IntoDataFrame] | None,
    num_samples: int | None,
    prediction: Mapping[str, object],
) -> Callable[[FoldSplit], FoldPrediction]:
    """Retain included-row coordinates and explicitly owned predictor inputs."""
    recipe = fitted.refit_recipe
    if recipe is None:
        raise ValueError("this fit has no retained refit recipe")
    if forbidden := set(prediction) & {
        "new_data",
        "predictor_data",
        "seed",
        "num_samples",
    }:
        raise ValueError(
            f"the refit adapter owns prediction arguments {sorted(forbidden)}"
        )
    inspect.signature(fitted.log_likelihood).bind(**prediction)
    frame = recipe.data
    size = len(nw.from_native(frame, eager_only=True))
    selection = fitted.formula.modifiers.subset
    expected = (
        selection.size
        if selection is not None
        else fitted.formula.response_values.shape[0]
    )
    if size != expected:
        raise ValueError(
            "retained original frame no longer has the fitted row layout; update before constructing refit partitions"
        )
    included = selection.positions if selection is not None else tuple(range(size))
    attachments: tuple[RowAttachment, ...] = ()
    predictor = recipe.method.get("predictor_data")
    if predictor is not None:
        if not isinstance(predictor, torch.Tensor):
            raise TypeError("retained predictor_data must be a tensor row attachment")
        attachment = RowAttachment(
            "predictor_data", "predictor_data", predictor.detach().clone()
        )
        attachment.validate(size)
        attachments = (attachment,)

    def select(data: IntoDataFrame, rows: tuple[int, ...]) -> IntoDataFrame:
        return nw.from_native(data, eager_only=True)[list(rows), :].to_native()

    choose = select if partition is None else partition

    def run(
        rows: AttachedRows[IntoDataFrame],
        settings: Mapping[str, object],
        seed: int | None,
    ) -> BayesianFit:
        for attachment in attachments:
            if settings.get(attachment.training_keyword) is not recipe.method.get(
                attachment.training_keyword
            ):
                raise ValueError(
                    f"method_factory must preserve retained row attachment {attachment.training_keyword!r}; the adapter partitions it"
                )
        return recipe.run(
            rows.data, method={**settings, **rows.keywords(scoring=False)}
        )

    # These objects already own snapshots. Reference ownership here avoids a
    # third frame copy while the generic callback retains one original view.
    wrapped = RefitRecipe(
        AttachedRows(frame, attachments),
        recipe.method,
        run,
        snapshot_policy="reference",
        provenance=recipe.provenance,
    )

    def split_rows(
        rows: AttachedRows[IntoDataFrame], positions: tuple[int, ...]
    ) -> AttachedRows[IntoDataFrame]:
        return rows.partition(original_positions(positions, included), choose)

    def score(
        result: BayesianFit, rows: AttachedRows[IntoDataFrame], seed: int
    ) -> Mapping[str, torch.Tensor]:
        arguments = {
            "new_data": rows.data,
            "seed": seed,
            "num_samples": num_samples,
            **rows.keywords(scoring=True),
            **prediction,
        }
        bound = inspect.signature(result.log_likelihood).bind(**arguments)
        return result.log_likelihood(*bound.args, **bound.kwargs)

    callback = refit_callback(wrapped, split_rows, score, method_factory=method_factory)

    def invoke(split: FoldSplit) -> FoldPrediction:
        validate_split_rows(split, len(included))
        return callback(split)

    return invoke
