"""Bind source prediction grids using checked observation-axis identity."""

from typing import TYPE_CHECKING

import torch

from quivers.diagnostics.conditional import PredictionGrid
from quivers.qiec import tensor_shape

if TYPE_CHECKING:
    from quivers.modeling.model import BoundModel


def grid_data(
    bound: BoundModel, grid: PredictionGrid, response: str
) -> dict[str, object]:
    """Replace explicit row inputs while retaining unrelated fitted constants.

    Parameters
    ----------
    bound : BoundModel
        Retained checked source binding, including its unspecialized schema.
    grid : PredictionGrid
        Explicit scalar effect and conditioning columns.
    response : str
        Selected declared observation defining the row-axis identity.

    Returns
    -------
    dict[str, object]
        New predictor data with responses released. Floating tensor inputs
        retain their fitted dtype when supplied floating coordinates.

    Raises
    ------
    ValueError
        If the response lacks a named row axis, required row inputs are
        absent, or a scalar/global input is varied across grid rows.
    """
    schema = bound.specification.schema
    selected = next(
        (
            site
            for site in schema.entry.sites
            if site.name == response and site.kind == "observe"
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"{response!r} is not a declared observation")
    parameters = {parameter.name: parameter for parameter in schema.entry.parameters}
    axes = parameters[response].axes if response in parameters else ()
    row_axis = axes[0] if axes else selected.batch[0].name if selected.batch else None
    if row_axis is None:
        raise ValueError(
            "source effects require a named observation-row axis; "
            "declare a FinSet and use it in the observation tensor"
        )
    row_parameters = {
        parameter.name
        for parameter in schema.entry.parameters
        if parameter.name != response
        and parameter.axes
        and parameter.axes[0] == row_axis
    }
    row_inputs = row_parameters - set(schema.observations)
    if missing := row_inputs - set(grid.columns):
        raise ValueError(
            f"grid requires explicit conditioning for row inputs {sorted(missing)}"
        )
    if extra := set(grid.columns) - (set(schema.types) - {response}):
        raise ValueError(f"grid columns are not predictor inputs: {sorted(extra)}")
    values = {
        name: value
        for name, value in bound.data.items()
        if name not in schema.observations
    }
    for name, column in grid.to_mapping().items():
        shape = tensor_shape(schema.types[name])
        if name in row_parameters:
            value: object = column
        elif shape is None or not shape[1]:
            if any(value != column[0] for value in column):
                raise ValueError(
                    f"scalar input {name!r} cannot vary across grid rows; declare a row tensor"
                )
            value = column[0]
        else:
            raise ValueError(
                f"input {name!r} does not use the selected observation-row axis"
            )
        original = bound.data.get(name)
        if isinstance(original, torch.Tensor) and original.is_floating_point():
            floating = any(isinstance(v, float) for v in column) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in column
            )
            if floating:
                value = torch.tensor(
                    value, dtype=original.dtype, device=original.device
                )
        values[name] = value
    return values
