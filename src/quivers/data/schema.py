"""Dataframe-to-QVR schema bridge.

`DatasetSchema` is the single point that turns "I have a
dataframe" into "I have the object cardinalities, the observations
dict, and the plate-index tensors a QVR program needs." It accepts
pandas, polars, or any other
[Narwhals](https://narwhals-dev.github.io/narwhals/)-compatible
dataframe and emits the two artefacts inference consumes:

* A ``.qvr`` declaration prelude with one ``object X : N`` line per
  declared object axis. ``N`` is derived from
  ``df[col].n_unique()``; the canonical ordering of categories is
  cached so plate indices are reproducible across reruns.

* An ``observations`` dict mapping observe-site / plate-index names
  to `torch.Tensor` values, ready to hand into
  `quivers.inference.MCMC.run` or
  `quivers.inference.SVI.step`.

The companion `compose` wraps `quivers.dsl.loads` so a
user can write a ``.qvr`` body without spelling out
``object Verb : 40`` when ``40`` came from a dataframe column anyway.
"""

from __future__ import annotations

from typing import Mapping

import didactic.api as dx
import narwhals as nw
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.data.encoding import (
    ColumnRole,
    MissingPolicy,
    encode_column,
)
from quivers.dsl import loads


class DatasetSchema(ValueModel):
    """Mapping from dataframe columns to QVR program artefacts.

    Schemas compare by value over their column mappings and missing
    policy, and by identity over ``df``: two schemas are equal only
    when they map the same columns the same way over the same
    dataframe object. Copying a schema keeps the dataframe object;
    deep-copying or pickling copies it, so the copy compares unequal.

    Attributes
    ----------
    df : IntoDataFrame
        Source dataframe; pandas, polars, modin, dask, pyarrow, or
        anything else Narwhals' ``from_native`` accepts. Stored as
        an opaque field compared by identity, so the schema does not
        depend on a specific dataframe flavour.
    objects : dict[str, str]
        Map from column name to the QVR object name. The object's
        cardinality is inferred from the column's number of unique
        values; the canonical ordering is the sorted set of unique
        values, so plate indices are deterministic across reruns.
    observations : dict[str, str]
        Map from column name to the QVR observe-site name. Categorical
        columns are encoded to ``LongTensor`` codes (using either
        their own object's category ordering, when the column is
        also listed under ``objects``, or a sorted-unique fallback);
        numeric columns to ``FloatTensor``.
    plate_indices : dict[str, str]
        Map from column name (which must also appear under
        ``objects``) to the per-row plate-index variable name.
        Encoded as ``LongTensor`` of category codes; one entry per
        row.
    covariates : dict[str, str]
        Map from numeric column name to the QVR variable name to
        bind the column's values to (as a ``FloatTensor``).
    missing_policy : MissingPolicy
        Policy applied to every column with nulls. Default
        `quivers.data.encoding.MissingPolicy.RAISE`.
    """

    df: IntoDataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    objects: dict[str, str] = dx.field(default_factory=dict)
    observations: dict[str, str] = dx.field(default_factory=dict)
    plate_indices: dict[str, str] = dx.field(default_factory=dict)
    covariates: dict[str, str] = dx.field(default_factory=dict)
    missing_policy: MissingPolicy = MissingPolicy.RAISE

    @dx.derived
    def _nw_df(self) -> nw.DataFrame:
        nw_df = nw.from_native(self.df, eager_only=True)
        columns = set(nw_df.columns)
        for spec_name, mapping in [
            ("objects", self.objects),
            ("observations", self.observations),
            ("plate_indices", self.plate_indices),
            ("covariates", self.covariates),
        ]:
            for col in mapping:
                if col not in columns:
                    raise ValueError(
                        f"DatasetSchema: column {col!r} referenced in "
                        f"{spec_name} not found in dataframe; available "
                        f"columns: {sorted(columns)}"
                    )
        for col in self.plate_indices:
            if col not in self.objects:
                raise ValueError(
                    f"DatasetSchema: plate index for column {col!r} "
                    f"requires the column to also appear under "
                    f"`objects` (so the axis cardinality is declared)"
                )
        return nw_df

    @dx.derived
    def _categories(self) -> dict[str, tuple[str, ...]]:
        """Sorted-unique canonical ordering of values for each object
        column.  Plate-index codes are assigned by
        ``categories.index(value)``; the ordering is deterministic
        across reruns of the same dataframe.
        """
        return {
            col: tuple(
                str(v) for v in self._nw_df[col].drop_nulls().unique().sort().to_list()
            )
            for col in self.objects
        }

    @dx.derived
    def cardinalities(self) -> Mapping[str, int]:
        """Inferred object cardinalities, keyed by QVR object name."""
        # Touch _nw_df to trigger validation even on schemas that
        # declare no object columns.
        _ = self._nw_df
        return {
            obj_name: len(self._categories[col])
            for col, obj_name in self.objects.items()
        }

    def categories(self, column: str) -> tuple[str, ...]:
        """Canonical ordering of values for an object-column.

        Codes are assigned as ``categories.index(value)``; the
        ordering is the column's sorted unique non-null values, so
        the same dataframe always produces the same indices.
        """
        if column not in self._categories:
            raise KeyError(
                f"DatasetSchema.categories: column {column!r} is not "
                f"declared as an object column"
            )
        return self._categories[column]

    def declarations(self) -> str:
        """Emit a ``.qvr`` declaration prelude.

        Lines are ``object <Name> : FinSet <cardinality>``, sorted
        by name for reproducibility. Suitable for prepending to a
        user's ``.qvr`` source via `compose`.
        """
        sorted_objs = sorted(self.objects.items(), key=lambda kv: kv[1])
        lines = [
            f"object {obj_name} : FinSet {self.cardinalities[obj_name]}"
            for _, obj_name in sorted_objs
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    def observations_dict(self) -> dict[str, torch.Tensor]:
        """Build the observations dict for inference.

        Contains entries for every observation, plate-index, and
        covariate column.  Categorical observations and plate
        indices use the canonical ordering returned by
        `categories`; numeric observations and covariates
        become ``FloatTensor``.
        """
        result: dict[str, torch.Tensor] = {}

        for col, site in self.observations.items():
            cats: tuple[str, ...] | None = None
            if col in self.objects:
                cats = self._categories[col]
            else:
                dtype = self._nw_df[col].dtype
                if dtype == nw.String:
                    cats = tuple(
                        str(v)
                        for v in self._nw_df[col].drop_nulls().unique().sort().to_list()
                    )
            result[site] = encode_column(
                self._nw_df,
                col,
                role=ColumnRole.OBSERVATION,
                categories=cats,
                missing_policy=self.missing_policy,
            )

        for col, var in self.plate_indices.items():
            result[var] = encode_column(
                self._nw_df,
                col,
                role=ColumnRole.PLATE_INDEX,
                categories=self._categories[col],
                missing_policy=self.missing_policy,
            )

        for col, var in self.covariates.items():
            result[var] = encode_column(
                self._nw_df,
                col,
                role=ColumnRole.COVARIATE,
                missing_policy=self.missing_policy,
            )

        return result


def compose(qvr_body: str, schema: DatasetSchema, **kwargs):
    """Compile a ``.qvr`` body against a dataset schema.

    Prepends the schema's ``object`` declarations to ``qvr_body``,
    then calls `quivers.dsl.loads`.  The user writes only the
    program body (latents, kernels, observations, return); object
    cardinalities inferred from the dataframe are slotted in
    automatically.  If the body re-declares an object that appears
    in the schema, the body's declaration wins.

    Parameters
    ----------
    qvr_body : str
        QVR source without the ``object`` declarations covered by
        ``schema.objects``.
    schema : DatasetSchema
        Dataframe schema providing cardinalities.
    **kwargs
        Forwarded to `quivers.dsl.loads` (e.g. ``data=...`` for
        ``from_data`` lookups).
    """
    body_declares: set[str] = set()
    for line in qvr_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("object "):
            after = stripped[len("object ") :].split(":")[0].split("=")[0]
            body_declares.add(after.strip())

    prelude_lines = []
    for _, obj_name in sorted(schema.objects.items(), key=lambda kv: kv[1]):
        if obj_name in body_declares:
            continue
        prelude_lines.append(
            f"object {obj_name} : FinSet {schema.cardinalities[obj_name]}"
        )
    prelude = "\n".join(prelude_lines)
    if prelude:
        prelude += "\n\n"
    return loads(prelude + qvr_body, **kwargs)


__all__ = ["DatasetSchema", "compose"]
