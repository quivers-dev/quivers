"""Tests for :mod:`quivers.data`: dataframe-to-QVR schema bridge.

The schema is validated against both pandas and polars dataframes
(supplied via Narwhals) for equivalent encoded outputs. Determinism
of categorical ordering is asserted explicitly so plate indices are
reproducible across re-runs.
"""

from __future__ import annotations

import copy
import pickle

import pandas as pd
import polars as pl
import pytest
import torch

from quivers.data import DatasetSchema, MissingPolicy, compose


@pytest.fixture
def df_pandas():
    return pd.DataFrame(
        {
            "verb": ["eat", "drink", "eat", "run", "drink", "eat"],
            "item": ["a", "b", "c", "a", "b", "c"],
            "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "response": [1, 0, 1, 1, 0, 1],
            "rt": [0.3, 0.5, 0.4, 0.2, 0.6, 0.35],
        }
    )


@pytest.fixture
def df_polars(df_pandas):
    return pl.from_pandas(df_pandas)


class TestDatasetSchemaPandas:
    def test_cardinalities(self, df_pandas):
        schema = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb", "item": "Item", "subject": "Subject"},
        )
        assert dict(schema.cardinalities) == {
            "Verb": 3,
            "Item": 3,
            "Subject": 3,
        }

    def test_declarations_sorted(self, df_pandas):
        schema = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb", "item": "Item", "subject": "Subject"},
        )
        decls = schema.declarations()
        # Sorted alphabetically by object name.
        assert (
            decls
            == "object Item : FinSet 3\nobject Subject : FinSet 3\nobject Verb : FinSet 3\n"
        )

    def test_observations_dict_categorical(self, df_pandas):
        schema = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb"},
            plate_indices={"verb": "verb_idx"},
            observations={"response": "y"},
        )
        obs = schema.observations_dict()
        # Plate index: sorted unique = ['drink', 'eat', 'run'] -> 0,1,2.
        assert obs["verb_idx"].dtype == torch.long
        assert obs["verb_idx"].tolist() == [1, 0, 1, 2, 0, 1]
        # Numeric observation passes through as float.
        assert obs["y"].dtype == torch.float32
        assert obs["y"].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    def test_observations_dict_covariate(self, df_pandas):
        schema = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb"},
            covariates={"rt": "x_rt"},
        )
        obs = schema.observations_dict()
        assert obs["x_rt"].dtype == torch.float32
        assert obs["x_rt"].shape == (6,)

    def test_categories_deterministic(self, df_pandas):
        s1 = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        s2 = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        assert s1.categories("verb") == s2.categories("verb")
        # Sorted lexicographic.
        assert s1.categories("verb") == ("drink", "eat", "run")

    def test_missing_column_raises(self, df_pandas):
        with pytest.raises(ValueError, match="not found in dataframe"):
            DatasetSchema(
                df=df_pandas, objects={"nonexistent": "X"}
            ).cardinalities  # trigger validation via derived

    def test_plate_index_requires_object(self, df_pandas):
        with pytest.raises(ValueError, match="requires the column"):
            DatasetSchema(
                df=df_pandas,
                objects={},
                plate_indices={"verb": "verb_idx"},
            ).cardinalities  # trigger derived


class TestDatasetSchemaPolars:
    def test_pandas_polars_equivalent(self, df_pandas, df_polars):
        schema_pd = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb", "item": "Item"},
            observations={"response": "y"},
            plate_indices={"verb": "verb_idx", "item": "item_idx"},
        )
        schema_pl = DatasetSchema(
            df=df_polars,
            objects={"verb": "Verb", "item": "Item"},
            observations={"response": "y"},
            plate_indices={"verb": "verb_idx", "item": "item_idx"},
        )
        assert dict(schema_pd.cardinalities) == dict(schema_pl.cardinalities)
        obs_pd = schema_pd.observations_dict()
        obs_pl = schema_pl.observations_dict()
        assert obs_pd.keys() == obs_pl.keys()
        for k in obs_pd:
            assert torch.equal(obs_pd[k], obs_pl[k]), k


class TestMissingPolicy:
    def test_raise_on_null(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0], "g": ["a", "b", "a"]})
        schema = DatasetSchema(
            df=df,
            objects={"g": "G"},
            covariates={"x": "x_covar"},
            missing_policy=MissingPolicy.RAISE,
        )
        with pytest.raises(ValueError, match="missing values"):
            schema.observations_dict()

    def test_mask_passes_through_nan(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0], "g": ["a", "b", "a"]})
        schema = DatasetSchema(
            df=df,
            objects={"g": "G"},
            covariates={"x": "x_covar"},
            missing_policy=MissingPolicy.MASK,
        )
        obs = schema.observations_dict()
        assert torch.isnan(obs["x_covar"][1])
        assert obs["x_covar"][0].item() == 1.0
        assert obs["x_covar"][2].item() == 3.0


class TestCompose:
    def test_compose_prepends_declarations(self, df_pandas):
        schema = DatasetSchema(
            df=df_pandas,
            objects={"verb": "Verb"},
            covariates={"rt": "x_rt"},
        )
        body = """
program demo : Verb -> Verb
    sample mu : Verb <- Normal(loc=0.0, scale=1.0)
    return mu

export demo
"""
        prog = compose(body, schema)
        # The compiled program is a MonadicProgram whose morphism
        # routes through the inferred Verb cardinality (3).
        assert prog._morphism is not None
        assert prog._morphism.domain.cardinality == 3

    def test_compose_respects_body_declaration(self, df_pandas):
        # If the body already declares the object, the schema's
        # prelude does not override it.
        schema = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        body = """
object Verb : FinSet 99

program demo : Verb -> Verb
    sample mu : Verb <- Normal(loc=0.0, scale=1.0)
    return mu

export demo
"""
        prog = compose(body, schema)
        # Body's `object Verb : 99` wins over the schema's `Verb : 3`.
        assert prog._morphism.domain.cardinality == 99


class TestDatasetSchemaValueSemantics:
    def test_schemas_over_one_frame_with_equal_mappings_are_equal(self, df_pandas):
        first = DatasetSchema(df=df_pandas, objects={"verb": "Verb", "item": "Item"})
        second = DatasetSchema(df=df_pandas, objects={"item": "Item", "verb": "Verb"})
        assert first == second
        assert hash(first) == hash(second)

    @pytest.mark.parametrize(
        "field", ["objects", "observations", "plate_indices", "covariates"]
    )
    def test_each_mapping_enters_equality(self, df_pandas, field):
        base = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        changed_mappings = {
            "objects": {"verb": "Verb", "item": "Item"},
            "observations": {"response": "y"},
            "plate_indices": {"verb": "verb_idx"},
            "covariates": {"rt": "rt"},
        }
        mappings: dict[str, dict[str, str]] = {
            "objects": {"verb": "Verb"},
            "observations": {},
            "plate_indices": {},
            "covariates": {},
        }
        mappings[field] = changed_mappings[field]
        changed = DatasetSchema(
            df=df_pandas,
            objects=mappings["objects"],
            observations=mappings["observations"],
            plate_indices=mappings["plate_indices"],
            covariates=mappings["covariates"],
        )
        assert changed != base
        assert hash(changed) != hash(base)

    def test_dataframe_compares_by_identity(self, df_pandas):
        schema = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        assert schema == DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        assert schema != DatasetSchema(df=df_pandas.copy(), objects={"verb": "Verb"})

    def test_copy_keeps_the_frame_and_pickle_restores_it(self, df_pandas):
        schema = DatasetSchema(df=df_pandas, objects={"verb": "Verb"})
        shallow = copy.copy(schema)
        assert shallow.df is df_pandas
        assert shallow == schema
        for clone in (copy.deepcopy(schema), pickle.loads(pickle.dumps(schema))):
            assert clone.df is not df_pandas
            assert isinstance(clone.df, pd.DataFrame)
            assert clone.df.equals(df_pandas)
            assert clone.objects == {"verb": "Verb"}
            assert dict(clone.cardinalities) == dict(schema.cardinalities)
