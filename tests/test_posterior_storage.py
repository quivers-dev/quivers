"""Retained analyses round-trip the existing labeled ArviZ tensor groups."""

import numpy as np
import pytest
import torch
import xarray as xr

from quivers.diagnostics import PosteriorAnalysis


@pytest.mark.parametrize("kind", ["mcmc", "variational", "prior"])
def test_datatree_and_netcdf_roundtrip_values_labels_provenance_and_stat_names(
    tmp_path, kind
):
    chains = 2 if kind == "mcmc" else 1
    values = torch.arange(chains * 5 * 3, dtype=torch.float64).reshape(chains, 5, 3) / 7
    scores = -values.square()
    analysis = PosteriorAnalysis(
        samples={"theta": values},
        log_likelihood={"y": scores},
        observed_data={"y": torch.tensor([2, 1, 0])},
        kind=kind,
        sample_stats={
            "diverged": torch.zeros(chains, 5, dtype=torch.bool),
            "num_steps": torch.full((chains, 5), 3, dtype=torch.long),
        }
        if kind == "mcmc"
        else {},
        coords={
            "coordinate": ["b", "a", "c"],
            "row": [12, 10, 11],
            "chain": list(range(chains)),
            "draw": [3, 5, 7, 9, 11],
        },
        dims={"theta": ["coordinate"], "y": ["row"]},
    )
    tree = analysis.to_datatree()
    outputs = [PosteriorAnalysis.from_datatree(tree)]
    path = tmp_path / "posterior.nc"
    analysis.to_netcdf(path)
    outputs.append(PosteriorAnalysis.from_netcdf(path))
    for restored in outputs:
        assert (
            restored.kind == kind
            and restored.coords == analysis.coords
            and restored.dims == analysis.dims
        )
        for field in ("samples", "log_likelihood", "observed_data", "sample_stats"):
            assert getattr(restored, field).keys() == getattr(analysis, field).keys()
            for name, expected in getattr(analysis, field).items():
                actual = getattr(restored, field)[name]
                torch.testing.assert_close(actual, expected)
                assert actual.data_ptr() != expected.data_ptr()
        torch.testing.assert_close(restored.waic().elpd_i, analysis.waic().elpd_i)
        covariance = restored.covariance()
        assert covariance.coordinate_labels == (("b",), ("a",), ("c",))
        torch.testing.assert_close(
            covariance.covariance, analysis.covariance().covariance
        )


def test_external_tree_requires_explicit_provenance_and_transposes_named_axes():
    array = np.arange(12, dtype=float).reshape(3, 2, 2)
    tree = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"theta": (("coordinate", "draw", "chain"), array)},
                coords={"coordinate": ["x", "y", "z"]},
            )
        }
    )
    with pytest.raises(ValueError, match="explicit valid"):
        PosteriorAnalysis.from_datatree(tree)
    restored = PosteriorAnalysis.from_datatree(tree, kind="mcmc")
    torch.testing.assert_close(
        restored.samples["theta"], torch.tensor(array.transpose(2, 1, 0))
    )
    assert restored.dims["theta"] == ["coordinate"]


def test_restoration_rejects_format_provenance_axis_and_stat_alias_conflicts():
    analysis = PosteriorAnalysis(
        samples={"a": torch.zeros(1, 3)}, log_likelihood={}, kind="variational"
    )
    tree = analysis.to_datatree()
    with pytest.raises(ValueError, match="disagrees"):
        PosteriorAnalysis.from_datatree(tree, kind="mcmc")
    tree.attrs["quivers_posterior_format"] = "qvr-posterior/v999"
    with pytest.raises(ValueError, match="unsupported"):
        PosteriorAnalysis.from_datatree(tree)
    external = xr.DataTree.from_dict(
        {"posterior": xr.Dataset({"a": (("sample",), np.zeros(3))})}
    )
    with pytest.raises(ValueError, match="chain and draw"):
        PosteriorAnalysis.from_datatree(external, kind="mcmc")
    aliases = PosteriorAnalysis(
        samples={"a": torch.zeros(1, 3)},
        log_likelihood={},
        sample_stats={"diverged": torch.zeros(1, 3), "diverging": torch.ones(1, 3)},
    )
    with pytest.raises(ValueError, match="aliases collide"):
        aliases.to_datatree()


def test_failed_netcdf_write_preserves_previous_file_and_removes_own_staging(
    tmp_path, monkeypatch
):
    target = tmp_path / "analysis.nc"
    target.write_bytes(b"previous valid artifact")
    analysis = PosteriorAnalysis(samples={"a": torch.zeros(1, 3)}, log_likelihood={})

    def fail(tree, path, **kwargs):
        path.write_bytes(b"interrupted")
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(xr.DataTree, "to_netcdf", fail)
    with pytest.raises(RuntimeError, match="write interrupted"):
        analysis.to_netcdf(target)
    assert target.read_bytes() == b"previous valid artifact"
    assert list(tmp_path.iterdir()) == [target]


def test_vector_observation_events_keep_group_specific_axes_after_likelihood_reduction(
    tmp_path,
):
    observed = torch.tensor([[1, 2, 0], [0, 1, 2]], dtype=torch.int64)
    analysis = PosteriorAnalysis(
        samples={"theta": torch.ones(1, 4)},
        log_likelihood={"y": torch.full((1, 4, 2), -1.0)},
        observed_data={"y": observed},
        kind="variational",
    )
    tree = analysis.to_datatree()
    restored = PosteriorAnalysis.from_datatree(tree)
    assert len(restored.group_dims["log_likelihood"]["y"]) == 1
    assert len(restored.group_dims["observed_data"]["y"]) == 2
    restored.to_netcdf(tmp_path / "vector-count.nc")
    final = PosteriorAnalysis.from_netcdf(tmp_path / "vector-count.nc")
    torch.testing.assert_close(final.observed_data["y"], observed)
    assert final.log_likelihood["y"].shape == (1, 4, 2)


def test_parameter_free_zero_observation_plate_restores_real_draw_axes(tmp_path):
    analysis = PosteriorAnalysis(
        samples={}, log_likelihood={"y": torch.empty(1, 3, 0)}, kind="variational"
    )
    analysis.to_netcdf(tmp_path / "empty.nc")
    restored = PosteriorAnalysis.from_netcdf(tmp_path / "empty.nc")
    assert not restored.samples and restored.log_likelihood["y"].shape == (1, 3, 0)
    assert restored.kind == "variational"


def test_existing_canonical_statistic_name_is_not_renamed_on_roundtrip():
    analysis = PosteriorAnalysis(
        samples={"a": torch.zeros(1, 3)},
        log_likelihood={},
        sample_stats={"diverging": torch.zeros(1, 3, dtype=torch.bool)},
    )
    assert PosteriorAnalysis.from_datatree(
        analysis.to_datatree()
    ).sample_stats.keys() == {"diverging"}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_posterior_roundtrip_is_exact_without_unsupported_numpy_dtype(
    tmp_path, dtype
):
    values = torch.tensor([[[0.25, -0.5], [0.75, 1.5], [2.0, -2.0]]], dtype=dtype)
    analysis = PosteriorAnalysis(
        samples={"a": values}, log_likelihood={}, kind="variational"
    )
    path = tmp_path / "low-precision.nc"
    analysis.to_netcdf(path)
    restored = PosteriorAnalysis.from_netcdf(path)
    torch.testing.assert_close(restored.samples["a"], values, rtol=0, atol=0)


def test_integer_prior_storage_preserves_original_capacity_before_summary_casting(
    tmp_path,
):
    values = torch.tensor([[2**53 + 1, 2**53 + 3]], dtype=torch.int64)
    analysis = PosteriorAnalysis(
        samples={"count": values}, log_likelihood={}, kind="prior"
    )
    analysis.to_netcdf(tmp_path / "counts.nc")
    restored = PosteriorAnalysis.from_netcdf(tmp_path / "counts.nc")
    torch.testing.assert_close(restored.samples["count"], values)
    with pytest.raises(ValueError, match="2\\*\\*53"):
        restored.summary()


def test_restore_rejects_sample_axis_reordering_between_groups_and_lossy_metadata():
    posterior = xr.Dataset(
        {"a": (("chain", "draw"), np.zeros((1, 3)))}, coords={"draw": [1, 2, 3]}
    )
    likelihood = xr.Dataset(
        {"y": (("chain", "draw", "row"), np.zeros((1, 3, 1)))},
        coords={"draw": [3, 2, 1]},
    )
    with pytest.raises(ValueError, match="disagree on labels"):
        PosteriorAnalysis.from_datatree(
            xr.DataTree.from_dict(
                {"posterior": posterior, "log_likelihood": likelihood}
            ),
            kind="mcmc",
        )
    tree = PosteriorAnalysis(
        samples={"a": torch.full((1, 3), 0.3)}, log_likelihood={}
    ).to_datatree()
    tree.attrs["quivers_tensor_dtypes"] = '{"posterior":{"a":"torch.float16"}}'
    with pytest.raises(ValueError, match="not exact"):
        PosteriorAnalysis.from_datatree(tree)


@pytest.mark.parametrize("name,value", [("diverged", 0.25), ("num_steps", 1.5)])
def test_canonical_statistic_encoding_refuses_silent_numeric_truncation(name, value):
    analysis = PosteriorAnalysis(
        samples={"a": torch.zeros(1, 3)},
        log_likelihood={},
        sample_stats={name: torch.full((1, 3), value)},
    )
    with pytest.raises(ValueError, match="requires"):
        analysis.to_datatree()


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([[2**63]], dtype=torch.uint64),
        torch.tensor([[-1]], dtype=torch.int64),
    ],
)
def test_canonical_count_statistics_check_original_integer_capacity(value):
    analysis = PosteriorAnalysis(
        samples={"a": torch.zeros(1, 1)},
        log_likelihood={},
        sample_stats={"num_steps": value},
    )
    with pytest.raises(ValueError, match="Int64 capacity"):
        analysis.to_datatree()
