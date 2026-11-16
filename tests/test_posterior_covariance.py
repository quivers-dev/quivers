"""Retained joint summaries agree with independent empirical calculations."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import pearsonr

from quivers.diagnostics import PosteriorAnalysis, PosteriorCovariance
from quivers.formulas import fit
from quivers.modeling import Model, VariationalMethod


def _analysis(samples, kind="mcmc"):
    return PosteriorAnalysis(samples=samples, log_likelihood={}, kind=kind)


@pytest.mark.parametrize("chainwise", [False, True])
@pytest.mark.parametrize("ddof", [0, 1, 3])
def test_joint_scalar_matrix_summary_matches_numpy_and_scipy(chainwise, ddof):
    rng = np.random.default_rng(81)
    raw = rng.normal(size=(3, 17, 5)) @ rng.normal(size=(5, 5))
    samples = {
        "scalar": torch.tensor(raw[..., 0]),
        "matrix": torch.tensor(raw[..., 1:].reshape(3, 17, 2, 2)),
    }
    result = _analysis(samples).covariance(
        ["matrix", "scalar"], ddof=ddof, chainwise=chainwise
    )
    assert isinstance(result, PosteriorCovariance)
    ordered = np.concatenate((raw[..., 1:], raw[..., :1]), axis=-1)
    values = ordered if chainwise else ordered.reshape(-1, 5)[None]
    cov = np.stack([np.cov(value, rowvar=False, ddof=ddof) for value in values])
    corr = np.stack([np.corrcoef(value, rowvar=False) for value in values])
    np.testing.assert_allclose(
        result.covariance, cov if chainwise else cov[0], rtol=3e-14, atol=3e-14
    )
    np.testing.assert_allclose(
        result.correlation, corr if chainwise else corr[0], rtol=3e-14, atol=3e-14
    )
    first = result.correlation[0] if chainwise else result.correlation
    assert first[0, 4].item() == pytest.approx(
        pearsonr(values[0, :, 0], values[0, :, 4]).statistic, abs=2e-14
    )
    assert result.coordinates == (
        ("matrix", (0, 0)),
        ("matrix", (0, 1)),
        ("matrix", (1, 0)),
        ("matrix", (1, 1)),
        ("scalar", ()),
    )
    assert result.event_shapes == ((2, 2), ())
    assert result.sample_shape == (3, 17) and result.kind == "mcmc"


def test_named_coordinates_keep_exact_site_names_labels_and_sample_axes():
    values = torch.arange(2 * 5 * 2, dtype=torch.float64).reshape(2, 5, 2)
    result = _analysis({"a[0]": values, "scalar": values[..., 0]}).covariance(
        coords={
            "component": ["left,[]", "right"],
            "chain": ["A", "B"],
            "draw": list("abcde"),
        },
        dims={"a[0]": ["component"]},
    )
    assert result.coordinates == (("a[0]", (0,)), ("a[0]", (1,)), ("scalar", ()))
    assert result.dimension_names == (("component",), ("component",), ())
    assert result.coordinate_labels == (("left,[]",), ("right",), ())
    assert result.chain_labels == ("A", "B") and result.draw_labels == tuple("abcde")
    assert result.variables == ("a[0]", "scalar")


def test_covariance_and_datatree_share_coordinate_validation_and_default_names():
    analysis = _analysis({"a": torch.randn(2, 5, 2, 3)})
    coords = {"row": ["one", "two"], "a_dim_0": [10, 20, 30]}
    dims = {"a": ["row"]}
    result = analysis.covariance(coords=coords, dims=dims)
    tree = analysis.to_datatree(coords=coords, dims=dims)
    assert result.dimension_names[0] == tuple(tree["posterior"].dataset["a"].dims[2:])
    assert result.coordinate_labels[4] == ("two", 20)
    for operation in (analysis.covariance, analysis.to_datatree):
        with pytest.raises(ValueError, match="requires 2 labels"):
            operation(coords={"row": ["one"]}, dims=dims)


@pytest.mark.parametrize("chainwise", [False, True])
def test_constant_coordinates_have_zero_covariance_and_undefined_correlations(
    chainwise,
):
    values = torch.arange(12, dtype=torch.float64).reshape(2, 6)
    result = _analysis(
        {"x": values, "constant": torch.full_like(values, 7)}
    ).covariance(chainwise=chainwise)
    assert bool((result.covariance[..., 1, :] == 0).all())
    assert bool((result.covariance[..., :, 1] == 0).all())
    assert bool(torch.isnan(result.correlation[..., 1, :]).all())
    assert bool(torch.isnan(result.correlation[..., :, 1]).all())
    torch.testing.assert_close(
        result.correlation[..., 0, 0], torch.ones_like(result.correlation[..., 0, 0])
    )


def test_finite_extreme_and_subnormal_samples_do_not_corrupt_correlation():
    tiny = torch.tensor([1e-200, 2e-200, 3e-200, 4e-200], dtype=torch.float64).reshape(
        1, 4
    )
    result = _analysis({"tiny": tiny, "large": tiny * 1e200}).covariance()
    assert result.covariance[0, 0] == 0  # Its mathematical variance underflows Float64.
    torch.testing.assert_close(
        result.correlation, torch.ones(2, 2, dtype=torch.float64)
    )
    constant = _analysis(
        {"x": torch.full((1, 4), 1e308, dtype=torch.float64)}
    ).covariance()
    assert constant.covariance.item() == 0 and bool(
        torch.isnan(constant.correlation).all()
    )
    with pytest.raises(ValueError, match="Float64 capacity"):
        _analysis(
            {"x": torch.tensor([[-1e308, 1e308]], dtype=torch.float64)}
        ).covariance()


def test_integer_and_boolean_prior_draws_preserve_numeric_carriers():
    counts = torch.tensor([[2**53 - 3, 2**53 - 2, 2**53 - 1, 2**53]], dtype=torch.int64)
    choices = torch.tensor([[False, True, False, True]])
    result = _analysis({"count": counts, "choice": choices}, kind="prior").covariance()
    expected = np.cov(np.stack([np.arange(4), [0, 1, 0, 1]]), ddof=1)
    np.testing.assert_allclose(result.covariance, expected, atol=1e-14)
    assert result.kind == "prior" and result.sample_shape == (1, 4)


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), -(2**63)])
def test_unrepresentable_original_integer_prior_draws_are_refused(value):
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _analysis(
            {"x": torch.tensor([[0, value]], dtype=torch.int64)}, kind="prior"
        ).covariance()


@pytest.mark.parametrize("var_names", [[], ["a", "a"], ["missing"]])
def test_invalid_selection_is_refused(var_names):
    with pytest.raises(ValueError):
        _analysis({"a": torch.zeros(1, 4)}).covariance(var_names)


@pytest.mark.parametrize("ddof", [-1, True, 1.5, 4])
def test_invalid_degrees_of_freedom_are_refused(ddof):
    with pytest.raises(ValueError):
        _analysis({"a": torch.zeros(1, 4)}).covariance(ddof=ddof)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_missing_values_are_refused_without_pairwise_deletion(value):
    with pytest.raises(ValueError, match="finite samples"):
        _analysis({"a": torch.tensor([[0.0, value, 1.0]])}).covariance()


@pytest.mark.parametrize(
    "dims,coords",
    [
        ({"a": ["row", "extra"]}, {}),
        ({"a": ["chain"]}, {}),
        ({"a": ["row"]}, {"row": [1]}),
    ],
)
def test_invalid_event_axis_metadata_is_refused(dims, coords):
    with pytest.raises(ValueError):
        _analysis({"a": torch.zeros(1, 4, 2)}).covariance(dims=dims, coords=coords)


def test_conflicting_sample_axes_and_chainwise_degrees_are_refused():
    with pytest.raises(ValueError, match="incompatible chain/draw"):
        _analysis({"a": torch.zeros(2, 4), "b": torch.zeros(2, 5)}).covariance()
    analysis = _analysis({"a": torch.zeros(2, 2)})
    assert analysis.covariance(ddof=2).covariance.shape == (1, 1)
    with pytest.raises(ValueError, match="reduced sample count"):
        analysis.covariance(ddof=2, chainwise=True)


@pytest.mark.parametrize("source", [False, True])
def test_actual_fits_reuse_one_retained_posterior_without_sampling_again(source):
    if source:
        model = Model("""object Row : FinSet _
object Out : Real 1
program p(x:Tensor[Real]([Row]), y:Tensor[Real]([Row])) : Row -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = a+b*x
    observe y:Row <- Normal(loc=mu, scale=1.0)
    return a
export p
""")
        fitted = model.fit(
            {"x": torch.arange(4.0), "y": torch.tensor([0.2, 0.4, 0.1, 0.7])},
            method=VariationalMethod(num_steps=1, seed=2),
        )
    else:
        fitted = fit(
            "y~x",
            data=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.2, 0.4, 0.1, 0.7]}),
            fixed_parameters={"sigma": 1.0},
            method="svi",
            num_steps=1,
            seed=2,
        )
    analysis = fitted.analysis(num_samples=8, seed=7)
    state = torch.get_rng_state().clone()
    with patch.object(
        type(fitted.posterior), "rsample", side_effect=AssertionError("replayed guide")
    ):
        result = analysis.covariance()
        repeated = analysis.covariance(chainwise=True)
        assert result.kind == repeated.kind == "variational"
        assert all(
            summary.convergence is None for summary in analysis.summary().values()
        )
    assert torch.equal(state, torch.get_rng_state())
    raw = np.concatenate(
        [value.double().reshape(8, -1).numpy() for value in analysis.samples.values()],
        axis=1,
    )
    np.testing.assert_allclose(
        result.covariance, np.cov(raw, rowvar=False), atol=1e-13, rtol=1e-13
    )
    torch.testing.assert_close(result.covariance, repeated.covariance[0])


@pytest.mark.parametrize(
    "value,valid", [(2**53, True), (2**53 + 1, False), (2**64 - 1, False)]
)
def test_unsigned_prior_capacity_is_checked_before_conversion(value, valid):
    analysis = _analysis(
        {"x": torch.tensor([[0, value]], dtype=torch.uint64)}, kind="prior"
    )
    if valid:
        assert bool(torch.isfinite(analysis.covariance().covariance).all())
    else:
        with pytest.raises(ValueError, match=r"2\*\*53"):
            analysis.covariance()


def test_shared_numeric_adapter_detaches_preserves_device_and_rejects_complex():
    from quivers.diagnostics._numeric import real_samples

    values = torch.ones(2, 3, dtype=torch.float32, requires_grad=True)
    converted = real_samples(values, label="example")
    assert converted.shape == values.shape and converted.device == values.device
    assert converted.dtype == torch.float64 and not converted.requires_grad
    with pytest.raises(TypeError, match="real sample tensor"):
        real_samples(torch.ones(2, dtype=torch.complex64), label="example")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dims": {"a": ["same", "same"]}},
        {"dims": {"a": ["same"], "b": ["same"]}},
        {"coords": {"chain": [0]}},
        {"coords": {"draw": [0]}},
        {"dims": {"b": "x"}},
    ],
)
def test_axis_labels_and_shared_named_extents_must_agree(kwargs):
    analysis = _analysis({"a": torch.zeros(2, 4, 2, 3), "b": torch.zeros(2, 4, 3)})
    with pytest.raises((ValueError, TypeError)):
        analysis.covariance(**kwargs)


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_recorded_sampler_chains_have_independent_covariance_axes(method):
    fitted = fit(
        "y~x",
        data=pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.2, 0.1, 0.7]}),
        fixed_parameters={"sigma": 1.0},
        method=method,
        num_warmup=2,
        num_samples=4,
        num_chains=2,
        seed=9,
    )
    analysis = fitted.analysis()
    result = analysis.covariance(chainwise=True)
    assert result.sample_shape == (2, 4) and result.kind == "mcmc"
    raw = np.concatenate(
        [
            values.double().reshape(2, 4, -1).numpy()
            for values in analysis.samples.values()
        ],
        axis=-1,
    )
    oracle = np.stack([np.cov(chain, rowvar=False) for chain in raw])
    np.testing.assert_allclose(result.covariance, oracle, atol=1e-13, rtol=1e-13)


def test_structural_coordinate_record_survives_copy_and_serialization():
    import copy
    import pickle

    result = _analysis(
        {"site[0]": torch.arange(8, dtype=torch.float64).reshape(1, 4, 2)}, kind="prior"
    ).covariance(dims={"site[0]": ["level"]}, coords={"level": ["a", "b"]})
    for restored in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        assert restored.coordinates == (("site[0]", (0,)), ("site[0]", (1,)))
        assert restored.coordinate_labels == (("a",), ("b",))
        assert restored.sample_shape == (1, 4)
        torch.testing.assert_close(restored.covariance, result.covariance)


def test_covariance_does_not_require_optional_arviz_packages():
    import os
    from pathlib import Path
    import subprocess
    import sys

    code = """import sys
import torch
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'arviz', 'xarray'}:
            raise ModuleNotFoundError('optional diagnostics blocked')
sys.meta_path.insert(0, BlockOptional())
from quivers.diagnostics import PosteriorAnalysis
analysis = PosteriorAnalysis(samples={'x':torch.tensor([[0.,1.,2.]])},log_likelihood={})
result = analysis.covariance()
assert result.covariance.item() == 1.0
assert result.correlation.item() == 1.0
assert 'arviz' not in sys.modules and 'xarray' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
