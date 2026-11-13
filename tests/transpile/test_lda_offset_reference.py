"""Scalar probability calculations verify the LDA offset reference."""

from pathlib import Path
import math
from unittest.mock import patch

import pytest
import torch

from tests.transpile import _gallery_data, _equivalence
from tests.transpile.probes._protocol import Point
from tests.transpile.test_expected_offsets import _qvr_log_densities


def _dirichlet(values: list[float], concentration: float) -> float:
    """Evaluate a symmetric Dirichlet density with scalar arithmetic.

    Parameters
    ----------
    values
        Represented simplex coordinates.
    concentration
        Positive shared concentration.

    Returns
    -------
    float
        Log density including its normalizer.
    """
    return (
        math.lgamma(len(values) * concentration)
        - len(values) * math.lgamma(concentration)
        + math.fsum((concentration - 1) * math.log(value) for value in values)
    )


def _scalar_joint(
    dataset: _gallery_data.GalleryDataset, point: Point, *, shared_topic: bool = False
) -> float:
    """Sum the indexed token groups and Dirichlet priors independently of QVR.

    Parameters
    ----------
    dataset
        Shapes of the topic weights and vocabulary distributions.
    point
        Fixed latent values, words, and document indices.
    shared_topic
        Use the deliberately wrong model with one topic per document.

    Returns
    -------
    float
        Joint log density from Python scalar operations.
    """
    theta = (
        torch.tensor(point.params["theta"], dtype=torch.float64)
        .reshape(dataset.params["theta"].shape)
        .tolist()
    )
    phi = (
        torch.tensor(point.params["phi"], dtype=torch.float64)
        .reshape(dataset.params["phi"].shape)
        .tolist()
    )
    alpha, beta = float(point.data["alpha"]), float(point.data["beta"])
    prior = math.fsum(_dirichlet(row, alpha) for row in theta) + math.fsum(
        _dirichlet(row, beta) for row in phi
    )
    theta_prob = [[value / math.fsum(row) for value in row] for row in theta]
    phi_prob = [[value / math.fsum(row) for value in row] for row in phi]
    words = [int(value) for value in point.data["w"]]
    documents = [int(value) for value in point.data["doc_idx"]]
    # Each observation maps to a token group. Repeated indices share its topic;
    # an empty group integrates its normalized topic prior to one.
    groups: list[list[int]] = [
        [] for _ in range(len(theta_prob) if shared_topic else len(documents))
    ]
    for word, token in zip(words, point.data["token_idx"], strict=True):
        group = documents[int(token)] if shared_topic else int(token)
        groups[group].append(word)
    terms = []
    for group, group_words in enumerate(groups):
        topic_weights = theta_prob[group if shared_topic else documents[group]]
        topic_scores = [
            math.log(weight)
            + math.fsum(math.log(phi_prob[topic][word]) for word in group_words)
            for topic, weight in enumerate(topic_weights)
        ]
        maximum = max(topic_scores)
        terms.append(
            maximum
            + math.log(math.fsum(math.exp(value - maximum) for value in topic_scores))
        )
    return prior + math.fsum(terms)


def test_lda_offset_reference_matches_scalar_joint(tmp_path: Path) -> None:
    """Check all six original points without changing the data or tolerance.

    Parameters
    ----------
    tmp_path
        Unused scratch directory required by the probe interface.
    """
    example = Path("docs/examples/source/lda.qvr")
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None
    points = _gallery_data.points_from_dataset(dataset)
    dtype = torch.get_default_dtype()
    actual = _qvr_log_densities(example, dataset, points, tmp_path)
    assert torch.get_default_dtype() == dtype
    expected = [_scalar_joint(dataset, point) for point in points]
    assert len(actual) == len(expected) == 6
    for left, right in zip(actual, expected, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-9)
    wrong = [_scalar_joint(dataset, point, shared_topic=True) for point in points]
    with pytest.raises(AssertionError, match="spread"):
        _equivalence.assert_log_density_match(expected, wrong, atol=5e-4)


def test_reference_precision_is_restored_after_a_probe_error(tmp_path: Path) -> None:
    """A rejected point must not change later gallery pages' arithmetic.

    Parameters
    ----------
    tmp_path
        Scratch directory required by the probe interface.
    """
    example = Path("docs/examples/source/lda.qvr")
    dataset = _gallery_data.load_gallery_data(example)
    assert dataset is not None
    points = _gallery_data.points_from_dataset(dataset)
    previous = torch.get_default_dtype()
    with (
        patch(
            "tests.transpile._gallery_data.QvrProbe.evaluate",
            side_effect=RuntimeError("clamping rejected"),
        ),
        pytest.raises(RuntimeError, match="clamping rejected"),
    ):
        _gallery_data.reference_log_densities(
            example, dataset, points, scratch=tmp_path
        )
    assert torch.get_default_dtype() == previous
