"""Edward2 half-t responses retain the folded law beyond a density shift."""

import math
from pathlib import Path

import pytest
from scipy import stats

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from tests.transpile import _docker

SOURCE = """
object Row : FinSet 3
program half_t(scale: Real) : Row -> Row
    observe y : Row <- HalfStudentT(df=3.0, scale=scale)
    return y
export half_t
"""
SCRIPT = Path(__file__).parent / "probes" / "_scripts" / "half_student_edward2.py"


def assert_half_student_measurement(result):
    scale = 1.5
    values = [0.0, 0.5, 1.0]
    expected = stats.t.logpdf(values, df=3, scale=scale) + math.log(2)
    assert result["density"] == pytest.approx(expected.tolist(), abs=2e-6)
    gradient = sum(
        -1 / scale + 4 * value**2 / (scale * (3 * scale**2 + value**2))
        for value in values
    )
    assert result["gradient"] == pytest.approx(gradient, abs=2e-6)
    assert result["negative_is_impossible"]
    assert result["generated_nonnegative"]
    assert result["sample_nonnegative"]
    assert result["sample_mean"] == pytest.approx(
        scale * 2 * math.sqrt(3) / math.pi, abs=0.04
    )
    expected_cdf = [
        2 * stats.t.cdf(value / scale, df=3) - 1 for value in [0.25, 1.0, 3.0, 8.0]
    ]
    assert result["sample_cdf"] == pytest.approx(expected_cdf, abs=0.006)


def test_edward2_half_student_density_support_derivatives_and_sampling(scratch):
    image = "panproto-test-edward2"
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    directory = scratch / "half-student-edward2"
    directory.mkdir(parents=True)
    (directory / "source.py").write_bytes(transpile(parse(SOURCE), target="edward2"))
    result = _docker.run_probe_script(image=image, script=SCRIPT, scratch=directory)
    assert_half_student_measurement(result)
