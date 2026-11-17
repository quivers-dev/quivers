"""Native vector and matrix event axes do not enter parameter batch matching."""

import pytest
import torch

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program
from quivers.modeling import Model


def _host(value):
    return tuple(_host(item) for item in value) if isinstance(value, list) else value


@pytest.mark.parametrize("rows,width", [(2, 3), (3, 3)])
def test_vector_location_and_covariance_have_only_row_batch_axes(rows, width):
    source = f"""object Row : FinSet {rows}
object Coordinate : FinSet {width}
program model(mu:Tensor[Real]([Row,Coordinate]), cov:Tensor[Real]([Row,Coordinate,Coordinate]), y:Tensor[Real]([Row,Coordinate])):Row->Row
    observe y : Row <- MultivariateNormal(loc=mu,covariance_matrix=cov) [over=Coordinate]
    return y
export model
"""
    mu = torch.arange(rows * width, dtype=torch.double).reshape(rows, width) / 5
    cov = torch.eye(width, dtype=torch.double).expand(rows, width, width).clone() * 1.3
    y = mu + 0.25
    expected = (
        torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        .log_prob(y)
        .sum()
    )
    recorded = Model(source).bind({"mu": mu, "cov": cov, "y": y}).trace({})
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    host = {
        "mu": _host(mu.tolist()),
        "cov": _host(cov.tolist()),
        "y": _host(y.tolist()),
    }
    result = run_program(lower_qvr_to_qiec(parse(source)), "model", data=host, sites={})
    assert result.log_joint == pytest.approx(expected.item(), abs=2e-10)


@pytest.mark.parametrize("rows,width", [(2, 3), (3, 3)])
def test_matrix_event_does_not_confuse_row_scalar_parameter_with_coordinate(
    rows, width
):
    source = f"""object Row : FinSet {rows}
object Coordinate : FinSet {width}
program model(df:Tensor[Real]([Row]), scale:Tensor[Real]([Row,Coordinate,Coordinate]), y:Tensor[Real]([Row,Coordinate,Coordinate])):Row->Row
    observe y : Row <- Wishart(df=df,scale_tril=scale) [over=[Coordinate,Coordinate]]
    return y
export model
"""
    df = torch.arange(rows, dtype=torch.double) + width + 1
    scale = torch.eye(width, dtype=torch.double).expand(rows, width, width).clone()
    y = scale * 2
    expected = torch.distributions.Wishart(df, scale_tril=scale).log_prob(y).sum()
    checked = lower_qvr_to_qiec(parse(source))
    result = run_program(
        checked,
        "model",
        data={
            "df": _host(df.tolist()),
            "scale": _host(scale.tolist()),
            "y": _host(y.tolist()),
        },
        sites={},
    )
    assert result.log_joint == pytest.approx(expected.item(), abs=2e-10)


def test_category_length_does_not_supply_a_missing_row_parameter_axis():
    source = """object Row : FinSet 2
object Category : FinSet 3
program model(n:Tensor[Int]([Category]), p:Tensor[Real]([Row,Category]), y:Tensor[Int]([Row,Category])):Row->Row
    observe y : Row <- Multinomial(total_count=n,probs=p) [over=Category]
    return y
export model
"""
    with pytest.raises(ValueError, match="total_count"):
        lower_qvr_to_qiec(parse(source))
