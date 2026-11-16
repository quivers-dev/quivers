"""Mathematical count contracts remain separate from host precision limits."""

import pytest
import torch
from torch.distributions import constraints

from quivers.continuous._closed_probability import ClosedBinomial
from quivers.continuous._zero_inflated_count import (
    BetaBinomial,
    ZeroInflatedBinomial,
    ZeroInflatedBetaBinomial,
)
from quivers.qiec.families import family


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (ClosedBinomial, {"probs": torch.tensor(0.4)}),
        (BetaBinomial, {"concentration1": 2.0, "concentration0": 3.0}),
        (ZeroInflatedBinomial, {"zero_prob": 0.2, "probs": 0.4}),
        (
            ZeroInflatedBetaBinomial,
            {"zero_prob": 0.2, "concentration1": 2.0, "concentration0": 3.0},
        ),
    ],
)
def test_trial_constraint_describes_law_and_constructor_checks_precision(
    cls, parameters
):
    declared = cls.arg_constraints["total_count"]
    assert isinstance(declared, type(constraints.nonnegative_integer))
    values = torch.tensor([-1.0, 0.0, 0.5, 3.0], dtype=torch.float64)
    torch.testing.assert_close(
        declared.check(values), torch.tensor([False, True, False, True])
    )
    # This is a host numerical-capacity refusal, independent of validate_args.
    with pytest.raises(ValueError, match=r"2\*\*53"):
        cls(total_count=2**53 + 1, validate_args=False, **parameters)
    assert cls(total_count=0, validate_args=True, **parameters).total_count.item() == 0


@pytest.mark.parametrize("cls", [ZeroInflatedBinomial, ZeroInflatedBetaBinomial])
def test_finite_mixture_support_is_dependent_in_registry_and_instances(cls):
    record = family(cls.__name__)
    assert record.support == "dependent"
    assert record.parameters[1].constraint == "nonnegative_integer"
    parameters = (
        {"probs": 0.4}
        if cls is ZeroInflatedBinomial
        else {"concentration1": 2.0, "concentration0": 3.0}
    )
    law = cls(0.2, torch.tensor([0, 3]), **parameters)
    values = torch.tensor([[0, 0], [1, 3], [2, 4]])
    torch.testing.assert_close(
        law.support.check(values),
        torch.tensor([[True, True], [False, True], [False, False]]),
    )
    expanded = law.expand((4, 2))
    assert expanded.support.check(torch.tensor(0)).all()
    assert not expanded.support.check(torch.tensor(4)).any()
