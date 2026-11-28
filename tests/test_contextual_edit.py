"""Complete-path probability and gradient oracles for contextual editing."""

from itertools import product

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.relations import ContextualEditChannel, RelationLimitError
from tests.evidence.counters import TensorCounter


class _BatchedInputContext(nn.Module):
    def __init__(self, insertion_bound=1, alphabet_size=2):
        super().__init__()
        self.calls = 0
        self.insertion_bound = insertion_bound
        self.alphabet_size = alphabet_size
        n_actions = 2 + 2 * alphabet_size
        self.bias = nn.Parameter(
            torch.linspace(-0.5, 0.5, n_actions, dtype=torch.float64)
        )
        self.context = nn.Parameter(
            torch.linspace(0.2, -0.1, n_actions, dtype=torch.float64)
        )

    def forward(self, inputs: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        self.calls += 1
        width = 1 + max(map(len, inputs))
        # A right-context feature changes the first edit when a later
        # input symbol changes. All examples share the two parameters.
        suffix = self.bias.new_tensor(
            [
                [sum(symbol + 1 for symbol in word[pos:]) for pos in range(width)]
                for word in inputs
            ]
        )
        past_insertions = torch.arange(self.insertion_bound + 1, dtype=self.bias.dtype)
        context = suffix[:, :, None, None] + 0.25 * past_insertions[None, None, :, None]
        return self.bias + context * self.context


def _channel(scorer, bound=1, output_size=2):
    input_tape = StringSet(alphabet=FinSet(name="inputs", cardinality=3))
    output_tape = StringSet(alphabet=FinSet(name="outputs", cardinality=output_size))
    return ContextualEditChannel(input_tape, output_tape, scorer, max_insertions=bound)


def _enumerate_all(inputs, source, example, alphabet_size, bound):
    """Visit all finite edit paths and group their total masses by output."""
    zero = inputs[example, 0, 0, 0].new_zeros(())
    one = zero + 1
    output_weights = {}
    pending = [(0, 0, (), one)]
    while pending:
        pos, run, output, value = pending.pop()
        choices = []
        if pos == len(source):
            choices.append((1, None))
        else:
            choices.append((0, (pos + 1, 0, output)))
            for symbol in range(alphabet_size):
                choices.append(
                    (2 + alphabet_size + symbol, (pos + 1, 0, (*output, symbol)))
                )
        if run < bound:
            for symbol in range(alphabet_size):
                choices.append((symbol + 2, (pos, run + 1, (*output, symbol))))
        probabilities = torch.stack(
            [inputs[example, pos, run, action].exp() for action, _ in choices]
        )
        normalizer = torch.sum(probabilities)
        for (_, target), mass in zip(choices, probabilities / normalizer, strict=True):
            weight = mass * value
            if target is None:
                output_weights[output] = output_weights.get(output, zero) + weight
            else:
                pending.append((*target, weight))
    return output_weights


def test_batched_weights_and_live_gradients_match_complete_path_enumeration():
    scorer = _BatchedInputContext()
    model = _channel(scorer)
    inputs = ((0, 1), (), (0, 2), (1,))
    outputs = ((1,), (1,), (1,), (0, 1))
    plan = model.plan(inputs, outputs, max_states=300, max_work=1000)
    got = plan.log_prob()
    assert scorer.calls == 1
    logits = scorer(inputs)
    expected = torch.stack(
        [
            _enumerate_all(logits, source, i, 2, 1)[target].log()
            for i, (source, target) in enumerate(zip(inputs, outputs, strict=True))
        ]
    )
    torch.testing.assert_close(got, expected, rtol=3e-14, atol=3e-14)
    assert got[0].item() != got[2].item()
    actual_grads = torch.autograd.grad(got.sum(), tuple(scorer.parameters()))
    expected_grads = torch.autograd.grad(expected.sum(), tuple(scorer.parameters()))
    for observed, oracle in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, oracle, rtol=3e-14, atol=3e-14)
    with torch.no_grad():
        scorer.bias[0] += 1
    assert not torch.equal(plan.log_prob(), got)


def test_whole_output_normalizes_and_impossible_examples_have_zero_gradient():
    scorer = _BatchedInputContext()
    model = _channel(scorer)
    all_outputs = tuple(word for n in range(4) for word in product(range(2), repeat=n))
    outputs = (*all_outputs, (1, 1, 1, 1))
    probabilities = model(
        ((0,),) * len(outputs), outputs, max_states=500, max_work=1500
    )
    assert probabilities[-1].item() == -torch.inf
    torch.testing.assert_close(probabilities.exp().sum(), scorer.bias.new_tensor(1.0))
    for grad in torch.autograd.grad(probabilities[-1], tuple(scorer.parameters())):
        assert torch.count_nonzero(grad).item() == 0


def test_log_prob_minus_one_thousand_remains_finite():
    scorer = _BatchedInputContext(insertion_bound=0, alphabet_size=1)
    with torch.no_grad():
        scorer.bias.copy_(scorer.bias.new_tensor((0, 0, 0, -1000)))
        scorer.context.zero_()
    model = _channel(scorer, bound=0, output_size=1)
    value = model(((0,),), ((0,),), max_states=8, max_work=20)
    assert value.tolist() == [-1000.0]
    assert torch.autograd.grad(value.sum(), scorer.bias)[0].tolist() == [
        -1.0,
        0.0,
        0.0,
        1.0,
    ]


def test_action_cells_and_graph_use_one_explicit_budget():
    scorer = _BatchedInputContext()
    model = _channel(scorer)
    with pytest.raises(RelationLimitError, match="action cells"):
        model.plan(((0,),), ((),), max_states=10, max_work=0)
    with pytest.raises(RelationLimitError, match="observations"):
        model.plan(((0,),), ((),), max_states=10, max_work=24)
    with pytest.raises(RelationLimitError, match="max_states"):
        model.plan(((0,),), ((),), max_states=1, max_work=100)
    assert scorer.calls == 0


def test_scorer_shape_and_words_are_checked_and_forbidden_actions_stay_masked():
    model = _channel(nn.Identity())
    with pytest.raises(ValueError, match="floating tensor of shape"):
        model(((0,),), ((),), max_states=8, max_work=100)
    with pytest.raises(ValueError, match="input tape"):
        model.plan(((3,),), ((),), max_states=8, max_work=100)
    with pytest.raises(ValueError, match="output tape"):
        model.plan(((0,),), ((True,),), max_states=8, max_work=100)
    scorer = _BatchedInputContext(insertion_bound=0)
    channel = _channel(scorer, bound=0)
    first = channel(((),), ((),), max_states=8, max_work=100)
    with torch.no_grad():
        scorer.bias[2] = 10000
    second = channel(((),), ((),), max_states=8, max_work=100)
    torch.testing.assert_close(first, second)
    assert second.tolist() == [0.0]


def test_shared_packed_schedule_uses_fewer_tensor_dispatches_than_singletons():
    scorer = _BatchedInputContext()
    model = _channel(scorer)
    inputs = ((0, 1), (1, 0)) * 16
    outputs = ((1, 0), (0, 1)) * 16
    policy = {"max_states": 1600, "max_work": 4000}
    batch = model.plan(inputs, outputs, **policy)
    singles = [
        model.plan((source,), (target,), **policy)
        for source, target in zip(inputs, outputs, strict=True)
    ]
    with torch.no_grad(), TensorCounter() as packed:
        together = batch.log_prob()
    with torch.no_grad(), TensorCounter() as separate:
        each = torch.cat([plan.log_prob() for plan in singles])

    torch.testing.assert_close(together, each)
    assert packed.counts["kernels"] * 8 < separate.counts["kernels"]
