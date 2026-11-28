"""Measure packed and singleton training for the same contextual edit channel."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from statistics import median
from time import perf_counter
import tracemalloc

import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.relations import ContextualEditChannel


type Word = tuple[int, ...]
type Score = Callable[[], torch.Tensor]


class _Scorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 12, dtype=torch.float64)
        self.last = nn.Linear(12, 6, dtype=torch.float64)

    def forward(self, inputs: tuple[Word, ...]) -> torch.Tensor:
        width = max(map(len, inputs)) + 1
        features = [
            [
                [
                    (
                        position / 6,
                        sum(word[position:]) / 6,
                        float(word[-1]),
                        float(insertions),
                    )
                    for insertions in range(2)
                ]
                for position in range(width)
            ]
            for word in inputs
        ]
        return self.last(self.first(self.first.weight.new_tensor(features)).tanh())


def _pairs(size: int) -> tuple[tuple[Word, ...], tuple[Word, ...]]:
    inputs = tuple(
        tuple(
            ((i + 5) * (position + 7) // (position + 2)) % 2
            for position in range(4 + i % 3)
        )
        for i in range(size)
    )
    outputs = tuple(
        (1 - word[0], *word[1 : len(word) - (i % 2)]) for i, word in enumerate(inputs)
    )
    return inputs, outputs


def _scores(model: ContextualEditChannel, size: int) -> tuple[Score, Score]:
    inputs, outputs = _pairs(size)
    limits = dict(max_states=50_000, max_work=500_000)
    together = model.plan(inputs, outputs, **limits)
    separate = tuple(
        model.plan((source,), (target,), **limits)
        for source, target in zip(inputs, outputs, strict=True)
    )

    def singletons() -> torch.Tensor:
        return torch.cat(tuple(plan.log_prob() for plan in separate))

    return together.log_prob, singletons


def _train(
    scorer: _Scorer,
    score: Score,
    steps: int,
    *,
    profile: bool = False,
) -> tuple[tuple[torch.Tensor, ...], float]:
    parameters = tuple(scorer.parameters())
    losses = []
    peak = 0
    saved: dict[int, int] = {}

    def pack(value: torch.Tensor) -> torch.Tensor:
        storage = value.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return value.detach()

    for _ in range(steps):
        if profile:
            saved.clear()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                loss = -score().mean()
            peak = max(peak, sum(saved.values()))
        else:
            loss = -score().mean()
        gradients = torch.autograd.grad(loss, parameters)
        losses.append(loss.detach())
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.add_(gradient, alpha=-0.2)
    return tuple(losses), peak / 1024


def main() -> None:
    """Compare refit time, peak Python memory, and distinct saved tensor storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(4, 16, 64))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(*args.sizes, args.steps, args.repeats) < 1:
        parser.error("sizes, steps, and repeats must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(518)
    scorer = _Scorer()
    tape = StringSet(alphabet=FinSet(name="Letter", cardinality=2))
    channel = ContextualEditChannel(tape, tape, scorer, max_insertions=1)
    initial = {
        name: value.detach().clone() for name, value in scorer.state_dict().items()
    }
    print(
        "Batch size | Mode | Update ms | Peak Python KiB | Saved tape KiB | Final loss"
    )
    print("--- | --- | --- | --- | --- | ---")
    for size in args.sizes:
        together, separate = _scores(channel, size)
        torch.testing.assert_close(together(), separate())
        # Equal initial scores and independent parameter calls compare
        # every fitted loss and the complete final parameter vectors.
        oracle_losses = None
        oracle_parameters = None
        for mode, score in (("Packed", together), ("Singletons", separate)):
            scorer.load_state_dict(initial)
            losses, _ = _train(scorer, score, args.steps)
            final = tuple(
                parameter.detach().clone() for parameter in scorer.parameters()
            )
            if oracle_losses is None:
                oracle_losses, oracle_parameters = losses, final
            else:
                torch.testing.assert_close(losses, oracle_losses)
                torch.testing.assert_close(final, oracle_parameters)
            times = []
            for _ in range(args.repeats):
                scorer.load_state_dict(initial)
                start = perf_counter()
                _train(scorer, score, args.steps)
                times.append(1000 * (perf_counter() - start) / args.steps)
            scorer.load_state_dict(initial)
            tracemalloc.start()
            try:
                _train(scorer, score, args.steps)
                _, python_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            scorer.load_state_dict(initial)
            _, tensor_peak = _train(scorer, score, args.steps, profile=True)
            print(
                f"{size} | {mode} | {median(times):.2f} | {python_peak / 1024:.1f} | {tensor_peak:.1f} | {losses[-1].item():.6f}"
            )


if __name__ == "__main__":
    main()
