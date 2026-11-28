"""A proper locally normalized contextual edit channel with batched path sums."""

from __future__ import annotations

from math import prod

import torch
from torch import nn

from quivers.core.algebras import LOG_PROB
from quivers.core.objects import StringSet
from quivers.core.relations._packed import (
    _Incoming,
    _PackedPaths,
    _pack_incoming,
    _pack_paths,
)
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Words

type _EditState = tuple[int, int, int]


def _validate_words(words: Words, tape: StringSet, role: str) -> None:
    size = tape.alphabet.cardinality
    if any(
        type(symbol) is not int or not 0 <= symbol < size
        for word in words
        for symbol in word
    ):
        raise ValueError(f"invalid symbol on the contextual edit {role} tape")


def _observation_arcs(
    coordinates: _EditState,
    output: tuple[int, ...],
    input_length: int,
    insertion_limit: int,
    alphabet_size: int,
) -> list[tuple[_EditState, int]]:
    consumed, run, emitted = coordinates
    arcs: list[tuple[_EditState, int]] = []
    if consumed < input_length:
        arcs.append(((consumed + 1, 0, emitted), 0))
        if emitted < len(output):
            arcs.append(
                ((consumed + 1, 0, emitted + 1), 2 + alphabet_size + output[emitted])
            )
    if run < insertion_limit and emitted < len(output):
        arcs.append(((consumed, run + 1, emitted + 1), 2 + output[emitted]))
    return arcs


def _compile(
    input_words: Words,
    output_words: Words,
    shape: tuple[int, int, int, int],
    max_states: int,
    max_work: int,
) -> tuple[_PackedPaths, _Incoming, list[bool]]:
    examples, positions, runs, actions = shape
    cells = prod(shape)
    alphabet_size = (actions - 2) // 2
    spent = cells
    if spent > max_work:
        raise RelationLimitError("contextual edit action cells exceed max_work")

    arcs: list[tuple[tuple[int, int], ...]] = []
    owner: list[int] = []
    initial: list[tuple[int, int]] = []
    final: list[tuple[int, int]] = []
    order: list[int] = []
    for example, (source, target) in enumerate(
        zip(input_words, output_words, strict=True)
    ):
        base = len(owner)
        if base == max_states:
            raise RelationLimitError("contextual edit graph exceeds max_states")
        origin = (0, 0, 0)
        states: list[_EditState] = [origin]
        identifiers: dict[_EditState, int] = {origin: base}
        owner.append(example)
        initial.append((base, cells))
        for offset, coordinates in enumerate(states):
            consumed, run, emitted = coordinates
            position = (((example * positions + consumed) * runs) + run) * actions
            if consumed == len(source) and emitted == len(target):
                final.append((base + offset, position + 1))
            edges = []
            for terminal, action in _observation_arcs(
                coordinates, target, len(source), runs - 1, alphabet_size
            ):
                if spent == max_work:
                    raise RelationLimitError(
                        "contextual edit observations exceed max_work"
                    )
                spent += 1
                state = identifiers.get(terminal)
                if state is None:
                    state = len(owner)
                    if state == max_states:
                        raise RelationLimitError(
                            "contextual edit graph exceeds max_states"
                        )
                    owner.append(example)
                    identifiers[terminal] = state
                    states.append(terminal)
                edges.append((state, position + action))
            arcs.append(tuple(edges))
        order.extend(identifiers[state] for state in sorted(states))
    accepted: list[int] = []
    packed = _pack_paths(order, arcs, tuple(initial), final, accept_order=accepted)
    groups: list[list[int]] = [[] for _ in range(examples)]
    for index, state in enumerate(accepted):
        groups[owner[state]].append(index)
    return (
        packed,
        _pack_incoming(groups, len(accepted)),
        [bool(group) for group in groups],
    )


class ContextualEditChannel(nn.Module):
    """A locally normalized edit channel scored from the entire fixed input.

    At a position in the input and a current insertion count, the legal
    moves either consume the next symbol (with or without an output),
    insert an output symbol, or stop at the end of the input. The
    consecutive insertion bound ensures every path terminates.

    Parameters
    ----------
    input_tape : StringSet
        The type of input words.
    output_tape : StringSet
        The type of output words.
    scorer : nn.Module
        A module called once with the tuple of input words. It returns a
        floating tensor with axes example, input position (including an
        end position), insertion count, and action. The input position
        dimension is padded to the longest input length plus one. The
        action order is delete, stop, the insertions in output-alphabet
        order, and the replacements in output-alphabet order. The
        scorer can use arbitrary context from the supplied input words.
        The logits for a given input must be unchanged by batching
        with other inputs to define one probability model per example.
    max_insertions : int
        Maximum consecutive insertions. A consuming edit resets the
        count. The same bound applies at the end of the input.
    """

    def __init__(
        self,
        input_tape: StringSet,
        output_tape: StringSet,
        scorer: nn.Module,
        *,
        max_insertions: int,
    ) -> None:
        super().__init__()
        if not isinstance(input_tape, StringSet) or not isinstance(
            output_tape, StringSet
        ):
            raise TypeError(
                "contextual edit input and output tapes must be StringSet objects"
            )
        if not isinstance(scorer, nn.Module):
            raise TypeError("contextual edit scoring requires a torch module")
        if type(max_insertions) is not int or max_insertions < 0:
            raise ValueError("max_insertions must be a nonnegative integer")
        self._input_tape = input_tape
        self._output_tape = output_tape
        self._scorer = scorer
        self._max_insertions = max_insertions

    def plan(
        self,
        input_words: Words,
        output_words: Words,
        *,
        max_states: int,
        max_work: int,
    ) -> ContextualEditPlan:
        """Plan a batch of observed pairs for repeated differentiable scoring."""
        return ContextualEditPlan(
            self, input_words, output_words, max_states=max_states, max_work=max_work
        )

    def forward(
        self,
        input_words: Words,
        output_words: Words,
        *,
        max_states: int,
        max_work: int,
    ) -> torch.Tensor:
        """Return one alignment-marginal log probability per input-output pair."""
        return self.plan(
            input_words, output_words, max_states=max_states, max_work=max_work
        ).log_prob()


class ContextualEditPlan:
    """A reusable packed evaluation of a fixed batch of edit observations.

    Parameters
    ----------
    channel : ContextualEditChannel
        The channel and its original scorer.
    input_words : Words
        A nonempty batch of fixed input words.
    output_words : Words
        One observed output word per input word.
    max_states : int
        Maximum total observation configurations across the batch.
    max_work : int
        Upper bound on the sum of padded action cells and compatible
        arcs in all observation graphs.
    """

    def __init__(
        self,
        channel: ContextualEditChannel,
        input_words: Words,
        output_words: Words,
        *,
        max_states: int,
        max_work: int,
    ) -> None:
        if max_states is None or max_work is None:
            raise ValueError(
                "contextual edit scoring requires explicit state and work limits"
            )
        _validate_limits(max_states, max_work)
        if not input_words or len(input_words) != len(output_words):
            raise ValueError(
                "contextual edits require a nonempty batch of paired inputs and outputs"
            )
        self._inputs = inputs = tuple(map(tuple, input_words))
        outputs = tuple(map(tuple, output_words))
        _validate_words(inputs, channel._input_tape, "input")
        _validate_words(outputs, channel._output_tape, "output")
        self._alphabet_size = channel._output_tape.alphabet.cardinality
        self._scorer = channel._scorer
        self._lengths = tuple(map(len, inputs))
        self._shape = (
            len(inputs),
            max(self._lengths) + 1,
            channel._max_insertions + 1,
            2 + 2 * self._alphabet_size,
        )
        self._packed, self._groups, self._supported = _compile(
            inputs, outputs, self._shape, max_states, max_work
        )

    def log_prob(self) -> torch.Tensor:
        """Sum all paths realizing each pair with the current local log probabilities."""
        logits = self._scorer(self._inputs)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != self._shape
            or not logits.is_floating_point()
        ):
            raise ValueError(
                f"contextual edit logits must be a floating tensor of shape {self._shape}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("contextual edit logits must be finite")
        _, positions, runs, actions = self._shape
        lengths = torch.tensor(self._lengths, device=logits.device)
        before_end = (
            torch.arange(positions, device=logits.device)[None, :, None, None]
            < lengths[:, None, None, None]
        )
        can_insert = (
            torch.arange(runs, device=logits.device)[None, None, :, None] < runs - 1
        )
        action = torch.arange(actions, device=logits.device)[None, None, None, :]
        allowed = (
            ((action == 0) | (action >= self._alphabet_size + 2)) & before_end
            | (action == 1) & ~before_end
            | ((action >= 2) & (action < self._alphabet_size + 2)) & can_insert
        )
        local = logits.masked_fill(~allowed, -torch.inf).log_softmax(dim=-1)
        if not bool((torch.isfinite(local) | ~allowed).all()):
            raise OverflowError(
                "a legal contextual edit log probability is outside the input precision"
            )
        packed = torch.cat((local.reshape(-1), logits.new_zeros(1)))
        result = self._packed.evaluate(packed, LOG_PROB, final_groups=self._groups)
        supported = torch.tensor(self._supported, device=logits.device)
        if not bool((torch.isfinite(result) | ~supported).all()):
            raise OverflowError(
                "an observed contextual edit log probability is outside the input precision"
            )
        return result


__all__ = ["ContextualEditChannel", "ContextualEditPlan"]
