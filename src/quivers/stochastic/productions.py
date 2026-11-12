"""Productions of a weighted grammar, grouped by left-hand side.

A deduction with a lexicon is a grammar: each rule and each lexicon
entry is a production that rewrites the category its conclusion names
(its left-hand side) into the items its premises name, or into a word.
A [probabilistic context-free
grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)
gives the productions that share a left-hand side a probability
distribution, so that

$$\\sum_{A \\to \\beta} P(A \\to \\beta) = 1$$

for every category $A$ ([Chi and Geman, 1998](<https://aclanthology.org/J98-2005/>)). `ProductionGroup` records
one left-hand side and its productions in a fixed order, and
`ProductionWeights` maps the grammar's free log-weights to the
per-left-hand-side log-probabilities a chart reads: each group's
log-weights pass through a
[`log_softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html),
so a group's probabilities sum to one whatever the free weights are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal

import didactic.api as dx
import torch


__all__ = [
    "Production",
    "ProductionGroup",
    "ProductionKey",
    "ProductionWeights",
]


type ProductionKey = tuple[Literal["rule", "lexicon"], int]
"""A production's address: its kind and its index among the rules, or
among the expanded lexicon entries."""


class Production(dx.Model):
    """One production of a grammar: a rule or an expanded lexicon entry.

    Parameters
    ----------
    kind : Literal["rule", "lexicon"]
        Whether the production is a rule or a lexicon entry.
    index : int
        The rule's position among the deduction's rules, or the entry's
        position among the expanded lexicon entries (one per word and
        candidate category, inline entries first, then the rows of a
        lexicon file).
    label : str
        The production's name within its group: the rule's name, or the
        entry's word in double quotes.
    learnable : bool
        Whether the production's log-weight is a learned parameter; a
        production that is not learnable has log-weight zero.
    """

    kind: Literal["rule", "lexicon"]
    index: int
    label: str
    learnable: bool

    @property
    def key(self) -> ProductionKey:
        """The production's address.

        Returns
        -------
        ProductionKey
            Its kind and index.
        """
        return (self.kind, self.index)


class ProductionGroup(dx.Model):
    """The productions that rewrite one left-hand side.

    Parameters
    ----------
    label : str
        The left-hand side: the category of a span item, or an item with
        its position slots written ``_``.
    productions : tuple[Production, ...]
        The productions, rules in declaration order, then lexicon
        entries in expansion order.
    """

    label: str
    productions: tuple[Production, ...]


class ProductionWeights:
    """The per-left-hand-side weights a grammar's chart reads.

    The object holds the grammar's production groups and a reader for
    each production's free log-weight (its learned parameter, or zero).
    `table` turns the free log-weights into the log-weights the chart
    reads: under per-left-hand-side normalization, each group's vector
    passes through ``log_softmax``; otherwise each production keeps its
    free log-weight. A normalized grammar may instead take its production
    probabilities from the caller, one probability vector per left-hand
    side with several productions (`bound_groups`), as a program does
    when it places a prior on the grammar's weights. A run of the
    deduction computes the table once and makes it `active`, so every
    rule firing and lexical axiom of that run reads one consistent
    table.

    Parameters
    ----------
    groups : tuple[ProductionGroup, ...]
        The production groups, one per left-hand side.
    free_weight : Callable[[Production], torch.Tensor]
        The free log-weight of a production, a scalar tensor.
    normalize : bool
        Whether each group's log-weights are normalized.
    """

    def __init__(
        self,
        groups: tuple[ProductionGroup, ...],
        free_weight: Callable[[Production], torch.Tensor],
        *,
        normalize: bool,
    ) -> None:
        self.groups = groups
        self.normalize = normalize
        self._free_weight = free_weight
        self._active: dict[ProductionKey, torch.Tensor] | None = None

    def group_vector(self, group: ProductionGroup) -> torch.Tensor:
        """The free log-weights of one group, in production order.

        Parameters
        ----------
        group : ProductionGroup
            The group.

        Returns
        -------
        torch.Tensor
            A vector with one entry per production.
        """
        return torch.stack(
            [
                self._free_weight(production).reshape(())
                for production in group.productions
            ]
        )

    @property
    def bound_groups(self) -> tuple[ProductionGroup, ...]:
        """The groups whose probabilities a caller supplies.

        Returns
        -------
        tuple[ProductionGroup, ...]
            The groups with more than one production; a production alone
            in its group has probability one.
        """
        return tuple(group for group in self.groups if len(group.productions) > 1)

    def check_probabilities(self, probabilities: Mapping[str, torch.Tensor]) -> None:
        """Refuse supplied probabilities that do not fit the groups.

        Parameters
        ----------
        probabilities : Mapping[str, torch.Tensor]
            One probability vector per bound group, by group label.

        Raises
        ------
        ValueError
            If the grammar is not normalized, a bound group is missing,
            a label names no bound group, a vector's shape is not
            ``(n,)`` for a group of ``n`` productions, or an entry is
            negative.
        """
        if not self.normalize:
            raise ValueError(
                "production probabilities replace the weights of a grammar "
                "normalized per left-hand side; declare normalize=lhs"
            )
        bound = {group.label: group for group in self.bound_groups}
        missing = [label for label in bound if label not in probabilities]
        extra = [label for label in probabilities if label not in bound]
        if missing or extra:
            singletons = [
                label
                for label in extra
                if any(group.label == label for group in self.groups)
            ]
            raise ValueError(
                "production probabilities must name every left-hand side with "
                f"several productions, {list(bound)}; missing {missing}, "
                f"unknown {extra}"
                + (
                    f" ({singletons} have one production, whose probability is one)"
                    if singletons
                    else ""
                )
            )
        for label, group in bound.items():
            vector = probabilities[label]
            size = len(group.productions)
            if tuple(vector.shape) != (size,):
                raise ValueError(
                    f"the probabilities of {label!r} have shape "
                    f"{tuple(vector.shape)}, but {label!r} has {size} productions "
                    f"({', '.join(p.label for p in group.productions)})"
                )
            if bool((vector.detach() < 0).any()):
                raise ValueError(
                    f"the probabilities of {label!r} have a negative entry: "
                    f"{vector.detach().tolist()}"
                )

    def table(
        self, probabilities: Mapping[str, torch.Tensor] | None = None
    ) -> dict[ProductionKey, torch.Tensor]:
        """The log-weight of every production.

        Parameters
        ----------
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities to read instead of the free
            log-weights, one vector per bound group; each is
            renormalized, so its logarithm passes through the same
            ``log_softmax``.

        Returns
        -------
        dict[ProductionKey, torch.Tensor]
            One scalar per production: its group's ``log_softmax`` entry
            under normalization, its free log-weight otherwise.

        Raises
        ------
        ValueError
            If the probabilities do not fit the groups
            (`check_probabilities`).
        """
        if probabilities is not None:
            self.check_probabilities(probabilities)
        out: dict[ProductionKey, torch.Tensor] = {}
        for group in self.groups:
            if probabilities is not None and len(group.productions) == 1:
                vector = torch.zeros(1)
            elif probabilities is not None:
                vector = torch.log(probabilities[group.label])
            else:
                vector = self.group_vector(group)
            if self.normalize:
                vector = torch.log_softmax(vector, dim=0)
            for position, production in enumerate(group.productions):
                out[production.key] = vector[position]
        return out

    @contextmanager
    def active(self, table: dict[ProductionKey, torch.Tensor]) -> Iterator[None]:
        """Serve one table to every read inside the block.

        Parameters
        ----------
        table : dict[ProductionKey, torch.Tensor]
            The table, from `table`.

        Yields
        ------
        None
            Control, with the table active.
        """
        previous = self._active
        self._active = table
        try:
            yield
        finally:
            self._active = previous

    def weight(self, key: ProductionKey) -> torch.Tensor:
        """The log-weight one production contributes.

        Parameters
        ----------
        key : ProductionKey
            The production's address.

        Returns
        -------
        torch.Tensor
            The entry of the active table, or of a fresh table when none
            is active.

        Raises
        ------
        KeyError
            If no group holds the production.
        """
        table = self._active if self._active is not None else self.table()
        return table[key]

    def log_probs(
        self, probabilities: Mapping[str, torch.Tensor] | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """The table by group and production label.

        Parameters
        ----------
        probabilities : Mapping[str, torch.Tensor] | None
            Production probabilities to read instead of the free
            log-weights, as for `table`.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            For each left-hand side, each production's log-weight.

        Raises
        ------
        ValueError
            If the probabilities do not fit the groups.
        """
        table = self.table(probabilities)
        return {
            group.label: {
                production.label: table[production.key]
                for production in group.productions
            }
            for group in self.groups
        }
