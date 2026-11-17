"""Explicit row attachments shared by refitting and held-out scoring adapters."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from quivers.diagnostics.cross_validation import FoldSplit


@dataclass(frozen=True, eq=False)
class RowAttachment:
    """A declared tensor axis and its fitting/scoring keyword roles."""

    training_keyword: str
    scoring_keyword: str
    value: torch.Tensor
    axis: int = 0

    def validate(self, original_rows: int) -> None:
        """Check declared row alignment and finite values before fitting."""
        if (
            type(self.axis) is not int
            or not 0 <= self.axis < self.value.ndim
            or self.value.shape[self.axis] != original_rows
        ):
            raise ValueError(
                f"row attachment {self.training_keyword!r} must align with {original_rows} original rows on its declared axis"
            )
        if not bool(torch.isfinite(self.value).all()):
            raise ValueError(
                f"row attachment {self.training_keyword!r} must contain finite values"
            )

    def select(self, positions: tuple[int, ...]) -> RowAttachment:
        """Select explicitly declared row coordinates without touching event axes."""
        index = torch.tensor(positions, dtype=torch.long, device=self.value.device)
        return RowAttachment(
            self.training_keyword,
            self.scoring_keyword,
            self.value.index_select(self.axis, index),
            self.axis,
        )


@dataclass(frozen=True, eq=False)
class AttachedRows[Data]:
    """A frame or mapping together with explicit out-of-frame row tensors."""

    data: Data
    attachments: tuple[RowAttachment, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous attachment keyword roles before any partition."""
        for attribute in ("training_keyword", "scoring_keyword"):
            names = [getattr(item, attribute) for item in self.attachments]
            if len(set(names)) != len(names) or any(
                not name.isidentifier() for name in names
            ):
                raise ValueError(
                    "row attachments require distinct valid training and scoring keywords"
                )

    def keywords(self, *, scoring: bool) -> dict[str, torch.Tensor]:
        """Expose only declared training or scoring roles for this partition."""
        return {
            item.scoring_keyword if scoring else item.training_keyword: item.value
            for item in self.attachments
        }

    def partition(
        self,
        positions: tuple[int, ...],
        select: Callable[[Data, tuple[int, ...]], Data],
    ) -> AttachedRows[Data]:
        """Apply one original-row selection to data and every declared attachment."""
        return AttachedRows(
            select(self.data, positions),
            tuple(item.select(positions) for item in self.attachments),
        )


def original_positions(
    rows: tuple[int, ...], included: tuple[int, ...]
) -> tuple[int, ...]:
    """Map included-row coordinates to original rows with explicit bounds."""
    if any(type(row) is not int or not 0 <= row < len(included) for row in rows):
        raise ValueError("fold rows must index the retained included-row positions")
    return tuple(included[row] for row in rows)


def validate_split_rows(split: FoldSplit, size: int) -> None:
    """Require a complete partition of exactly the retained included rows."""
    if sorted((*split.train, *split.omitted)) != list(range(size)) or not set(
        split.test
    ) <= set(split.omitted):
        raise ValueError(
            f"fold partition must describe exactly {size} retained included rows"
        )
