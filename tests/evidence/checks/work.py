"""Checks of the work bounds of complexity families."""

from __future__ import annotations

from functools import partial

from tests.evidence import scaling
from tests.evidence.checks._registry import add


def _register() -> None:
    """Register every family."""
    for family in (*scaling.FAMILIES, *scaling.REPEATED):
        slug = family.name.replace(" ", "-")
        add(
            f"work/{slug}",
            family.executor,
            family.question,
            partial(scaling.check_family, family),
            slow=family.slow,
        )


_register()
