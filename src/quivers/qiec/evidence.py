"""Kernel-generated equality evidence.

The first QIEC kernel has no arbitrary user proof language.  Evidence is
either reflexivity or a rigid branch given introduced by GADT elimination.
"""

from __future__ import annotations

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
import didactic.api as dx

from typing import Literal

from quivers.qiec.identifiers import EqualityId
from quivers.qiec.types import EqualityType


class Reflexivity(QiecModel):
    """Evidence that an equality holds because both sides are the same term.

    Parameters
    ----------
    equality
        The proposition witnessed, whose sides must be equal.
    tag
        The serialization discriminator; always ``"reflexivity"``.
    """

    equality: EqualityType = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["reflexivity"] = "reflexivity"


class BranchGiven(QiecModel):
    """Evidence granted by a case branch's constructor refinement.

    Parameters
    ----------
    id
        The stable identity of the given, derived from the branch scope.
    equality
        The proposition the branch's constructor makes available.
    tag
        The serialization discriminator; always ``"branch_given"``.
    """

    id: EqualityId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    equality: EqualityType = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    tag: Literal["branch_given"] = "branch_given"


type EqualityEvidence = Reflexivity | BranchGiven
"""Kernel-generated evidence of a static equality: reflexivity or a branch's given."""


__all__ = [
    "BranchGiven",
    "EqualityEvidence",
    "Reflexivity",
]
