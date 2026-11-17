"""Family lookup by brms family name.

brms reads a family name case-insensitively, expands the prefixes ``zi_``
and ``hu_`` to ``zero_inflated_`` and ``hurdle_``, and reads ``normal`` as
``gaussian`` (``brmsfamily``). Three of its names differ from the names of
this package's registry: ``student``, ``negbinomial`` and
``inverse.gaussian``. The registry here resolves every such spelling to one
registered family, so the family objects, their parameters and the names
they report stay those of the registry.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import re

#: brms family names whose registered name differs.
BRMS_FAMILY_NAMES: Mapping[str, str] = {
    "student": "student_t",
    "negbinomial": "negative_binomial",
    "inverse.gaussian": "inverse_gaussian",
}

#: brms's own rewrites of an authored family name, applied in order.
_BRMS_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^normal$"), "gaussian"),
    (re.compile(r"^zi_"), "zero_inflated_"),
    (re.compile(r"^hu_"), "hurdle_"),
)


def canonical_family_name(name: str, registered: Mapping[str, object]) -> str:
    """Resolve a brms spelling of a family name to its registered name.

    Parameters
    ----------
    name : str
        Authored family name, such as ``"negbinomial"``, ``"Gamma"``,
        ``"student"`` or ``"zi_poisson"``.
    registered : Mapping[str, object]
        Families keyed by registered name.

    Returns
    -------
    str
        The registered name when the spelling resolves to one, and the
        authored name unchanged otherwise.

    Notes
    -----
    A registered name is returned as it is. Any other name is lowercased,
    rewritten as brms's ``brmsfamily`` rewrites it, and mapped through
    `BRMS_FAMILY_NAMES`. Cost is constant.
    """
    if name in registered:
        return name
    resolved = name.lower()
    for pattern, replacement in _BRMS_REWRITES:
        resolved = pattern.sub(replacement, resolved)
    resolved = BRMS_FAMILY_NAMES.get(resolved, resolved)
    return resolved if resolved in registered else name


class FamilyRegistry[V](Mapping[str, V]):
    """Families keyed by registered name and looked up by any brms spelling.

    Parameters
    ----------
    families : Mapping[str, V]
        Families keyed by registered name.

    Notes
    -----
    Iteration and length cover the registered names only, so a listing of
    the registry names each family once. Lookup and membership accept every
    spelling `canonical_family_name` resolves.
    """

    def __init__(self, families: Mapping[str, V]) -> None:
        """Retain the registered families.

        Parameters
        ----------
        families : Mapping[str, V]
            Families keyed by registered name.
        """
        self._families = dict(families)

    def __getitem__(self, name: str) -> V:
        """Return the family a name resolves to.

        Parameters
        ----------
        name : str
            Registered name or brms spelling.

        Returns
        -------
        V
            The registered family.

        Raises
        ------
        KeyError
            If the name resolves to no registered family.
        """
        return self._families[canonical_family_name(name, self._families)]

    def __contains__(self, name: object) -> bool:
        """Report whether a name resolves to a registered family.

        Parameters
        ----------
        name : object
            Candidate name.

        Returns
        -------
        bool
            True for a registered name or a brms spelling of one.
        """
        return (
            isinstance(name, str)
            and canonical_family_name(name, self._families) in self._families
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered names.

        Returns
        -------
        Iterator[str]
            Registered names in registration order.
        """
        return iter(self._families)

    def __len__(self) -> int:
        """Return the number of registered families.

        Returns
        -------
        int
            Count of registered names.
        """
        return len(self._families)


__all__: list[str] = []
