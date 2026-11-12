"""Tier 5: distribution-family × backend support matrix.

For every family in
[`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry]
and every registered backend:

- If `FAMILY_META[family].target_names` contains `backend`, the
  transpile must succeed and the emitted source must reference the
  backend-specific distribution name (e.g. Stan `beta`, NumPyro
  `Beta`, BUGS `dbeta`).
- If the family/backend pair is absent from `target_names`, the
  transpile must raise
  `UnsupportedConstruct(kinds=["family:<Family>"])`.

The matrix is the source of truth for the docs page "which backends
support which families."
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.renderers.jags import _ZEROS_TRICK_FAMILIES
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


def _backend_target_name(family: str, backend: str) -> str | None:
    """Return the per-backend distribution name for ``family`` from
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], or
    ``None`` if the backend is not registered for the family.

    The new Lower + Renderer pipeline derives every backend-specific
    distribution name from this central table; renderer modules no
    longer carry per-backend `_FAMILIES` dicts.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        return None
    return meta.target_names.get(backend)


def _family_fixtures() -> list[_load.Fixture]:
    return _load.load_families()


def _family_for_fixture(fixture: _load.Fixture) -> str:
    """Each `families/<family>.qvr` fixture's stem is the lowercased
    family name; recover the canonical mixed-case name from the
    registry."""
    from quivers.dsl.compiler._prelude import _get_family_registry

    registry = _get_family_registry()
    canonical = {name.lower(): name for name in registry}
    if fixture.name not in canonical:
        raise AssertionError(
            f"fixture {fixture.name!r} does not match any family name "
            f"in the registry (lowercased): {sorted(canonical)}"
        )
    return canonical[fixture.name]


#: Per backend, the families it renders without a native
#: distribution name, by writing the density out longhand.
#:
#: `FAMILY_META[family].target_names` answers "which distribution does
#: this target spell the family as", and a `None` there usually means
#: the target cannot express the family at all. JAGS has a third
#: option: the zeros trick writes a closed-form log-density into the
#: rate of a `dpois` on a carrier of zeros, so the family is emitted
#: with no distribution name to look for. Read from the renderer's own
#: set rather than restated, so a family entering or leaving that path
#: cannot drift from what this test expects.
_LONGHAND_FAMILIES: dict[str, frozenset[str]] = {
    "jags": _ZEROS_TRICK_FAMILIES,
}

#: What a zeros-trick emission looks like: a Poisson on a carrier of
#: zeros, standing in for the distribution name the other branch reads.
_ZEROS_TRICK_MARKERS: tuple[bytes, ...] = (b"dpois", b"zeros")


#: Cells whose transpile refuses for a reason that has nothing to do
#: with the family under test.
#:
#: A fixture exercises its family at an observed site, and may reach
#: for other constructs around it. When a backend refuses one of those
#: the cell says nothing about the family, so the pin records the kind
#: the refusal must carry: a closed gap surfaces as a raise that no
#: longer happens, and a regression as a raise of a different kind.
_ORTHOGONAL_REFUSALS: dict[tuple[str, str], str] = {
    # `wishart.qvr` writes its scale matrix as a literal, which JAGS's
    # data block has no syntax for; the family itself is one JAGS has.
    ("jags", "wishart"): "arg:matrix-literal",
    # BUGS's data block has no literal vector or matrix syntax either,
    # and its argument check runs before its family check.
    ("bugs", "wishart"): "arg:matrix-literal",
    ("bugs", "inversewishart"): "arg:matrix-literal",
    ("bugs", "lowrankmvn"): "arg:list-literal",
    # `categorical.qvr` observes the class codes `cls`, which the data
    # holds counting from zero and `dcat` reads from one; BUGS has no
    # data block to shift them in and no free log-density statement to
    # score them otherwise.
    ("bugs", "categorical"): "observe:class-code",
    # `bernoulli.qvr` draws its outcome as a latent, and Stan's samplers
    # carry no discrete parameter; Stan scores the family at an observed
    # site.
    ("stan", "bernoulli"): "sample:discrete-latent",
}


@pytest.mark.parametrize("fixture", _family_fixtures(), ids=lambda f: f.name)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_family_backend_cell(fixture: _load.Fixture, backend: str) -> None:
    """One cell of the (family × backend) matrix.

    Two pre-declared outcomes:

    - `backend_name is None` (family absent from
      `FAMILY_META[family].target_names` for this backend): the
      pipeline MUST raise `UnsupportedConstruct` with at least one
      kind whose tag begins with ``"family:"``.
    - `backend_name is not None`: the pipeline MUST emit non-empty
      bytes that contain the backend-specific distribution name
      (e.g. Stan ``beta``, NumPyro ``Beta``).
    """
    family = _family_for_fixture(fixture)
    backend_name = _backend_target_name(family, backend)
    module = parse(fixture.source)

    orthogonal = _ORTHOGONAL_REFUSALS.get((backend, fixture.name))
    if orthogonal is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(orthogonal) for k in kinds), (
            f"backend {backend!r} on family {family!r}: expected a "
            f"refusal carrying {orthogonal!r}, got kinds={kinds!r}. "
            f"The cell is pinned to a gap beside the family; if that "
            f"gap closed, drop the row so the family is exercised "
            f"again."
        )
        return

    if backend_name is None and family in _LONGHAND_FAMILIES.get(backend, frozenset()):
        # No native name, and none is wanted: the density is written
        # out longhand, so the cell asserts the shape of that emission
        # rather than a distribution name.
        output = transpile(module, target=backend)
        assert output, (
            f"backend {backend!r} on family {family!r}: the renderer "
            f"lists this family as one it writes out longhand, and it "
            f"emitted nothing."
        )
        missing = [marker for marker in _ZEROS_TRICK_MARKERS if marker not in output]
        assert not missing, (
            f"backend {backend!r} on family {family!r}: emitted "
            f"output is missing {missing!r}, so it does not carry the "
            f"zeros-trick form this family is rendered through. "
            f"Either the renderer found a native distribution for it, "
            f"in which case add {backend!r} to "
            f"`FAMILY_META[{family!r}].target_names`, or the longhand "
            f"path changed shape."
        )
        return

    if backend_name is None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith("family:") for k in kinds), (
            f"backend {backend!r} on family {family!r}: expected "
            f"`UnsupportedConstruct` with a `family:`-prefixed kind, "
            f"got kinds={kinds!r}. Either add {backend!r} to "
            f"`FAMILY_META[{family!r}].target_names` (if the renderer "
            f"now supports the family) or fix the renderer to raise "
            f"a family-tagged kind."
        )
        return

    output = transpile(module, target=backend)
    assert output, f"backend {backend!r} on family {family!r}: empty bytes"
    assert backend_name.encode("utf-8") in output, (
        f"backend {backend!r} on family {family!r}: emitted output "
        f"does not contain expected distribution name "
        f"{backend_name!r}; got: "
        f"{output.decode('utf-8', errors='replace')[:200]!r}"
    )
