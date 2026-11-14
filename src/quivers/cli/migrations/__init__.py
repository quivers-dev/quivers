"""Registry of structural one-hop ``.qvr`` source migrations.

Each adjacent-pair migrator is a callable ``bytes -> bytes`` built
on the panproto pipeline in `quivers.cli.migrations._common`:
parse with the source revision's tree-sitter parser, build the
target schema by walking the source parse tree and emitting target
vertices/edges/constraints, then emit canonical target bytes via
the target revision's ``emit_pretty``. The third stage is grammar-
bound by panproto: a target schema that does not satisfy the target
grammar's rules cannot emit.

The `CHAIN` tuple lists releases in chronological order with
``"HEAD"`` always last. Each adjacent pair ``(CHAIN[i], CHAIN[i+1])``
must have a registered migrator in `MIGRATORS`.
`compose_migration` walks ``CHAIN`` between any two listed
releases and composes the intermediate hops into a single
``bytes -> bytes`` callable.

To add the next release:

1. Tag the release in git.
2. Run ``python grammars/qvr/vcs/build_schemas.py`` and
   ``python grammars/qvr/vcs/build_parsers.py``.
3. Replace the current ``v<latest> -> HEAD`` edge with the pinned
   ``v<latest> -> v<next>`` hop, then add a new ``v<next> -> HEAD``
   hop covering the next batch of grammar changes.  Use the explicit
   identity migrator when the authored grammar is unchanged.
4. Insert the new release immediately before ``"HEAD"`` in `CHAIN`
   and register both affected edges in `MIGRATORS`.

The ``qvr migrate`` CLI consumes `compose_migration` and is
agnostic to which pairs are registered.
"""

from __future__ import annotations

from typing import Callable

import panproto

from quivers.cli.migrations import _identity
from quivers.cli.migrations import v0_2_0_to_v0_3_0 as _hop_2_3
from quivers.cli.migrations import v0_3_0_to_v0_4_0 as _hop_3_4
from quivers.cli.migrations import v0_4_0_to_v0_5_0 as _hop_4_5
from quivers.cli.migrations import v0_5_0_to_v0_6_0 as _hop_5_6
from quivers.cli.migrations import v0_6_0_to_v0_7_0 as _hop_6_7
from quivers.cli.migrations import v0_7_0_to_v0_9_0 as _hop_7_9
from quivers.cli.migrations import v0_10_0_to_v0_11_0 as _hop_10_11
from quivers.cli.migrations import v0_11_0_to_v0_14_0 as _hop_11_14
from quivers.cli.migrations import v0_14_0_to_v0_15_0 as _hop_14_15
from quivers.cli.migrations import v0_18_0_to_v0_19_0 as _hop_18_19
from quivers.cli.migrations import v0_19_0_to_head as _hop_19_head
from quivers.cli.migrations._common import MigrationError
from quivers.cli.migrations._manifest import SCHEMA_COMMITS, revisions_are_identical
from quivers.cli.migrations._vcs import (
    BlameReport,
    DiffCoverageReport,
    _open_repo,
    blame_kind,
    check_chain_coverage,
    commit_id_for,
    diff_coverage,
)

_Migrator = Callable[[bytes], bytes]


CHAIN: tuple[str, ...] = (
    "v0.2.0",
    "v0.3.0",
    "v0.4.0",
    "v0.5.0",
    "v0.6.0",
    "v0.7.0",
    "v0.9.0",
    "v0.10.0",
    "v0.11.0",
    "v0.14.0",
    "v0.15.0",
    "v0.16.0",
    "v0.17.0",
    "v0.18.0",
    "v0.19.0",
    "HEAD",
)
"""The QVR grammar releases a migration can start or end at, oldest
first, with ``"HEAD"`` (the current grammar) always last."""


# Commit-id index: maps each `CHAIN` release name to its
# panproto VCS commit id, computed once at module-import time.
# Released tags resolve directly; releases whose grammar is byte-
# identical to a predecessor share that predecessor's commit;
# the explicit final ``HEAD`` entry resolves to the VCS head.
_COMMIT_IDS: dict[str, str] = {}


def _build_commit_index() -> dict[str, str]:
    """Resolve `CHAIN` through its explicit schema-commit manifest."""
    if set(CHAIN) != set(SCHEMA_COMMITS):
        raise MigrationError(
            "migration chain and schema-commit manifest have different revisions"
        )
    out: dict[str, str] = {}
    for ref in CHAIN:
        expected = SCHEMA_COMMITS[ref]
        resolved = commit_id_for(ref)
        if resolved and resolved != expected:
            raise MigrationError(
                f"schema ref {ref!r} resolves to {resolved}, but the migration "
                f"manifest pins {expected}"
            )
        # Some releases deliberately share a schema commit and therefore
        # have no separate panproto tag. The manifest is the only permitted
        # alias mechanism; never infer an alias from tag ordering.
        out[ref] = expected
    return out


def commit_id(ref: str) -> str:
    """Resolve a release name to its pinned panproto VCS commit id.

    The commit ids come from the schema-commit manifest and are
    computed once, on first call.

    Parameters
    ----------
    ref : str
        A release name in `CHAIN`, such as ``"v0.19.0"`` or ``"HEAD"``.

    Returns
    -------
    str
        The commit id, or ``""`` when ``ref`` is not in `CHAIN`.

    Raises
    ------
    MigrationError
        When the manifest disagrees with `CHAIN` or with the VCS tags.
    """
    if not _COMMIT_IDS:
        _COMMIT_IDS.update(_build_commit_index())
    return _COMMIT_IDS.get(ref, "")


MIGRATORS: dict[tuple[str, str], _Migrator] = {
    ("v0.2.0", "v0.3.0"): _hop_2_3.migrate,
    ("v0.3.0", "v0.4.0"): _hop_3_4.migrate,
    ("v0.4.0", "v0.5.0"): _hop_4_5.migrate,
    ("v0.5.0", "v0.6.0"): _hop_5_6.migrate,
    ("v0.6.0", "v0.7.0"): _hop_6_7.migrate,
    ("v0.7.0", "v0.9.0"): _hop_7_9.migrate,
    ("v0.9.0", "v0.10.0"): _identity.migrator(
        "v0.9.0", "v0.10.0", resolve_commit=commit_id
    ),
    ("v0.10.0", "v0.11.0"): _hop_10_11.migrate,
    ("v0.11.0", "v0.14.0"): _hop_11_14.migrate,
    ("v0.14.0", "v0.15.0"): _hop_14_15.migrate,
    ("v0.15.0", "v0.16.0"): _identity.migrator(
        "v0.15.0", "v0.16.0", resolve_commit=commit_id
    ),
    ("v0.16.0", "v0.17.0"): _identity.migrator(
        "v0.16.0", "v0.17.0", resolve_commit=commit_id
    ),
    ("v0.17.0", "v0.18.0"): _identity.migrator(
        "v0.17.0", "v0.18.0", resolve_commit=commit_id
    ),
    ("v0.18.0", "v0.19.0"): _hop_18_19.migrate,
    ("v0.19.0", "HEAD"): _hop_19_head.migrate,
}
"""The one-hop migrator registered for each adjacent pair of `CHAIN`,
a ``bytes -> bytes`` rewrite of ``.qvr`` source."""

COVERAGE: dict[tuple[str, str], frozenset[str]] = {
    ("v0.2.0", "v0.3.0"): _hop_2_3.SOURCE_RULE_COVERAGE,
    ("v0.3.0", "v0.4.0"): _hop_3_4.SOURCE_RULE_COVERAGE,
    ("v0.4.0", "v0.5.0"): _hop_4_5.SOURCE_RULE_COVERAGE,
    ("v0.5.0", "v0.6.0"): _hop_5_6.SOURCE_RULE_COVERAGE,
    ("v0.6.0", "v0.7.0"): _hop_6_7.SOURCE_RULE_COVERAGE,
    ("v0.7.0", "v0.9.0"): _hop_7_9.SOURCE_RULE_COVERAGE,
    ("v0.9.0", "v0.10.0"): _identity.SOURCE_RULE_COVERAGE,
    ("v0.10.0", "v0.11.0"): _hop_10_11.SOURCE_RULE_COVERAGE,
    ("v0.11.0", "v0.14.0"): _hop_11_14.SOURCE_RULE_COVERAGE,
    ("v0.14.0", "v0.15.0"): _hop_14_15.SOURCE_RULE_COVERAGE,
    ("v0.15.0", "v0.16.0"): _identity.SOURCE_RULE_COVERAGE,
    ("v0.16.0", "v0.17.0"): _identity.SOURCE_RULE_COVERAGE,
    ("v0.17.0", "v0.18.0"): _identity.SOURCE_RULE_COVERAGE,
    ("v0.18.0", "v0.19.0"): _hop_18_19.SOURCE_RULE_COVERAGE,
    ("v0.19.0", "HEAD"): _hop_19_head.SOURCE_RULE_COVERAGE,
}
"""The source-side grammar rules each hop's converters cover, keyed by
adjacent pair of `CHAIN`. `vcs_coverage_report` checks these against
the schema diff between the two revisions."""


def _declared_identity_pairs() -> frozenset[tuple[str, str]]:
    """Recover identity edges from the callables registered for each hop."""
    pairs: set[tuple[str, str]] = set()
    for registry_pair, migration in MIGRATORS.items():
        declared_pair = getattr(migration, "_qvr_identity_pair", None)
        if declared_pair is None:
            continue
        if declared_pair != registry_pair:
            raise MigrationError(
                f"identity migrator for {declared_pair!r} is registered as "
                f"{registry_pair!r}"
            )
        pairs.add(registry_pair)
    return frozenset(pairs)


IDENTITY_PAIRS: frozenset[tuple[str, str]] = _declared_identity_pairs()
"""The adjacent pairs of `CHAIN` whose grammars are identical, so that
their registered migrator returns its source unchanged."""


def _validate_registry() -> frozenset[tuple[str, str]]:
    """Fail closed when the executable chain drifts from its manifests."""
    expected_edges = set(zip(CHAIN, CHAIN[1:]))
    migration_edges = set(MIGRATORS)
    coverage_edges = set(COVERAGE)
    if migration_edges != expected_edges:
        raise MigrationError(
            "migration registry does not match adjacent CHAIN edges: "
            f"missing={sorted(expected_edges - migration_edges)!r}, "
            f"extra={sorted(migration_edges - expected_edges)!r}"
        )
    if coverage_edges != expected_edges:
        raise MigrationError(
            "coverage registry does not match adjacent CHAIN edges: "
            f"missing={sorted(expected_edges - coverage_edges)!r}, "
            f"extra={sorted(coverage_edges - expected_edges)!r}"
        )
    declared_identity = _declared_identity_pairs()
    manifest_identity = frozenset(
        pair for pair in expected_edges if revisions_are_identical(*pair)
    )
    if declared_identity != manifest_identity:
        raise MigrationError(
            "registered identity migrators do not match the grammar/schema "
            f"manifest: declared={sorted(declared_identity)!r}, "
            f"manifest={sorted(manifest_identity)!r}"
        )
    return declared_identity


def vcs_coverage_report() -> list[DiffCoverageReport]:
    """Check every registered hop against the grammar's schema diff.

    For each adjacent pair of `CHAIN`, the report carries the schema
    diff between the two revisions and the removed source rules that
    the hop's `COVERAGE` entry does not cover. ``qvr migrate --check``
    runs this, and a CI test can run it to catch a migrator that has
    fallen behind a grammar change.

    Returns
    -------
    list[DiffCoverageReport]
        One report per adjacent pair of `CHAIN`, in chain order.

    Raises
    ------
    MigrationError
        When `MIGRATORS`, `COVERAGE`, or `IDENTITY_PAIRS` disagree with
        `CHAIN` or with the grammar manifest.
    """
    identity_pairs = _validate_registry()
    reports: list[DiffCoverageReport] = []
    for i in range(len(CHAIN) - 1):
        from_ref = CHAIN[i]
        to_ref = CHAIN[i + 1]
        if (from_ref, to_ref) in identity_pairs:
            _identity.validate_identity_pair(from_ref, to_ref, resolve_commit=commit_id)
        decl = COVERAGE.get((from_ref, to_ref), frozenset())
        from_id = commit_id(from_ref)
        to_id = commit_id(to_ref)
        if from_id == to_id:
            reports.append(
                DiffCoverageReport(
                    from_ref=from_ref,
                    to_ref=to_ref,
                    added_rules=(),
                    removed_rules=(),
                    uncovered_removed=(),
                )
            )
            continue
        # Re-implement diff_coverage's body using the resolved
        # commit ids directly to avoid double-resolution.

        repo = _open_repo()
        src_schema = repo.schema_at(from_id)
        tgt_schema = repo.schema_at(to_id)
        diff_dict = panproto.diff_schemas(src_schema, tgt_schema).to_dict()
        added = tuple(sorted(diff_dict.get("added_vertices", [])))
        removed = tuple(sorted(diff_dict.get("removed_vertices", [])))
        uncovered = tuple(r for r in removed if r not in decl)
        reports.append(
            DiffCoverageReport(
                from_ref=from_ref,
                to_ref=to_ref,
                added_rules=added,
                removed_rules=removed,
                uncovered_removed=uncovered,
            )
        )
    return reports


def _resolve_ref(ref: str) -> str:
    """Normalize a user-facing revision string against `CHAIN`.

    ``"HEAD"`` is an explicit final entry of `CHAIN`; every value must
    therefore already be a member of the chain."""
    return ref


def _chain_slice(from_ref: str, to_ref: str) -> list[tuple[str, str]]:
    """Return the ordered list of adjacent ``(src, dst)`` pairs that
    walk `CHAIN` from ``from_ref`` to ``to_ref``."""
    from_ref = _resolve_ref(from_ref)
    to_ref = _resolve_ref(to_ref)
    if from_ref not in CHAIN:
        raise MigrationError(
            f"unknown source revision {from_ref!r}; known: {CHAIN}",
        )
    if to_ref not in CHAIN:
        raise MigrationError(
            f"unknown target revision {to_ref!r}; known: {CHAIN}",
        )
    i_from = CHAIN.index(from_ref)
    i_to = CHAIN.index(to_ref)
    if i_to < i_from:
        raise MigrationError(
            f"backward migration ({from_ref} -> {to_ref}) is not yet "
            "implemented; only forward composition through CHAIN is "
            "supported",
        )
    return [(CHAIN[i], CHAIN[i + 1]) for i in range(i_from, i_to)]


def compose_migration(from_ref: str, to_ref: str) -> _Migrator:
    """Compose the hops between two releases into one migrator.

    Parameters
    ----------
    from_ref : str
        The source release, a member of `CHAIN`.
    to_ref : str
        The target release, a member of `CHAIN` no earlier than
        ``from_ref``.

    Returns
    -------
    Callable[[bytes], bytes]
        A rewrite of ``.qvr`` source from ``from_ref``'s grammar to
        ``to_ref``'s, applying every registered hop in between.

    Raises
    ------
    MigrationError
        When either release is not in `CHAIN`, when ``to_ref`` precedes
        ``from_ref``, or when a hop between them has no migrator.
    """
    pairs = _chain_slice(from_ref, to_ref)
    if not pairs:
        return _identity.migrator(from_ref, to_ref, resolve_commit=commit_id)
    missing = [pair for pair in pairs if pair not in MIGRATORS]
    if missing:
        raise MigrationError(
            "missing migrator(s) for pair(s): "
            + ", ".join(f"{a} -> {b}" for a, b in missing),
        )
    migrators = [MIGRATORS[pair] for pair in pairs]

    def _composite(source: bytes) -> bytes:
        for step in migrators:
            source = step(source)
        return source

    return _composite


def available_targets(from_ref: str) -> tuple[str, ...]:
    """List the releases a migration from ``from_ref`` can target.

    Parameters
    ----------
    from_ref : str
        The source release.

    Returns
    -------
    tuple[str, ...]
        ``from_ref`` and every later release of `CHAIN`, or the empty
        tuple when ``from_ref`` is not in `CHAIN`.
    """
    if from_ref not in CHAIN:
        return ()
    i = CHAIN.index(from_ref)
    return CHAIN[i:]


__all__ = [
    "BlameReport",
    "CHAIN",
    "COVERAGE",
    "DiffCoverageReport",
    "IDENTITY_PAIRS",
    "MIGRATORS",
    "MigrationError",
    "available_targets",
    "blame_kind",
    "check_chain_coverage",
    "commit_id",
    "compose_migration",
    "diff_coverage",
    "vcs_coverage_report",
]
