from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest

from quivers.cli import migrate as migrate_cli
from quivers.cli.migrations import (
    CHAIN,
    COVERAGE,
    IDENTITY_PAIRS,
    MIGRATORS,
    MigrationError,
    commit_id,
    compose_migration,
    vcs_coverage_report,
)
from quivers.cli.migrations import _assets, _grammar, _identity, _manifest
from quivers.cli.migrations import v0_14_0_to_v0_15_0 as hop_14_15
from quivers.cli.migrations import v0_18_0_to_v0_19_0 as hop_18_19
from quivers.cli.migrations import v0_19_0_to_v0_62_0 as hop_19_62


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chain_has_every_release_and_explicit_head_terminus() -> None:
    assert CHAIN[-7:] == (
        "v0.15.0",
        "v0.16.0",
        "v0.17.0",
        "v0.18.0",
        "v0.19.0",
        "v0.62.0",
        "HEAD",
    )
    pairs = set(zip(CHAIN, CHAIN[1:]))
    assert set(MIGRATORS) == pairs
    assert set(COVERAGE) == pairs


def test_released_identity_segment_shares_one_grammar_commit() -> None:
    commits = {commit_id(ref) for ref in CHAIN[-7:-3]}
    assert len(commits) == 1
    assert "" not in commits


def test_v014_hop_uses_pinned_v015_parser() -> None:
    assert hop_14_15._TARGET_REV == "v0.15.0"
    snapshot = REPO_ROOT / "grammars/qvr/vcs/parsers/v0.15.0/source/src"
    assert (snapshot / "parser.c").is_file()
    assert (snapshot / "grammar.json").is_file()
    assert (snapshot / "node-types.json").is_file()


def test_v015_to_v018_aliases_use_one_canonical_regenerated_snapshot() -> None:
    aliases = ("v0.15.0", "v0.16.0", "v0.17.0", "v0.18.0")
    assert {_manifest.snapshot_for(revision) for revision in aliases} == {"v0.15.0"}

    assert {_grammar.grammar_identity(revision) for revision in aliases} == {
        _manifest.GRAMMAR_ASSETS["v0.15.0"]
    }
    assert _grammar.grammar_identity("HEAD") != _manifest.GRAMMAR_ASSETS["v0.15.0"]


def test_v019_has_the_parser_that_was_head_at_release() -> None:
    assert _manifest.snapshot_for("v0.19.0") == "v0.19.0"
    assert _grammar.grammar_identity("v0.19.0") != _grammar.grammar_identity("HEAD")


def test_hop_composition_is_coherent_across_identity_releases() -> None:
    source = b"bundle demo = [first, second]\n"
    direct = compose_migration("v0.14.0", "v0.18.0")(source)
    one_hop = compose_migration("v0.14.0", "v0.15.0")(source)
    composed = compose_migration("v0.15.0", "v0.18.0")(one_hop)
    assert direct == composed == b"bundle demo : [first, second]\n"
    assert compose_migration("v0.18.0", "HEAD")(direct) == direct


def test_zero_hop_rejects_missing_recovery_vertices() -> None:
    with pytest.raises(MigrationError, match="missing .* token"):
        compose_migration("HEAD", "HEAD")(b"object A :\n")


def test_vcs_coverage_spans_every_chain_edge() -> None:
    reports = vcs_coverage_report()
    assert [(report.from_ref, report.to_ref) for report in reports] == list(
        zip(CHAIN, CHAIN[1:]),
    )
    assert all(not report.uncovered_removed for report in reports)
    identity_reports = [
        report
        for report in reports
        if (report.from_ref, report.to_ref) in IDENTITY_PAIRS
    ]
    assert all(not report.added_rules for report in identity_reports)
    assert all(not report.removed_rules for report in identity_reports)

    head_report = reports[-1]
    assert (head_report.from_ref, head_report.to_ref) == ("v0.62.0", "HEAD")
    assert not head_report.removed_rules
    assert ("v0.62.0", "HEAD") not in IDENTITY_PAIRS


def test_v018_to_v019_additive_hop_is_byte_preserving_and_validated() -> None:
    source = b"bundle demo : [first, second]\n"
    assert hop_18_19.migrate(source) == source
    with pytest.raises(MigrationError, match="does not parse"):
        hop_18_19.migrate(b"\x00\x00\x00")


def test_v019_to_v062_layout_hop_preserves_unaffected_source_and_validates() -> None:
    source = b"define f(x : Real) : Real !{} =\n    return exp(x)\n"
    assert hop_19_62.migrate(source) == source
    with pytest.raises(MigrationError, match="does not parse"):
        hop_19_62.migrate(b"\x00\x00\x00")


def test_v019_to_v062_names_categorical_probability_arguments() -> None:
    source = b"""\
object K : FinSet 3
morphism prior : K -> K ~ Categorical(weights)
program p : K -> K
    observe y <- Categorical(weights)
    return y
"""
    expected = source.replace(b"Categorical(weights)", b"Categorical(probs=weights)")
    assert hop_19_62.migrate(source) == expected


def test_v019_to_v062_bundles_categorical_spread_probabilities() -> None:
    source = b"""\
object K : FinSet 3
program p : K -> K
    sample y <- Categorical(0.2, 0.3, 0.5)
    return y
"""
    expected = source.replace(
        b"Categorical(0.2, 0.3, 0.5)",
        b"Categorical(probs=[0.2, 0.3, 0.5])",
    )
    assert hop_19_62.migrate(source) == expected


def test_v019_to_v062_names_all_family_construction_levels() -> None:
    source = b"""\
object K : FinSet 3
morphism prior : K -> K ~ Normal(0.0, 1.0)
program p : K -> K
    sample weights <- Dirichlet(1.0, 2.0, 3.0)
    observe y <- Restrict(Normal(0.0, 1.0), 0.0, 2.0)
    return y
"""
    expected = b"""\
object K : FinSet 3
morphism prior : K -> K ~ Normal(loc=0.0, scale=1.0)
program p : K -> K
    sample weights <- Dirichlet(concentration=[1.0, 2.0, 3.0])
    observe y <- Restrict(base=Normal(loc=0.0, scale=1.0), low=0.0, high=2.0)
    return y
"""
    assert hop_19_62.migrate(source) == expected


def test_manifest_identity_drift_is_checked_for_v09_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ("v0.9.0", "v0.10.0") in IDENTITY_PAIRS
    original = _identity.grammar_identity
    changed = _manifest.GrammarAsset(
        grammar_json_sha256="0" * 64,
        node_types_sha256="1" * 64,
        parser_c_sha256="2" * 64,
        compiled_inputs_sha256="3" * 64,
    )
    monkeypatch.setattr(
        _identity,
        "grammar_identity",
        lambda revision: changed if revision == "v0.10.0" else original(revision),
    )

    with pytest.raises(MigrationError, match="parser/grammar delta"):
        vcs_coverage_report()


def test_coverage_gate_rejects_missing_executable_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(MIGRATORS, ("v0.11.0", "v0.14.0"))
    with pytest.raises(MigrationError, match="migration registry"):
        vcs_coverage_report()


def test_coverage_gate_rejects_identity_migrator_on_additive_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        MIGRATORS,
        ("v0.11.0", "v0.14.0"),
        _identity.migrator("v0.11.0", "v0.14.0", resolve_commit=commit_id),
    )
    with pytest.raises(MigrationError, match="identity migrators do not match"):
        vcs_coverage_report()


def test_v02_category_patterns_migrate_as_v03_type_expressions() -> None:
    source = b"rule app(X, Y) : (X/Y) * Y => X\n"
    assert compose_migration("v0.2.0", "v0.3.0")(source) == source


def test_v03_draw_forms_and_output_migrate_to_v04_surface() -> None:
    source = b"program P : A -> B\n  draw x ~ F(y)\n  z <- G(x)\nreturn z\noutput z\n"
    migrated = compose_migration("v0.3.0", "v0.4.0")(source)
    assert migrated == (
        b"program P : A -> B\n  x <- F(y)\n  z <- G(x)\nreturn z\nexport z\n"
    )


def test_v04_continuous_and_stochastic_decls_migrate_to_kernels() -> None:
    source = b"continuous c : A -> B ~ Normal\nstochastic s : A -> B\n"
    migrated = compose_migration("v0.4.0", "v0.5.0")(source)
    assert migrated == b"kernel c : A -> B ~ Normal\nkernel s : A -> B\n"


def test_v06_quantale_decl_migrates_to_algebra_decl() -> None:
    source = b"quantale ProductFuzzy\nsemigroupoid Path\n"
    migrated = compose_migration("v0.6.0", "v0.7.0")(source)
    assert migrated == b"algebra ProductFuzzy\nsemigroupoid Path\n"


def test_v02_historical_fixture_composes_through_head() -> None:
    source = (
        b"quantale ProductFuzzy\n"
        b"category Cat\n"
        b"rule app(X, Y) : (X/Y) * Y => X\n"
        b"object A : 2\n"
        b"object B : 3\n"
        b"continuous c : A -> B ~ Normal\n"
        b"stochastic s : A -> B\n"
        b"program P : A -> B\n"
        b"  draw x ~ c\n"
        b"  y <- s(x)\n"
        b"return y\n"
        b"output y\n"
    )
    migrated = compose_migration("v0.2.0", "HEAD")(source)
    assert b"composition ProductFuzzy [level=algebra]\n" in migrated
    assert b"rule app(X, Y) : (X/Y) * Y |- X\n" in migrated
    assert b"morphism c : A -> B [role=kernel] ~ Normal\n" in migrated
    assert b"morphism s : A -> B [role=kernel]\n" in migrated
    assert b"    sample x <- c\n" in migrated
    assert migrated.endswith(b"export y\n")


def test_package_assets_take_precedence_over_repository_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "installed/quivers/cli/migrations/_grammar_vcs"
    repository_root = tmp_path / "checkout/grammars/qvr/vcs"
    (package_root / ".panproto").mkdir(parents=True)
    (repository_root / ".panproto").mkdir(parents=True)
    monkeypatch.setattr(_assets, "_PACKAGED_VCS_ROOT", package_root)
    monkeypatch.setattr(_assets, "_REPOSITORY_VCS_ROOT", repository_root)
    assert _assets.vcs_root() == package_root


def test_packaged_source_snapshot_compiles_without_prebuilt_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = REPO_ROOT / "grammars/qvr/vcs/parsers/v0.15.0/source/src"
    package_root = tmp_path / "installed/_grammar_vcs"
    target = package_root / "parsers/v0.15.0/source/src"
    shutil.copytree(source, target)
    monkeypatch.setattr(_grammar, "candidate_vcs_roots", lambda: (package_root,))
    monkeypatch.setattr(
        _grammar.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    _grammar._REGISTRIES.clear()
    _grammar._LIB_KEEPALIVE.clear()

    registry = _grammar.registry_for("v0.18.0")
    schema = registry.lens("qvr").parse(b"bundle demo : [first]\n")
    assert not [vertex for vertex in schema.vertices if vertex.kind == "ERROR"]


def test_source_fallback_reports_missing_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot/source/src"
    snapshot.mkdir(parents=True)
    (snapshot / "parser.c").write_text("", encoding="utf-8")
    monkeypatch.setattr(_grammar.shutil, "which", lambda _name: None)
    with pytest.raises(
        _grammar.HistoricalGrammarUnavailable, match="supported C compiler"
    ):
        _grammar._compile_parser(tmp_path / "snapshot", "v0.test")


def test_cli_reports_unavailable_historical_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "model.qvr"
    source.write_bytes(b"bundle demo = [first]\n")

    def unavailable(_source: bytes) -> bytes:
        raise _grammar.HistoricalGrammarUnavailable("install a C compiler")

    monkeypatch.setattr(
        migrate_cli,
        "compose_migration",
        lambda _from_ref, _to_ref: unavailable,
    )
    args = _args(source, tmp_path / "out")
    args.from_ref = "v0.14.0"

    assert migrate_cli.main(args) == 2
    assert "install a C compiler" in capsys.readouterr().err


def _args(source: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        check=False,
        from_ref="v0.18.0",
        to_ref="HEAD",
        paths=[str(source)],
        output=str(output),
        dry_run=False,
    )


def test_recursive_output_preserves_relative_topology(tmp_path: Path) -> None:
    source = tmp_path / "models"
    nested = source / "family/variant/model.qvr"
    nested.parent.mkdir(parents=True)
    nested.write_text("bundle demo : [first]\n", encoding="utf-8")
    output = tmp_path / "migrated"

    assert migrate_cli.main(_args(source, output)) == 0
    assert (output / "family/variant/model.qvr").read_bytes() == nested.read_bytes()
    assert not (output / "model.qvr").exists()


def test_unchanged_in_place_source_is_not_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.qvr"
    original = b"bundle demo : [first]\n"
    source.write_bytes(original)
    args = _args(source, tmp_path / "unused")
    args.output = None

    def fail_write(
        _target: Path,
        _data: bytes,
        *,
        source_mode: int | None = None,
    ) -> None:
        del source_mode
        raise AssertionError("unchanged source should not be rewritten")

    monkeypatch.setattr(migrate_cli, "_atomic_write", fail_write)
    assert migrate_cli.main(args) == 0
    assert source.read_bytes() == original


def test_output_collision_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left/model.qvr"
    right = tmp_path / "right/model.qvr"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("bundle left : [first]\n", encoding="utf-8")
    right.write_text("bundle right : [first]\n", encoding="utf-8")
    args = _args(left, tmp_path / "out")
    args.paths = [str(left), str(right)]

    assert migrate_cli.main(args) == 2
    assert "output collision" in capsys.readouterr().err


def test_output_directory_cannot_alias_an_explicit_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "model.qvr"
    original = b"bundle demo = [first]\n"
    source.write_bytes(original)

    assert migrate_cli.main(_args(source, tmp_path)) == 2
    assert source.read_bytes() == original
    assert "aliases input" in capsys.readouterr().err


def test_atomic_write_failure_preserves_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.qvr"
    target.write_bytes(b"original\n")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr(migrate_cli.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        migrate_cli._atomic_write(target, b"replacement\n")

    assert target.read_bytes() == b"original\n"
    assert list(tmp_path.glob(".model.qvr.*.tmp")) == []


def test_cli_requires_explicit_source_revision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "model.qvr"
    source.write_text("bundle demo : [first]\n", encoding="utf-8")
    args = _args(source, tmp_path / "out")
    args.from_ref = None

    assert migrate_cli.main(args) == 2
    assert "--from is required" in capsys.readouterr().err
    assert not (tmp_path / "out/model.qvr").exists()


def test_identity_hop_rejects_schema_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_identity, "schemas_identical", lambda _src, _dst: False)

    with pytest.raises(MigrationError, match="panproto schema delta"):
        _identity.migrate(
            b"bundle demo : [first]\n", "v0.17.0", "v0.18.0", resolve_commit=commit_id
        )


def test_identity_hop_rejects_parser_hash_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _identity.grammar_identity
    changed = _manifest.GrammarAsset(
        grammar_json_sha256="0" * 64,
        node_types_sha256="1" * 64,
        parser_c_sha256="2" * 64,
        compiled_inputs_sha256="3" * 64,
    )
    monkeypatch.setattr(
        _identity,
        "grammar_identity",
        lambda revision: changed if revision == "v0.18.0" else original(revision),
    )

    with pytest.raises(MigrationError, match="parser/grammar delta"):
        _identity.migrate(
            b"bundle demo : [first]\n", "v0.17.0", "v0.18.0", resolve_commit=commit_id
        )


def test_identity_hop_validates_target_parse() -> None:
    with pytest.raises(MigrationError, match="does not parse under identity target"):
        _identity.migrate(
            b"\x00\x00\x00", "v0.17.0", "v0.18.0", resolve_commit=commit_id
        )


@pytest.mark.parametrize(
    ("source_revision", "target_revision"),
    [
        ("v0.18.0", "v0.18.0"),
        ("v0.9.0", "v0.10.0"),
        ("v0.11.0", "v0.14.0"),
        ("v0.10.0", "v0.11.0"),
    ],
)
def test_no_op_additive_and_structural_hops_reject_invalid_source(
    source_revision: str,
    target_revision: str,
) -> None:
    migrate = compose_migration(source_revision, target_revision)
    with pytest.raises(MigrationError, match="does not parse"):
        migrate(b"\x00\x00\x00")


def test_manifest_pins_all_chain_revisions_and_snapshot_bytes() -> None:
    assert set(_manifest.REVISION_SNAPSHOTS) == set(CHAIN)
    assert set(_manifest.SCHEMA_COMMITS) == set(CHAIN)
    for revision in CHAIN:
        assert commit_id(revision) == _manifest.SCHEMA_COMMITS[revision]
        snapshot = _manifest.snapshot_for(revision)
        assert _grammar.grammar_identity(revision) == _manifest.GRAMMAR_ASSETS[snapshot]


def test_tampered_snapshot_is_rejected_by_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = REPO_ROOT / "grammars/qvr/vcs/parsers/v0.15.0/source/src"
    package_root = tmp_path / "assets"
    target = package_root / "parsers/v0.15.0/source/src"
    shutil.copytree(source, target)
    (target / "grammar.json").write_bytes(
        (target / "grammar.json").read_bytes() + b"\n"
    )
    monkeypatch.setattr(_grammar, "candidate_vcs_roots", lambda: (package_root,))

    with pytest.raises(
        _grammar.HistoricalGrammarUnavailable,
        match="does not match its hash manifest",
    ):
        _grammar._snapshot_dir("v0.18.0")


@pytest.mark.parametrize(
    "relative_path",
    [
        "scanner.c",
        "tree_sitter/alloc.h",
        "tree_sitter/array.h",
        "tree_sitter/parser.h",
    ],
)
def test_manifest_and_cache_fingerprint_cover_every_compiled_support_file(
    relative_path: str,
    tmp_path: Path,
) -> None:
    source = REPO_ROOT / "grammars/qvr/vcs/parsers/v0.15.0/source/src"
    target = tmp_path / "source/src"
    shutil.copytree(source, target)
    original_fingerprint = _grammar._source_fingerprint(target)

    changed = target / relative_path
    changed.write_bytes(changed.read_bytes() + b"\n/* drift */\n")

    assert _grammar._source_fingerprint(target) != original_fingerprint
    with pytest.raises(ValueError, match="does not match its hash manifest"):
        _manifest.verify_snapshot("v0.15.0", target)


def test_new_output_preserves_source_mode(tmp_path: Path) -> None:
    source = tmp_path / "model.qvr"
    source.write_text("bundle demo : [first]\n", encoding="utf-8")
    source.chmod(0o751)
    output = tmp_path / "out"

    assert migrate_cli.main(_args(source, output)) == 0
    assert stat.S_IMODE((output / "model.qvr").stat().st_mode) == 0o751


def test_recursive_migration_excludes_nested_output_on_repeated_runs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "models"
    source.mkdir()
    (source / "model.qvr").write_text(
        "bundle demo : [first]\n",
        encoding="utf-8",
    )
    output = source / "migrated"
    args = _args(source, output)

    assert migrate_cli.main(args) == 0
    assert migrate_cli.main(args) == 0
    assert (output / "model.qvr").is_file()
    assert not (output / "migrated").exists()


def test_concurrent_parser_compilation_uses_one_locked_unique_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    src = snapshot / "source/src"
    src.mkdir(parents=True)
    (src / "parser.c").write_text("/* parser */", encoding="utf-8")
    (src / "grammar.json").write_text("{}", encoding="utf-8")
    (src / "node-types.json").write_text("[]", encoding="utf-8")
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(_grammar.tempfile, "gettempdir", lambda: str(cache_root))
    monkeypatch.setattr(_grammar, "_compiler_for_platform", lambda: "/usr/bin/cc")

    calls: list[Path] = []
    calls_lock = threading.Lock()

    def compile_once(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        output = Path(command[-1])
        with calls_lock:
            calls.append(output)
        time.sleep(0.05)
        output.write_bytes(b"shared-library")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_grammar.subprocess, "run", compile_once)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _index: _grammar._compile_parser(snapshot, "v0.test"),
                range(2),
            )
        )

    assert results[0] == results[1]
    assert results[0].read_bytes() == b"shared-library"
    assert len(calls) == 1
    assert calls[0].parent.name.startswith("build-")


def test_windows_compile_commands_emit_dlls_without_posix_flags() -> None:
    src = Path("C:/qvr/src")
    sources = (src / "parser.c", src / "scanner.c")
    output = Path("C:/cache/build/qvr.dll")

    msvc = _grammar._compile_command(
        "cl.exe",
        src,
        sources,
        output,
        platform_name="win32",
    )
    assert "/LD" in msvc
    assert f"/OUT:{output}" in msvc
    assert "-fPIC" not in msvc

    mingw = _grammar._compile_command(
        "gcc.exe",
        src,
        sources,
        output,
        platform_name="win32",
    )
    assert "-shared" in mingw
    assert "-fPIC" not in mingw
    assert mingw[-1].endswith("qvr.dll")


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_macos_compile_command_builds_for_the_loading_architecture(
    machine: str,
) -> None:
    src = Path("/qvr/src")
    command = _grammar._compile_command(
        "cc",
        src,
        (src / "parser.c",),
        Path("/cache/build/qvr.dylib"),
        platform_name="darwin",
        machine=machine,
    )
    position = command.index("-arch")
    assert command[position + 1] == machine


def test_linux_compile_command_names_no_architecture() -> None:
    src = Path("/qvr/src")
    command = _grammar._compile_command(
        "cc",
        src,
        (src / "parser.c",),
        Path("/cache/build/qvr.so"),
        platform_name="linux",
        machine="x86_64",
    )
    assert "-arch" not in command


def test_compiled_parsers_are_cached_per_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    src = snapshot / "source/src"
    src.mkdir(parents=True)
    (src / "parser.c").write_text("/* parser */", encoding="utf-8")
    (src / "grammar.json").write_text("{}", encoding="utf-8")
    (src / "node-types.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        _grammar.tempfile, "gettempdir", lambda: str(tmp_path / "cache")
    )
    monkeypatch.setattr(_grammar, "_compiler_for_platform", lambda: "/usr/bin/cc")
    built: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        built.append(command)
        Path(command[-1]).write_bytes(b"shared-library")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_grammar.subprocess, "run", fake_run)
    paths = []
    for machine in ("arm64", "x86_64"):
        monkeypatch.setattr(_grammar.platform, "machine", lambda m=machine: m)
        paths.append(_grammar._compile_parser(snapshot, "v0.test"))
    assert paths[0] != paths[1]
    assert paths[0].parent.name.endswith("-arm64")
    assert paths[1].parent.name.endswith("-x86_64")
    assert len(built) == 2
