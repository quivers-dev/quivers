"""Memoisation of probe measurements, and what it refuses to reuse.

[`run_probe`][tests.transpile._docker.run_probe] can skip a container
when it has already measured the same thing. The saving is large, since
a probe pays for a language runtime rather than for the container, but
a memoised measurement is only as sound as the key that selects it.

The tests here are mostly about rejection. Each one changes a single
input and requires the key to move, so that a stale number cannot be
served after the thing it measured has changed. The one thing they do
not check by mutation is whether a hit really skips the container: that
is asserted directly, by making the container impossible to launch.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tests.transpile import _docker


_SCRIPT_DIR = pathlib.Path(__file__).parent / "probes" / "_scripts"


def _key(**overrides: object) -> str:
    """The key for a baseline call, with `overrides` applied."""
    base: dict[str, object] = {
        "image": "panproto-test-numpyro",
        "resolved_image_id": "sha256:baseline",
        "source": b"model = 1",
        "source_ext": "py",
        "points": [{"params": {"theta": [0.5]}, "data": {"y": [1.0]}}],
        "shapes": {"y": [1]},
        "dtypes": {"y": "float"},
        "script": _SCRIPT_DIR / "numpyro.py",
        "side_files": {},
    }
    base.update(overrides)
    return _docker._probe_cache_key(**base)  # type: ignore[arg-type]


def test_the_key_is_stable_for_identical_inputs() -> None:
    """Two calls on the same inputs agree, or nothing ever hits."""
    assert _key() == _key()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "panproto-test-pyro"),
        ("resolved_image_id", "sha256:rebuilt"),
        ("source", b"model = 2"),
        ("source_ext", "jl"),
        ("points", [{"params": {"theta": [0.25]}, "data": {"y": [1.0]}}]),
        ("shapes", {"y": [2]}),
        ("dtypes", {"y": "int"}),
        ("script", _SCRIPT_DIR / "pyro.py"),
        ("side_files", {"export_names.json": b'["theta"]'}),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_changing_one_input_changes_the_key(field: str, value: object) -> None:
    """Every input that can move the measurement moves the key.

    `resolved_image_id` is the subtle one. A tag is a mutable pointer,
    so rebuilding an image from an edited Dockerfile leaves `image`
    identical while every measurement taken under the old one is now
    wrong.
    """
    assert _key(**{field: value}) != _key(), (
        f"changing {field!r} left the probe cache key unchanged, so a "
        f"measurement taken before the change would still be served "
        f"after it."
    )


def test_editing_a_probe_script_changes_the_key(tmp_path: pathlib.Path) -> None:
    """The script is hashed by content, not by name."""
    script = tmp_path / "probe.py"
    script.write_bytes(b"print(1)\n")
    before = _key(script=script)
    script.write_bytes(b"print(2)\n")
    assert _key(script=script) != before


@pytest.mark.parametrize("helper", ["_reshape.py", "_reshape.jl"])
def test_editing_a_reshape_helper_changes_the_key(
    helper: str, tmp_path: pathlib.Path
) -> None:
    """The helpers `run_probe` copies in beside the script are hashed
    too. They are what turns a flat point payload back into the arrays
    the probe scores, so an edit to one changes the number the
    container reports while leaving the script itself untouched."""
    script = tmp_path / "probe.py"
    script.write_bytes(b"print(1)\n")
    before = _key(script=script)
    (tmp_path / helper).write_bytes(b"# reshape\n")
    assert _key(script=script) != before


def test_caching_is_off_unless_the_variable_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No directory, no cache. The default is to run the container."""
    monkeypatch.delenv(_docker.PROBE_CACHE_ENV, raising=False)
    assert _docker.probe_cache_dir() is None


def test_the_directory_is_created_on_demand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A configured path that does not exist yet is made, so a fresh
    checkout does not have to prepare one."""
    target = tmp_path / "nested" / "probe-cache"
    monkeypatch.setenv(_docker.PROBE_CACHE_ENV, str(target))
    assert _docker.probe_cache_dir() == target
    assert target.is_dir()


def _write_cache_entry(cache_dir: pathlib.Path, key: str, payload: dict) -> None:
    (cache_dir / f"{key}.json").write_text(json.dumps(payload))


def test_a_hit_does_not_launch_a_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The point of the cache: on a hit, `docker run` never happens.

    Asserted by making the launch raise. A test that merely timed the
    call could pass on a fast container; this one cannot pass unless
    the container is genuinely skipped.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv(_docker.PROBE_CACHE_ENV, str(cache_dir))
    monkeypatch.setattr(_docker, "image_id", lambda tag: "sha256:pinned")

    script = tmp_path / "probe.py"
    script.write_bytes(b"print(1)\n")
    source = b"model = 1"
    points = [{"params": {}, "data": {}}]
    key = _docker._probe_cache_key(
        image="panproto-test-numpyro",
        resolved_image_id="sha256:pinned",
        source=source,
        source_ext="py",
        points=points,
        shapes=None,
        dtypes=None,
        script=script,
        side_files={},
    )
    _write_cache_entry(cache_dir, key, {"log_densities": [-1.5, -2.5]})

    def _explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("a cache hit must not launch a container")

    monkeypatch.setattr(_docker.subprocess, "run", _explode)
    result = _docker.run_probe(
        image="panproto-test-numpyro",
        script=script,
        source=source,
        source_ext="py",
        points=points,
        scratch=tmp_path / "scratch",
    )
    assert result == {"log_densities": [-1.5, -2.5]}


def test_a_changed_source_misses_the_entry_written_for_the_old_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The rejection that matters most, end to end.

    An entry recorded for one emitted program must not answer for a
    different one. The container is again made to raise, so reaching it
    is the observable signal that the cache correctly declined.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv(_docker.PROBE_CACHE_ENV, str(cache_dir))
    monkeypatch.setattr(_docker, "image_id", lambda tag: "sha256:pinned")

    script = tmp_path / "probe.py"
    script.write_bytes(b"print(1)\n")
    points = [{"params": {}, "data": {}}]
    key = _docker._probe_cache_key(
        image="panproto-test-numpyro",
        resolved_image_id="sha256:pinned",
        source=b"model = 1",
        source_ext="py",
        points=points,
        shapes=None,
        dtypes=None,
        script=script,
        side_files={},
    )
    _write_cache_entry(cache_dir, key, {"log_densities": [-1.5]})

    def _explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("reached the container")

    monkeypatch.setattr(_docker.subprocess, "run", _explode)
    with pytest.raises(RuntimeError, match="reached the container"):
        _docker.run_probe(
            image="panproto-test-numpyro",
            script=script,
            source=b"model = 2",
            source_ext="py",
            points=points,
            scratch=tmp_path / "scratch",
        )


def test_a_file_added_to_the_mounted_directory_misses_the_entry_without_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A probe's instructions in the mounted directory are part of the key.

    An entry recorded for a plain scoring call must not answer a call
    that also ships a list of names to export: the cached result holds
    no exports. The container is made to raise, so reaching it shows
    the cache declined.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv(_docker.PROBE_CACHE_ENV, str(cache_dir))
    monkeypatch.setattr(_docker, "image_id", lambda tag: "sha256:pinned")

    script = tmp_path / "probe.py"
    script.write_bytes(b"print(1)\n")
    source = b"model = 1"
    points = [{"params": {}, "data": {}}]
    key = _docker._probe_cache_key(
        image="panproto-test-numpyro",
        resolved_image_id="sha256:pinned",
        source=source,
        source_ext="py",
        points=points,
        shapes=None,
        dtypes=None,
        script=script,
        side_files={},
    )
    _write_cache_entry(cache_dir, key, {"log_densities": [-1.5]})

    class _Reached(Exception):
        pass

    def _reach(*args: object, **kwargs: object) -> object:
        raise _Reached

    monkeypatch.setattr(_docker.subprocess, "run", _reach)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "export_names.json").write_text(json.dumps(["theta"]))
    with pytest.raises(_Reached):
        _docker.run_probe(
            image="panproto-test-numpyro",
            script=script,
            source=source,
            source_ext="py",
            points=points,
            scratch=scratch,
        )
