"""Every opaque didactic field in quivers declares how it enters equality.

didactic's own `Model.__eq__` and `Model.__hash__` ignore opaque fields,
so a model with an opaque field and no declared policy silently treats
values that differ in that field as equal. These checks fail when any
such field appears:

- statically, every ``dx.field(..., opaque=True, ...)`` call in the
  source tree passes ``extras=OPAQUE_BY_VALUE``,
  ``extras=OPAQUE_BY_IDENTITY``, or ``extras=opaque_by_value(freeze)``,
  including modules the host cannot import;
- at runtime, every `didactic.api.Model` subclass defined in quivers
  that has an opaque field is a `quivers._values.ValueModel`, declares
  a valid policy for each opaque field, and does not replace the shared
  equality, hashing, or pickling.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from collections.abc import Iterator
from pathlib import Path

import didactic.api as dx
import pytest

import quivers
from quivers._values import ValueModel, opaque_policy

_SRC = Path(quivers.__file__).resolve().parent
_POLICIES = {"OPAQUE_BY_VALUE", "OPAQUE_BY_IDENTITY", "opaque_by_value"}


def _opaque_field_calls(tree: ast.AST) -> Iterator[ast.Call]:
    """Yield every call that passes ``opaque=True``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "opaque"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                yield node


def _declared_policy(call: ast.Call) -> str | None:
    """The policy constant, or policy helper, passed as ``extras``, if any."""
    for keyword in call.keywords:
        if keyword.arg != "extras":
            continue
        if isinstance(keyword.value, ast.Name):
            return keyword.value.id
        if isinstance(keyword.value, ast.Call) and isinstance(
            keyword.value.func, ast.Name
        ):
            return keyword.value.func.id
    return None


def _source_files() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_every_opaque_field_in_the_source_declares_a_policy():
    missing: list[str] = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in _opaque_field_calls(tree):
            if _declared_policy(call) not in _POLICIES:
                missing.append(f"{path.relative_to(_SRC.parent)}:{call.lineno}")
    assert not missing, (
        "opaque fields without extras=OPAQUE_BY_VALUE, extras=OPAQUE_BY_IDENTITY, "
        "or extras=opaque_by_value(...): " + ", ".join(missing)
    )


def test_the_static_scan_detects_an_undeclared_opaque_field():
    source = (
        "class M(dx.Model):\n"
        "    a: int = dx.field(opaque=True)\n"
        "    b: int = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)\n"
        "    c: int = dx.field(opaque=True, extras={'k': 'v'})\n"
        "    d: int = dx.field(opaque=True, extras=opaque_by_value(f))\n"
        "    e: int = dx.field(opaque=True, extras=dict(k='v'))\n"
    )
    calls = list(_opaque_field_calls(ast.parse(source)))
    assert [_declared_policy(c) in _POLICIES for c in calls] == [
        False,
        True,
        False,
        True,
        False,
    ]


def _import_all_modules() -> list[str]:
    """Import every quivers module; return the names that fail to import."""
    failed: list[str] = []
    for info in pkgutil.walk_packages(quivers.__path__, "quivers."):
        try:
            importlib.import_module(info.name)
        except ImportError, NameError:
            failed.append(info.name)
    return failed


def _model_subclasses(root: type[dx.Model]) -> Iterator[type[dx.Model]]:
    for sub in root.__subclasses__():
        yield sub
        yield from _model_subclasses(sub)


def test_modules_the_host_cannot_import_define_no_models():
    """The modules that need a target language's runtime to import are
    covered by the static scan alone, so they must not define models."""
    for name in _import_all_modules():
        path = _SRC.parent.joinpath(*name.split(".")).with_suffix(".py")
        if not path.exists():
            path = _SRC.parent.joinpath(*name.split("."), "__init__.py")
        source = path.read_text(encoding="utf-8")
        assert "didactic" not in source, f"{name} defines didactic models"


def _quivers_models() -> list[type[dx.Model]]:
    _import_all_modules()
    return sorted(
        {
            cls
            for cls in _model_subclasses(dx.Model)
            if cls.__module__.startswith("quivers.")
        },
        key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        cls
        for cls in _quivers_models()
        if any(spec.is_opaque for spec in cls.__field_specs__.values())
    ],
    ids=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
)
def test_every_model_with_an_opaque_field_uses_value_semantics(
    model_cls: type[dx.Model],
):
    assert issubclass(model_cls, ValueModel), (
        f"{model_cls.__qualname__} has opaque fields but is not a ValueModel"
    )
    for name, spec in model_cls.__field_specs__.items():
        if spec.is_opaque:
            assert opaque_policy(model_cls, name) is not None, (
                f"{model_cls.__qualname__}.{name} declares no equality policy"
            )
    assert model_cls.__eq__ is ValueModel.__eq__
    assert model_cls.__hash__ is ValueModel.__hash__
    assert model_cls.__reduce__ is ValueModel.__reduce__
