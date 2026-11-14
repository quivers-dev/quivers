"""Constructor documentation follows local model inheritance."""

from __future__ import annotations

from pathlib import Path

from tools.docstring_audit import audit


def test_indirect_model_fields_are_constructor_parameters(tmp_path: Path) -> None:
    source = tmp_path / "models.py"
    source.write_text('''class Frame(QiecModel):
    """A model base."""

class Intermediate(Frame):
    """An intermediate model base."""

class Bind(Intermediate):
    """A bind frame.

    Parameters
    ----------
    binder
        The binding name.
    """

    binder: str
''')
    assert audit(source) == []
    source.write_text(source.read_text().replace("    binder\n", "    invented\n"))
    findings = audit(source)
    assert len(findings) == 1
    assert findings[0][0] == "Bind"
    assert findings[0][2] == ["documents absent parameters: invented"]


def test_ordinary_inheritance_does_not_create_constructor_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ordinary.py"
    source.write_text('''class Base:
    """An ordinary base."""

class Child(Base):
    """An ordinary class.

    Parameters
    ----------
    name
        A class annotation, not a constructor argument.
    """

    name: str
''')
    findings = audit(source)
    assert len(findings) == 1
    assert findings[0][2] == ["documents absent parameters: name"]


def test_explicit_constructor_takes_precedence_over_model_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "explicit.py"
    source.write_text('''class Base(QiecModel):
    """A model base."""

class Child(Base):
    """A custom constructor.

    Parameters
    ----------
    actual
        The explicit argument.
    """

    field: str

    def __init__(self, actual: str) -> None:
        self.field = actual
''')
    assert audit(source) == []
