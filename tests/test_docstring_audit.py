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


def test_check_directory_audits_nested_python_modules(tmp_path: Path, capsys) -> None:
    from tools.docstring_audit import main

    package = tmp_path / "package"
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "complete.py").write_text('''def complete():
    """Perform no operation."""
    pass
''')
    (nested / "incomplete.py").write_text('''def incomplete(value):
    """Return the supplied value."""
    return value
''')
    assert main(["--check", str(package)]) == 1
    summary = capsys.readouterr().out
    assert "1 incomplete docstring(s) across 1 file(s) with findings" in summary
    assert "audited 2 Python file(s)" in summary
    assert "incomplete.py" in summary


def test_clean_audit_reports_number_of_examined_files(tmp_path: Path, capsys) -> None:
    from tools.docstring_audit import main

    (tmp_path / "complete.py").write_text('''def complete():
    """Perform no operation."""
    pass
''')
    assert main(["--check", str(tmp_path)]) == 0
    summary = capsys.readouterr().out
    assert "0 incomplete docstring(s) across 0 file(s) with findings" in summary
    assert "audited 1 Python file(s)" in summary


def test_missing_audit_directory_is_an_error(tmp_path: Path, capsys) -> None:
    import pytest

    from tools.docstring_audit import main

    with pytest.raises(SystemExit) as error:
        main(["--check", str(tmp_path / "missing")])
    assert error.value.code == 2
    assert "audit paths do not exist" in capsys.readouterr().err


def test_check_empty_audit_directory_cannot_report_success(
    tmp_path: Path, capsys
) -> None:
    import pytest

    from tools.docstring_audit import main

    with pytest.raises(SystemExit) as error:
        main(["--check", str(tmp_path)])
    assert error.value.code == 2
    assert "no Python modules found" in capsys.readouterr().err
