"""The repository's static import graph, and what a test file depends on.

The graph has one node per Python module of the package, the tests, and the
tools. An edge means that running the first module can run the second. Edges
are read from the source without importing it, and the reading only ever
over-approximates:

* ``import a.b`` and ``from a.b import c`` reach ``a``, ``a.b``, and ``a.b.c``
  where that is a module, wherever the statement stands: at the top of the
  file, inside a function, or under ``TYPE_CHECKING``.
* A module name written in a string reaches that module, so a lazy export
  table, an ``importlib.import_module`` call, and a program given to a child
  interpreter are all read. A name with a formatted tail reaches every module
  under its constant head. Docstrings are prose and are not read.
* A string that names a Python file, or a pattern of them, reaches every module
  whose file ends that way, so a module read as data counts as one imported. A
  formatted part of the name stands for any text.
* A module that imports by a name no string in it spells depends on every
  module.
* A test module also depends on the ``conftest.py`` of each directory above it.

A file that is not Python belongs to the package or the test directory that
holds it.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Iterable, Iterator
from functools import cached_property
from pathlib import Path, PurePosixPath

ROOTS = {"quivers": "src/quivers", "tests": "tests", "tools": "tools"}
"""The directory of each importable root of the repository."""

NAME = re.compile(rf"\b(?:{'|'.join(ROOTS)})(?:\.[A-Za-z_]\w*)+")
"""A dotted module name under one of the roots, in running text."""

RELATIVE = re.compile(r"\.{0,2}[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
"""A whole string that may name a module relative to a package."""

SOURCE_FILE = re.compile(r"(?:[\w*?-]+/)*(?:[\w*?-]+\.py|[\w*?-]*\w{4}[\w*?-]*\.\*)")
"""A whole string that names a Python file, or a pattern of them.

The extension may be formatted into a name that has some text of its own.
"""

IMPORTERS = frozenset({"import_module", "__import__", "run_module", "find_spec"})
"""The calls that import a module by a name computed at run time."""


def module_name(path: str) -> str | None:
    """The dotted name a Python file is imported under.

    Parameters
    ----------
    path : str
        The file, relative to the repository.

    Returns
    -------
    str | None
        The module name, or ``None`` for a file outside every root. The
        ``__init__.py`` of a package has the package's name.
    """
    if not path.endswith(".py"):
        return None
    for root, directory in ROOTS.items():
        if path.startswith(f"{directory}/"):
            parts = PurePosixPath(path).relative_to(directory).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join((root, *parts))
    return None


def _docstrings(tree: ast.Module) -> set[int]:
    """The identities of the string nodes that are docstrings.

    Parameters
    ----------
    tree : ast.Module
        The parsed module.

    Returns
    -------
    set[int]
        The ``id`` of every constant that opens a module, class, or function.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


class ImportGraph:
    """The import graph of one checkout.

    Parameters
    ----------
    repository : Path
        The root of the checkout.
    files : Iterable[str]
        Every file of the checkout, relative to it.
    """

    def __init__(self, repository: Path, files: Iterable[str]) -> None:
        self._repository = repository
        self.files = frozenset(files)
        self.modules: dict[str, str] = {}
        for path in sorted(self.files):
            name = module_name(path)
            if name is not None:
                self.modules[name] = path

    @cached_property
    def edges(self) -> dict[str, frozenset[str]]:
        """The modules each module can run, by module name.

        Returns
        -------
        dict[str, frozenset[str]]
            The direct dependencies of every module of the graph.
        """
        return {
            name: frozenset(self._dependencies(name, path))
            for name, path in self.modules.items()
        }

    def prefixes(self, name: str) -> Iterator[str]:
        """The modules that importing a dotted name runs.

        Parameters
        ----------
        name : str
            A dotted name, which may go past a module into its attributes.

        Yields
        ------
        str
            Each module of the graph the name passes through.
        """
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            prefix = ".".join(parts[:count])
            if prefix in self.modules:
                yield prefix

    def named_in(self, text: str) -> set[str]:
        """The modules running text mentions.

        Parameters
        ----------
        text : str
            A document, a program, or any other text.

        Returns
        -------
        set[str]
            Every module a dotted name in the text passes through, and every
            root the text names on its own.
        """
        found: set[str] = set()
        for match in NAME.finditer(text):
            found.update(self.prefixes(match.group()))
        for root in ROOTS:
            if root in self.modules and re.search(
                rf"\b(?:import|from)\s+{root}\b", text
            ):
                found.add(root)
        return found

    def _below(self, head: str) -> set[str]:
        """The modules whose names begin with a formatted name's constant head.

        Parameters
        ----------
        head : str
            The text before the first formatted part.

        Returns
        -------
        set[str]
            Every module under that head, with the packages above it, when the
            head is a dotted name that ends at a dot; otherwise none.
        """
        if not head.endswith(".") or head.split(".")[0] not in ROOTS:
            return set()
        found = {name for name in self.modules if name.startswith(head)}
        found.update(self.prefixes(head.rstrip(".")))
        return found

    def _relative(self, name: str, text: str) -> set[str]:
        """The modules a string names relative to the module holding it.

        Parameters
        ----------
        name : str
            The module holding the string.
        text : str
            The string.

        Returns
        -------
        set[str]
            The module the string names from the package of ``name``, if it
            names one.
        """
        if not RELATIVE.fullmatch(text):
            return set()
        package = name if self._is_package(name) else name.rpartition(".")[0]
        bare = text.lstrip(".")
        if text.startswith(".."):
            package = package.rpartition(".")[0]
        candidate = f"{package}.{bare}"
        return set(self.prefixes(candidate)) if candidate in self.modules else set()

    def _files_named(self, text: str) -> set[str]:
        """The modules whose files a string names.

        Parameters
        ----------
        text : str
            The string, with ``*`` for each part that is formatted into it.

        Returns
        -------
        set[str]
            Every module whose path ends in the name or fits the pattern.
        """
        if not SOURCE_FILE.fullmatch(text):
            return set()
        return {
            name
            for name, path in self.modules.items()
            if fnmatch.fnmatchcase(f"/{path}", f"*/{text}")
        }

    def _is_package(self, name: str) -> bool:
        """Whether a module is a package.

        Parameters
        ----------
        name : str
            The module.

        Returns
        -------
        bool
            Whether the file of the module is an ``__init__.py``.
        """
        return self.modules[name].endswith("/__init__.py")

    def _dependencies(self, name: str, path: str) -> set[str]:
        """The modules one module can run directly.

        Parameters
        ----------
        name : str
            The module.
        path : str
            Its file.

        Returns
        -------
        set[str]
            Its direct dependencies, without itself.
        """
        tree = ast.parse((self._repository / path).read_text(encoding="utf-8"), path)
        prose = _docstrings(tree)
        found = set(self.prefixes(name))
        # A module outside a package is imported with its directory's packages.
        for parent in PurePosixPath(path).parents:
            conftest = module_name(str(parent / "conftest.py"))
            if path.startswith("tests/") and conftest in self.modules:
                found.add(conftest)
        dynamic = False
        spelled: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.update(self.prefixes(alias.name))
            elif isinstance(node, ast.ImportFrom):
                base = self._imported_from(name, node)
                for alias in node.names:
                    found.update(self.prefixes(f"{base}.{alias.name}"))
                    if alias.name == "*":
                        found.update(self.prefixes(base))
            elif isinstance(node, ast.JoinedStr):
                head = node.values[0] if node.values else None
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    spelled.update(self._below(head.value))
                spelled.update(
                    self._files_named(
                        "".join(
                            part.value
                            if isinstance(part, ast.Constant)
                            and isinstance(part.value, str)
                            else "*"
                            for part in node.values
                        )
                    )
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in prose:
                    spelled.update(self.named_in(node.value))
                    spelled.update(self._relative(name, node.value))
                    spelled.update(self._files_named(node.value))
            elif isinstance(node, ast.Call):
                called = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else ""
                )
                first = node.args[0] if node.args else None
                literal = isinstance(first, ast.Constant) and isinstance(
                    first.value, str
                )
                dynamic = dynamic or (called in IMPORTERS and not literal)
        found |= spelled
        if dynamic and not spelled:
            found |= set(self.modules)
        return found - {name}

    def _imported_from(self, name: str, node: ast.ImportFrom) -> str:
        """The absolute name a ``from`` statement imports from.

        Parameters
        ----------
        name : str
            The module holding the statement.
        node : ast.ImportFrom
            The statement.

        Returns
        -------
        str
            The absolute dotted name.
        """
        if not node.level:
            return node.module or ""
        package = name if self._is_package(name) else name.rpartition(".")[0]
        for _ in range(node.level - 1):
            package = package.rpartition(".")[0]
        return f"{package}.{node.module}" if node.module else package

    def reachable(self, names: Iterable[str]) -> frozenset[str]:
        """Everything running some modules can run.

        Parameters
        ----------
        names : Iterable[str]
            The modules to start from.

        Returns
        -------
        frozenset[str]
            Those modules and all their transitive dependencies.
        """
        reached: set[str] = set()
        frontier = [name for name in names if name in self.modules]
        while frontier:
            name = frontier.pop()
            if name not in reached:
                reached.add(name)
                frontier.extend(self.edges[name] - reached)
        return frozenset(reached)

    def owner(self, path: str) -> str | None:
        """The package or test directory a file that is not Python belongs to.

        Parameters
        ----------
        path : str
            The file, relative to the repository.

        Returns
        -------
        str | None
            The nearest package above a file of the package tree, or ``None``
            for a file anywhere else.
        """
        if not path.startswith("src/quivers/") or path.endswith(".py"):
            return None
        for parent in PurePosixPath(path).parents:
            name = module_name(str(parent / "__init__.py"))
            if name in self.modules:
                return name
        return None

    def closure_files(self, names: Iterable[str]) -> frozenset[str]:
        """The files that running some modules can read as code or package data.

        Parameters
        ----------
        names : Iterable[str]
            The modules to start from.

        Returns
        -------
        frozenset[str]
            The file of every reachable module, and every data file whose
            package holds one.
        """
        reached = self.reachable(names)
        files = {self.modules[name] for name in reached}
        for path in self.files:
            owner = self.owner(path)
            if owner is not None and (
                owner in reached
                or any(name.startswith(f"{owner}.") for name in reached)
            ):
                files.add(path)
        return frozenset(files)


__all__ = ["ImportGraph", "module_name"]
