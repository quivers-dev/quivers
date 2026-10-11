"""quivers DSL: parse .qvr files into trainable PyTorch models.

The DSL provides a declarative syntax for specifying V-enriched
categorical morphism networks that compile to ``quivers.Program``
instances (``nn.Module`` subclasses). Parsing is delegated to panproto
via the ``qvr`` tree-sitter grammar; the AST is a tree of
[`quivers.dsl.ast_nodes`][quivers.dsl.ast_nodes] didactic Models.

Morphism roles need not be declared: a morphism drawn with ``sample``
compiles as latent, one drawn with ``observe`` as observed, and any
other morphism as a kernel. An explicit ``[role=...]`` option always
wins over inference.

Quick start
-----------
::

    from quivers.dsl import load, loads

    program = loads('''
        object Resp : FinSet 8
        morphism bias : Resp -> Real 1 ~ Normal(0.0, 5.0)
        program m : Resp -> Resp
            sample b <- bias
            observe y : Resp <- Normal(b, 1.0)
            return y
        export m
    ''')

    program = load("model.qvr")

    optimizer = torch.optim.Adam(program.parameters())
"""

from pathlib import Path

from quivers.dsl.ast_nodes import Module
from quivers.dsl.compiler import Compiler, CompileError
from quivers.dsl.constraints import Violation, check_constraints
from quivers.dsl.emit import EmitError, module_to_source, static_argument_to_source
from quivers.dsl.family_schemas import (
    DISTRIBUTION_FAMILIES,
    FAMILY_ALIASES,
    family_parameter_names,
    family_parameterizations,
)
from quivers.dsl.let_expr_traversal import (
    free_let_names,
    let_expr_children,
    substitute_let_expr,
    walk_let_expr,
)
from quivers.dsl.parser import ParseError, parse, parse_file
from quivers.dsl.program_theory import (
    QVR_DEDUCTION_PROTOCOL,
    QVR_PROGRAM_PROTOCOL,
    extract_deduction_schema,
    extract_program_schema,
)
from quivers.dsl.pygments_lexer import QvrLexer
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import (
    QIEC_STATEMENT_TYPES,
    QVR_SOURCE_VERSION,
    CheckedQvrQiec,
    QvrQiecLowerer,
    QvrQiecSource,
    has_qiec_surface,
    lower_qvr_to_qiec,
    non_qiec_projection,
    qiec_projection,
)
from quivers.program import Program


def loads(
    source: str,
    *,
    data: dict | None = None,
) -> Program:
    """Compile .qvr source text into a trainable Program.

    Parameters
    ----------
    source : str
        The ``.qvr`` source.
    data : dict, optional
        Maps string keys to tensors (or tensor-like objects) for
        any ``from_data("KEY")`` initializers in the source. The
        compiler looks each key up at compile time; an unknown key
        raises `CompileError`.

    Returns
    -------
    Program
        The compiled program.

    Raises
    ------
    ParseError
        If ``source`` does not parse.
    CompileError
        If the parsed module does not compile.
    """
    ast = parse(source)
    compiler = Compiler(ast, module_name="source", file_path="<source>")
    if data is not None:
        compiler.bind_data(data)
    return compiler.compile()


def load(
    path: str | Path,
    *,
    data: dict | None = None,
) -> Program:
    """Load and compile a .qvr file into a trainable Program.

    Parameters
    ----------
    path : str | Path
        The ``.qvr`` file. Its stem names the compiled module.
    data : dict, optional
        Maps string keys to tensors for any ``from_data("KEY")``
        initializers in the file, as in `loads`.

    Returns
    -------
    Program
        The compiled program.

    Raises
    ------
    OSError
        If the file cannot be read.
    ParseError
        If the file does not parse.
    CompileError
        If the parsed module does not compile.
    """
    ast = parse_file(path)
    compiler = Compiler(ast, module_name=Path(path).stem, file_path=str(path))
    if data is not None:
        compiler.bind_data(data)
    return compiler.compile()


__all__ = [
    "DISTRIBUTION_FAMILIES",
    "FAMILY_ALIASES",
    "QIEC_STATEMENT_TYPES",
    "QVR_DEDUCTION_PROTOCOL",
    "QVR_PROGRAM_PROTOCOL",
    "QVR_SOURCE_VERSION",
    "CheckedQvrQiec",
    "CompileError",
    "Compiler",
    "EmitError",
    "Module",
    "ParseError",
    "QiecDiagnosticError",
    "QvrLexer",
    "QvrQiecLowerer",
    "QvrQiecSource",
    "Violation",
    "check_constraints",
    "extract_deduction_schema",
    "extract_program_schema",
    "family_parameter_names",
    "family_parameterizations",
    "free_let_names",
    "has_qiec_surface",
    "let_expr_children",
    "load",
    "loads",
    "lower_qvr_to_qiec",
    "module_to_source",
    "non_qiec_projection",
    "parse",
    "parse_file",
    "qiec_projection",
    "static_argument_to_source",
    "substitute_let_expr",
    "walk_let_expr",
]
