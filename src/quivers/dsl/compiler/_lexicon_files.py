"""Lexicon files of ``deduction`` declarations."""

from __future__ import annotations

from pathlib import Path

from quivers.dsl.ast_nodes import (
    DeductionDecl,
    LetExprNode,
    LetExprVar,
    LetStep,
    ProgramDecl,
    TypeName,
)
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.parser import parse as _parse_qvr


def load_lexicon_tsv(
    path: str, decl: DeductionDecl
) -> list[tuple[str, TypeName, LetExprNode]]:
    """Read a lexicon file into words, categories, and logical forms.

    Each row has three tab-separated columns: the word, the category,
    an atom name, and the logical form, a let expression; blank lines
    and lines starting with ``#`` are skipped. The logical form is read
    by the live grammar, as the let step of a synthetic program, so the
    file's syntax is the module's.

    Parameters
    ----------
    path : str
        The file, absolute or relative to the working directory.
    decl : DeductionDecl
        The declaring deduction, for diagnostics.

    Returns
    -------
    list[tuple[str, TypeName, LetExprNode]]
        One row per entry: the word, the category as a type name, and
        the logical form's expression.

    Raises
    ------
    CompileError
        If the file is missing, a row has fewer than three columns, or
        a logical form does not parse.
    """
    file = Path(path)
    if not file.exists():
        raise CompileError(
            f"deduction {decl.name!r}: lexicon file {path!r} not found",
            decl.line,
            decl.col,
        )
    rows: list[tuple[str, TypeName, LetExprNode]] = []
    with file.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: expected 3 tab-separated "
                    f"columns (word, category, lf), got {len(parts)}",
                    decl.line,
                    decl.col,
                )
            word, cat_text, lf_text = parts[0], parts[1], parts[2]
            category = TypeName(name=cat_text)
            if "(" not in lf_text:
                rows.append((word, category, LetExprVar(name=lf_text)))
                continue
            syn_src = (
                "object LexTerm : FinSet 1\n"
                "morphism lex_source : LexTerm -> LexTerm "
                "[role=parameter]\n"
                "program lex_program : LexTerm -> LexTerm\n"
                "    sample lex_value : LexTerm <- lex_source\n"
                f"    let lex_lf = {lf_text}\n"
                "    return lex_value\n"
            )
            syn_mod = _parse_qvr(syn_src.encode(), "<lex-lf>")
            prog = next(
                (
                    s
                    for s in syn_mod.statements
                    if isinstance(s, ProgramDecl) and s.name == "lex_program"
                ),
                None,
            )
            if prog is None:
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: LF template {lf_text!r} "
                    f"did not parse to the synthetic program",
                    decl.line,
                    decl.col,
                )
            let_step = prog.draws[1]
            if not isinstance(let_step, LetStep):
                raise CompileError(
                    f"deduction {decl.name!r}: lexicon file "
                    f"{path!r}:{lineno}: LF template {lf_text!r} "
                    f"did not parse to a let binding",
                    decl.line,
                    decl.col,
                )
            rows.append((word, category, let_step.value))
    return rows


__all__ = ["load_lexicon_tsv"]
