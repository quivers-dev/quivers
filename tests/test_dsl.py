"""End-to-end tests for the quivers DSL (lexer, parser, compiler, loader)."""

from __future__ import annotations
import textwrap

import torch
import pytest

from quivers.continuous.programs import MonadicProgram
from quivers.core.algebras import BOOLEAN
from quivers.core.objects import FinSet
from quivers.dsl import CompileError, ParseError, load, loads, parse
from quivers.dsl.ast_nodes import (
    DefineDecl,
    ExportDecl,
    ExprCompose,
    ExprIdent,
    ExprMarginalize,
    ExprTensorProduct,
    Module,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
)
from quivers.dsl.compiler import Compiler
from quivers.program import Program


class TestParser:
    def _parse(self, source: str) -> Module:
        return parse(source)

    def test_composition_decl(self):
        """Parse a composition declaration at the algebra level."""
        mod = self._parse("composition product_fuzzy [level=algebra]\n")
        assert len(mod.statements) == 1

    def test_object_decl_finset(self):
        """Parse an object declaration with a FinSet initializer."""
        mod = self._parse("object X : FinSet 3\n")
        assert len(mod.statements) == 1
        stmt = mod.statements[0]
        assert isinstance(stmt, ObjectDecl)
        assert stmt.names == ("X",)

    def test_latent_morphism_role(self):
        """Parse a morphism declaration with role=parameter."""
        mod = self._parse("object X : FinSet 3\nmorphism f : X -> X [role=parameter]\n")
        stmt = mod.statements[1]
        assert isinstance(stmt, MorphismDecl)
        assert stmt.names == ("f",)

    def test_let_compose(self):
        """Parse a let binding with composition."""
        source = (
            "object X : FinSet 3\n"
            "morphism f : X -> X [role=parameter]\n"
            "morphism g : X -> X [role=parameter]\n"
            "define h = f >> g\n"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[3]
        assert isinstance(let_stmt, DefineDecl)
        assert isinstance(let_stmt.expr, ExprCompose)

    def test_let_tensor_product(self):
        """Parse a let binding with tensor product."""
        source = (
            "object X : FinSet 3\n"
            "morphism f : X -> X [role=parameter]\n"
            "morphism g : X -> X [role=parameter]\n"
            "define h = f @ g\n"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[3]
        assert isinstance(let_stmt.expr, ExprTensorProduct)

    def test_let_marginalize(self):
        """Parse a let binding with marginalization."""
        source = (
            "object X : FinSet 3\n"
            "morphism f : X -> X [role=parameter]\n"
            "define m = f.marginalize(X)\n"
        )
        mod = self._parse(source)
        let_stmt = mod.statements[2]
        assert isinstance(let_stmt.expr, ExprMarginalize)
        assert let_stmt.expr.names == ("X",)

    def test_export_decl(self):
        """Parse an export declaration."""
        source = "object X : FinSet 3\nmorphism f : X -> X [role=parameter]\nexport f\n"
        mod = self._parse(source)
        out = mod.statements[2]
        assert isinstance(out, ExportDecl)
        assert isinstance(out.expr, ExprIdent)

    def test_parse_error_invalid_object(self):
        """ParseError on malformed object initializer."""
        with pytest.raises(ParseError):
            self._parse("object X : >>\n")

    def test_parse_returns_module(self):
        """The parse function returns a Module AST."""
        mod = parse(
            "object X : FinSet 3\nmorphism f : X -> X [role=parameter]\nexport f\n"
        )
        assert isinstance(mod, Module)
        assert len(mod.statements) == 3


class TestCompiler:
    def test_simple_latent(self):
        """Compile a single latent morphism."""
        prog = loads(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "morphism f : X -> Y [role=parameter]\n"
            "export f\n"
        )
        assert isinstance(prog, Program)
        assert prog().shape == torch.Size([3, 4])

    def test_composition(self):
        """Compile sequential composition."""
        prog = loads(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "object Z : FinSet 2\n"
            "morphism f : X -> Y [role=parameter]\n"
            "morphism g : Y -> Z [role=parameter]\n"
            "export f >> g\n"
        )
        assert prog().shape == torch.Size([3, 2])

    def test_tensor_product(self):
        """Compile tensor product."""
        prog = loads(
            "object X : FinSet 2\n"
            "object Y : FinSet 3\n"
            "morphism f : X -> X [role=parameter]\n"
            "morphism g : Y -> Y [role=parameter]\n"
            "export f @ g\n"
        )
        assert prog().shape == torch.Size([2, 3, 2, 3])

    def test_let_binding(self):
        """Let bindings can be referenced later."""
        prog = loads(
            "object X : FinSet 3\n"
            "morphism f : X -> X [role=parameter]\n"
            "define g = f >> f\n"
            "export g\n"
        )
        assert prog().shape == torch.Size([3, 3])

    def test_composition_algebra_boolean(self):
        """Compile with the boolean algebra."""
        prog = loads(
            "composition boolean [level=algebra]\n"
            "object X : FinSet 2\n"
            "morphism h : X -> X [role=observed] ~ identity(X)\n"
            "export h\n"
        )
        out = prog()
        torch.testing.assert_close(out, torch.eye(2))

    def test_trainable(self):
        """Compiled program has trainable parameters."""
        prog = loads(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "morphism f : X -> Y [role=parameter]\n"
            "export f\n"
        )
        params = list(prog.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_gradient_flow(self):
        """Gradients flow through composed morphisms."""
        prog = loads(
            "object X : FinSet 2\n"
            "object Y : FinSet 3\n"
            "object Z : FinSet 2\n"
            "morphism f : X -> Y [role=parameter]\n"
            "morphism g : Y -> Z [role=parameter]\n"
            "export f >> g\n"
        )
        out = prog()
        loss = out.sum()
        loss.backward()
        for p in prog.parameters():
            assert p.grad is not None

    def test_init_scale_option(self):
        """Morphism scale option is respected."""
        prog = loads(
            "object X : FinSet 3\n"
            "morphism f : X -> X [role=parameter, scale=0.1]\n"
            "export f\n"
        )
        assert prog().shape == torch.Size([3, 3])

    def test_undefined_object_error(self):
        """CompileError for undefined object reference."""
        with pytest.raises(CompileError, match="undefined object"):
            loads("morphism f : X -> Y [role=parameter]\nexport f\n")

    def test_undefined_morphism_error(self):
        """CompileError for undefined morphism reference."""
        with pytest.raises(CompileError, match="undefined morphism"):
            loads("object X : FinSet 3\nexport f\n")


class TestLoader:
    def test_load_file(self, tmp_path):
        """Load a .qvr file from disk."""
        f = tmp_path / "test_model.qvr"
        f.write_text(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "morphism f : X -> Y [role=parameter]\n"
            "export f\n"
        )
        prog = load(f)
        assert isinstance(prog, Program)
        assert prog().shape == torch.Size([3, 4])

    def test_load_string_path(self, tmp_path):
        """Load accepts string paths."""
        f = tmp_path / "model.qvr"
        f.write_text(
            "object X : FinSet 2\nmorphism f : X -> X [role=parameter]\nexport f\n"
        )
        prog = load(str(f))
        assert isinstance(prog, Program)

    def test_file_not_found(self):
        """FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/model.qvr")


class TestCompileEnv:
    def test_compile_env_objects(self):
        """compile_env returns objects in the environment."""
        ast = parse(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "morphism f : X -> Y [role=parameter]\n"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "X" in env
        assert isinstance(env["X"], FinSet)
        assert env["X"].cardinality == 3

    def test_compile_env_morphisms(self):
        """compile_env returns morphisms in the environment."""
        ast = parse(
            "object X : FinSet 3\n"
            "object Y : FinSet 4\n"
            "morphism f : X -> Y [role=parameter]\n"
        )
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert "f" in env

    def test_compile_env_algebra(self):
        """compile_env includes the active algebra."""
        ast = parse("composition boolean [level=algebra]\nobject X : FinSet 2\n")
        compiler = Compiler(ast)
        env = compiler.compile_env()
        assert env["__algebra__"] is BOOLEAN


class TestContinuousSurface:
    """Compile a kernel morphism over a continuous codomain."""

    def test_kernel_normal(self):
        """A kernel morphism with the Normal family compiles."""
        src = (
            "object X : FinSet 5\n"
            "object R3 : Real 3\n"
            "morphism f : X -> R3 [role=kernel] ~ Normal\n"
            "export f\n"
        )
        prog = loads(textwrap.dedent(src))
        assert isinstance(prog, Program)
        x = torch.arange(5)
        y = prog.rsample(x)
        assert y.shape == (5, 3)

    def test_kernel_log_normal(self):
        """A kernel morphism with the LogNormal family compiles."""
        src = (
            "object X : FinSet 3\n"
            "object R : Real 2\n"
            "morphism g : X -> R [role=kernel] ~ LogNormal\n"
            "export g\n"
        )
        prog = loads(textwrap.dedent(src))
        x = torch.tensor([0, 1, 2])
        y = prog.rsample(x)
        assert y.shape == (3, 2)

    def test_unknown_family_error(self):
        """CompileError when the named family is unknown."""
        with pytest.raises(CompileError):
            loads(
                "object X : FinSet 3\n"
                "object R : Real 2\n"
                "morphism f : X -> R [role=kernel] ~ Nonexistent\n"
                "export f\n"
            )


class TestProgramSurface:
    """Compile and run a monadic program decl."""

    def test_simple_program_compiles(self):
        """A program with one sample and a return compiles."""
        src = (
            "object X : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : X -> R [role=kernel] ~ Normal\n"
            "program p : X -> R\n"
            "    sample y <- f\n"
            "    return y\n"
        )
        env = Compiler(parse(textwrap.dedent(src))).compile_env()
        assert "p" in env
        assert isinstance(env["p"], MonadicProgram)

    def test_program_chained_samples(self):
        """A program with chained sample steps runs end-to-end."""
        src = (
            "object X : FinSet 3\n"
            "object R : Real 2\n"
            "object S : Real 4\n"
            "morphism f : X -> R [role=kernel] ~ Normal\n"
            "morphism g : R -> S [role=kernel] ~ Normal\n"
            "program chain : X -> S\n"
            "    sample y <- f\n"
            "    sample z <- g(y)\n"
            "    return z\n"
        )
        env = Compiler(parse(textwrap.dedent(src))).compile_env()
        prog = env["chain"]
        out = prog.rsample(torch.tensor([0, 1, 2]))
        assert out.shape == (3, 4)

    def test_program_as_export(self):
        """A program can be the exported expression."""
        src = (
            "object X : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : X -> R [role=kernel] ~ Normal\n"
            "program model : X -> R\n"
            "    sample y <- f\n"
            "    return y\n"
            "export model\n"
        )
        prog = Compiler(parse(textwrap.dedent(src))).compile()
        assert isinstance(prog, Program)
        out = prog.rsample(torch.tensor([0, 1, 2]))
        assert out.shape == (3, 2)

    def test_program_decl_ast(self):
        """ProgramDecl is the AST node for a program declaration."""
        src = (
            "object X : FinSet 3\n"
            "object R : Real 2\n"
            "morphism f : X -> R [role=kernel] ~ Normal\n"
            "program p : X -> R\n"
            "    sample y <- f\n"
            "    return y\n"
        )
        mod = parse(textwrap.dedent(src))
        prog_stmt = mod.statements[-1]
        assert isinstance(prog_stmt, ProgramDecl)
        assert prog_stmt.name == "p"
        assert prog_stmt.return_vars == ("y",)


class TestComments:
    def test_comments_and_whitespace(self):
        """Plain `#` comments and extra whitespace are handled."""
        src = (
            "# a model with comments\n"
            "\n"
            "object X : FinSet 3   # input\n"
            "object Y : FinSet 4   # output\n"
            "\n"
            "morphism f : X -> Y [role=parameter]\n"
            "\n"
            "export f\n"
        )
        prog = loads(textwrap.dedent(src))
        assert prog().shape == torch.Size([3, 4])
