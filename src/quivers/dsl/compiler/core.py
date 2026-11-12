"""Compiler: transform a quivers DSL AST into a trainable Program."""

from __future__ import annotations

from dataclasses import replace
import torch.nn as nn
from quivers.dsl.markov_chains import chain_cells
from quivers.continuous.deterministic import LearnedFunction
from quivers.core.algebras import PRODUCT_FUZZY, Algebra
from quivers.continuous.morphisms import AnySpace
from quivers.continuous.programs import MonadicProgram
from quivers.core.objects import SetObject
from quivers.program import Program
from quivers.dsl.ast_nodes import (
    BundleDecl,
    CategoryDecl,
    CompositionDecl,
    ContractionDecl,
    DecoderDecl,
    DeductionDecl,
    DefineDecl,
    EncoderDecl,
    ExportDecl,
    Expr,
    ExprIdent,
    LossDecl,
    Module,
    MorphismDecl,
    ProgramDecl,
    RuleDecl,
    SchemaDecl,
    SignatureDecl,
    Statement,
    ObjectDecl,
    ObjectExpr,
)
from quivers.dsl.latent_weights import LatentWeightsError, expand_latent_weights
from quivers.dsl.program_elaboration import LATENT_WEIGHT_HEADS
from quivers.dsl.compiler._prelude import (
    CompileError,
    _CompiledContraction,
    _build_default_trans_constructors,
    _build_default_trans_singletons,
)
from quivers.dsl.compiler.declarations import _DeclarationsMixin
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.compiler.structural import _StructuralMixin
from quivers.dsl.compiler.deductions import _DeductionsMixin
from quivers.dsl.compiler.resolution import _ResolutionMixin
from quivers.dsl.compiler.expressions import _ExpressionsMixin
from quivers.dsl.program_elaboration import GAP_CODE
from quivers.dsl.qiec_lowering import (
    QiecDiagnosticError,
    has_qiec_surface,
    lower_qvr_to_qiec,
    non_qiec_projection,
)
from quivers.qiec import QiecModule
from quivers.qiec.terms import PlateAxis
from quivers.qiec.types import IndexLiteral
from quivers.qiec.entries import EntryPoint, entry_point, entry_points
from quivers.qiec.execution import ExecutionDiagnostic, ExecutionFailure


def _plate_axes(
    batch: tuple[PlateAxis, ...],
) -> tuple[tuple[str, int | None], ...]:
    """A checked site's batch axes as ``(plate, extent)`` pairs.

    Parameters
    ----------
    batch : tuple[PlateAxis, ...]
        The site's batch axes, outermost first.

    Returns
    -------
    tuple[tuple[str, int | None], ...]
        One pair per axis: its extent when the program fixes it as a
        number, ``None`` when the program leaves it to its data or its
        static arguments.
    """
    return tuple(
        (
            axis.name,
            axis.size.value
            if isinstance(axis.size, IndexLiteral) and isinstance(axis.size.value, int)
            else None,
        )
        for axis in batch
    )


class Compiler(
    _DeclarationsMixin,
    _ProgramsMixin,
    _StructuralMixin,
    _DeductionsMixin,
    _ResolutionMixin,
    _ExpressionsMixin,
):
    """Compile a quivers DSL AST into a Program.

    The compiler maintains three environments:

    - objects: name -> SetObject (discrete finite sets)
    - spaces: name -> ContinuousSpace
    - morphisms: name -> Morphism or ContinuousMorphism (any morphism-like)

    It processes statements in order and compiles the output
    expression into a Program wrapping the morphism DAG.

    Parameters
    ----------
    module : Module
        The parsed AST.
    lower_qiec : bool
        Whether to lower the checked QIEC projection. Tooling sets this to
        ``False`` only for its classic-compiler recovery pass after QIEC has
        already produced a diagnostic.
    """

    def __init__(
        self,
        module: Module,
        *,
        module_name: str | None = None,
        file_path: str = "<source>",
        lower_qiec: bool = True,
    ) -> None:
        self._qiec_module: QiecModule | None = None
        # The QIEC projection is lowered on first use, after the
        # statements' own checks when `compile` drives it, so a source
        # error the statement compiler names is reported by it.
        self._qiec_source: Module | None = None
        self._qiec_module_name = module_name
        self._qiec_file_path = file_path
        try:
            module = expand_latent_weights(module, LATENT_WEIGHT_HEADS)
        except LatentWeightsError as error:
            raise CompileError(str(error), error.line, error.col) from error
        if lower_qiec and has_qiec_surface(module):
            self._qiec_source = module
            module = non_qiec_projection(module)
        self._module = module
        self._algebra: Algebra = PRODUCT_FUZZY
        self._categories: list[str] = []
        self._rules: dict = {}
        self._bundles: dict[str, tuple[str, ...]] = {}
        self._aliases: dict[str, ObjectExpr] = {}
        self._alias_names: set[str] = set()
        # Built-in transformation catalog: singletons looked up by
        # bare name (``expectation``, ``log_prob``, …) and
        # constructors invoked with arguments (``softmax(B)``,
        # ``bayes_invert(prior)``).  Disjoint from
        # `_transformations`, which holds user define-bound
        # transformations defined inside the module.
        self._trans_singletons: dict = _build_default_trans_singletons()
        self._trans_constructors: dict = _build_default_trans_constructors()
        # User-defined transformations bound via ``define t = …``.
        # Disjoint from `_morphisms`: a ``define`` whose RHS
        # resolves to a transformation lands here; a ``define`` whose
        # RHS resolves to a morphism lands in ``_morphisms``.
        self._transformations: dict = {}
        self._objects: dict[str, SetObject] = {}
        self._spaces: dict = {}
        self._morphisms: dict = {}
        self._groups: dict[str, list[str]] = {}
        self._output_expr: Expr | None = None
        # Parametric-program templates: dependent kernels Π(p:P).Kern(dom(p),cod(p))
        # stored as their unsubstituted AST decl. Instantiated at each call
        # site by parameter substitution + α-renaming of internal latents.
        self._program_templates: dict[str, ProgramDecl] = {}
        # The located refusal of the first call each program keeps
        # because it cannot be lowered to compiled runtime steps.
        self._call_refusals: dict[str, CompileError] = {}
        # Every ``program`` declaration of the module by name, parametric
        # or not: the scope model a program's sites are addressed through.
        self._program_decls: dict[str, ProgramDecl] = {}
        # The parameter spaces of the program being compiled, innermost
        # last: its host data and its lets, by name.
        self._parameter_space_stack: list[dict[str, AnySpace]] = []
        # The checked extents of the values the program being compiled
        # binds, innermost last, by name.
        self._value_extents_stack: list[dict[str, tuple[int, ...]]] = []
        # Operadic contraction declarations. Each entry is callable
        # from the DSL at define-binding sites; the value records the
        # compiled `EinsumWiring` plus the declared domain /
        # codomain typing for shape-checking at invocation time.
        self._contractions: dict[str, "_CompiledContraction"] = {}
        # The step programs the expansion of a scan over a program cell
        # declares, their compiled forms, and the learned initial state
        # of each scan that declares one, shared by name as the checked
        # elaboration shares its `weight` input.
        self._scan_step_declarations: dict[str, ProgramDecl] = {}
        self._scan_step_programs: dict[str, MonadicProgram] = {}
        self._scan_initial_states: dict[str, nn.Parameter] = {}
        # The network of each latent-weight kernel, which a draw through
        # the kernel applies at the run's drawn weights.
        self._latent_networks: dict[str, LearnedFunction] = {}

    @property
    def categories(self) -> list[str]:
        """The declared category atoms."""
        return list(self._categories)

    @property
    def rules(self) -> dict:
        """The compiled rule schemas from ``rule`` declarations."""
        return dict(self._rules)

    @property
    def objects(self) -> dict[str, SetObject]:
        """The compiled object environment."""
        return dict(self._objects)

    @property
    def spaces(self) -> dict:
        """The compiled space environment."""
        return dict(self._spaces)

    @property
    def morphisms(self) -> dict:
        """The compiled morphism environment."""
        return dict(self._morphisms)

    @property
    def algebra(self) -> Algebra:
        """The active algebra."""
        return self._algebra

    @property
    def programs(self) -> dict[str, ProgramDecl]:
        """Every declared program, by name.

        A program with typed parameters (a template) and one without
        both appear: each is a scope whose parameters and sites are
        addressed by ``program::site`` paths. A program without typed
        parameters also compiles to a morphism, listed in `morphisms`.
        """
        return dict(getattr(self, "_program_decls", {}))

    @property
    def deductions(self) -> dict:
        """Declared deduction systems (``deduction NAME : ...``)."""
        return dict(getattr(self, "_deductions", {}))

    @property
    def signatures(self) -> dict:
        """Declared signatures (``signature NAME``)."""
        return dict(getattr(self, "_signatures", {}))

    @property
    def encoders(self) -> dict:
        """Declared encoders (``encoder NAME : Sig``)."""
        return dict(getattr(self, "_encoders", {}))

    @property
    def decoders(self) -> dict:
        """Declared decoders (``decoder NAME : Sig``)."""
        return dict(getattr(self, "_decoders", {}))

    @property
    def losses(self) -> dict:
        """Declared loss heads (``loss NAME : ... [on=...]``).

        Keyed by entry name; values are the registered :class:`LossEntry`
        records. Empty when no ``loss`` decl appears in the module.
        """
        reg = getattr(self, "_loss_registry", None)
        if reg is None:
            return {}
        return {entry.name: entry for entry in reg.entries}

    @property
    def bundles(self) -> dict[str, tuple[str, ...]]:
        """Declared bundle aliases (``bundle NAME = R1 | R2 | ...``)."""
        return dict(self._bundles)

    @property
    def contractions(self) -> dict:
        """Declared tensor-network contractions
        (``contraction NAME : ... [wiring=...]``)."""
        return dict(getattr(self, "_contractions", {}))

    @property
    def transformations(self) -> dict:
        """User-declared transformation constructors / singletons."""
        return dict(self._transformations)

    @property
    def qiec_module(self) -> QiecModule | None:
        """The checked QIEC module, or ``None`` when the source has no QIEC surface.

        Returns
        -------
        QiecModule | None
            The module, lowered on first access.

        Raises
        ------
        QiecDiagnosticError
            If the source's QIEC surface fails to lower, other than by a
            program gap, which the module records instead.
        """
        if self._qiec_source is not None and self._qiec_module is None:
            self._qiec_module = self._lower_qiec(self._qiec_source)
        return self._qiec_module

    def entry_points(self) -> tuple[EntryPoint, ...]:
        """The executable entry points of the checked module.

        Returns
        -------
        tuple[EntryPoint, ...]
            Every ``program`` and ``define`` computation, in declaration
            order; empty when the source has no checked module.
        """
        module = self.qiec_module
        if module is None:
            return ()
        return entry_points(module)

    def entry(self, name: str) -> EntryPoint:
        """One entry point of the checked module, by name.

        Parameters
        ----------
        name : str
            The entry's source name.

        Returns
        -------
        EntryPoint
            The entry.

        Raises
        ------
        ExecutionFailure
            With code ``qiec-run-module`` if the source has no checked
            module, or ``qiec-run-computation`` for an unknown name.
        """
        module = self.qiec_module
        if module is None:
            raise ExecutionFailure(
                ExecutionDiagnostic(
                    "qiec-run-module",
                    "the source has no checked module and no entry points",
                    name,
                )
            )
        if module.gap and not any(point.name == name for point in entry_points(module)):
            raise ExecutionFailure(
                ExecutionDiagnostic(
                    "qiec-program-gap",
                    f"entry point {name!r} was not lowered: {module.gap}",
                    name,
                )
            )
        return entry_point(module, name)

    def _lower_qiec(self, module: Module) -> QiecModule:
        """Lower the source's QIEC surface to a checked module.

        Parameters
        ----------
        module : Module
            The parsed source, QIEC surface included.

        Returns
        -------
        QiecModule
            The checked module; when a program falls in a gap of the
            elaboration, the module of the other declarations with the
            gap recorded.

        Raises
        ------
        QiecDiagnosticError
            If the surface fails to lower other than by a program gap.
        """
        try:
            return lower_qvr_to_qiec(
                module,
                module_name=self._qiec_module_name,
                file_path=self._qiec_file_path,
            )
        except QiecDiagnosticError as error:
            if error.code != GAP_CODE:
                raise
            return replace(
                lower_qvr_to_qiec(
                    module,
                    module_name=self._qiec_module_name,
                    file_path=self._qiec_file_path,
                    elaborate_programs=False,
                ),
                gap=error.message,
            )

    def compile(self) -> Program:
        """Compile the module into a trainable Program.

        Returns
        -------
        Program
            The compiled nn.Module wrapping the morphism DAG.

        Raises
        ------
        CompileError
            On semantic errors (undefined names, type mismatches, etc.).
        """
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        qiec_module = self.qiec_module
        if self._output_expr is None:
            # A module may declare only structural artifacts
            # (signatures, encoders, decoders, losses) with no
            # exported morphism; the returned Program is a container
            # carrying those artifacts.
            program = Program(None)
        elif isinstance(
            self._output_expr, ExprIdent
        ) and self._output_expr.name in getattr(self, "_program_templates", {}):
            # The export names a parametric program template. A
            # template has no root morphism on its own (a function
            # ``Pi (p : P). Kern(dom(p), cod(p))`` rather than a
            # single Kleisli arrow); the returned Program is a
            # container holding the template, which the caller
            # instantiates by attribute access.
            tmpl_name = self._output_expr.name
            program = Program(None)
            invoker = self._make_template_invoker(tmpl_name)
            program.templates = {tmpl_name: invoker}
            # Convenience: ``program.<name>(...)`` works directly.
            object.__setattr__(program, tmpl_name, invoker)
        else:
            root_morphism = self._compile_expr(self._output_expr)
            program = Program(root_morphism)
        # Exports after the first become named accessors on the compiled
        # Program: ``export a`` then ``export b`` makes ``a`` the primary
        # morphism (``program.morphism`` / ``program(...)``) and
        # ``program.b`` the compiled form of the second.
        self._attach_secondary_exports(program)
        # Attach the compiler's deduction and posterior registries to
        # the Program so downstream callers can reach them after
        # `quivers.dsl.load(...)`.
        program.deductions = getattr(self, "_deductions", {})
        program.posteriors = getattr(self, "_posteriors", {})
        program.signatures = getattr(self, "_signatures", {})
        program.encoders = getattr(self, "_encoders", {})
        program.decoders = getattr(self, "_decoders", {})
        program.losses = getattr(self, "_loss_registry", None)
        # Structural components are ordinary PyTorch modules. Register them
        # under stable source-derived names as well as exposing the ergonomic
        # dictionaries above, so ``Program.parameters()``, ``state_dict()``,
        # optimizers, and checkpointing see every learned tensor.
        for role, components in (
            ("encoder", program.encoders),
            ("decoder", program.decoders),
        ):
            for name, component in components.items():
                if isinstance(component, nn.Module):
                    program.add_module(f"_qvr_{role}_{name}", component)
        # Preserve the checked projection on the compiled container.  A
        # QIEC-only source thus becomes an inspection/evaluation container;
        # its computation graph is never misrepresented as a PyTorch morphism.
        object.__setattr__(program, "qiec", qiec_module)
        if qiec_module is not None:
            self._attach_host_data_types(qiec_module)
        # Wire the loss registry into every compiled deduction so the
        # agenda's rule-firing and chart-completion paths can fire
        # rule-attached and chart-attached losses automatically.
        if program.losses is not None:
            for name, system in program.deductions.items():
                system._loss_registry = program.losses  # type: ignore[attr-defined]
                system._deduction_name = name  # type: ignore[attr-defined]
        return program

    def _attach_host_data_types(self, module: QiecModule) -> None:
        """Give each compiled program the checked types of its host data.

        A program's checked entry types every free name its steps read
        (its ``data`` parameters) and every group map (its ``fibration``
        parameters); the compiled program refuses a run whose supplied
        data do not inhabit those types, so it computes only the model
        the checked module describes. An input that fills a parameter of
        a plated draw directly may also be supplied without its plate
        axis, as one value every row shares.

        Parameters
        ----------
        module : QiecModule
            The checked module.
        """
        compiled = {
            **getattr(self, "_posteriors", {}),
            **self._morphisms,
        }
        for entry in module.entries:
            target = compiled.get(entry.name)
            if not isinstance(target, MonadicProgram):
                continue
            target.host_data_types = {
                parameter.name: parameter.type
                for parameter in entry.parameters
                if parameter.role in ("data", "fibration")
            }
            target.host_data_shared = frozenset(
                parameter.name for parameter in entry.parameters if parameter.per_row
            )
            target.site_plates = {
                site.name: _plate_axes(site.batch) for site in entry.sites
            }

    def _attach_secondary_exports(self, program: "Program") -> None:
        """Expose every export after the first as a named attribute.

        The first export is the module's primary output (already wrapped
        as ``program``). Each later ``export <name>`` is compiled and
        attached under its name so callers reach it via
        ``program.<name>``: a program module exporting both a plain
        composite and its probabilistic-program surface makes the
        composite the callable primary while keeping the program body
        reachable by name for tracing.

        Attachment goes through ``object.__setattr__`` so a
        morphism-valued export is stored as a plain attribute rather than
        an ``nn`` submodule, which would otherwise fold its parameters
        into the primary's optimiser view. Names colliding with an
        existing Program attribute (``morphism``, ``domain``, ...) are
        skipped so a secondary export never shadows the public surface.
        """
        exports = getattr(self, "_exports", [])
        templates = getattr(self, "_program_templates", {})
        for expr in exports[1:]:
            if not isinstance(expr, ExprIdent):
                continue
            name = expr.name
            if hasattr(program, name):
                continue
            if name in templates:
                value: object = self._make_template_invoker(name)
            else:
                value = self._compile_expr(expr)
            object.__setattr__(program, name, value)

    def compile_env(self) -> dict:
        """Compile all statements and return the full environment.

        Useful for inspection without requiring an output declaration.

        Returns
        -------
        dict
            Combined environment of every declared atom: the active
            algebra, objects, spaces, morphisms, rules, programs
            (parametric templates), deductions, signatures, encoders,
            decoders, losses, bundles, contractions, transformations.
        """
        for stmt in self._module.statements:
            self._compile_statement(stmt)
        env: dict = {}
        env["__algebra__"] = self._algebra
        qiec_module = self.qiec_module
        if qiec_module is not None:
            env["__qiec__"] = qiec_module
        for name, obj in self._objects.items():
            env[name] = obj
        for name, space in self._spaces.items():
            env[name] = space
        for name, morph in self._morphisms.items():
            env[name] = morph
        for name, rule in self._rules.items():
            env[name] = rule
        for name, tmpl in getattr(self, "_program_templates", {}).items():
            env[name] = tmpl
        for name, system in getattr(self, "_deductions", {}).items():
            env[name] = system
        for name, sig in getattr(self, "_signatures", {}).items():
            env[name] = sig
        for name, enc in getattr(self, "_encoders", {}).items():
            env[name] = enc
        for name, dec in getattr(self, "_decoders", {}).items():
            env[name] = dec
        reg = getattr(self, "_loss_registry", None)
        if reg is not None:
            for entry in reg.entries:
                env[entry.name] = entry
        for name, bundle in self._bundles.items():
            env[name] = bundle
        for name, contr in getattr(self, "_contractions", {}).items():
            env[name] = contr
        return env

    def _compile_statement(self, stmt: Statement) -> None:
        """Dispatch to the appropriate statement compiler."""
        if isinstance(stmt, CompositionDecl):
            self._compile_composition(stmt)
        elif isinstance(stmt, CategoryDecl):
            self._compile_category(stmt)
        elif isinstance(stmt, RuleDecl):
            self._compile_rule(stmt)
        elif isinstance(stmt, SchemaDecl):
            self._compile_schema(stmt)
        elif isinstance(stmt, ObjectDecl):
            self._compile_type(stmt)
        elif isinstance(stmt, BundleDecl):
            self._compile_bundle(stmt)
        elif isinstance(stmt, MorphismDecl):
            self._compile_morphism(stmt)
        elif isinstance(stmt, ProgramDecl):
            self._program_decls[stmt.name] = stmt
            if stmt.name in chain_cells(self._module):
                # The cell of a marginalized chain has no value of its
                # own; the scan that runs it compiles it.
                return
            self._compile_program(stmt)
        elif isinstance(stmt, ContractionDecl):
            self._compile_contraction(stmt)
        elif isinstance(stmt, DefineDecl):
            self._compile_define(stmt)
        elif isinstance(stmt, ExportDecl):
            self._compile_export(stmt)
        elif isinstance(stmt, DeductionDecl):
            self._compile_deduction(stmt)
        elif isinstance(stmt, SignatureDecl):
            self._compile_signature(stmt)
        elif isinstance(stmt, EncoderDecl):
            self._compile_encoder(stmt)
        elif isinstance(stmt, DecoderDecl):
            self._compile_decoder(stmt)
        elif isinstance(stmt, LossDecl):
            self._compile_loss(stmt)
        else:
            raise CompileError(f"unknown statement type: {type(stmt).__name__}")


__all__ = ["Compiler"]
