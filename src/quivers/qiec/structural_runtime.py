"""Run host-backed structural computations on the reference machine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_shape
from quivers.qiec.effects import EffectRef, HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    RuntimeAttachments,
    RuntimeHandler,
)
from quivers.qiec.execution import (
    ExecutionResult,
    RuntimeConfiguration,
    RuntimeValidator,
    run_named,
)
from quivers.qiec.identifiers import TypeId
from quivers.qiec.module import NamedEffectInstance, QiecModule
from quivers.qiec.types import REAL, TypeApplication, TypeExpr
from quivers.structural import Term


def _tensor_value(type_: TypeExpr, value: object) -> bool:
    """Return whether a torch tensor has the checked tensor shape.

    Parameters
    ----------
    type_
        The closed QIEC type whose tensor shape supplies the expected rank
        and any literal dimensions.
    value
        The host value to validate.

    Returns
    -------
    bool
        Whether ``value`` is a tensor matching every statically known
        dimension of ``type_``.
    """

    shape = tensor_shape(type_)
    if shape is None or not isinstance(value, torch.Tensor):
        return False
    dimensions = shape[1]
    if value.ndim != len(dimensions):
        return False
    for actual, expected in zip(value.shape, dimensions, strict=True):
        literal = getattr(expected, "value", None)
        if isinstance(literal, int) and actual != literal:
            return False
    return True


class StructuralRuntimeProvider(QiecModel):
    """PyTorch components and validators for one structural invocation.

    Parameters
    ----------
    runtimes
        The host handlers implementing the module's structural computations.
    term_types
        Type identities whose host values are structural :class:`Term`
        instances.
    name
        The provider name recorded in execution results and traces.
    """

    runtimes: tuple[RuntimeHandler, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    term_types: frozenset[TypeId] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str = "structural"

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach each host computation handler.

        Parameters
        ----------
        module
            The checked module, unused because the handlers were selected
            while constructing this provider.
        attachments
            The invocation's attachment table.
        """

        del module
        for runtime in self.runtimes:
            attachments.bind_handler(runtime)

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Return the validator for structural terms and torch values.

        Parameters
        ----------
        type_
            The closed QIEC type a host argument or result must inhabit.

        Returns
        -------
        RuntimeValidator | None
            A predicate for scalar weights, structural terms, tensors, and
            structural products, or ``None`` when the provider does not cover
            ``type_``.
        """

        if type_ == REAL or type_ == LOG_WEIGHT:
            return lambda value: (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                or isinstance(value, torch.Tensor)
                and value.shape == ()
            )
        if _is_term_type(type_, self.term_types):
            return lambda value: isinstance(value, Term)
        if tensor_shape(type_) is not None:
            return lambda value: _tensor_value(type_, value)
        if isinstance(type_, TypeApplication) and type_.constructor.name.startswith(
            "Product"
        ):
            components = tuple(
                item for item in type_.arguments if isinstance(item, TypeApplication)
            )
            validators = tuple(self.validator_for(item) for item in components)
            if len(validators) != len(type_.arguments) or any(
                validator is None for validator in validators
            ):
                return None

            def product(value: object) -> bool:
                """Check a structural product componentwise.

                Parameters
                ----------
                value
                    The host value to validate against the product type.

                Returns
                -------
                bool
                    Whether ``value`` is a tuple whose components pass the
                    corresponding validators.
                """

                return (
                    isinstance(value, tuple)
                    and len(value) == len(validators)
                    and all(
                        validator is not None and validator(item)
                        for validator, item in zip(validators, value, strict=True)
                    )
                )

            return product
        return None


def _is_term_type(type_: TypeExpr, identities: frozenset[TypeId]) -> bool:
    """Return whether a type is one of the module's structural families.

    Parameters
    ----------
    type_
        The closed QIEC type to inspect.
    identities
        Type-constructor identities belonging to structural term families.

    Returns
    -------
    bool
        Whether ``type_`` applies one of those family constructors.
    """

    return isinstance(type_, TypeApplication) and type_.constructor.id in identities


def _validator_for(
    type_: TypeExpr,
    term_types: frozenset[TypeId],
) -> RuntimeValidator | None:
    """Return the structural validator for one closed runtime type.

    Parameters
    ----------
    type_
        The closed QIEC type to validate at the host boundary.
    term_types
        Type identities whose host values are structural terms.

    Returns
    -------
    RuntimeValidator | None
        The matching predicate, or ``None`` when the structural provider does
        not cover ``type_``.
    """

    provider = StructuralRuntimeProvider(runtimes=(), term_types=term_types)
    return provider.validator_for(type_)


def structural_handler_definition(
    effect: EffectRef,
    result: TypeExpr,
    *,
    instance_name: str,
    computation_name: str,
) -> HandlerDef:
    """Build the stable checked handler signature for one neural attachment.

    Parameters
    ----------
    effect
        The applied ``Compute`` interface served by the attachment.
    result
        The closed result type returned by the host component.
    instance_name
        The source-derived effect-instance name.
    computation_name
        The source-derived structural computation name.

    Returns
    -------
    HandlerDef
        A stable foreign handler declaration for the component.
    """

    runtime = compute_handler(
        lambda argument, _context: argument,
        effect=effect,
        result_validator=lambda _value: True,
        answer_type=result,
        key=f"structural-{instance_name}-{computation_name}",
    )
    return runtime.definition.with_(
        name=f"__structural_{instance_name}_{computation_name}",
    )


def _loss_value(
    program: quivers.program.Program, name: str, argument: object
) -> torch.Tensor:
    """Evaluate one named loss entry under its term argument.

    Parameters
    ----------
    program
        The compiled program containing the structural loss registry.
    name
        The loss entry to evaluate.
    argument
        The structural term supplied to the loss body and optional weight.

    Returns
    -------
    torch.Tensor
        The weighted loss value, retaining its autograd graph.
    """

    registry = program.losses
    entry = next(item for item in registry.entries if item.name == name)
    environment = {"term": argument}
    value = entry.body(environment)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(float(value))
    if entry.weight is None:
        return value
    weight = entry.weight(environment)
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(float(weight))
    return value * weight


def _component_function(
    program: quivers.program.Program,
    instance: NamedEffectInstance,
) -> Callable[[object, ClauseContext], object]:
    """Resolve one structural instance to its compiled PyTorch operation.

    Parameters
    ----------
    program
        The compiled container supplying encoder, decoder, and loss objects.
    instance
        The QIEC effect instance whose source provenance names the component.

    Returns
    -------
    Callable[[object, ClauseContext], object]
        The host operation installed as the instance's ``Compute.run``
        clause.

    Raises
    ------
    KeyError
        If the structural registry has no component named by ``instance``.
    ValueError
        If the instance lacks structural provenance or names an unknown role.
    """

    path = instance.origin.structural_path
    if len(path) < 4 or path[0] != "structural":
        raise ValueError(f"instance {instance.name!r} is not structural")
    role = str(path[1])
    name = str(path[2])
    if role == "encoder":
        encoder = program.encoders[name]
        return lambda argument, _context: encoder(argument)
    if role == "decoder":
        decoder = program.decoders[name]
        return lambda argument, _context: decoder(argument)
    if role == "decoder-nll":
        decoder_name = name.removesuffix("__nll")
        decoder = program.decoders[decoder_name]

        def nll(argument: object, _context: ClauseContext) -> torch.Tensor:
            """Score one observed term and code as negative log likelihood.

            Parameters
            ----------
            argument
                A ``(term, code)`` pair supplied to the decoder likelihood.
            _context
                The enclosing clause context, unused by this pure host call.

            Returns
            -------
            torch.Tensor
                The decoder's negative log probability, retaining its
                autograd graph.

            Raises
            ------
            TypeError
                If ``argument`` is not a pair.
            """

            if not isinstance(argument, tuple) or len(argument) != 2:
                raise TypeError("decoder negative log likelihood takes (term, code)")
            term, code = argument
            return -decoder.log_prob(term, code)

        return nll
    if role == "loss":
        return lambda argument, _context: _loss_value(program, name, argument)
    raise ValueError(f"unknown structural component role {role!r}")


def structural_runtime_provider(
    module: QiecModule,
    program: quivers.program.Program,
) -> StructuralRuntimeProvider:
    """Build the typed neural-attachment provider for a compiled module.

    Parameters
    ----------
    module
        The checked QIEC module whose structural instances need handlers.
    program
        The compiled container supplying the corresponding PyTorch objects.

    Returns
    -------
    StructuralRuntimeProvider
        A provider containing one validated runtime handler per structural
        instance.

    Raises
    ------
    TypeError
        If a structural component returns a type with no host validator.
    """

    term_types = frozenset(
        family.type_constructor.id
        for family in module.families
        if family.name.endswith("__Term")
    )
    runtimes: list[RuntimeHandler] = []
    for instance in module.instances:
        if instance.origin.structural_path[:1] != ("structural",):
            continue
        result_type = instance.entry.effect.arguments[1]
        assert isinstance(result_type, TypeApplication)
        component_name = str(instance.origin.structural_path[2])
        validator = _validator_for(result_type, term_types)
        if validator is None:
            raise TypeError(
                f"structural component {component_name!r} has no validator for "
                f"{result_type!r}"
            )
        runtime = compute_handler(
            _component_function(program, instance),
            effect=instance.entry.effect,
            result_validator=validator,
            answer_type=result_type,
            key=f"structural-{instance.name}-{component_name}",
        )
        runtime.definition = structural_handler_definition(
            instance.entry.effect,
            result_type,
            instance_name=instance.name,
            computation_name=component_name,
        )
        runtimes.append(runtime)
    return StructuralRuntimeProvider(runtimes=tuple(runtimes), term_types=term_types)


def structural_runtime_configuration(
    module: QiecModule,
    program: quivers.program.Program,
) -> RuntimeConfiguration:
    """Return the standard runtime configuration for structural entries.

    Parameters
    ----------
    module
        The checked QIEC module whose structural entries may be invoked.
    program
        The compiled container supplying their PyTorch attachments.

    Returns
    -------
    RuntimeConfiguration
        A configuration selecting the structural provider when the module has
        structural runtimes, or no providers otherwise.
    """

    provider = structural_runtime_provider(module, program)
    return RuntimeConfiguration(
        providers=(provider,) if provider.runtimes else (),
    )


def run_structural(
    module: QiecModule,
    program: quivers.program.Program,
    name: str,
    arguments: Sequence[object],
    *,
    fuel: int | None = None,
) -> ExecutionResult:
    """Run a structural computation through the compiled PyTorch attachments.

    Parameters
    ----------
    module
        The checked QIEC module.
    program
        The classic compiled container supplying encoders, decoders, losses,
        and their learned tensors.
    name
        The structural computation name.
    arguments
        Its host arguments.
    fuel
        Optional reference-machine step budget.

    Returns
    -------
    ExecutionResult
        The typed result and trace. Torch tensor results retain their autograd
        graph back to the attached component parameters.

    Raises
    ------
    KeyError
        If ``name`` is not a structural computation.
    """

    computation = next(
        (
            item
            for item in module.computations
            if item.name == name and item.origin.structural_path[:1] == ("structural",)
        ),
        None,
    )
    if computation is None:
        raise KeyError(
            f"module {module.module!r} has no structural computation {name!r}"
        )
    return run_named(
        module,
        computation.name,
        tuple(arguments),
        runtime=structural_runtime_configuration(module, program),
        fuel=fuel,
    )


__all__ = [
    "StructuralRuntimeProvider",
    "run_structural",
    "structural_handler_definition",
    "structural_runtime_configuration",
    "structural_runtime_provider",
]

# `quivers.program` imports this module to configure the entry points it
# runs, and the annotations above name its `Program`. The import binds the
# package path rather than the class and sits at the foot of the file, so
# whichever module loads first, the other finds every name it needs, and
# the annotations resolve once both have finished loading.
import quivers.program  # noqa: E402
