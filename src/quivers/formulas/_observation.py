"""The observation of a response before it is written as a program step.

An `Observation` is the site a response is scored at: its kernel, the binding
passed to each kernel parameter, the plate of its rows, the event axis of a
vector response, and the recorded roles that modify how it is scored (row
weights, censoring and truncation columns). Every emitter of a response builds
one through `observation`, so the roles are read from the response modifiers
in one place; readers that compose responses (residual correlation across
responses) take the arguments from the value rather than from emitted source.
"""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
    LetExprNode,
    ObserveStep,
    OptionEntry,
    OptionName,
    TypeName,
    atom_to_draw_arg,
)
from quivers.dsl.program_templates import draw_argument_expression
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _draw
from quivers.formulas._response_roles import event_option_columns
from quivers.formulas.formula import ResponseModifiers

WEIGHTS = "formula_weights"
"""Binding of the row weights of a response."""


class Observation(ValueModel):
    """One scored response site before it is written.

    Parameters
    ----------
    site : str
        Observed site.
    kernel : str
        Native or compilation-local law.
    parameters : tuple[str, ...]
        Kernel parameters, in schema order.
    values : tuple[str or int or float, ...]
        The binding or literal passed to each parameter.
    plate : str
        Finite axis of the rows.
    event : str or None
        Event axis of a vector response (``over =``), or None.
    weights : str or None
        Binding of the row weights, or None.
    event_options : tuple[str, ...]
        Recorded-event options (censoring and truncation).
    event_columns : tuple[str, ...]
        The binding of each recorded-event option.
    catalog : FamilyCatalog or None
        Compilation-local laws the kernel is registered in.
    """

    site: str
    kernel: str
    parameters: tuple[str, ...]
    values: tuple[str | int | float, ...]
    plate: str = "Resp"
    event: str | None = None
    weights: str | None = None
    event_options: tuple[str, ...] = ()
    event_columns: tuple[str, ...] = ()
    catalog: FamilyCatalog | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def arguments(self) -> tuple[tuple[str, str | int | float], ...]:
        """Pair each kernel parameter with its binding.

        Returns
        -------
        tuple[tuple[str, str or int or float], ...]
            ``(parameter, binding)`` in schema order.
        """
        return tuple(zip(self.parameters, self.values, strict=True))

    @property
    def events(self) -> tuple[tuple[str, str], ...]:
        """Pair each recorded-event option with its binding.

        Returns
        -------
        tuple[tuple[str, str], ...]
            ``(option, binding)`` in option order.
        """
        return tuple(zip(self.event_options, self.event_columns, strict=True))

    def with_arguments(
        self, arguments: tuple[tuple[str, str | int | float], ...]
    ) -> Observation:
        """Return the observation passing other arguments.

        Parameters
        ----------
        arguments : tuple[tuple[str, str or int or float], ...]
            ``(parameter, binding)`` in schema order.

        Returns
        -------
        Observation
            The same site with the arguments.
        """
        return self.with_(
            parameters=tuple(name for name, _ in arguments),
            values=tuple(value for _, value in arguments),
        )

    def with_events(self, events: tuple[tuple[str, str], ...]) -> Observation:
        """Return the observation with other recorded-event bindings.

        Parameters
        ----------
        events : tuple[tuple[str, str], ...]
            ``(option, binding)`` in option order.

        Returns
        -------
        Observation
            The same site with the events.
        """
        return self.with_(
            event_options=tuple(key for key, _ in events),
            event_columns=tuple(column for _, column in events),
        )

    def argument(self, parameter: str) -> LetExprNode:
        """Read the expression passed to one kernel parameter.

        Parameters
        ----------
        parameter : str
            Kernel parameter.

        Returns
        -------
        LetExprNode
            The binding or literal as an expression.

        Raises
        ------
        KeyError
            If the kernel takes no such parameter.
        """
        bound = dict(self.arguments)
        if parameter not in bound:
            raise KeyError(f"{self.kernel} takes no parameter {parameter!r}")
        return draw_argument_expression(atom_to_draw_arg(bound[parameter]))

    def with_argument(self, parameter: str, value: str | int | float) -> Observation:
        """Return the observation passing another binding to a parameter.

        Parameters
        ----------
        parameter : str
            Kernel parameter.
        value : str or int or float
            New binding or literal.

        Returns
        -------
        Observation
            The same site with the replaced argument.
        """
        return self.with_arguments(
            tuple(
                (name, value if name == parameter else bound)
                for name, bound in self.arguments
            )
        )

    def step(self) -> ObserveStep:
        """Write the observation as a program step.

        Returns
        -------
        ObserveStep
            The step, with its weights, event axis, and recorded events as
            options in that order.
        """
        options = (
            *(
                ()
                if self.weights is None
                else (OptionEntry(key="weights", value=OptionName(value=self.weights)),)
            ),
            *(
                ()
                if self.event is None
                else (OptionEntry(key="over", value=OptionName(value=self.event)),)
            ),
            *(
                OptionEntry(key=key, value=OptionName(value=column))
                for key, column in self.events
            ),
        )
        return ObserveStep(
            vars=(self.site,),
            morphism=self.kernel,
            args=tuple(
                DrawArgNamed(parameter=name, value=atom_to_draw_arg(bound))
                for name, bound in self.arguments
            ),
            index=TypeName(name=self.plate),
            options=options,
        )


def observation(
    site: str,
    kernel: str,
    arguments: tuple[str | int | float, ...],
    modifiers: ResponseModifiers | None,
    *,
    plate: str = "Resp",
    event: str | None = None,
    events: bool = True,
    parameterization: tuple[str, ...] | None = None,
    catalog: FamilyCatalog | None = None,
) -> Observation:
    """Build the observation of a response from its modifiers.

    Parameters
    ----------
    site : str
        Observed site.
    kernel : str
        Native or compilation-local law.
    arguments : tuple[str or int or float, ...]
        Bindings in the order of the kernel's parameters.
    modifiers : ResponseModifiers or None
        Recorded roles of the response; None for none.
    plate : str
        Finite axis of the rows.
    event : str or None
        Event axis of a vector response.
    events : bool
        Whether recorded-event columns are options of this site; False when
        the kernel takes them otherwise.
    parameterization : tuple[str, ...] or None
        Parameter names of an explicit schema of the kernel.
    catalog : FamilyCatalog or None
        Compilation-local laws the kernel is registered in.

    Returns
    -------
    Observation
        The observation with the response's weights and recorded events.
    """
    written = _draw(
        site,
        kernel,
        arguments,
        index=TypeName(name=plate),
        mode="score",
        parameterization=parameterization,
        family_catalog=catalog,
    )
    parameters = tuple(
        argument.parameter
        for argument in written.args or ()
        if isinstance(argument, DrawArgNamed)
    )
    recorded = (
        ()
        if modifiers is None or not events
        else tuple(
            (key, column.qvr_name) for key, column in event_option_columns(modifiers)
        )
    )
    return Observation(
        site=site,
        kernel=kernel,
        parameters=parameters,
        values=arguments,
        plate=plate,
        event=event,
        weights=WEIGHTS
        if modifiers is not None and modifiers.weights is not None
        else None,
        event_options=tuple(key for key, _ in recorded),
        event_columns=tuple(column for _, column in recorded),
        catalog=catalog,
    )


__all__: list[str] = []
