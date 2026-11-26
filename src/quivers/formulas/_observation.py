"""The observation of a response before it is written as a program step.

An `Observation` is the site a response is scored at: its kernel, the binding
passed to each kernel parameter, the plate of its rows, the event axis of a
vector response, the component laws of a mixture of laws, and the recorded
roles that modify how it is scored (row weights, censoring and truncation
columns), which the modifiers of the response attach. Readers that compose
responses (mixtures, residual correlation across responses, missing
responses) take the arguments from the value rather than from emitted source.
"""

from __future__ import annotations

from collections.abc import Mapping

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgDist,
    DrawArgList,
    DrawArgName,
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

WEIGHTS = "formula_weights"
"""Binding of the row weights of a response."""

_TRUNCATION = frozenset({"trunc_lower", "trunc_upper"})
"""Recorded-event options that truncate the law rather than censor a row."""


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
    components : tuple[Observation, ...]
        Component laws of a mixture of laws, passed as the list argument
        ``components``; empty for any other law.
    factorized : bool
        Whether the rows are independent given the parameters, so each has a
        likelihood factor of its own.
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
    components: tuple[Observation, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    factorized: bool = True

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

    def law(self) -> DrawArgDist:
        """Write the law of the site as a distribution argument.

        A truncated component is written normalized on its own interval,
        ``Normalize(Restrict(law, low, high))``, which is how brms mixes
        truncated components.

        Returns
        -------
        DrawArgDist
            The kernel over its arguments.
        """
        law = DrawArgDist(family=self.kernel, args=self._arguments())
        bounds = dict(self.events)
        restriction = tuple(
            DrawArgNamed(parameter=parameter, value=DrawArgName(text=bounds[option]))
            for option, parameter in (("trunc_lower", "low"), ("trunc_upper", "high"))
            if option in bounds
        )
        if not restriction:
            return law
        return DrawArgDist(
            family="Normalize",
            args=(
                DrawArgNamed(
                    parameter="base",
                    value=DrawArgDist(
                        family="Restrict",
                        args=(DrawArgNamed(parameter="base", value=law), *restriction),
                    ),
                ),
            ),
        )

    def renamed(
        self, names: Mapping[str, str], kernels: Mapping[str, str] | None = None
    ) -> Observation:
        """Return the observation in another program's names.

        Parameters
        ----------
        names : Mapping[str, str]
            New name of each binding; others are kept.
        kernels : Mapping[str, str] or None
            New name of each compilation-local law; others are kept.

        Returns
        -------
        Observation
            The observation with renamed site, bindings, and laws.
        """

        def named(value: str | int | float) -> str | int | float:
            return names.get(value, value) if isinstance(value, str) else value

        return self.with_(
            site=names.get(self.site, self.site),
            kernel=(kernels or {}).get(self.kernel, self.kernel),
            values=tuple(named(value) for value in self.values),
            weights=None
            if self.weights is None
            else names.get(self.weights, self.weights),
            event_columns=tuple(
                names.get(column, column) for column in self.event_columns
            ),
            components=tuple(
                component.renamed(names, kernels) for component in self.components
            ),
        )

    def with_recorded_events(self, events: tuple[tuple[str, str], ...]) -> Observation:
        """Return the observation recording responses as bounds or truncated.

        A mixture of laws truncates each component on its own interval and
        keeps the censoring on the site; any other law takes every option on
        the site.

        Parameters
        ----------
        events : tuple[tuple[str, str], ...]
            ``(option, binding)`` of each recorded-event column.

        Returns
        -------
        Observation
            The observation with the events.
        """
        if not self.components:
            return self.with_events(events)
        truncation = tuple(
            (key, column) for key, column in events if key in _TRUNCATION
        )
        return self.with_events(
            tuple((key, column) for key, column in events if key not in _TRUNCATION)
        ).with_(
            components=tuple(
                component.with_events(truncation) for component in self.components
            )
        )

    def _arguments(self) -> tuple[DrawArgNamed, ...]:
        """Write the arguments of the kernel.

        Returns
        -------
        tuple[DrawArgNamed, ...]
            One named argument per parameter, then the component laws of a
            mixture of laws.
        """
        return (
            *(
                DrawArgNamed(parameter=name, value=atom_to_draw_arg(bound))
                for name, bound in self.arguments
            ),
            *(
                ()
                if not self.components
                else (
                    DrawArgNamed(
                        parameter="components",
                        value=DrawArgList(
                            items=tuple(
                                component.law() for component in self.components
                            )
                        ),
                    ),
                )
            ),
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
            args=self._arguments(),
            index=TypeName(name=self.plate),
            options=options,
        )


def observation(
    site: str,
    kernel: str,
    arguments: tuple[str | int | float, ...],
    *,
    plate: str = "Resp",
    event: str | None = None,
    parameterization: tuple[str, ...] | None = None,
    catalog: FamilyCatalog | None = None,
) -> Observation:
    """Build the observation of a native law over its arguments.

    The response terms that change how the site is scored (weights,
    recorded events) are attached by the modifiers of the response.

    Parameters
    ----------
    site : str
        Observed site.
    kernel : str
        Native or compilation-local law.
    arguments : tuple[str or int or float, ...]
        Bindings in the order of the kernel's parameters.
    plate : str
        Finite axis of the rows.
    event : str or None
        Event axis of a vector response.
    parameterization : tuple[str, ...] or None
        Parameter names of an explicit schema of the kernel.
    catalog : FamilyCatalog or None
        Compilation-local laws the kernel is registered in.

    Returns
    -------
    Observation
        The observation, with its parameters in schema order.
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
    return Observation(
        site=site,
        kernel=kernel,
        parameters=tuple(
            argument.parameter
            for argument in written.args or ()
            if isinstance(argument, DrawArgNamed)
        ),
        values=arguments,
        plate=plate,
        event=event,
        catalog=catalog,
    )


__all__: list[str] = []
