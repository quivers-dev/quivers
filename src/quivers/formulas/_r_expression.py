"""R and Stan arithmetic expressions read into QVR value expressions.

brms writes the right-hand side of a nonlinear formula and each side of a
``hypothesis()`` string as an R expression whose functions are Stan's. This
module reads that syntax (``^`` and ``**`` powers, R operator precedence,
numeric literals as doubles, Stan's mathematical functions such as
``inv_logit`` and ``log1m``) and writes the equivalent QVR value expression,
which the canonical QVR parser and type checker then read. One reader serves
nonlinear formulas and hypotheses.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
import math
import re

from quivers._values import ValueModel

_SQRT2 = repr(math.sqrt(2.0))


class RFunction(ValueModel):
    """One R or Stan function and its QVR spelling.

    Parameters
    ----------
    arities : tuple[int, ...]
        Accepted argument counts.
    template : str
        QVR expression with ``{0}``, ``{1}``, ... standing for the translated
        arguments, each already parenthesized where it needs to be.
    """

    arities: tuple[int, ...]
    template: str


def _same(name: str, arity: int = 1) -> RFunction:
    """Build the entry of a function QVR spells as R and Stan do.

    Parameters
    ----------
    name : str
        Function name shared by both languages.
    arity : int
        Argument count.

    Returns
    -------
    RFunction
        The pass-through entry.
    """
    arguments = ", ".join("{" + str(index) + "}" for index in range(arity))
    return RFunction(arities=(arity,), template=f"{name}({arguments})")


#: Stan's mathematical functions, the R functions brms defines beside them
#: (``logm1``, ``expp1``, ``inv_logit_scaled``, ``logit_scaled``), and the QVR
#: builtins of scalar real arithmetic, each with its QVR expression.
R_FUNCTIONS: Mapping[str, RFunction] = {
    **{
        name: _same(name)
        for name in (
            "exp",
            "expm1",
            "log",
            "log1p",
            "sqrt",
            "abs",
            "square",
            "sign",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "floor",
            "ceil",
            "round",
            "trunc",
            "erf",
            "erfc",
            "erfinv",
            "lgamma",
            "digamma",
            "sigmoid",
            "softplus",
            "logsigmoid",
            "reciprocal",
            "rsqrt",
            "log1mexp",
            "log_ndtr",
            "real",
            "int",
        )
    },
    "pow": _same("pow", 2),
    "min": _same("min", 2),
    "max": _same("max", 2),
    "select": _same("select", 3),
    "log2": RFunction(arities=(0, 1), template="log2({0})"),
    "log10": RFunction(arities=(0, 1), template="log10({0})"),
    "fabs": RFunction(arities=(1,), template="abs({0})"),
    "fmin": RFunction(arities=(2,), template="min({0}, {1})"),
    "fmax": RFunction(arities=(2,), template="max({0}, {1})"),
    "fdim": RFunction(arities=(2,), template="max({0} - {1}, 0.0)"),
    "fmod": RFunction(arities=(2,), template="({0} - trunc({0} / {1}) * {1})"),
    "fma": RFunction(arities=(3,), template="({0} * {1} + {2})"),
    "hypot": RFunction(arities=(2,), template="sqrt(square({0}) + square({1}))"),
    "inv": RFunction(arities=(1,), template="reciprocal({0})"),
    "inv_sqrt": RFunction(arities=(1,), template="rsqrt({0})"),
    "inv_square": RFunction(arities=(1,), template="reciprocal(square({0}))"),
    "cbrt": RFunction(arities=(1,), template="(sign({0}) * pow(abs({0}), 1.0 / 3.0))"),
    "exp2": RFunction(arities=(1,), template="pow(2.0, {0})"),
    "inv_logit": RFunction(arities=(1,), template="sigmoid({0})"),
    "logit": RFunction(arities=(1,), template="(log({0}) - log1p(-{0}))"),
    "log_inv_logit": RFunction(arities=(1,), template="logsigmoid({0})"),
    "log1m_inv_logit": RFunction(arities=(1,), template="logsigmoid(-{0})"),
    "cloglog": RFunction(arities=(1,), template="log(-log1p(-{0}))"),
    "inv_cloglog": RFunction(arities=(1,), template="(-expm1(-exp({0})))"),
    "Phi": RFunction(arities=(1,), template=f"(0.5 * erfc(-{{0}} / {_SQRT2}))"),
    "inv_Phi": RFunction(
        arities=(1,), template=f"({_SQRT2} * erfinv(2.0 * {{0}} - 1.0))"
    ),
    "Phi_approx": RFunction(
        arities=(1,), template="sigmoid(0.07056 * pow({0}, 3.0) + 1.5976 * {0})"
    ),
    "log1m": RFunction(arities=(1,), template="log1p(-{0})"),
    "log1p_exp": RFunction(arities=(1,), template="softplus({0})"),
    "log1m_exp": RFunction(arities=(1,), template="log1mexp({0})"),
    "log_sum_exp": RFunction(
        arities=(2,),
        template="(max({0}, {1}) + log1p(exp(-abs({0} - {1}))))",
    ),
    "log_diff_exp": RFunction(arities=(2,), template="({0} + log1mexp({1} - {0}))"),
    "lbeta": RFunction(
        arities=(2,), template="(lgamma({0}) + lgamma({1}) - lgamma({0} + {1}))"
    ),
    "multiply_log": RFunction(arities=(2,), template="({0} * log({1}))"),
    "lmultiply": RFunction(arities=(2,), template="({0} * log({1}))"),
    "step": RFunction(arities=(1,), template="select({0} < 0.0, 0.0, 1.0)"),
    "logm1": RFunction(arities=(1,), template="log({0} - 1.0)"),
    "expp1": RFunction(arities=(1,), template="(exp({0}) + 1.0)"),
    "inv_logit_scaled": RFunction(
        arities=(1, 3), template="({1} + ({2} - {1}) * sigmoid({0}))"
    ),
    "logit_scaled": RFunction(
        arities=(1, 3),
        template="(log(({0} - {1}) / ({2} - {1})) - log1p(-(({0} - {1}) / ({2} - {1}))))",
    ),
    "pi": RFunction(arities=(0,), template=repr(math.pi)),
    "e": RFunction(arities=(0,), template=repr(math.e)),
    "sqrt2": RFunction(arities=(0,), template=_SQRT2),
}

#: Value of a function called without arguments, and default trailing
#: arguments of the functions that have them.
_ZERO_ARGUMENT: Mapping[str, str] = {
    "log2": repr(math.log(2.0)),
    "log10": repr(math.log(10.0)),
}
_DEFAULT_ARGUMENTS: Mapping[str, tuple[str, ...]] = {
    "inv_logit_scaled": ("0.0", "1.0"),
    "logit_scaled": ("0.0", "1.0"),
}

#: Functions whose arguments keep their integer carrier.
_INTEGER_ARGUMENT: frozenset[str] = frozenset({"real", "int"})

#: R's named constants, used when no variable has the name.
_CONSTANTS: Mapping[str, str] = {"pi": repr(math.pi)}

#: R binary operators by precedence level, loosest first, with the QVR
#: operator each is written as.
_BINARY_LEVELS: tuple[Mapping[str, str], ...] = (
    {"||": "||", "|": "||"},
    {"&&": "&&", "&": "&&"},
    {"==": "==", "!=": "!=", "<=": "<=", ">=": ">=", "<": "<", ">": ">"},
    {"+": "+", "-": "-"},
    {"*": "*", "/": "/"},
)

_NUMBER = re.compile(r"(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)L?")
_OPERATOR = re.compile(r"\*\*|\|\||&&|==|!=|<=|>=|[-+*/^()\[\],<>|&]")

#: Names of a nonlinear formula: R syntactic names.
R_NAME = re.compile(r"[A-Za-z.][A-Za-z0-9._]*")

#: Names of a brms hypothesis: letters, digits, ``_``, ``.`` and ``:``, with
#: an optional bracketed index, as brms's ``find_vars`` reads them.
HYPOTHESIS_NAME = re.compile(r"[A-Za-z.][A-Za-z0-9_.:]*(?:\[[^\],]+(?:,[^\],]+)*\])?")


class RTranslation(ValueModel):
    """A QVR value expression read from an R expression.

    Parameters
    ----------
    expression : str
        QVR value expression.
    names : tuple[str, ...]
        Variable names of the R expression in order of first use, as
        authored.
    """

    expression: str
    names: tuple[str, ...]


class _Reader:
    """A recursive-descent reader of one R expression.

    Parameters
    ----------
    text : str
        R expression.
    context : str
        Description of the argument holding the expression, named in errors.
    name_pattern : re.Pattern[str]
        Pattern of a variable name.
    rename : Callable[[str], str]
        QVR spelling of an authored variable name.
    integer_names : Collection[str]
        Authored names of integer or Boolean carriers, read as doubles.
    variables : Collection[str] or None
        Names known to be variables, which take precedence over R's
        constants. None reads every name as a variable.
    """

    def __init__(
        self,
        text: str,
        *,
        context: str,
        name_pattern: re.Pattern[str],
        rename: Callable[[str], str],
        integer_names: Collection[str],
        variables: Collection[str] | None,
    ) -> None:
        """Start reading at the first character.

        Parameters
        ----------
        text : str
            R expression.
        context : str
            Description of the argument holding the expression.
        name_pattern : re.Pattern[str]
            Pattern of a variable name.
        rename : Callable[[str], str]
            QVR spelling of an authored variable name.
        integer_names : Collection[str]
            Authored names read as doubles through ``real()``.
        variables : Collection[str] or None
            Names known to be variables.
        """
        self.text = text
        self.context = context
        self.name_pattern = name_pattern
        self.rename = rename
        self.integer_names = integer_names
        self.variables = variables
        self.position = 0
        self.names: dict[str, None] = {}

    def fail(self, message: str, position: int | None = None) -> ValueError:
        """Build the located error of a rejected expression.

        Parameters
        ----------
        message : str
            What is wrong.
        position : int or None
            Zero-based character offset, or None for the current one.

        Returns
        -------
        ValueError
            Error naming the argument, the column, and the expression.
        """
        column = (self.position if position is None else position) + 1
        return ValueError(
            f"{self.context}: {message} at column {column} of {self.text!r}"
        )

    def skip(self) -> None:
        """Advance past white space."""
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def operator(self, choices: Collection[str]) -> str | None:
        """Consume the next operator when it is one of ``choices``.

        Parameters
        ----------
        choices : Collection[str]
            Operators accepted here.

        Returns
        -------
        str or None
            The consumed operator, or None when the next token is not one.
        """
        self.skip()
        match = _OPERATOR.match(self.text, self.position)
        if match is None or match.group() not in choices:
            return None
        self.position = match.end()
        return match.group()

    def expect(self, token: str) -> None:
        """Consume one required token.

        Parameters
        ----------
        token : str
            The token.

        Raises
        ------
        ValueError
            If the next token differs.
        """
        if self.operator((token,)) is None:
            raise self.fail(f"expected {token!r}")

    def binary(self, level: int, integral: bool) -> str:
        """Read a left-associative chain of one precedence level.

        Parameters
        ----------
        level : int
            Index into the binary precedence levels.
        integral : bool
            Keep integer literals as integers (inside an index).

        Returns
        -------
        str
            QVR expression.
        """
        if level == len(_BINARY_LEVELS):
            return self.unary(integral)
        operators = _BINARY_LEVELS[level]
        left = self.binary(level + 1, integral)
        while True:
            found = self.operator(operators)
            if found is None:
                return left
            right = self.binary(level + 1, integral)
            left = f"({left} {operators[found]} {right})"

    def unary(self, integral: bool) -> str:
        """Read a signed power; R binds ``^`` tighter than a unary sign.

        Parameters
        ----------
        integral : bool
            Keep integer literals as integers.

        Returns
        -------
        str
            QVR expression.
        """
        sign = self.operator(("-", "+"))
        if sign is None:
            return self.power(integral)
        operand = self.unary(integral)
        return f"(-{operand})" if sign == "-" else operand

    def power(self, integral: bool) -> str:
        """Read a right-associative power, ``a ^ b`` or ``a ** b``.

        Parameters
        ----------
        integral : bool
            Keep integer literals as integers.

        Returns
        -------
        str
            QVR expression.
        """
        base = self.primary(integral)
        if self.operator(("^", "**")) is None:
            return base
        # The exponent admits its own sign: ``a ^ -b``.
        return f"pow({base}, {self.unary(False)})"

    def arguments(self, integral: bool) -> tuple[str, ...]:
        """Read a parenthesized, comma-separated argument list.

        Parameters
        ----------
        integral : bool
            Keep integer literals and integer carriers as they are.

        Returns
        -------
        tuple[str, ...]
            Translated arguments.
        """
        if self.operator((")",)) is not None:
            return ()
        values = [self.binary(0, integral)]
        while self.operator((",",)) is not None:
            values.append(self.binary(0, integral))
        self.expect(")")
        return tuple(values)

    def call(self, name: str, start: int) -> str:
        """Read the arguments of one function call and write its QVR form.

        Parameters
        ----------
        name : str
            Function name.
        start : int
            Offset of the name, for errors.

        Returns
        -------
        str
            QVR expression.

        Raises
        ------
        ValueError
            If a listed function has the wrong argument count.
        """
        function = R_FUNCTIONS.get(name)
        values = self.arguments(name in _INTEGER_ARGUMENT)
        if function is None:
            # A name outside the table is left to the QVR checker, which
            # knows every builtin and reports an unknown one.
            return f"{name}({', '.join(values)})"
        if len(values) not in function.arities:
            raise self.fail(
                f"{name}() takes {' or '.join(map(str, function.arities))} "
                f"arguments, got {len(values)}",
                start,
            )
        # A name outside the integer carriers is already a double.
        if (
            name == "real"
            and R_NAME.fullmatch(values[0])
            and values[0] not in self.integer_names
        ):
            return values[0]
        if not values and name in _ZERO_ARGUMENT:
            return _ZERO_ARGUMENT[name]
        defaults = _DEFAULT_ARGUMENTS.get(name, ())
        if defaults and len(values) == 1:
            values = (*values, *defaults)
        return function.template.format(*values)

    def primary(self, integral: bool) -> str:
        """Read a literal, a variable, a call, or a parenthesized expression.

        Parameters
        ----------
        integral : bool
            Keep integer literals and integer carriers as they are.

        Returns
        -------
        str
            QVR expression.

        Raises
        ------
        ValueError
            If no operand starts here.
        """
        self.skip()
        if self.operator(("(",)) is not None:
            inner = self.binary(0, integral)
            self.expect(")")
            return f"({inner})"
        start = self.position
        number = _NUMBER.match(self.text, start)
        if number is not None:
            self.position = number.end()
            literal = number.group().rstrip("L")
            if integral and literal.isdigit():
                return literal
            return repr(float(literal))
        quoted = self.text.startswith("`", start)
        if quoted:
            end = self.text.find("`", start + 1)
            if end < 0:
                raise self.fail("unterminated backquoted name")
            name, self.position = self.text[start + 1 : end], end + 1
        else:
            match = self.name_pattern.match(self.text, start)
            if match is None:
                raise self.fail(
                    "expected a number, a name, or '('"
                    if start < len(self.text)
                    else "the expression ends where an operand is required"
                )
            name, self.position = match.group(), match.end()
        if not quoted and self.operator(("(",)) is not None:
            return self.call(name, start)
        if (
            name in _CONSTANTS
            and not quoted
            and self.variables is not None
            and name not in self.variables
        ):
            return _CONSTANTS[name]
        self.names.setdefault(name)
        value = self.rename(name)
        if self.operator(("[",)) is not None:
            indices = [self.binary(0, True)]
            while self.operator((",",)) is not None:
                indices.append(self.binary(0, True))
            self.expect("]")
            return f"{value}[{', '.join(indices)}]"
        if name in self.integer_names and not integral:
            return f"real({value})"
        return value

    def read(self) -> RTranslation:
        """Read the whole expression.

        Returns
        -------
        RTranslation
            QVR expression and the variable names it uses.

        Raises
        ------
        ValueError
            If the text is empty or has trailing tokens.
        """
        if not self.text.strip():
            raise self.fail("an expression is required")
        expression = self.binary(0, False)
        self.skip()
        if self.position != len(self.text):
            raise self.fail(f"unexpected {self.text[self.position]!r}")
        return RTranslation(expression=expression, names=tuple(self.names))


def translate_r_expression(
    text: str,
    *,
    context: str,
    name_pattern: re.Pattern[str] = R_NAME,
    rename: Callable[[str], str] | None = None,
    integer_names: Collection[str] = (),
    variables: Collection[str] | None = None,
) -> RTranslation:
    """Read an R expression over Stan's functions into a QVR value expression.

    Parameters
    ----------
    text : str
        R expression: numbers, names, ``+ - * /``, the powers ``^`` and
        ``**``, comparisons, ``&`` and ``|``, parentheses, indexing, and
        calls of the functions in `R_FUNCTIONS`.
    context : str
        Description of the argument holding the expression, named in errors.
    name_pattern : re.Pattern[str]
        Pattern of a variable name. `R_NAME` by default, `HYPOTHESIS_NAME`
        for brms hypothesis strings.
    rename : Callable[[str], str] or None
        QVR spelling of each authored variable name. None keeps the name.
    integer_names : Collection[str]
        Authored names with an integer or Boolean carrier. R reads numeric
        data as doubles, so each use is written ``real(name)``, except as an
        argument of ``real()`` or ``int()`` and inside an index.
    variables : Collection[str] or None
        Names known to be variables. With a collection, the R constant
        ``pi`` is read as its value unless a variable has that name. None
        reads every name as a variable.

    Returns
    -------
    RTranslation
        The QVR expression and the variable names in order of first use.

    Raises
    ------
    ValueError
        If the expression is empty, malformed, or gives a listed function
        the wrong number of arguments. A function outside `R_FUNCTIONS` is
        written as it is, for the QVR checker to accept or refuse. The message names
        ``context`` and the column.

    Notes
    -----
    Numeric literals become doubles, as in R. R's precedence holds: a power
    binds tighter than a unary sign and associates to the right, so
    ``-x^2`` is ``-(x^2)`` and ``a^b^c`` is ``a^(b^c)``. Cost is linear in
    the length of the text.
    """
    reader = _Reader(
        text,
        context=context,
        name_pattern=name_pattern,
        rename=(lambda name: name) if rename is None else rename,
        integer_names=integer_names,
        variables=variables,
    )
    return reader.read()


__all__: list[str] = []
