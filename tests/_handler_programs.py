"""Sources with authored handlers, shared by the discharge and fitting tests.

`HANDLERS` declares one interface and handler per way a clause may use its
continuation, and `random_body` writes a computation that requests from all of
them, so a test can compare two readings of one program over many programs.
"""

from __future__ import annotations

import random

__all__: list[str] = []

HANDLERS = """
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

effect Read
    ask : Unit -> Real

effect Tell
    tell : Real -> Unit

effect Stop
    stop : Real -> Real

effect Maybe
    check : Real -> Real

effect Flip
    flip : Real -> Real

effect After
    after : Real -> Real

instance st : State[Real]
instance rd : Read
instance wr : Tell
instance ab : Stop
instance mb : Maybe
instance ch : Flip
instance af : After
instance ct : Flip

handler cell for State[Real] : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x + 0.5 * s
    get resumes 1 =>
        resume(s)
    put(v : Real) resumes 1 =>
        resume() with (v)

handler reader for Read : Real -> Real with (r : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    ask resumes 1 =>
        resume(r)

handler writer for Tell : Real -> Real with (total : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x + total
    tell(v : Real) resumes 1 =>
        resume() with (total + v)

handler stopped for Stop : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    stop(v : Real) resumes 0 =>
        return 0.25 * v

handler checked for Maybe : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x - 0.125
    check(v : Real) resumes aff =>
        if v > 0.0 then
            resume(v)
        else
            return 1.0 - v

handler mixed for Flip : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    flip(v : Real) resumes omega =>
        let a <- resume(v)
        let b <- resume(0.0 - v)
        return 0.25 * a + 0.75 * b

handler doubled for After : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return 0.5 * x
    after(v : Real) resumes 1 =>
        let rest <- resume(v)
        return rest + 0.5 * v

handler counting for Flip : Real -> Real with (n : Real, m : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x + 0.0625 * n
    flip(v : Real) resumes omega =>
        let a <- resume(n) with (n + 1.0, m)
        let b <- resume(m * v) with (0.5 * n, m + v)
        return a - 0.5 * b
"""

#: Each handler installation, and whether its operation may copy the rest of
#: the computation.
_INSTALLS = {
    "st": "cell(a)",
    "rd": "reader(b)",
    "wr": "writer(0.0)",
    "ab": "stopped",
    "mb": "checked",
    "ch": "mixed",
    "af": "doubled",
    "ct": "counting(1.0, a)",
}
_COPYING = ("ch", "ct")


def _row(lines: list[str]) -> str:
    """The row of a body, from the instances its requests name."""
    return ", ".join(
        name for name in _INSTALLS if any(f"perform {name}." in line for line in lines)
    )


def _expression(rng: random.Random, names: list[str]) -> str:
    """An arithmetic expression over some names."""
    left, right = rng.choice(names), rng.choice(names)
    constant = rng.choice(("0.5", "0.25", "1.5", "0.75"))
    return rng.choice(
        (
            f"{left} + {constant} * {right}",
            f"{constant} * {left} - {right}",
            f"{constant} * {left} * {right}",
            f"{left} + {constant}",
        )
    )


def _statements(
    rng: random.Random, names: list[str], count: int, copies: list[int], indent: str
) -> list[str]:
    """Statements that bind names, extending ``names`` with what they bind."""
    lines: list[str] = []
    for _ in range(count):
        kind = rng.choice(("let", "get", "put", "ask", "tell", "mb", "ch", "af", "ct"))
        name = f"v{len(names)}"
        value = _expression(rng, names)
        if kind in _COPYING:
            if copies[0] == 0:
                kind = "let"
            else:
                copies[0] -= 1
        if kind == "put":
            lines.append(f"{indent}perform st.put({value})")
            continue
        if kind == "tell":
            lines.append(f"{indent}perform wr.tell({value})")
            continue
        lines.append(
            {
                "let": f"{indent}let {name} = {value}",
                "get": f"{indent}let {name} <- perform st.get()",
                "ask": f"{indent}let {name} <- perform rd.ask()",
                "mb": f"{indent}let {name} <- perform mb.check({value})",
                "ch": f"{indent}let {name} <- perform ch.flip({value})",
                "af": f"{indent}let {name} <- perform af.after({value})",
                "ct": f"{indent}let {name} <- perform ct.flip({value})",
            }[kind]
        )
        names.append(name)
    return lines


def random_body(seed: int) -> str:
    """A computation ``handled(a, b)`` under every handler, in a random order.

    Parameters
    ----------
    seed : int
        The generator's seed.

    Returns
    -------
    str
        Definitions to follow `HANDLERS`: a conditional helper, the requesting
        body, and ``handled``, whose row is empty.
    """
    rng = random.Random(seed)
    copies = [2]
    helper = [f"    if a > {rng.choice(('0.0', '0.5', '-0.5'))} then"]
    for branch in range(2):
        names = ["a", "b"]
        helper += _statements(rng, names, rng.randint(0, 3), copies, " " * 8)
        if rng.random() < 0.3:
            helper.append(
                f"        let w <- perform ab.stop({_expression(rng, names)})"
            )
            names.append("w")
        helper.append(f"        return {_expression(rng, names)}")
        if branch == 0:
            helper.append("    else")
    names = ["a", "b"]
    body: list[str] = []
    for _ in range(rng.randint(1, 3)):
        body += _statements(rng, names, rng.randint(1, 3), copies, " " * 4)
        name = f"v{len(names)}"
        body.append(
            f"    let {name} <- helper({_expression(rng, names)}, {rng.choice(names)})"
        )
        names.append(name)
    body.append(f"    return {_expression(rng, names)}")
    helper.insert(0, f"define helper(a : Real, b : Real) : Real !{{{_row(helper)}}} =")
    body.insert(
        0, f"define body(a : Real, b : Real) : Real !{{{_row(helper + body)}}} ="
    )
    order = [name for name in _INSTALLS if name in _row(helper + body).split(", ")]
    rng.shuffle(order)
    handled = ["define handled(a : Real, b : Real) : Real !{} ="]
    for depth, instance in enumerate(order):
        pad = " " * (4 * (depth + 1))
        handled.append(f"{pad}handle {instance} with {_INSTALLS[instance]} in")
    pad = " " * (4 * (len(order) + 1))
    handled += [f"{pad}let r <- body(a, b)", f"{pad}return r"]
    return "\n\n".join("\n".join(lines) for lines in (helper, body, handled)) + "\n"
