"""Render plate graphs as tables, Mermaid, Graphviz, TikZ, or daft output.

Each renderer consumes only `PlateGraph` and does not inspect the source AST.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.analysis.plate_graph import PlateGraph, PlateNode

if TYPE_CHECKING:
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
else:
    try:
        from rich.table import Table
        from rich.text import Text
    except ModuleNotFoundError as error:
        if error.name != "rich":
            raise
        _RICH_AVAILABLE = False
    else:
        _RICH_AVAILABLE = True


# ---------------------------------------------------------------------------
# Rich table view (in-TUI default)
# ---------------------------------------------------------------------------


_KIND_BADGE: dict[str, str] = {
    "latent": "○",
    "observed": "●",
    "marginalized": "⊘",
    "deterministic": "·",
}

_KIND_STYLE: dict[str, str] = {
    "latent": "",
    "observed": "bold",
    "marginalized": "italic",
    "deterministic": "dim",
}


def render_table(graph: PlateGraph) -> Table:
    """Return a `rich.table.Table` describing ``graph``.

    The caller is responsible for printing it (`Console.print` for
    the plain prompt; `RichLog.write` for the Textual TUI).

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to tabulate.

    Returns
    -------
    rich.table.Table
        One row per node, with its kind, family, plates, and parents.

    Raises
    ------
    ModuleNotFoundError
        If the optional Rich dependency is absent.
    """
    if not _RICH_AVAILABLE:
        raise ModuleNotFoundError(
            "Rich table rendering requires quivers[repl]", name="rich"
        )
    title = Text()
    title.append("plate graph of ", style="dim")
    title.append(graph.program_name, style="bold")
    if graph.domain or graph.codomain:
        title.append(f"  ({graph.domain} -> {graph.codomain})", style="dim")

    table = Table(title=title, show_header=True, expand=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("kind", justify="left")
    table.add_column("variable", justify="left")
    table.add_column("family", justify="left")
    table.add_column("plates", justify="left")
    table.add_column("parents", justify="left")

    parents_by_dst: dict[str, list[str]] = {}
    for e in graph.edges:
        parents_by_dst.setdefault(e.dst, []).append(e.src)

    for i, node in enumerate(graph.nodes, start=1):
        badge = _KIND_BADGE.get(node.kind, "·")
        kind_cell = Text(f"{badge} {node.kind}", style=_KIND_STYLE.get(node.kind, ""))
        plates = " ⨯ ".join(node.plates) if node.plates else "—"
        parents = ", ".join(parents_by_dst.get(node.name, ())) or "—"
        family = node.family or "—"
        table.add_row(
            str(i),
            kind_cell,
            node.name,
            family,
            plates,
            parents,
        )
    return table


def render_table_plain(graph: PlateGraph) -> str:
    """Plain-text rendering of the table, for non-Rich callers.

    Used by `:plate` from `qvr repl --plain` and from the Jupyter
    kernel.

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to tabulate.

    Returns
    -------
    str
        The table as aligned plain text.
    """
    parents_by_dst: dict[str, list[str]] = {}
    for e in graph.edges:
        parents_by_dst.setdefault(e.dst, []).append(e.src)

    cols = ("#", "kind", "variable", "family", "plates", "parents")
    rows: list[tuple[str, ...]] = []
    for i, node in enumerate(graph.nodes, start=1):
        plates = " x ".join(node.plates) if node.plates else "-"
        parents = ", ".join(parents_by_dst.get(node.name, ())) or "-"
        family = node.family or "-"
        rows.append(
            (
                str(i),
                f"{_KIND_BADGE.get(node.kind, '.')} {node.kind}",
                node.name,
                family,
                plates,
                parents,
            )
        )

    # Compute column widths
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out_lines = [
        f"plate graph of {graph.program_name}"
        + (
            f"  ({graph.domain} -> {graph.codomain})"
            if graph.domain or graph.codomain
            else ""
        ),
        "",
        _fmt(cols),
        "  ".join("-" * w for w in widths),
    ]
    out_lines.extend(_fmt(row) for row in rows)

    if graph.plates:
        out_lines.append("")
        out_lines.append("plates:")
        for p in graph.plates:
            card = f" [{p.cardinality}]" if p.cardinality else ""
            parent = f" parent={p.parent}" if p.parent else ""
            out_lines.append(f"  {p.name}{card}{parent}")

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def _safe_id(s: str) -> str:
    """Mermaid node ids can't contain ``::``, spaces, or other
    punctuation that QVR scope paths produce. Encode them so the
    output is well-formed regardless of input."""
    return s.replace("::", "_").replace(" ", "_").replace("-", "_")


def render_mermaid(graph: PlateGraph) -> str:
    """Return Mermaid ``graph TD`` source for ``graph``.

    Each plate becomes a ``subgraph`` cluster. Each node is
    drawn as a circle (``( )``) if latent, a double-circle
    (``(( ))``) if observed, an octagon (``{{ }}``) if
    marginalized, and a hexagon (``([ ])``) if deterministic.
    Edges follow the graph's ``edges`` field.

    Mermaid lacks a native plate / nested-plate primitive; the
    subgraph blocks approximate it well enough that
    GitHub-rendered markdown and mermaid.live both display the
    structure.

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to draw.

    Returns
    -------
    str
        Mermaid ``graph TD`` source.
    """
    lines: list[str] = []
    lines.append("graph TD")
    # Node declarations, grouped by their innermost plate.
    by_plate: dict[tuple[str, ...], list[PlateNode]] = {}
    for n in graph.nodes:
        by_plate.setdefault(n.plates, []).append(n)

    def _node_decl(n: PlateNode) -> str:
        nid = _safe_id(n.scope_path or n.name)
        label = n.name
        if n.family:
            label = f"{n.name}<br/>{n.family}"
        if n.kind == "latent":
            return f"    {nid}(({label}))"
        if n.kind == "observed":
            return f"    {nid}(((({label}))))"
        if n.kind == "marginalized":
            return f"    {nid}{{{{ {label} }}}}"
        return f"    {nid}([{label}])"

    # Emit plate clusters in nesting order (longest plate stacks
    # last so inner plates appear inside outer ones in the source).
    plate_stacks = sorted({tuple(n.plates) for n in graph.nodes}, key=len)
    seen_cluster_keys: set[tuple[str, ...]] = set()
    for stack in plate_stacks:
        if not stack:
            # Outer scope: emit naked.
            for n in by_plate.get(stack, ()):
                lines.append(_node_decl(n))
            continue
        if stack in seen_cluster_keys:
            continue
        seen_cluster_keys.add(stack)
        label = " x ".join(stack)
        cid = _safe_id("_x_".join(stack))
        lines.append(f'    subgraph {cid} ["{label}"]')
        for n in by_plate.get(stack, ()):
            lines.append(_node_decl(n))
        lines.append("    end")

    # Edges.
    for e in graph.edges:
        # Some edges reference variables that aren't in the
        # graph's nodes (e.g. external program inputs). Emit them
        # anyway; Mermaid creates an implicit node.
        src = _safe_id(e.src)
        dst_node = next((n for n in graph.nodes if n.name == e.dst), None)
        dst = _safe_id(dst_node.scope_path) if dst_node else _safe_id(e.dst)
        lines.append(f"    {src} --> {dst}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT
# ---------------------------------------------------------------------------


def render_dot(graph: PlateGraph) -> str:
    """Return Graphviz DOT source. Plates become ``cluster_*``
    subgraphs; nodes carry shape attributes based on kind.

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to draw.

    Returns
    -------
    str
        Graphviz DOT source.
    """
    shape_for: dict[str, str] = {
        "latent": "circle",
        "observed": "doublecircle",
        "marginalized": "octagon",
        "deterministic": "box",
    }
    fillcolor_for: dict[str, str] = {
        "latent": "white",
        "observed": "lightgray",
        "marginalized": "white",
        "deterministic": "white",
    }
    lines: list[str] = [
        f"digraph {_safe_id(graph.program_name)} {{",
        "  graph [rankdir=TB, fontsize=10]",
        "  node  [fontsize=10, style=filled]",
        "  edge  [fontsize=9]",
    ]

    by_plate: dict[tuple[str, ...], list[PlateNode]] = {}
    for n in graph.nodes:
        by_plate.setdefault(n.plates, []).append(n)

    def _node_id(n: PlateNode) -> str:
        return _safe_id(n.scope_path or n.name)

    def _emit_node(n: PlateNode, indent: str) -> str:
        nid = _node_id(n)
        label = n.name
        if n.family:
            label = f"{n.name}\\n{n.family}"
        shape = shape_for.get(n.kind, "box")
        fc = fillcolor_for.get(n.kind, "white")
        if n.kind == "marginalized":
            return f'{indent}{nid} [label="{label}", shape={shape}, fillcolor={fc}, style="filled,dashed"]'
        return f'{indent}{nid} [label="{label}", shape={shape}, fillcolor={fc}]'

    # Cluster per plate stack.
    plate_stacks = sorted({tuple(n.plates) for n in graph.nodes}, key=len)
    for stack in plate_stacks:
        if not stack:
            for n in by_plate.get(stack, ()):
                lines.append(_emit_node(n, "  "))
            continue
        cid = _safe_id("_x_".join(stack))
        label = " x ".join(stack)
        lines.append(f"  subgraph cluster_{cid} {{")
        lines.append(f'    label="{label}";')
        lines.append('    style="rounded,dashed";')
        for n in by_plate.get(stack, ()):
            lines.append(_emit_node(n, "    "))
        lines.append("  }")

    for e in graph.edges:
        src = _safe_id(e.src)
        dst_node = next((n for n in graph.nodes if n.name == e.dst), None)
        dst = _node_id(dst_node) if dst_node else _safe_id(e.dst)
        lines.append(f"  {src} -> {dst}")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TikZ (LaTeX)
# ---------------------------------------------------------------------------


def render_tikz(graph: PlateGraph) -> str:
    """Return a LaTeX snippet using ``tikz`` and the ``bayesnet``
    package conventions. Computes positions by laying nodes out
    in columns (one per depth level) and rows (one per plate
    stack); good enough for diagrams up to ~20 nodes.

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to draw.

    Returns
    -------
    str
        A LaTeX ``tikzpicture`` snippet.
    """
    # Topological depth = longest path of edges into a node.
    depth: dict[str, int] = {}
    for n in graph.nodes:
        depth[n.name] = 0
    changed = True
    while changed:
        changed = False
        for e in graph.edges:
            new_d = depth.get(e.src, 0) + 1
            if new_d > depth.get(e.dst, 0):
                depth[e.dst] = new_d
                changed = True

    # Group nodes by (depth, plate_stack)
    positions: dict[str, tuple[float, float]] = {}
    by_col: dict[int, list[PlateNode]] = {}
    for n in graph.nodes:
        by_col.setdefault(depth.get(n.name, 0), []).append(n)
    for col, members in by_col.items():
        for row, n in enumerate(members):
            positions[n.name] = (col * 2.0, -row * 1.5)

    lines: list[str] = [
        "% Plate diagram of " + graph.program_name,
        "% Requires \\usepackage{tikz} and \\usetikzlibrary{bayesnet}",
        "\\begin{tikzpicture}",
    ]
    for n in graph.nodes:
        x, y = positions[n.name]
        nid = _safe_id(n.scope_path or n.name)
        label = n.name
        if n.kind == "observed":
            lines.append(f"  \\node[obs] ({nid}) at ({x:.1f},{y:.1f}) {{${label}$}};")
        elif n.kind == "marginalized":
            lines.append(
                f"  \\node[latent, dashed] ({nid}) at ({x:.1f},{y:.1f}) {{${label}$}};"
            )
        elif n.kind == "deterministic":
            lines.append(f"  \\node[det] ({nid}) at ({x:.1f},{y:.1f}) {{${label}$}};")
        else:
            lines.append(
                f"  \\node[latent] ({nid}) at ({x:.1f},{y:.1f}) {{${label}$}};"
            )

    for e in graph.edges:
        src_node = next((n for n in graph.nodes if n.name == e.src), None)
        if src_node is None:
            continue
        dst_node = next((n for n in graph.nodes if n.name == e.dst), None)
        if dst_node is None:
            continue
        src = _safe_id(src_node.scope_path or src_node.name)
        dst = _safe_id(dst_node.scope_path or dst_node.name)
        lines.append(f"  \\edge {{{src}}} {{{dst}}};")

    # Plates: each plate becomes a \plate around its members.
    plate_stacks = sorted({tuple(n.plates) for n in graph.nodes}, key=len)
    plate_id = 0
    for stack in plate_stacks:
        if not stack:
            continue
        members = [n for n in graph.nodes if n.plates == stack]
        if not members:
            continue
        nodes = " ".join(_safe_id(n.scope_path or n.name) for n in members)
        label = " \\times ".join(stack)
        lines.append(f"  \\plate {{p{plate_id}}} {{({nodes})}} {{${label}$}};")
        plate_id += 1

    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# daft (matplotlib-backed pgm renderer)
# ---------------------------------------------------------------------------


def render_daft(graph: PlateGraph) -> str:
    """Return a Python script that uses ``daft`` to render the
    plate diagram. The script defines a ``build_pgm()`` function
    returning a ``daft.PGM`` instance, ready for the user to
    ``render()`` and save.

    Parameters
    ----------
    graph : PlateGraph
        The plate graph to draw.

    Returns
    -------
    str
        Python source defining ``build_pgm()``.
    """
    # Topological depth -> column position.
    depth: dict[str, int] = {n.name: 0 for n in graph.nodes}
    changed = True
    while changed:
        changed = False
        for e in graph.edges:
            new_d = depth.get(e.src, 0) + 1
            if new_d > depth.get(e.dst, 0):
                depth[e.dst] = new_d
                changed = True

    by_col: dict[int, list[PlateNode]] = {}
    for n in graph.nodes:
        by_col.setdefault(depth.get(n.name, 0), []).append(n)
    positions: dict[str, tuple[float, float]] = {}
    for col, members in by_col.items():
        for row, n in enumerate(members):
            positions[n.name] = (col + 1.0, -row + len(members) / 2.0)

    width = max(depth.values(), default=0) + 2
    height = max(len(m) for m in by_col.values()) + 1 if by_col else 1

    lines: list[str] = [
        "import daft",
        "",
        "",
        "def build_pgm():",
        f'    """Build the plate diagram for {graph.program_name}."""',
        f"    pgm = daft.PGM(shape=[{width}, {height}])",
    ]
    for n in graph.nodes:
        x, y = positions[n.name]
        observed = "True" if n.kind == "observed" else "False"
        fixed = "True" if n.kind == "deterministic" else "False"
        lines.append(
            f'    pgm.add_node("{n.name}", r"${n.name}$", {x:.2f}, {y:.2f}, '
            f"observed={observed}, fixed={fixed})"
        )
    for e in graph.edges:
        if any(n.name == e.src for n in graph.nodes):
            lines.append(f'    pgm.add_edge("{e.src}", "{e.dst}")')

    # Plates: bounding box per plate stack, derived from node positions.
    plate_stacks = sorted({tuple(n.plates) for n in graph.nodes}, key=len)
    for stack in plate_stacks:
        if not stack:
            continue
        members = [n for n in graph.nodes if n.plates == stack]
        if not members:
            continue
        xs = [positions[m.name][0] for m in members]
        ys = [positions[m.name][1] for m in members]
        x0 = min(xs) - 0.5
        y0 = min(ys) - 0.5
        w = max(xs) - x0 + 0.5
        h = max(ys) - y0 + 0.5
        label = " \\times ".join(stack)
        lines.append(
            f"    pgm.add_plate([{x0:.2f}, {y0:.2f}, {w:.2f}, {h:.2f}], "
            f'label=r"${label}$")'
        )

    lines.append("    return pgm")
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    pgm = build_pgm()")
    lines.append("    pgm.render()")
    lines.append('    pgm.savefig("plate.png")')
    return "\n".join(lines)


__all__ = [
    "render_dot",
    "render_daft",
    "render_mermaid",
    "render_table",
    "render_table_plain",
    "render_tikz",
]
