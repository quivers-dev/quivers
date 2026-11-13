"""Tests for the qvr_program panproto protocol and Schema extraction.

For every example program, compile to populate the resolution environment,
extract the corresponding :class:`panproto.Schema`, and assert it validates
against :data:`~quivers.dsl.program_theory.QVR_PROGRAM_PROTOCOL`. Then
spot-check that ``panproto.diff_schemas`` produces a non-trivial diff
between two structurally distinct example programs (sanity that the
extracted schemas carry enough structure to be diffable).
"""

from pathlib import Path

import panproto
import pytest

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse, parse_file
from quivers.dsl.program_theory import QVR_PROGRAM_PROTOCOL, extract_program_schema


EXAMPLES_DIR = Path(__file__).parent.parent / "docs/examples/source"
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.qvr"))


def _compile_to_env(path: Path) -> Compiler:
    """Parse the program and run :meth:`Compiler.compile_env` to populate
    the resolved object/space/morphism dictionaries."""
    module = parse_file(path)
    compiler = Compiler(module)
    compiler.compile_env()
    return compiler


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_program_schema_validates(path: Path) -> None:
    """Every example produces a Schema that validates against qvr_program."""
    compiler = _compile_to_env(path)
    schema = extract_program_schema(compiler)
    assert schema.protocol == "qvr_program"
    schema.validate(QVR_PROGRAM_PROTOCOL)


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_program_schema_has_root(path: Path) -> None:
    """Every extracted schema has the ``program`` root vertex."""
    compiler = _compile_to_env(path)
    schema = extract_program_schema(compiler)
    kinds = {v.kind for v in schema.vertices}
    assert "program" in kinds


def test_program_schema_object_decls_for_hmm() -> None:
    """hmm.qvr's object declarations appear as object_decl vertices
    with the right names and bind to finset vertices.

    The two finite objects play different roles, and the schema
    carries both: `State` is the hidden chain's support and `Symbol`
    the alphabet an emission draws from. `StateProbs` is the real
    vector the program returns, so it binds a euclidean vertex rather
    than a finset one.
    """
    compiler = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    schema = extract_program_schema(compiler)

    object_decl_ids = [v.id for v in schema.vertices if v.kind == "object_decl"]
    decl_names = []
    for vid in object_decl_ids:
        for c in schema.constraints_for(vid):
            if c.sort == "name":
                decl_names.append(c.value)
    assert set(decl_names) == {"State", "Symbol"}

    finset_ids = [v.id for v in schema.vertices if v.kind == "finset"]
    finset_props = []
    for fid in finset_ids:
        props = {c.sort: c.value for c in schema.constraints_for(fid)}
        finset_props.append(props)
    cardinalities = sorted(int(p["cardinality"]) for p in finset_props)
    assert cardinalities == [2, 4]


def test_output_decl_recorded_for_hmm() -> None:
    """The compiler's output expression surfaces as an output_decl vertex."""
    compiler = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    schema = extract_program_schema(compiler)
    output_ids = [v.id for v in schema.vertices if v.kind == "output_decl"]
    assert len(output_ids) == 1, "expected exactly one output_decl"
    output_constraints = {
        c.sort: c.value for c in schema.constraints_for(output_ids[0])
    }
    # hmm.qvr's first export (`export hmm`) is the module's output
    # entry point; it resolves to ExprIdent(name='hmm').
    assert output_constraints["name"] == "hmm"
    # the program -> output_decl edge with kind 'output' is present
    output_edges = [
        e for e in schema.edges if e.src == "program" and e.kind == "output"
    ]
    assert len(output_edges) == 1
    assert output_edges[0].tgt == output_ids[0]


def test_diff_distinguishes_distinct_programs() -> None:
    """panproto.diff_schemas produces a non-trivial diff between two
    structurally distinct example programs."""
    compiler_a = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    compiler_b = _compile_to_env(EXAMPLES_DIR / "pcfg.qvr")
    schema_a = extract_program_schema(compiler_a)
    schema_b = extract_program_schema(compiler_b)

    diff = panproto.diff_schemas(schema_a, schema_b).to_dict()
    # at minimum, the two schemas should differ — they declare different
    # objects (State/Symbol vs N/T) and different morphisms.
    assert len(diff["added_vertices"]) > 0 or len(diff["removed_vertices"]) > 0


def test_identical_compilation_produces_equal_schemas() -> None:
    """Compiling the same .qvr file twice yields identical Schemas."""
    a = extract_program_schema(_compile_to_env(EXAMPLES_DIR / "hmm.qvr"))
    b = extract_program_schema(_compile_to_env(EXAMPLES_DIR / "hmm.qvr"))
    diff = panproto.diff_schemas(a, b).to_dict()
    # vertex/edge counts should match; vertex IDs are deterministic in the
    # extractor (object_decl::Name, then sequential numeric suffixes for the
    # set-object subtrees) so the schemas should be structurally identical.
    assert a.vertex_count == b.vertex_count
    assert a.edge_count == b.edge_count
    assert len(diff["added_vertices"]) == 0
    assert len(diff["removed_vertices"]) == 0


PROGRAM_ONLY = """\
object Trial : FinSet 6
object Rate : Real 1

program coin : Trial -> Rate
    sample theta <- Beta(concentration1=2.0, concentration0=2.0)
    observe y : Trial <- Bernoulli(probs=theta)
    return theta
export coin
"""

MIXED = """\
object Obs : FinSet 4

define shift(x : Real, by : Real) : Real !{} =
    return x + by

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

instance random : Random

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y <- Normal(loc=c, scale=0.5)
    return c
export prog

define twice(y : Real) : Real !{random, score} =
    let first <- prog(y)
    let second <- prog(y)
    return first + second
"""


def _names(schema: panproto.Schema, kind: str) -> dict[str, str]:
    """Map every vertex of a kind to its ``name`` constraint."""
    names: dict[str, str] = {}
    for vertex in schema.vertices:
        if vertex.kind != kind:
            continue
        for constraint in schema.constraints_for(vertex.id):
            if constraint.sort == "name":
                names[vertex.id] = constraint.value
    return names


def _edges(schema: panproto.Schema, kind: str) -> set[tuple[str, str]]:
    """The source and target of every edge of a kind."""
    return {(edge.src, edge.tgt) for edge in schema.edges if edge.kind == kind}


def test_program_only_module_yields_the_kernel_graph() -> None:
    """A module holding one program produces its computation, entry, sites,
    and the canonical instances the entry addresses."""
    compiler = Compiler(parse(PROGRAM_ONLY))
    compiler.compile_env()
    schema = extract_program_schema(compiler)
    schema.validate(QVR_PROGRAM_PROTOCOL)

    assert set(_names(schema, "computation_decl").values()) == {"coin"}
    assert set(_names(schema, "program_entry").values()) == {"coin"}
    assert set(_names(schema, "program_site").values()) == {"theta", "y"}
    assert set(_names(schema, "instance_decl").values()) == {"random", "score"}
    assert set(_names(schema, "effect_decl").values()) >= {"Random", "Score"}
    assert set(_names(schema, "program_parameter").values()) == {"y"}

    assert _edges(schema, "entry") == {("program", "program_entry::coin")}
    assert _edges(schema, "body") == {("program_entry::coin", "computation_decl::coin")}
    assert _edges(schema, "random") == {
        ("program_entry::coin", "instance_decl::random")
    }
    assert _edges(schema, "score") == {("program_entry::coin", "instance_decl::score")}
    assert _edges(schema, "performs") == {
        ("computation_decl::coin", "instance_decl::random"),
        ("computation_decl::coin", "instance_decl::score"),
    }
    assert _edges(schema, "row") == _edges(schema, "performs")
    sites = {
        name: {c.sort: c.value for c in schema.constraints_for(vid)}
        for vid, name in _names(schema, "program_site").items()
    }
    assert sites["theta"]["site_kind"] == "sample"
    assert sites["theta"]["family"] == "Beta"
    assert sites["y"]["site_kind"] == "observe"
    assert sites["y"]["family"] == "Bernoulli"
    assert sites["y"]["batch"] == "Trial:6"


def test_mixed_module_is_one_call_graph() -> None:
    """Computations and programs calling each other share one graph."""
    compiler = Compiler(parse(MIXED))
    schema = extract_program_schema(compiler)
    schema.validate(QVR_PROGRAM_PROTOCOL)

    assert set(_names(schema, "computation_decl").values()) == {
        "shift",
        "noisy",
        "prog",
        "twice",
    }
    assert _edges(schema, "calls") == {
        ("computation_decl::prog", "computation_decl::noisy"),
        ("computation_decl::prog", "computation_decl::shift"),
        ("computation_decl::twice", "computation_decl::prog"),
    }
    assert ("computation_decl::noisy", "instance_decl::random") in _edges(
        schema, "performs"
    )
    twice = {c.sort: c.value for c in schema.constraints_for("computation_decl::twice")}
    assert twice["type"] == "Real"
    parameters = _names(schema, "computation_parameter")
    assert parameters["computation_decl::shift/parameter::by"] == "by"


def test_diff_sees_a_changed_site() -> None:
    """Changing one site's family changes the extracted schema."""
    changed = PROGRAM_ONLY.replace("Bernoulli(probs=theta)", "Geometric(probs=theta)")
    first = Compiler(parse(PROGRAM_ONLY))
    second = Compiler(parse(changed))
    schema_a = extract_program_schema(first)
    schema_b = extract_program_schema(second)
    families_a = {
        c.value
        for c in schema_a.constraints_for("program_entry::coin/site::y")
        if c.sort == "family"
    }
    families_b = {
        c.value
        for c in schema_b.constraints_for("program_entry::coin/site::y")
        if c.sort == "family"
    }
    assert families_a == {"Bernoulli"}
    assert families_b == {"Geometric"}
    assert panproto.diff_schemas(schema_a, schema_b).to_dict() != (
        panproto.diff_schemas(schema_a, schema_a).to_dict()
    )
