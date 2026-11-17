"""All tools retain canonical telescope identities and parser boundaries."""

import pytest
import torch

from quivers.analysis.scope import ScopedRef, scope_children
from quivers.cli.check import check_file
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse
from quivers.dsl.ast_nodes import ProgramDecl, QiecValueParameter
from quivers.dsl._data_inference import infer_data_objects
from quivers.kernel.quivers_kernel import _split_cell
from quivers.lsp.document import DocumentState
from quivers.modeling import Model


SCALAR = """object R:Real 1
program p(x:Real,y:Real):R->R
    sample b <- Normal(loc=x,scale=1.0)
    observe y <- Normal(loc=b,scale=1.0)
    return b
export p
"""


def test_scalar_header_keeps_its_canonical_declaration_in_scope_and_lsp():
    declaration = next(
        node for node in parse(SCALAR).statements if isinstance(node, ProgramDecl)
    )
    children = scope_children(
        ScopedRef(name="p", kind="program", path="p", node=declaration)
    )
    assert all(children[name].kind == "data-input" for name in ("x", "y"))
    assert all(
        isinstance(children[name].node, QiecValueParameter) for name in ("x", "y")
    )
    session = ReplSession()
    assert session.dispatch(SCALAR).ok
    assert "data-input" in session.dispatch(":info p::y").body
    assert "Real" in session.dispatch(":type p::y").body
    doc = DocumentState(uri="file:///tmp/scalar-telescope.qvr")
    doc.update(source=SCALAR, version=1)
    visible = doc.visible_qiec_locals(line=3, col=15)
    assert [node.name for node in visible] == ["x", "y"]
    assert all(isinstance(node, QiecValueParameter) for node in visible)


@pytest.mark.parametrize(
    "header",
    [
        "program p[\n    n:Nat,\n\n](x:Tensor[Real]([n])) : R->R",
        "program p(\n    x:Real,\n\n) : R->R",
    ],
)
def test_notebook_preserves_hanging_telescope_and_executes_same_source(header):
    source = (
        "object R:Real 1\n"
        + header
        + "\n    sample b <- Normal(loc=0.0,scale=1.0)\n    return b\nexport p\n"
    )
    parse(source)
    chunks = _split_cell(source + "\n:type p\n")
    assert len(chunks) == 2 and chunks[1] == ":type p"
    session = ReplSession()
    for chunk in chunks:
        response = session.dispatch(chunk)
        assert response.ok, response.diagnostics
    assert "p" in response.body


def test_cli_and_lsp_do_not_skip_unrelated_errors_for_inferred_objects(tmp_path):
    source = "object Row:FinSet _\nobject R:Real 1\ndefine broken = missing\n"
    path = tmp_path / "inferred.qvr"
    path.write_text(source)
    checked = check_file(path)
    doc = DocumentState(uri=path.as_uri())
    doc.update(source=source, version=1)
    assert len(checked) == len(doc.diagnostics) == 1
    assert checked[0].code == doc.diagnostics[0].code == "compile"
    assert checked[0].line == doc.diagnostics[0].line == 3
    assert "undefined morphism 'missing'" in checked[0].message


def test_inferred_extent_substitutes_constructor_options_and_dimensions_together():
    source = """object R:Real 1
program p[n:Nat](x:Tensor[Real]([n])) : R -> Real n {low=0, high=n}
    return x
export p
"""
    module = infer_data_objects(parse(source), {"x": torch.tensor([0.25, 1.5])})
    declaration = next(
        node for node in module.statements if isinstance(node, ProgramDecl)
    )
    assert not declaration.binders
    assert declaration.codomain.args == ("2",)
    assert declaration.codomain.kwargs == {"low": 0, "high": 2}
    bound = Model(source).bind(
        {"x": torch.tensor([0.25, 1.5])}, input=torch.zeros(1, 1)
    )
    torch.testing.assert_close(
        bound.trace().output.flatten(), torch.tensor([0.25, 1.5])
    )


def test_dataframe_composition_preserves_every_authored_grouped_object(monkeypatch):
    import pandas as pd
    from quivers.data import DatasetSchema, compose
    from quivers.data import schema as schema_module
    from quivers.dsl import loads
    from quivers.dsl.ast_nodes import ObjectDecl

    compiled_inputs = []

    def inspected_loads(source, **options):
        compiled_inputs.append(parse(source))
        return loads(source, **options)

    monkeypatch.setattr(schema_module, "loads", inspected_loads)

    frame = pd.DataFrame({"a": ["u", "v"], "b": ["p", "q"]})
    schema = DatasetSchema(df=frame, objects={"a": "A", "b": "B"})
    compiled = compose(
        """object A, B : FinSet 7
morphism f : A -> B [role=parameter]
export f
""",
        schema,
    )
    assert compiled.morphism.domain.cardinality == 7
    assert compiled.morphism.codomain.cardinality == 7
    names = [
        name
        for declaration in compiled_inputs[0].statements
        if isinstance(declaration, ObjectDecl)
        for name in declaration.names
    ]
    assert names == ["A", "B"]


@pytest.mark.parametrize(
    "source,expected",
    [
        ("object R:Real 1\nexport missing\n", "undefined morphism"),
        (
            "object A:FinSet 2\nobject B:FinSet 3\nobject C:FinSet 4\nmorphism f:A->B\nmorphism g:C->A\nexport f >> g\n",
            "cannot compose",
        ),
    ],
)
def test_cli_and_lsp_validate_exports_through_the_same_compiler(
    source, expected, tmp_path
):
    path = tmp_path / "export.qvr"
    path.write_text(source)
    diagnostics = check_file(path)
    doc = DocumentState(uri=path.as_uri())
    doc.update(source=source, version=1)
    assert diagnostics and doc.diagnostics
    assert expected in diagnostics[0].message.lower()
    assert diagnostics[0].message == doc.diagnostics[0].message


@pytest.mark.parametrize(
    "source",
    [
        "object R:Real 1\nprogram p[n:Nat](x:Tensor[Real]([n])):R->R\n    sample b <- Normal(loc=0.0,scale=1.0)\n    return b\nexport p\n",
        "object Row:FinSet _\nobject R:Real 1\nprogram p(x:Tensor[Real]([Row])):Row->R\n    sample b <- Normal(loc=0.0,scale=1.0)\n    return b\nexport p\n",
    ],
)
def test_generic_and_deferred_exports_remain_inspectable(source, tmp_path):
    path = tmp_path / "generic.qvr"
    path.write_text(source)
    assert not check_file(path)
    doc = DocumentState(uri=path.as_uri())
    doc.update(source=source, version=1)
    assert not doc.diagnostics


def test_shared_analysis_validates_exports_without_reinitializing_parameters():
    from quivers.dsl import Compiler
    from quivers.dsl.qiec_tooling import analyze_module

    source = "object A:FinSet 2\nobject B:FinSet 3\nmorphism f:A->B [role=parameter]\nexport f\n"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        analysis = analyze_module(parse(source), module_name="audit")
        inspected = torch.random.get_rng_state().clone()
        torch.manual_seed(73)
        Compiler(parse(source), module_name="audit").compile()
        compiled = torch.random.get_rng_state().clone()
    assert analysis.compile_error is None
    assert torch.equal(inspected, compiled)
