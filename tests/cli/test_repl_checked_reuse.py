"""Type probes reuse checked evidence while retaining isolated native state."""

from unittest.mock import patch

import torch

from quivers.cli.repl_session import ReplSession
from quivers.dsl.compiler import core as compiler_core


SOURCE = """\
object X : FinSet 3
object Y : FinSet 4
morphism f : X -> Y [role=parameter]
define answer() : Int !{} =
    return 42
"""


def test_kind_probe_reuses_identical_checked_source():
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    with patch.object(
        compiler_core,
        "_lower_qvr_with_companions",
        wraps=compiler_core._lower_qvr_with_companions,
    ) as lower:
        response = session.dispatch(":kind FinSet 3")
        again = session.dispatch(":kind FinSet 3")
        lower.assert_not_called()
        assert ReplSession().dispatch(SOURCE.replace("return 42", "return 43")).ok
        lower.assert_called()
    assert response.ok and again.body == response.body == "FinSet 3 :: FinSet 3"


def test_scratch_compilers_share_only_checked_immutable_evidence():
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    first, second = session._scratch_compiler(), session._scratch_compiler()
    assert first is not second and first is not session._compiler
    assert first.qiec_module is second.qiec_module is session._qiec_module
    first_parameters = list(first.morphisms["f"].module().parameters())
    second_parameters = list(second.morphisms["f"].module().parameters())
    retained_parameters = list(session._compiler.morphisms["f"].module().parameters())
    assert first_parameters and second_parameters and retained_parameters
    retained_values = [parameter.detach().clone() for parameter in retained_parameters]
    for left, right in zip(first_parameters, second_parameters, strict=True):
        assert left is not right and left.data_ptr() != right.data_ptr()
    with torch.no_grad():
        first_parameters[0].fill_(17.0)
    for parameter, value in zip(retained_parameters, retained_values, strict=True):
        torch.testing.assert_close(parameter, value)
