"""A gathered row or slice of a table keeps the table's remaining axes.

`let row_table = table[idx]` with `table` over `(Group, Class)` is one row
of `Class` entries per observation. The plan names the gathered value's
axes after the index and the table's axes after the first, so a target
that computes it entry by entry reads `table[idx[n], k]` at each class
`k`; BUGS and JAGS once subscripted the table with the row index alone,
which JAGS refuses as a dimension mismatch.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile._entry import Lower, expand_composite_lets
from quivers.transpile.ir import IRDeterministic

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "let_expressions"

#: The program of the report: a row of a two-axis factor table gathered
#: by a host index, with a second object of the class axis's extent.
_REPORTED = """\
object Kind : FinSet 3
object Class : FinSet 4
object Resp : FinSet 6
object Out : Real 4

program p : Resp -> Out
    sample e : Kind <- Normal(loc=0.0, scale=1.0)
    let table = factor k : Kind, c : Class in sigmoid(e[k] + real(c))
    let row_probs = table[idx]
    observe y : Resp <- Categorical(probs=row_probs)
    return table

export p
"""


def _compact(source: str, target: str) -> str:
    """A program's target source with every space removed.

    Parameters
    ----------
    source : str
        The QVR source.
    target : str
        The target.

    Returns
    -------
    str
        The rendered program without spaces.
    """
    return re.sub(r"\s+", "", transpile(parse(source), target=target).decode())


def _axes(source: str, name: str) -> tuple[str, ...]:
    """The axis names the plan gives a let.

    Parameters
    ----------
    source : str
        The QVR source.
    name : str
        The let's name.

    Returns
    -------
    tuple[str, ...]
        Its batch then event axis names.
    """
    module = expand_composite_lets(parse(source), target="jags")
    ir = Lower().forward(module, target="jags")
    node = next(n for n in ir.body if isinstance(n, IRDeterministic) and n.name == name)
    return tuple(str(d.name) for d in (*node.plate.batch_dims, *node.plate.event_dims))


def test_the_gather_is_named_after_its_index_and_the_table() -> None:
    assert _axes(_REPORTED, "table") == ("Kind", "Class")
    assert _axes(_REPORTED, "row_probs") == ("Resp", "Class")


def test_jags_reads_the_reported_gather_at_each_class() -> None:
    jags = _compact(_REPORTED, "jags")
    assert "row_probs[m_Resp,m_Class]<-table[idx[m_Resp]+1,m_Class]" in jags


@pytest.mark.parametrize(
    ("stem", "target", "relation"),
    [
        (
            "let_row_gather",
            "jags",
            "row_table[m_Row,m_Class]<-table[group_idx[m_Row]+1,m_Class]",
        ),
        (
            "let_slice_gather",
            "jags",
            "row_slice[m_Row,m_Part,m_Class]<-table[group_idx[m_Row]+1,m_Part,m_Class]",
        ),
        (
            "let_slice_gather",
            "bugs",
            "row_slice[m_Row,m_Part,m_Class]<-table[group_idx[m_Row]+1,m_Part,m_Class]",
        ),
        (
            "let_slice_gather",
            "stan",
            "row_slice[m_Row,m_Part,m_Class]=table[group_idx[m_Row]+1,m_Part,m_Class];",
        ),
    ],
)
def test_target_reads_every_remaining_axis(
    stem: str, target: str, relation: str
) -> None:
    source = (_FIXTURES / f"{stem}.qvr").read_text()
    assert relation in _compact(source, target)
