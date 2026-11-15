"""Notebook execution over real Jupyter channels, including silent errors."""

from __future__ import annotations

import sys
import os
from pathlib import Path

from jupyter_client import KernelManager
import pytest

from quivers import __version__
import quivers


@pytest.mark.timeout(90)
def test_kernel_protocol_executes_qiec_and_preserves_silent_errors(tmp_path) -> None:
    manager = KernelManager(connection_file=str(tmp_path / "kernel.json"))
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "quivers.kernel",
        "_run_f",
        "{connection_file}",
    ]
    # The notebook changes directory; retain the package tested by this
    # process even when PYTHONPATH contains a relative source directory.
    package_root = str(Path(quivers.__file__).resolve().parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (package_root, environment.get("PYTHONPATH", ""))
    )
    manager.start_kernel(cwd=str(tmp_path), env=environment)
    client = manager.blocking_client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        info = client.kernel_info(reply=True, timeout=15)["content"]
        assert info["implementation_version"] == __version__
        assert info["language_info"]["version"] == __version__
        reply = client.execute(
            "define answer() : Int !{} =\n    return 42\n:run answer",
            reply=True,
            timeout=20,
        )["content"]
        assert reply["status"] == "ok"
        reply = client.execute(
            "object Row : FinSet _\n"
            "object Out : Real 1\n"
            "program regression(x : Tensor[Real]([Row])) : Row -> Out\n"
            "    sample b <- Normal(loc=0.0, scale=1.0)\n"
            "    let mu = b * x\n"
            "    observe y : Row <- Normal(loc=mu, scale=1.0)\n"
            "    return b\n"
            "export regression\n"
            ":type regression::x\n"
            ":run regression --data x=[1.0,2.0] --data y=[0.5,1.0]",
            reply=True,
            timeout=20,
        )["content"]
        assert reply["status"] == "ok"
        inspected = client.inspect("regression::x", 13, reply=True, timeout=15)[
            "content"
        ]
        assert inspected["found"]
        assert "Tensor[Real]([Row])" in inspected["data"]["text/plain"]
        reply = client.execute("@@@ invalid @@@", silent=True, reply=True, timeout=15)[
            "content"
        ]
        assert reply["status"] == "error"
        assert reply["ename"] and reply["evalue"] and reply["traceback"]
        reply = client.complete(":transpile sta", reply=True, timeout=15)["content"]
        assert ":transpile stan" in reply["matches"]
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
