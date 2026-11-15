"""Language-server initialization and document updates over stdio JSON-RPC."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from quivers import __version__


@pytest.mark.timeout(60)
def test_lsp_stdio_handshake_diagnostics_and_close() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from quivers.cli.lsp import _entry; raise SystemExit(_entry())",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None

    def send(method, params, identifier=None):
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if identifier is not None:
            message["id"] = identifier
        body = json.dumps(message).encode()
        process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        process.stdin.flush()

    def receive():
        headers = {}
        while (line := process.stdout.readline()) not in (b"\r\n", b"\n"):
            assert line, "server closed stdout before returning a protocol message"
            key, value = line.decode().split(":", 1)
            headers[key.lower()] = value.strip()
        return json.loads(process.stdout.read(int(headers["content-length"])))

    try:
        send(
            "initialize",
            {
                "processId": None,
                "rootUri": None,
                "capabilities": {"general": {"positionEncodings": ["utf-8", "utf-16"]}},
            },
            1,
        )
        response = receive()
        assert response["result"]["serverInfo"]["version"] == __version__
        assert response["result"]["capabilities"]["positionEncoding"] == "utf-16"
        send("initialized", {})
        uri = "untitled:tooling.qvr"
        send(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "qvr",
                    "version": 1,
                    "text": "define answer() : Int !{} =\n    return 42\n",
                }
            },
        )
        assert receive()["params"]["diagnostics"] == []
        send(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": 2},
                "contentChanges": [{"text": "@@@ invalid @@@"}],
            },
        )
        assert receive()["params"]["diagnostics"]
        send("textDocument/didClose", {"textDocument": {"uri": uri}})
        assert receive()["params"]["diagnostics"] == []
        send("shutdown", None, 2)
        assert receive()["id"] == 2
        send("exit", None)
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
