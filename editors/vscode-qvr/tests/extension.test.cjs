const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function extension(settings = {}, files = [], env = {}) {
  const clients = [];
  let listener;
  const vscode = {
    workspace: {
      workspaceFolders: [{ uri: { fsPath: "/project" } }],
      getConfiguration: () => ({ get: (key, fallback) => settings[key] ?? fallback }),
      onDidChangeConfiguration: (handler) => { listener = handler; return { dispose() {} }; },
    },
    window: { showErrorMessage: (text) => { throw new Error(text); } },
  };
  class LanguageClient {
    constructor(id, name, server, options) { Object.assign(this, {server, options, stops: 0}); clients.push(this); }
    async start() { this.started = true; }
    async stop() { this.stops++; }
  }
  const exports = {};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../out/extension.js"), "utf8"), {
    exports, process: { env }, require: (name) => ({
      fs: { existsSync: (name) => files.includes(name) }, path, vscode,
      "vscode-languageclient/node": { LanguageClient, TransportKind: { stdio: 0 } },
    })[name],
  });
  return { ...exports, clients, change: (key) => listener({ affectsConfiguration: (name) => name === key }) };
}

test("unset server setting discovers project and active environments", () => {
  assert.equal(extension({}, ["/project/.venv/bin/qvr-lsp"]).resolveServerCommand(), "/project/.venv/bin/qvr-lsp");
  assert.equal(extension({}, ["/active/Scripts/qvr-lsp.exe"], {VIRTUAL_ENV: "/active"}).resolveServerCommand(), "/active/Scripts/qvr-lsp.exe");
  assert.equal(extension().resolveServerCommand(), "qvr-lsp");
});
test("explicit server commands and workspace paths are honored", () => {
  assert.equal(extension({"lsp.path": "custom-qvr"}).resolveServerCommand(), "custom-qvr");
  assert.equal(extension({"lsp.path": "${workspaceFolder}/env/qvr-lsp"}).resolveServerCommand(), "/project/env/qvr-lsp");
});
test("activation forwards target and supports untitled documents; settings restart the client", async () => {
  const settings = {"lsp.args": ["--stdio"], transpileTarget: "stan"};
  const ext = extension(settings);
  await ext.activate({ subscriptions: [] });
  assert.deepEqual(Array.from(ext.clients[0].server.args), ["--stdio", "--target", "stan"]);
  assert.equal(ext.clients[0].options.documentSelector[0].scheme, undefined);
  settings["lsp.enabled"] = false;
  ext.change("qvr.lsp.enabled");
  await new Promise(setImmediate);
  assert.equal(ext.clients[0].stops, 1);
  assert.equal(ext.clients.length, 1);
  settings["lsp.enabled"] = true;
  ext.change("qvr.lsp.enabled");
  await new Promise(setImmediate);
  assert.equal(ext.clients.length, 2);
});

test("deactivation cannot restart a client queued by settings", async () => {
  const ext = extension();
  await ext.activate({ subscriptions: [] });
  ext.change("qvr.lsp.path");
  await ext.deactivate();
  assert.equal(ext.clients.length, 1);
  assert.ok(ext.clients[0].stops >= 1);
});
