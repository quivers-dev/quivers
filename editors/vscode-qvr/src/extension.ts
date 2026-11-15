/**
 * VS Code extension entry point for QVR.
 *
 * Starts a `vscode-languageclient` against the `qvr-lsp` executable
 * shipped by the `quivers[lsp]` Python extra. The TM grammar continues
 * to handle initial highlighting; semantic tokens and diagnostics
 * arrive from the language server.
 */

import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
let pending: Promise<void> = Promise.resolve();
let deactivating = false;

/**
 * Locate the `qvr-lsp` executable.
 *
 * Resolution order:
 *  1. `qvr.lsp.path` setting (literal path; `${workspaceFolder}` is
 *     expanded). An explicit command is honored, including PATH commands.
 *  2. `<workspace>/.venv/bin/qvr-lsp` (uv / venv convention).
 *  3. `<workspace>/.venv/Scripts/qvr-lsp.exe` (Windows venv).
 *  4. `VIRTUAL_ENV/bin/qvr-lsp` if the env var is set.
 *  5. The literal string `"qvr-lsp"`, letting the OS PATH resolve it.
 *     (Useful for system-wide installs.)
 */
export function resolveServerCommand(): string {
  const config = vscode.workspace.getConfiguration("qvr");
  const folders = vscode.workspace.workspaceFolders;
  const workspaceRoot = folders && folders.length > 0 ? folders[0].uri.fsPath : "";

  const configured = config.get<string>("lsp.path", "");
  if (configured) {
    const expanded = configured.replace(
      /\$\{workspaceFolder\}/g,
      workspaceRoot
    );
    return expanded;
  }

  const candidates: string[] = [];
  if (workspaceRoot) {
    candidates.push(path.join(workspaceRoot, ".venv", "bin", "qvr-lsp"));
    candidates.push(
      path.join(workspaceRoot, ".venv", "Scripts", "qvr-lsp.exe")
    );
  }
  const virtualEnv = process.env.VIRTUAL_ENV;
  if (virtualEnv) {
    candidates.push(path.join(virtualEnv, "bin", "qvr-lsp"));
    candidates.push(path.join(virtualEnv, "Scripts", "qvr-lsp.exe"));
  }
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return "qvr-lsp";
}

async function startClient(): Promise<void> {
  const config = vscode.workspace.getConfiguration("qvr");
  if (!config.get<boolean>("lsp.enabled", true)) {
    return;
  }
  const command = resolveServerCommand();
  const args = [...config.get<string[]>("lsp.args", [])];
  const target = config.get<string>("transpileTarget", "").trim();
  if (target) {
    args.push("--target", target);
  }

  const serverOptions: ServerOptions = {
    command,
    args,
    transport: TransportKind.stdio,
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ language: "qvr" }],
    synchronize: {
      configurationSection: "qvr",
    },
  };

  client = new LanguageClient(
    "qvr",
    "QVR Language Server",
    serverOptions,
    clientOptions
  );
  await client.start();
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  deactivating = false;
  const restart = (): Promise<void> => {
    pending = pending.then(async () => {
      await client?.stop();
      client = undefined;
      if (!deactivating) {
        await startClient();
      }
    }).catch((error: unknown) => {
      void vscode.window.showErrorMessage(`QVR language server: ${String(error)}`);
    });
    return pending;
  };
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((event) => {
    if (["qvr.lsp.path", "qvr.lsp.args", "qvr.lsp.enabled"].some((key) => event.affectsConfiguration(key))) {
      void restart();
    }
  }));
  await restart();
}

export async function deactivate(): Promise<void> {
  deactivating = true;
  await pending;
  await client?.stop();
  client = undefined;
}
