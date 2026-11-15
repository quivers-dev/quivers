"use strict";
/**
 * VS Code extension entry point for QVR.
 *
 * Starts a `vscode-languageclient` against the `qvr-lsp` executable
 * shipped by the `quivers[lsp]` Python extra. The TM grammar continues
 * to handle initial highlighting; semantic tokens and diagnostics
 * arrive from the language server.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.resolveServerCommand = resolveServerCommand;
exports.activate = activate;
exports.deactivate = deactivate;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const vscode = __importStar(require("vscode"));
const node_1 = require("vscode-languageclient/node");
let client;
let pending = Promise.resolve();
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
function resolveServerCommand() {
    const config = vscode.workspace.getConfiguration("qvr");
    const folders = vscode.workspace.workspaceFolders;
    const workspaceRoot = folders && folders.length > 0 ? folders[0].uri.fsPath : "";
    const configured = config.get("lsp.path", "");
    if (configured) {
        const expanded = configured.replace(/\$\{workspaceFolder\}/g, workspaceRoot);
        return expanded;
    }
    const candidates = [];
    if (workspaceRoot) {
        candidates.push(path.join(workspaceRoot, ".venv", "bin", "qvr-lsp"));
        candidates.push(path.join(workspaceRoot, ".venv", "Scripts", "qvr-lsp.exe"));
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
async function startClient() {
    const config = vscode.workspace.getConfiguration("qvr");
    if (!config.get("lsp.enabled", true)) {
        return;
    }
    const command = resolveServerCommand();
    const args = [...config.get("lsp.args", [])];
    const target = config.get("transpileTarget", "").trim();
    if (target) {
        args.push("--target", target);
    }
    const serverOptions = {
        command,
        args,
        transport: node_1.TransportKind.stdio,
    };
    const clientOptions = {
        documentSelector: [{ language: "qvr" }],
        synchronize: {
            configurationSection: "qvr",
        },
    };
    client = new node_1.LanguageClient("qvr", "QVR Language Server", serverOptions, clientOptions);
    await client.start();
}
async function activate(context) {
    deactivating = false;
    const restart = () => {
        pending = pending.then(async () => {
            await client?.stop();
            client = undefined;
            if (!deactivating) {
                await startClient();
            }
        }).catch((error) => {
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
async function deactivate() {
    deactivating = true;
    await pending;
    await client?.stop();
    client = undefined;
}
//# sourceMappingURL=extension.js.map