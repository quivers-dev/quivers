use zed_extension_api::{self as zed, settings::LspSettings, Result};

struct QvrExtension;

impl zed::Extension for QvrExtension {
    fn new() -> Self {
        Self
    }

    fn language_server_command(
        &mut self,
        language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<zed::Command> {
        let settings = LspSettings::for_worktree(language_server_id.as_ref(), worktree)?;
        let command = match settings.binary.as_ref().and_then(|binary| binary.path.clone()) {
            Some(path) => path,
            None => worktree.which("qvr-lsp").ok_or_else(|| {
                "qvr-lsp was not found. Install quivers[lsp] and set lsp.qvr-lsp.binary.path or launch Zed from that environment.".to_string()
            })?,
        };
        let mut env: std::collections::HashMap<String, String> =
            worktree.shell_env().into_iter().collect();
        if let Some(overrides) = settings
            .binary
            .as_ref()
            .and_then(|binary| binary.env.as_ref())
        {
            env.extend(overrides.clone());
        }
        Ok(zed::Command {
            command,
            args: settings
                .binary
                .and_then(|binary| binary.arguments)
                .unwrap_or_default(),
            env: env.into_iter().collect(),
        })
    }

    fn language_server_workspace_configuration(
        &mut self,
        language_server_id: &zed::LanguageServerId,
        worktree: &zed::Worktree,
    ) -> Result<Option<zed::serde_json::Value>> {
        Ok(LspSettings::for_worktree(language_server_id.as_ref(), worktree)?.settings)
    }
}

zed::register_extension!(QvrExtension);
