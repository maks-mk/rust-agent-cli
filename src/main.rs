mod agent;
mod app;
mod background;
mod inline;
mod mcp;
mod session;
#[cfg(any(windows, test))]
mod terminal_input;
mod ui;

use anyhow::{bail, Result};
const HELP: &str = "\
rust-agent: a terminal coding agent for any OpenAI-compatible API

USAGE:
    rust-agent [--model <name>] [--base-url <url>]

ENVIRONMENT (or a .env file next to the executable):
    OPENAI_API_KEY    API key (optional for local servers such as Ollama)
    OPENAI_BASE_URL   default: https://api.openai.com/v1
    OPENAI_MODEL      default: gpt-4o-mini
    API_MODE          chat (default) or responses
    REASONING_EFFORT  optional thinking level (e.g. low, medium, high);
                      sent in the format expected by API_MODE (chat/responses)
    REASONING_EFFORTS comma-separated levels cycled with Tab
                      (default: none,minimal,low,medium,high,xhigh,max);
                      which levels a model accepts is up to the provider
    AGENT_MAX_STEPS   max tool steps per turn (default: 25)
    AGENT_CONTEXT_TOKENS
                      context compression threshold in tokens
                      (default: 100000; 0 disables compression)
    AGENT_AUTO_APPROVE
                      initial auto-approval mode: on/off (default: off)
    RUST_AGENT_ASCII  set to 1 to force ASCII-only UI glyphs (default: Unicode)

The agent reads and writes files inside the current workspace. Use /workspace <path>
to switch workspace; shell commands run there and need approval (see /auto).
";

/// Configuration files live next to the executable. When the binary is built
/// and run from source, the crate directory is used instead.
fn config_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf));
    pick_config_dir(exe_dir.as_deref(), &manifest)
}

/// Prefers a directory that actually holds a config file: the executable
/// directory first, the crate directory second, the executable directory last.
fn pick_config_dir(
    exe_dir: Option<&std::path::Path>,
    manifest: &std::path::Path,
) -> std::path::PathBuf {
    let has_config =
        |dir: &std::path::Path| dir.join(".env").is_file() || dir.join("mcp.json").is_file();
    if let Some(dir) = exe_dir.filter(|dir| has_config(dir)) {
        return dir.to_path_buf();
    }
    if has_config(manifest) {
        return manifest.to_path_buf();
    }
    exe_dir.unwrap_or(manifest).to_path_buf()
}

/// Loads `.env` from the config directory. Without one there, the previous
/// behaviour applies: search the working directory and its parents.
fn load_env_file(config_dir: &std::path::Path) {
    let path = config_dir.join(".env");
    if path.is_file() {
        // Variables already set in the environment keep priority over the file.
        let _ = dotenvy::from_path(&path);
    } else {
        let _ = dotenvy::dotenv();
    }
}

fn load_config() -> Result<agent::Config> {
    let config_dir = config_dir();
    load_env_file(&config_dir);
    let mut model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
    let mut base_url =
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into());
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let api_mode = agent::api_mode_from_env()?;
    // Empty or unset means: do not send reasoning_effort at all.
    let reasoning_effort = std::env::var("REASONING_EFFORT")
        .ok()
        .filter(|v| !v.trim().is_empty());
    // Levels cycled with Tab; the list is chosen by the user.
    let reasoning_efforts = agent::reasoning_efforts_from_env();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" | "-m" => model = args.next().unwrap_or(model),
            "--base-url" | "-u" => base_url = args.next().unwrap_or(base_url),
            "--help" | "-h" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other} (try --help)"),
        }
    }

    if api_key.is_empty() && base_url.contains("api.openai.com") {
        bail!(
            "OPENAI_API_KEY is not set.\n\
             Put it in a .env file (OPENAI_API_KEY=sk-...) or set it in your shell.\n\
             For other providers also set OPENAI_BASE_URL and OPENAI_MODEL."
        );
    }

    Ok(agent::Config {
        api_mode,
        base_url,
        api_key,
        model,
        reasoning_effort,
        reasoning_efforts,
        config_dir,
        root: std::env::current_dir()?,
        max_steps: agent::max_steps_from_env(),
        context_tokens: agent::context_tokens_from_env(),
        auto_approve: agent::auto_approve_from_env(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    inline::start(load_config()?).await
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
