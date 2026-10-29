mod agent;
mod app;
mod background;
mod inline;
mod mcp;
mod session;
#[cfg(any(windows, test))]
mod terminal_input;
mod ui;

use anyhow::{bail, Context, Result};
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

.env is loaded only next to the executable and overrides inherited environment
variables. Command-line flags take precedence over .env.

The agent reads and writes files inside the current workspace. Use /workspace <path>
to switch workspace; shell commands run there and need approval (see /auto).
";

/// Configuration files and sessions live next to the executable.
fn config_dir() -> Result<std::path::PathBuf> {
    let exe = std::env::current_exe().context("cannot determine the executable path")?;
    pick_config_dir(exe.parent())
}

fn pick_config_dir(exe_dir: Option<&std::path::Path>) -> Result<std::path::PathBuf> {
    exe_dir
        .map(std::path::Path::to_path_buf)
        .context("cannot determine the executable directory")
}

/// Loads only the executable-adjacent `.env`, overriding inherited variables.
fn load_env_file(config_dir: &std::path::Path) -> Result<()> {
    let path = config_dir.join(".env");
    match dotenvy::from_path_override(&path) {
        Ok(()) => Ok(()),
        Err(dotenvy::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(dotenvy::Error::Io(error)) => {
            Err(error).with_context(|| format!("cannot read .env file {}", path.display()))
        }
        // Parse errors can contain secrets from the file; do not display them.
        Err(_) => bail!("invalid .env file {}; check its syntax", path.display()),
    }
}

fn load_config() -> Result<agent::Config> {
    let config_dir = config_dir()?;
    load_env_file(&config_dir)?;
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
             Put it in the .env next to the executable ({}) or set it in your shell.\n\
             For other providers also set OPENAI_BASE_URL and OPENAI_MODEL.",
            config_dir.join(".env").display()
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
