mod agent;
mod app;
mod background;
mod mcp;
mod session;
#[cfg(any(windows, test))]
mod terminal_input;
mod ui;

use anyhow::{bail, Result};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEventKind,
};
#[cfg(not(windows))]
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::mpsc;

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

fn should_capture_mouse(modern_terminal: bool) -> bool {
    !cfg!(windows) || modern_terminal
}

fn set_mouse_capture(enabled: bool) -> io::Result<()> {
    #[cfg(windows)]
    {
        // Crossterm's Windows backend uses Win32 console input records. Quick Edit
        // (native text selection) and mouse events conflict, so toggle both modes.
        win_console::set_mouse_capture(enabled)
    }
    #[cfg(not(windows))]
    {
        let mut stdout = io::stdout();
        if enabled {
            execute!(stdout, EnableMouseCapture)
        } else {
            execute!(stdout, DisableMouseCapture)
        }
    }
}

fn restore_terminal() {
    let mut stdout = io::stdout();
    let _ = execute!(stdout, DisableBracketedPaste);
    // Always turn mouse capture off first. On Windows this restores native Quick Edit;
    // disable_raw_mode then restores the console's original input mode.
    let _ = set_mouse_capture(false);
    let _ = disable_raw_mode();
    let _ = execute!(stdout, LeaveAlternateScreen);
}

#[cfg(windows)]
mod win_console {
    use std::{ffi::c_void, io};

    type Handle = *mut c_void;
    const STD_INPUT_HANDLE: u32 = (-10i32) as u32;
    const ENABLE_MOUSE_INPUT: u32 = 0x0010;
    const ENABLE_QUICK_EDIT_MODE: u32 = 0x0040;
    const ENABLE_EXTENDED_FLAGS: u32 = 0x0080;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn GetStdHandle(kind: u32) -> Handle;
        fn GetConsoleMode(handle: Handle, mode: *mut u32) -> i32;
        fn SetConsoleMode(handle: Handle, mode: u32) -> i32;
    }

    pub fn set_mouse_capture(enabled: bool) -> io::Result<()> {
        // Keep all other console input flags (including the raw-mode flags) unchanged.
        unsafe {
            let handle = GetStdHandle(STD_INPUT_HANDLE);
            if handle.is_null() || handle as isize == -1 {
                return Err(io::Error::last_os_error());
            }

            let mut mode = 0u32;
            if GetConsoleMode(handle, &mut mode) == 0 {
                return Err(io::Error::last_os_error());
            }

            mode |= ENABLE_EXTENDED_FLAGS;
            if enabled {
                mode |= ENABLE_MOUSE_INPUT;
                mode &= !ENABLE_QUICK_EDIT_MODE;
            } else {
                mode &= !ENABLE_MOUSE_INPUT;
                mode |= ENABLE_QUICK_EDIT_MODE;
            }

            if SetConsoleMode(handle, mode) == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = load_config()?;

    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut app = app::App::new(cfg, tx)?;

    // Classic Windows Console Host is common when the EXE is launched by double-click.
    // Start there with mouse capture off so native text selection remains available.
    // F2 toggles between native text selection and in-app mouse scrolling.
    let capture_mouse = Arc::new(AtomicBool::new(should_capture_mouse(
        std::env::var_os("WT_SESSION").is_some() || std::env::var_os("TERM_PROGRAM").is_some(),
    )));
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    // Configure mouse/selection mode explicitly. On classic Windows consoles this
    // enables Quick Edit by default; on modern terminals it enables mouse events.
    set_mouse_capture(capture_mouse.load(Ordering::SeqCst))?;
    // Bracketed paste is not supported by every Windows console; ignore failures.
    let _ = execute!(stdout, EnableBracketedPaste);

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prev_hook(info);
    }));

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    let result = run(&mut terminal, &mut app, &mut rx, &capture_mouse).await;
    restore_terminal();
    app.shutdown().await;
    result
}

async fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut app::App,
    rx: &mut mpsc::UnboundedReceiver<agent::AgentEvent>,
    capture_mouse: &Arc<AtomicBool>,
) -> Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(80));

    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(event)) => {
                    #[cfg(windows)]
                    let batch = terminal_input::collect_burst(&mut events, event).await?;
                    #[cfg(not(windows))]
                    let batch = vec![event];
                    for event in batch {
                        let toggle_mouse = matches!(
                            &event,
                            Event::Key(key)
                                if key.kind == KeyEventKind::Press && key.code == KeyCode::F(2)
                        );
                        if toggle_mouse {
                            let enabled = !capture_mouse.load(Ordering::SeqCst);
                            let changed = set_mouse_capture(enabled);
                            match changed {
                                Ok(()) => {
                                    capture_mouse.store(enabled, Ordering::SeqCst);
                                    let message = if enabled {
                                        "Mouse mode ON: use the wheel to scroll. Press F2 for text selection."
                                    } else {
                                        "Mouse mode OFF: select/copy text with the terminal. Press F2 to enable mouse scrolling."
                                    };
                                    app.entries.push(app::Entry::Notice(message.into()));
                                }
                                Err(e) => app.entries.push(app::Entry::Error(format!(
                                    "Could not change mouse mode: {e}"
                                ))),
                            }
                        } else {
                            app.on_event(event);
                        }
                    }
                },
                Some(Err(_)) | None => break,
            },
            Some(msg) = rx.recv() => {
                app.on_agent(msg);
                // Drain whatever else is queued so streaming bursts cost one redraw.
                while let Ok(m) = rx.try_recv() {
                    app.on_agent(m);
                }
            }
            _ = tick.tick() => {
                app.tick = app.tick.wrapping_add(1);
            }
        }

        app.finish_cancel(rx).await;
        if app.should_quit {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
