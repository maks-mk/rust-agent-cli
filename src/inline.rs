//! Fullscreen terminal lifecycle and event-driven rendering.
use crate::{agent, app, ui};
use anyhow::Result;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, EventStream, KeyCode, KeyEventKind,
    },
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use futures_util::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{
    io::{self, Write},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

type Console = Terminal<CrosstermBackend<io::Stdout>>;

fn restore_terminal() {
    let _ = execute!(
        io::stdout(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen,
        crossterm::cursor::Show,
        crossterm::style::ResetColor
    );
    let _ = disable_raw_mode();
}

pub async fn start(cfg: agent::Config) -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut app = app::App::new(cfg, tx)?;
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        prev_hook(info);
    }));
    enable_raw_mode()?;
    let result = async {
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            Clear(ClearType::All),
            crossterm::cursor::MoveTo(0, 0),
            EnableMouseCapture
        )?;
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        let _ = execute!(terminal.backend_mut(), EnableBracketedPaste);
        run(&mut terminal, &mut app, &mut rx).await
    }
    .await;
    restore_terminal();
    let saved = app.shutdown().await;
    result.and(saved)
}

fn frame_log() -> Option<std::fs::File> {
    let path = std::path::PathBuf::from(std::env::var_os("RUST_AGENT_FRAME_LOG")?);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|p| !matches!(p, std::path::Component::Normal(_)))
    {
        return None;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

async fn run(
    terminal: &mut Console,
    app: &mut app::App,
    rx: &mut mpsc::UnboundedReceiver<agent::AgentEvent>,
) -> Result<()> {
    let mut events = EventStream::new();
    let mut dirty = true;
    let mut last_frame = Instant::now() - Duration::from_millis(34);
    let mut next_spin = Instant::now() + Duration::from_millis(80);
    let mut log = frame_log();
    loop {
        let working = app.running || app.rendered.has_running_tool();
        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(event)) => {
                    #[cfg(windows)]
                    let batch = crate::terminal_input::collect_burst(&mut events, event).await?;
                    #[cfg(not(windows))]
                    let batch = vec![event];
                    for event in batch {
                        if matches!(&event, Event::Mouse(mouse) if !matches!(mouse.kind, crossterm::event::MouseEventKind::ScrollUp | crossterm::event::MouseEventKind::ScrollDown))
                            || matches!(&event, Event::Key(key) if key.kind == KeyEventKind::Release)
                        {
                            continue;
                        }
                        dirty = true;
                        if matches!(&event, Event::Key(key) if key.kind == KeyEventKind::Press && key.code == KeyCode::F(2)) {
                            if app.capture_mouse {
                                execute!(terminal.backend_mut(), DisableMouseCapture)?;
                            } else {
                                execute!(terminal.backend_mut(), EnableMouseCapture)?;
                            }
                            app.capture_mouse = !app.capture_mouse;
                            app.entries.push(app::Entry::Notice(if app.capture_mouse {
                                "Mouse capture ON: wheel scrolls history. F2 enables terminal selection."
                            } else {
                                "Mouse capture OFF: use terminal selection/copy. F2 enables wheel scrolling."
                            }.into()));
                        } else {
                            app.on_event(event);
                        }
                    }
                },
                Some(Err(e)) => return Err(e.into()),
                None => break,
            },
            Some(msg) = rx.recv() => {
                app.on_agent(msg);
                while let Ok(msg) = rx.try_recv() { app.on_agent(msg); }
                dirty = true;
            },
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next_spin)), if working => {
                app.tick = app.tick.wrapping_add(1);
                next_spin = Instant::now() + Duration::from_millis(80);
                dirty = true;
            },
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(last_frame + Duration::from_millis(34))), if dirty => {
                let started = Instant::now();
                if app.clear_terminal {
                    terminal.clear()?;
                    app.clear_terminal = false;
                }
                terminal.draw(|f| ui::draw(f, app))?;
                if let Some(log) = &mut log {
                    let _ = writeln!(log, "frame_us={} drawn_rows={} formatted_entries={} cached_rows={} entries={}", started.elapsed().as_micros(), app.rendered.drawn_rows(), app.rendered.formatted_entries(), app.rendered.line_count(), app.entries.len());
                }
                last_frame = Instant::now();
                dirty = false;
            }
        }
        app.finish_cancel(rx).await;
        app.poll_session_save();
        if app.should_quit {
            break;
        }
    }
    Ok(())
}
