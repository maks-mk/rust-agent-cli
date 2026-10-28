use super::*;
use crate::agent::{AgentEvent, DiffLine, TokenUsage};
use crate::app::ToolEntry;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::Terminal;
use std::path::PathBuf;

fn config() -> Config {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let config_dir = std::env::temp_dir().join(format!(
        "rust-agent-ui-test-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    Config {
        api_mode: crate::agent::ApiMode::Chat,
        base_url: "https://example.invalid/v1".into(),
        api_key: "test".into(),
        model: "test-model".into(),
        reasoning_effort: Some("high".into()),
        reasoning_efforts: vec!["low".into(), "high".into()],
        config_dir,
        root: PathBuf::from("."),
        max_steps: crate::agent::DEFAULT_MAX_STEPS,
        context_tokens: crate::agent::DEFAULT_CONTEXT_TOKENS,
        auto_approve: false,
    }
}

#[test]
fn ascii_requires_explicit_one() {
    assert!(ascii_requested(Some("1")));
    assert!(ascii_requested(Some(" 1 ")));
    for value in [None, Some(""), Some("0"), Some("true"), Some("yes")] {
        assert!(
            !ascii_requested(value),
            "unexpected ASCII mode for {value:?}"
        );
    }
}

#[test]
fn display_path_hides_the_windows_verbatim_prefix() {
    assert_eq!(display_path(r"\\?\D:\t1"), r"D:\t1");
    assert_eq!(display_path(r"\\?\UNC\server\share"), r"\\server\share");
    assert_eq!(display_path(r"D:\t1"), r"D:\t1");
}

#[test]
fn session_menu_hides_the_verbatim_workspace_prefix() {
    let mut app = app();
    app.session_menu = Some(crate::app::SessionMenu {
        items: vec![crate::session::SessionSummary {
            id: "s1".into(),
            title: "saved session".into(),
            workspace: r"\\?\D:\t1".into(),
            model: "test-model".into(),
            updated_at: "2026-10-28T01:07:00Z".into(),
        }],
        selected: 0,
    });

    let rows = render(&mut app, 80, 24);
    assert!(
        rows.iter().any(|row| row.contains(r"D:\t1")),
        "expected the readable workspace path: {rows:?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains(r"\\?\")),
        "expected no verbatim prefix: {rows:?}"
    );
}

fn usage(prompt: u64, completion: u64, cached: Option<u64>) -> TokenUsage {
    TokenUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cached_tokens: cached,
    }
}

fn app() -> App {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    App::new(config(), tx).expect("app")
}

fn render_buffer(app: &mut App, width: u16, height: u16) -> Buffer {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
    terminal.draw(|f| draw(f, app)).expect("draw");
    terminal.backend().buffer().clone()
}

fn buffer_rows(buffer: &Buffer) -> Vec<String> {
    (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| {
                    buffer
                        .cell((x, y))
                        .map(|c| c.symbol().to_string())
                        .unwrap_or_default()
                })
                .collect::<String>()
        })
        .collect()
}

fn render(app: &mut App, width: u16, height: u16) -> Vec<String> {
    buffer_rows(&render_buffer(app, width, height))
}

fn assert_text_style(buffer: &Buffer, text: &str, foreground: Color, background: Color) {
    for (y, row) in buffer_rows(buffer).iter().enumerate() {
        if let Some(start) = row.find(text) {
            let x: usize = row[..start].chars().map(cw).sum();
            for offset in 0..text.chars().map(cw).sum::<usize>() {
                let cell = buffer.cell(((x + offset) as u16, y as u16)).unwrap();
                assert_eq!(cell.fg, foreground, "foreground of {text:?}");
                assert_eq!(cell.bg, background, "background of {text:?}");
            }
            return;
        }
    }
    panic!("text {text:?} not found in {:?}", buffer_rows(buffer));
}

#[test]
fn message_roles_and_warnings_have_distinct_styles() {
    let mut app = app();
    app.entries = vec![
        Entry::User("user-message".into()),
        Entry::Assistant("agent-message".into()),
        Entry::Tool(ToolEntry {
            id: "read".into(),
            label: "Read".into(),
            target: "Cargo.toml".into(),
            status: ToolStatus::Ok,
            diff: vec![],
            result: Some("Read 10 lines".into()),
            output: None,
        }),
        Entry::Notice("notice-message".into()),
        Entry::Warning("warning-message".into()),
        Entry::Error("error-message".into()),
    ];
    let buffer = render_buffer(&mut app, 90, 30);
    assert_text_style(&buffer, "user-message", USER_TEXT, SURFACE);
    assert!(!buffer_rows(&buffer).join("\n").contains("Agent:"));
    assert_text_style(
        &buffer,
        if ascii_mode() { "* " } else { "● " },
        ACCENT,
        BACKGROUND,
    );
    assert_text_style(&buffer, "agent-message", TEXT, BACKGROUND);
    assert_text_style(&buffer, "Read", READ, BACKGROUND);
    assert_text_style(&buffer, "Cargo.toml", MUTED, BACKGROUND);
    assert_text_style(&buffer, "Read 10 lines", OK, BACKGROUND);
    assert_text_style(&buffer, "notice-message", MUTED, BACKGROUND);
    assert_text_style(&buffer, "warning-message", WARN, BACKGROUND);
    assert_text_style(&buffer, "error-message", ERR, BACKGROUND);
}

#[test]
fn assistant_comments_wrap_with_only_a_two_column_gutter() {
    let mut app = app();
    app.entries = vec![Entry::Assistant("Проверю файлы проекта и тесты".into())];
    for width in [12, 20, 40] {
        let lines = build_chat_lines(&app, width);
        assert!(lines.iter().all(|line| line.width() <= width));
        let rows: Vec<String> = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();
        let prefix = if ascii_mode() { "* " } else { "● " };
        assert!(rows[0].starts_with(&format!("{prefix}Проверю")));
        for row in &rows[1..] {
            assert!(row.starts_with("  "));
            assert!(!row.starts_with("   "), "extra gutter: {row:?}");
        }
        let text: String = rows
            .join("")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert_eq!(text, format!("{}Проверюфайлыпроектаитесты", prefix.trim()));
    }
}

#[test]
fn wrapped_user_message_keeps_background_without_label() {
    let mut app = app();
    app.entries = vec![Entry::User(
        "alpha beta gamma delta epsilon zeta eta theta tail-word".into(),
    )];
    let lines = build_chat_lines(&app, 30);
    assert!(lines.len() > 1);
    let bar_width = glyphs().user_bar.chars().map(cw).sum::<usize>();
    let buffer = render_buffer(&mut app, 32, 20);
    assert_text_style(&buffer, "tail-word", USER_TEXT, SURFACE);
    for y in 0..lines.len() {
        for x in 1..31 {
            assert_eq!(buffer.cell((x, y as u16)).unwrap().bg, SURFACE);
        }
        // No label indent: text starts right after the bar on every row.
        let cell = buffer.cell(((1 + bar_width) as u16, y as u16)).unwrap();
        assert_eq!(cell.fg, USER_TEXT);
        assert_ne!(cell.symbol(), " ");
    }
}

#[test]
fn user_messages_have_a_green_vertical_bar() {
    let mut app = app();
    app.entries = vec![Entry::User(
        "alpha beta gamma delta epsilon zeta eta theta tail-word".into(),
    )];
    let width = 30;
    let lines = build_chat_lines(&app, width);
    assert!(lines.len() > 1);
    for line in &lines {
        assert!(line.width() <= width);
        let bar = line.spans.first().expect("user bar span");
        assert_eq!(bar.content.as_ref(), glyphs().user_bar);
        assert_eq!(bar.style.fg, Some(USER_BAR));
        assert_eq!(bar.style.bg, Some(SURFACE));
    }

    let buffer = render_buffer(&mut app, 32, 20);
    let rows = buffer_rows(&buffer);
    let start = rows
        .iter()
        .position(|row| row.contains("alpha"))
        .expect("user message");
    let bar = glyphs().user_bar.chars().next().unwrap().to_string();
    for y in start..start + lines.len() {
        let cell = buffer.cell((1, y as u16)).unwrap();
        assert_eq!(cell.symbol(), bar);
        assert_eq!(cell.fg, USER_BAR);
        assert_eq!(cell.bg, SURFACE);
    }
}

#[test]
fn wrapping_respects_narrow_widths_and_preserves_span_styles() {
    let style = fg(TEXT).bg(SURFACE);
    let line = Line::from(vec![Span::styled("界 abc 界\tde", fg(CODE))]).style(style);
    for width in 1..=20 {
        let lines = wrap_line(&line, width, 9);
        assert!(lines.iter().all(|line| line.width() <= width));
        let mut text = String::new();
        for line in lines {
            assert_eq!(line.style, style);
            for span in line.spans {
                assert_eq!(span.style.bg, Some(SURFACE));
                for c in span.content.chars().filter(|c| !c.is_whitespace()) {
                    assert_eq!(span.style.fg, Some(CODE));
                    text.push(c);
                }
            }
        }
        assert_eq!(text, if width == 1 { "?abc?de" } else { "界abc界de" });
    }
}

#[test]
fn shell_preview_is_not_duplicated_and_states_are_explicit() {
    for (status, state, color) in [
        (ToolStatus::Running, "running", ACCENT),
        (ToolStatus::Waiting, "approval", WARN),
        (ToolStatus::Ok, "done", OK),
        (ToolStatus::Failed, "failed", ERR),
        (ToolStatus::Denied, "denied", WARN),
    ] {
        let mut app = app();
        app.entries = vec![Entry::Tool(ToolEntry {
            id: "shell".into(),
            label: "Shell".into(),
            target: "echo unique-command".into(),
            status,
            diff: vec![DiffLine {
                kind: DiffKind::Context,
                text: "$ echo unique-command".into(),
            }],
            result: Some("result-marker".into()),
            output: Some("exit code: 0\nshell-output-marker".into()),
        })];
        let buffer = render_buffer(&mut app, 90, 20);
        let screen = buffer_rows(&buffer).join("\n");
        assert_eq!(screen.matches("unique-command").count(), 1);
        assert_text_style(&buffer, "Shell", WARN, BACKGROUND);
        assert_text_style(&buffer, &format!("[{state}]"), color, BACKGROUND);
        assert_text_style(&buffer, "$ echo unique-command", MUTED, BACKGROUND);
        assert!(screen.contains("shell-output-marker"), "{screen}");
        assert_text_style(&buffer, "result-marker", color, BACKGROUND);
    }
}

#[test]
fn bar_columns_never_overlap_or_leave_the_area() {
    for width in 0..=120 {
        for right_width in [0, 5, 16, 1000] {
            let area = Rect::new(3, 2, width, 1);
            let (left, right) = bar_areas(area, right_width);
            assert_eq!(left.x, area.x);
            assert_eq!(right.right(), area.right());
            assert!(left.right() <= right.x);
            assert!(right.x >= area.x);
            if right.width > 0 {
                assert!(left.right() < right.x);
            }
        }
    }
}

#[test]
fn the_footer_shows_the_reasoning_level_next_to_the_model() {
    let mut app = app();
    let buffer = render_buffer(&mut app, 120, 20);
    let rows = buffer_rows(&buffer);
    let footer = rows.last().unwrap();
    assert!(footer.contains("test-model"), "{footer}");
    assert!(footer.contains("HIGH"), "{footer}");
    // The level is shown on its own, without the word "effort".
    assert!(!footer.contains("effort"), "{footer}");
}

#[test]
fn narrow_footer_prioritizes_stop_and_denial_over_model_name() {
    let mut app = app();
    app.cfg.model = "a-very-long-model-name-that-must-not-overwrite-controls".into();
    app.running = true;
    app.auto.store(true, std::sync::atomic::Ordering::Relaxed);
    let buffer = render_buffer(&mut app, 40, 20);
    let rows = buffer_rows(&buffer);
    let footer = rows.last().unwrap();
    assert!(footer.contains("Esc stop"));
    assert!(footer.contains("AUTO ON"));
    assert!(!footer.contains("model"));
    assert_text_style(&buffer, "AUTO ON", BACKGROUND, WARN);

    let (reply, _) = tokio::sync::oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell".into(),
        prompt: "Run: echo hello".into(),
        reply,
    });
    for can_approve in [false, true] {
        app.pending.as_mut().unwrap().can_approve = can_approve;
        for width in 20..=80 {
            let mut terminal = Terminal::new(TestBackend::new(width, 1)).unwrap();
            terminal.draw(|f| draw_footer(f, &app, f.area())).unwrap();
            let footer = buffer_rows(terminal.backend().buffer()).join("");
            assert!(footer.contains("n no"), "{width}: {footer}");
            if width >= 40 {
                assert!(footer.contains(if can_approve { "y yes" } else { "PgDn" }));
            }
        }
    }
}

#[test]
fn finished_tools_do_not_leave_a_running_tools_status() {
    let mut app = app();
    app.running = true;
    app.entries = vec![Entry::Tool(ToolEntry {
        id: "tool".into(),
        label: "Read".into(),
        target: String::new(),
        status: ToolStatus::Running,
        diff: vec![],
        result: None,
        output: None,
    })];
    assert!(render(&mut app, 90, 20)
        .join("\n")
        .contains("Running tools"));
    for status in [ToolStatus::Ok, ToolStatus::Failed, ToolStatus::Denied] {
        if let Entry::Tool(tool) = &mut app.entries[0] {
            tool.status = status;
        }
        let screen = render(&mut app, 90, 20).join("\n");
        assert!(screen.contains("Waiting for model"));
        assert!(!screen.contains("Running tools"));
        assert!(screen.contains("You can draft the next message here..."));
    }
}

#[test]
fn context_usage_has_warning_and_critical_thresholds() {
    let mut app = app();
    app.entries.clear();
    app.context_limit = 100;
    for (tokens, color) in [
        (0, READ),
        (74, READ),
        (75, WARN),
        (89, WARN),
        (90, ERR),
        (110, ERR),
    ] {
        app.context_tokens = tokens;
        let buffer = render_buffer(&mut app, 100, 15);
        assert_text_style(&buffer, "Context", color, BACKGROUND);
    }
    app.context_limit = 0;
    assert!(!render(&mut app, 100, 15).join("\n").contains("Context"));
}

#[test]
fn approval_shows_thirteenth_command_line() {
    let mut app = app();
    let (reply, _rx) = tokio::sync::oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell".into(),
        prompt: format!("Run: {}HIDDEN_TAIL", "Write-Output visible\n".repeat(12)),
        reply,
    });
    let screen = render(&mut app, 100, 50).join("\n");
    assert!(screen.contains("HIDDEN_TAIL"));
    assert!(app.pending.as_ref().unwrap().can_approve);
}

#[test]
fn mcp_trust_never_enables_auto_approval() {
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    let mut app = app();
    let (reply, mut rx) = tokio::sync::oneshot::channel();
    app.on_agent(AgentEvent::McpApproval {
        prompt: "Trust server?".into(),
        reply,
    });
    let screen = render(&mut app, 80, 20).join("\n");
    assert!(screen.contains("Trust MCP server"));
    assert!(!screen.contains("a always"));
    assert!(app.pending.as_ref().unwrap().can_approve);
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Char('a'),
        KeyModifiers::NONE,
    )));
    assert!(rx.try_recv().is_err());
    assert!(!app.auto.load(std::sync::atomic::Ordering::Relaxed));
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Char('y'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(rx.try_recv(), Ok(crate::agent::Approval::Yes)));
    assert!(!app.auto.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn approval_requires_scrolling_long_scripts_and_single_lines() {
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
    for command in [
        format!("{}TAIL", "line\n".repeat(100)),
        format!("{}TAIL", "long ".repeat(300)),
    ] {
        let mut app = app();
        let (reply, mut rx) = tokio::sync::oneshot::channel();
        app.on_agent(AgentEvent::ToolApproval {
            id: "shell".into(),
            prompt: format!("Run: {command}"),
            reply,
        });
        let first = render(&mut app, 40, 15).join("\n");
        assert!(!first.contains("TAIL"));
        assert!(!app.pending.as_ref().unwrap().can_approve);
        let key = |code| Event::Key(KeyEvent::new(code, KeyModifiers::NONE));
        app.on_event(key(KeyCode::Enter));
        assert!(rx.try_recv().is_err());
        let mut tail_seen = false;
        for _ in 0..200 {
            app.on_event(key(KeyCode::PageDown));
            let page = render(&mut app, 40, 15).join("\n");
            tail_seen |= page.contains("TAIL");
            if app.pending.as_ref().unwrap().can_approve {
                break;
            }
        }
        assert!(tail_seen);
        assert!(app.pending.as_ref().unwrap().can_approve);
        app.on_event(key(KeyCode::Enter));
        assert!(matches!(rx.try_recv(), Ok(crate::agent::Approval::Yes)));
    }
}

#[test]
fn tiny_approval_stays_locked_and_control_characters_are_visible() {
    let mut app = app();
    let (reply, _) = tokio::sync::oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell".into(),
        prompt: "Run: hidden\rvisible\u{202e}tail".into(),
        reply,
    });
    for (width, height) in [(1, 1), (20, 8), (80, 3)] {
        render(&mut app, width, height);
        assert!(!app.pending.as_ref().unwrap().can_approve);
    }
    let screen = render(&mut app, 80, 15).join("\n");
    assert!(screen.contains("hidden\\rvisible\\u{202e}tail"));
}

#[test]
fn chat_gap_stays_empty_with_long_responses_and_scrolling() {
    for height in [8, 12, 24] {
        for running in [false, true] {
            let mut app = app();
            app.running = running;
            app.entries = vec![Entry::Assistant(
                (0..60)
                    .map(|i| format!("response-line {i}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )];
            render(&mut app, 80, height);
            for scroll in [0, 5] {
                app.scroll = scroll;
                let buffer = render_buffer(&mut app, 80, height);
                let rows = buffer_rows(&buffer);
                let status = rows
                    .iter()
                    .position(|row| row.contains(if running { "Writing" } else { "Ready" }))
                    .expect("status row");
                assert_eq!(app.chat_height, status - 1);
                assert!(rows[status - 2].contains("response-line"));
                assert!(rows[status - 1].trim().is_empty());
                for x in 0..buffer.area.width {
                    assert_eq!(
                        buffer.cell((x, (status - 1) as u16)).unwrap().bg,
                        BACKGROUND
                    );
                }
            }
        }
    }
}

#[test]
fn short_terminal_omits_gap_to_keep_last_chat_line() {
    let mut app = app();
    app.entries = vec![Entry::Assistant("last-response".into())];
    let rows = render(&mut app, 80, 7);
    assert_eq!(app.chat_height, 1);
    assert!(rows[0].contains("last-response"));
    assert!(rows[1].contains("Ready"));
    assert!(rows[2].contains('╭'));
}

#[test]
fn usage_line_sits_between_the_input_box_and_the_footer() {
    let mut app = app();
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });
    let rows = render(&mut app, 80, 24);

    let idx = rows
        .iter()
        .position(|r| r.contains("Chat tokens:"))
        .expect("usage row");
    assert!(
        rows[idx].contains("Chat tokens: in 10 | out 4 | total 14 | cache hit 2"),
        "{}",
        rows[idx]
    );
    assert!(
        rows[idx - 1].contains('╰'),
        "expected the input border above the usage line: {:?}",
        rows[idx - 1]
    );
    assert!(
        rows[idx + 1].contains("enter send"),
        "expected the footer below the usage line: {:?}",
        rows[idx + 1]
    );
}

#[test]
fn unknown_values_are_rendered_as_placeholders() {
    let app = app();
    let lines = token_usage_lines(&app.token_usage, 78, true);
    assert_eq!(lines.len(), 1);
    assert_eq!(
        lines[0].to_string(),
        "Chat tokens: in - | out - | total - | cache hit -"
    );

    let lines = token_usage_lines(&app.token_usage, 78, false);
    assert_eq!(
        lines[0].to_string(),
        "Chat tokens: in — | out — | total — | cache hit —"
    );
}

#[test]
fn partial_values_are_rendered_as_lower_bounds() {
    let mut app = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: false,
    });

    let lines = token_usage_lines(&app.token_usage, 78, true);
    assert_eq!(
        lines[0].to_string(),
        "Chat tokens: in >=10 | out >=4 | total >=14 | cache hit >=2"
    );

    let lines = token_usage_lines(&app.token_usage, 78, false);
    assert_eq!(
        lines[0].to_string(),
        "Chat tokens: in ≥10 | out ≥4 | total ≥14 | cache hit ≥2"
    );
}

#[test]
fn known_zero_cache_is_rendered_as_zero() {
    let mut app = app();
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(0)),
        complete: true,
    });

    let lines = token_usage_lines(&app.token_usage, 78, true);
    assert_eq!(
        lines[0].to_string(),
        "Chat tokens: in 10 | out 4 | total 14 | cache hit 0"
    );
}

#[test]
fn usage_line_wraps_on_a_narrow_terminal() {
    let mut app = app();
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });

    let lines = token_usage_lines(&app.token_usage, 30, true);
    assert!(
        lines.len() > 1,
        "expected the usage line to wrap: {lines:?}"
    );
}

#[test]
fn the_command_menu_lists_the_available_commands() {
    let mut app = app();
    app.input.set("/");
    app.sync_slash();
    assert!(app.slash_menu_visible());

    let rows = render(&mut app, 80, 24);
    let menu: Vec<&String> = rows.iter().filter(|r| r.contains("Commands /")).collect();
    assert_eq!(menu.len(), 1, "expected one menu header: {rows:?}");

    for cmd in [
        "/model",
        "/auto",
        "/clear",
        "/session",
        "/workspace",
        "/help",
        "/exit",
    ] {
        assert!(
            rows.iter().any(|r| r.contains(cmd)),
            "expected {cmd} in the menu: {rows:?}"
        );
    }
    assert!(
        rows.iter().any(|r| r.contains("show or change the model")),
        "expected the description next to the command: {rows:?}"
    );
}

#[test]
fn enter_on_a_typed_session_command_shows_the_session_list() {
    use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

    let mut app = app();
    app.input.set("/session");
    app.sync_slash();
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));

    let rows = render(&mut app, 80, 24);
    assert!(
        rows.iter().any(|row| row.contains("Sessions")),
        "expected the session list: {rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.contains(crate::session::UNTITLED)),
        "expected the saved session entry: {rows:?}"
    );
}

#[test]
fn the_command_menu_is_hidden_without_a_slash() {
    let mut app = app();
    app.input.set("hello");
    app.sync_slash();
    let rows = render(&mut app, 80, 24);
    assert!(
        !rows.iter().any(|r| r.contains("Commands /")),
        "expected no menu: {rows:?}"
    );
}

#[test]
fn short_terminal_keeps_the_input_and_the_usage_visible() {
    let mut app = app();
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });
    let rows = render(&mut app, 40, 6);

    assert!(
        rows.iter().any(|r| r.contains("Chat tokens:")),
        "expected a usage row: {rows:?}"
    );
    assert!(
        rows.iter().any(|r| r.contains("╭")),
        "expected the input box: {rows:?}"
    );
}

fn span_text(spans: &[Span<'static>]) -> String {
    spans.iter().map(|s| s.content.as_ref()).collect()
}

fn table_borders(line: &str) -> Vec<usize> {
    let ascii = glyphs().ascii;
    let marks: &[char] = if ascii {
        &['+', '|']
    } else {
        &['┌', '┬', '┐', '├', '┼', '┤', '└', '┴', '┘', '│']
    };
    line.chars()
        .enumerate()
        .filter(|(_, c)| marks.contains(c))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn markdown_tables_render_as_aligned_boxes() {
    let mut app = app();
    // Rows deliberately use uneven cell widths and inline bold.
    app.entries = vec![Entry::Assistant(
        "Results:\n\n| Name | Status | Owner |\n|:-----|:------:|------:|\n| parser | ok | alice |\n| ui | **done** | bob |\n\nafter"
            .into(),
    )];

    let width = 60;
    let lines = build_chat_lines(&app, width);
    let texts: Vec<String> = lines.iter().map(Line::to_string).collect();
    assert!(texts.iter().any(|t| t.ends_with("Results:")), "{texts:?}");
    assert!(texts.iter().any(|t| t.ends_with("after")), "{texts:?}");

    let boxed: Vec<&String> = texts
        .iter()
        .filter(|t| table_borders(t).len() > 1)
        .collect();
    // top border, header, separator, two rows, bottom border
    assert_eq!(boxed.len(), 6, "expected a bordered table: {texts:?}");

    let expected = table_borders(boxed[0]);
    let boxed_width = boxed[0].chars().count();
    for line in &boxed {
        assert!(
            line.chars().count() <= width,
            "table line {line:?} overflows width {width}"
        );
        assert_eq!(line.chars().count(), boxed_width, "ragged table: {texts:?}");
        assert_eq!(table_borders(line), expected, "columns drifted: {texts:?}");
    }

    assert!(texts.iter().any(|t| t.contains("parser")));
    assert!(
        texts.iter().any(|t| t.contains("done")),
        "inline bold should be styled, not printed: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("**")),
        "markdown markers leaked into the table: {texts:?}"
    );
}

#[test]
fn narrow_tables_shrink_columns_instead_of_breaking_alignment() {
    let mut app = app();
    app.entries = vec![Entry::Assistant(
        "| Key | Description |\n|-----|-------------|\n| alpha | a description that is far too long for a narrow chat |\n| beta | short |"
            .into(),
    )];

    for width in [26usize, 34, 48, 80] {
        let lines = build_chat_lines(&app, width);
        for line in &lines {
            assert!(
                line.width() <= width,
                "line {line:?} overflows width {width}"
            );
        }
        let texts: Vec<String> = lines.iter().map(Line::to_string).collect();
        let boxed: Vec<&String> = texts
            .iter()
            .filter(|t| table_borders(t).len() > 1)
            .collect();
        assert_eq!(
            boxed.len(),
            6,
            "expected a six-line table at width {width}: {texts:?}"
        );
        let expected = table_borders(boxed[0]);
        for line in &boxed {
            assert_eq!(
                table_borders(line),
                expected,
                "columns drifted at width {width}: {texts:?}"
            );
        }
        assert!(texts.iter().any(|t| t.contains("alpha")));
    }

    let narrow: Vec<String> = build_chat_lines(&app, 26)
        .iter()
        .map(Line::to_string)
        .collect();
    assert!(
        narrow.iter().any(|t| t.contains(glyphs().ellipsis)),
        "expected long cells to be truncated, not wrapped: {narrow:?}"
    );
}

#[test]
fn table_alignment_follows_the_separator_row() {
    let rendered = render_md(
        "| Item | Qty |\n|:---|---:|\n| apples | 12 |\n| pears | 3 |",
        40,
    );
    let texts: Vec<String> = rendered.iter().map(|(spans, _)| span_text(spans)).collect();
    assert_eq!(texts.len(), 6, "{texts:?}");

    // `Item` is left aligned, `Qty` is right aligned with the numbers below it.
    let item = texts[3].find("apples").expect("apples");
    let pears = texts[4].find("pears").expect("pears");
    assert_eq!(item, pears, "left column should line up: {texts:?}");

    let header_qty = texts[1].find("Qty").expect("Qty") + 3;
    let twelve = texts[3].find("12").expect("12") + 2;
    let three = texts[4].find("3").expect("3") + 1;
    assert_eq!(twelve, three, "right column should line up: {texts:?}");
    assert_eq!(
        header_qty, twelve,
        "header should share the right edge: {texts:?}"
    );
}

#[test]
fn table_cells_split_on_unescaped_pipes_only() {
    assert_eq!(split_cells("| a | b \\| c |"), vec!["a", "b | c"]);
    assert_eq!(split_cells("x | y"), vec!["x", "y"]);
    assert!(is_table_separator("| --- | :---: |"));
    assert!(!is_table_separator("| a | b |"));
}

#[test]
fn pasted_tables_in_user_messages_line_up() {
    let mut app = app();
    app.entries = vec![Entry::User(
        "compare:\n| a | b |\n|---|---|\n| 1 | 2 |".into(),
    )];

    let width = 40;
    let lines = build_chat_lines(&app, width);
    // The first span of a user line is the green bar.
    let texts: Vec<String> = lines
        .iter()
        .map(|l| l.spans.iter().skip(1).map(|s| s.content.as_ref()).collect())
        .collect();

    assert!(texts.iter().any(|t| t == "compare:"), "{texts:?}");
    let boxed: Vec<&String> = texts
        .iter()
        .filter(|t| table_borders(t).len() > 1)
        .collect();
    // top border, header, separator, one row, bottom border
    assert_eq!(boxed.len(), 5, "expected a bordered table: {texts:?}");

    let expected = table_borders(boxed[0]);
    for line in &boxed {
        assert_eq!(table_borders(line), expected, "columns drifted: {texts:?}");
        assert_eq!(
            line.chars().count(),
            boxed[0].chars().count(),
            "ragged table: {texts:?}"
        );
    }
    for line in &lines {
        assert!(line.width() <= width, "line overflows: {line:?}");
        assert_eq!(line.style.bg, Some(SURFACE));
    }
}

#[test]
fn pipe_text_without_a_separator_stays_plain() {
    let rendered = render_md("choose a | b here", 40);
    let text: String = rendered.iter().map(|(spans, _)| span_text(spans)).collect();
    assert_eq!(text, "choose a | b here");
    assert!(!text.contains('│'), "unexpected table markup: {text:?}");
}
