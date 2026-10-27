use super::*;
use std::path::PathBuf;

fn config() -> Config {
    Config {
        api_mode: crate::agent::ApiMode::Chat,
        base_url: "https://example.invalid/v1".into(),
        api_key: "test".into(),
        model: "test-model".into(),
        reasoning_effort: None,
        reasoning_efforts: vec!["low".into(), "high".into()],
        config_dir: PathBuf::from("."),
        root: PathBuf::from("."),
        max_steps: crate::agent::DEFAULT_MAX_STEPS,
        context_tokens: crate::agent::DEFAULT_CONTEXT_TOKENS,
        auto_approve: false,
    }
}

fn usage(prompt: u64, completion: u64, cached: Option<u64>) -> TokenUsage {
    TokenUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        cached_tokens: cached,
    }
}

fn app() -> (App, tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    (App::new(config(), tx).expect("app"), rx)
}

#[tokio::test]
async fn windows_multiline_paste_stays_in_the_editor_until_explicit_enter() {
    use futures_util::stream;

    let (mut app, _rx) = app();
    app.input.set("prefix suffix");
    app.input.cursor = 7;
    let mut events = stream::iter([
        Ok(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        ))),
        Ok(Event::Key(KeyEvent::new(
            KeyCode::Char('я'),
            KeyModifiers::NONE,
        ))),
        Ok(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        ))),
    ]);
    let first = Event::Key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
    for event in crate::terminal_input::collect_burst(&mut events, first)
        .await
        .unwrap()
    {
        app.on_event(event);
    }
    assert_eq!(app.input.text(), "prefix a\nя\nsuffix");
    assert_eq!(app.input.cursor, 11);
    assert!(!app.running);
    assert!(app.history.is_empty());
    assert!(!app
        .entries
        .iter()
        .any(|entry| matches!(entry, Entry::User(_))));
}

#[tokio::test]
async fn windows_paste_does_not_answer_an_approval_prompt() {
    use futures_util::stream;

    let (mut app, _rx) = app();
    let (reply, mut receiver) = oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell".into(),
        prompt: "Run command".into(),
        reply,
    });
    app.pending.as_mut().unwrap().can_approve = true;
    let first = Event::Key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
    let mut events = stream::iter([Ok(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )))]);
    for event in crate::terminal_input::collect_burst(&mut events, first)
        .await
        .unwrap()
    {
        app.on_event(event);
    }
    assert!(app.pending.is_some());
    assert!(receiver.try_recv().is_err());
    assert!(app.input.is_empty());
}

#[test]
fn bracketed_paste_normalizes_line_endings_without_submitting() {
    let (mut app, _rx) = app();
    app.on_event(Event::Paste("one\r\ntwo\rthree\n\tend\n".into()));
    assert_eq!(app.input.text(), "one\ntwo\nthree\n    end\n");
    assert!(!app.running);
    assert!(app.history.is_empty());
}

#[test]
fn queued_approvals_are_shown_one_at_a_time_and_resolve_the_matching_tool() {
    let (mut app, _rx) = app();
    for id in ["shell-1", "shell-2"] {
        app.on_agent(AgentEvent::ToolStart {
            id: id.into(),
            label: "Shell".into(),
            target: id.into(),
        });
    }
    let (reply1, mut receiver1) = oneshot::channel();
    let (reply2, mut receiver2) = oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell-1".into(),
        prompt: "Run first".into(),
        reply: reply1,
    });
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell-2".into(),
        prompt: "Run second".into(),
        reply: reply2,
    });
    assert_eq!(app.pending.as_ref().unwrap().id, "shell-1");
    assert_eq!(app.pending_approval_count(), 2);
    app.pending.as_mut().unwrap().can_approve = true;
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(matches!(receiver1.try_recv(), Ok(Approval::Yes)));
    assert_eq!(app.pending.as_ref().unwrap().id, "shell-2");
    assert_eq!(app.pending_approval_count(), 1);
    assert!(matches!(
        app.tool_mut("shell-1").unwrap().status,
        ToolStatus::Running
    ));
    assert!(matches!(
        app.tool_mut("shell-2").unwrap().status,
        ToolStatus::Waiting
    ));
    app.pending.as_mut().unwrap().can_approve = true;
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Char('n'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(receiver2.try_recv(), Ok(Approval::No)));
    assert!(app.pending.is_none());
    assert_eq!(app.pending_approval_count(), 0);
}

#[test]
fn interleaved_chunks_keep_words_and_whitespace_in_one_response() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::CompletionStart);
    app.on_agent(AgentEvent::Text(" ".into()));
    app.on_agent(AgentEvent::Reasoning("Ду".into()));
    app.on_agent(AgentEvent::Text("при".into()));
    app.on_agent(AgentEvent::Reasoning("маю".into()));
    app.on_agent(AgentEvent::Text("вет мир".into()));
    assert_eq!(app.entries.len(), 3);
    assert!(matches!(&app.entries[1], Entry::Assistant(s) if s == " привет мир"));
    assert!(matches!(&app.entries[2], Entry::Thinking(s) if s == "Думаю"));
}

#[test]
fn retry_discards_only_current_attempt_and_preserves_notices() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::CompletionStart);
    app.on_agent(AgentEvent::Text("previous round".into()));
    app.on_agent(AgentEvent::ToolStart {
        id: "read".into(),
        label: "Read".into(),
        target: "file".into(),
    });
    app.on_agent(AgentEvent::CompletionStart);
    app.on_agent(AgentEvent::Reasoning("partial thought".into()));
    app.on_agent(AgentEvent::Notice("keep notice".into()));
    app.on_agent(AgentEvent::Text("partial word".into()));
    app.on_agent(AgentEvent::CompletionDiscard);
    assert_eq!(app.entries.len(), 4);
    assert_eq!(app.streamed_chars, "previous round".chars().count());
    assert!(matches!(&app.entries[1], Entry::Assistant(s) if s == "previous round"));
    assert!(matches!(&app.entries[2], Entry::Tool(_)));
    assert!(matches!(&app.entries[3], Entry::Notice(s) if s == "keep notice"));
    app.on_agent(AgentEvent::CompletionStart);
    app.on_agent(AgentEvent::Text("complete".into()));
    assert!(matches!(&app.entries[4], Entry::Assistant(s) if s == "complete"));
}

#[test]
fn ctrl_end_jumps_to_latest_without_changing_the_draft() {
    let (mut app, _rx) = app();
    app.scroll = 50;
    app.input.set("draft message");
    app.input.home();
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::End,
        KeyModifiers::CONTROL,
    )));
    assert_eq!(app.scroll, 0);
    assert_eq!(app.input.text(), "draft message");
    assert_eq!(app.input.cursor, 0);

    app.scroll = 12;
    app.on_event(Event::Key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE)));
    assert_eq!(app.scroll, 12);
    assert_eq!(app.input.cursor, app.input.buf.len());
}

#[test]
fn ctrl_end_does_not_bypass_approval_review() {
    let (mut app, _rx) = app();
    let (reply, mut rx) = oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell".into(),
        prompt: format!("Run: {}hidden-tail", "line\n".repeat(100)),
        reply,
    });
    app.scroll = 10;
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::End,
        KeyModifiers::CONTROL,
    )));
    let pending = app.pending.as_ref().unwrap();
    assert!(!pending.can_approve);
    assert_eq!(pending.scroll, 0);
    assert_eq!(app.scroll, 10);
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(rx.try_recv().is_err());
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Char('n'),
        KeyModifiers::NONE,
    )));
    assert!(matches!(rx.try_recv(), Ok(Approval::No)));
}

#[test]
fn enabling_auto_approval_adds_a_warning() {
    let (mut app, _rx) = app();
    app.command("/auto");
    assert!(app.auto.load(Ordering::Relaxed));
    assert!(matches!(app.entries.last(), Some(Entry::Warning(text)) if text.contains("ON")));
    app.command("/auto");
    assert!(!app.auto.load(Ordering::Relaxed));
    assert!(matches!(app.entries.last(), Some(Entry::Notice(text)) if text.contains("OFF")));
}

#[test]
fn configured_auto_approval_initializes_session_and_remains_toggleable() {
    for enabled in [false, true] {
        let mut cfg = config();
        cfg.auto_approve = enabled;
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = App::new(cfg, tx).expect("app");
        assert_eq!(app.auto.load(Ordering::Relaxed), enabled);
        app.command("/auto");
        assert_eq!(app.auto.load(Ordering::Relaxed), !enabled);
        app.command("/auto");
        assert_eq!(app.auto.load(Ordering::Relaxed), enabled);
    }
}

#[test]
fn tool_output_event_is_attached_to_matching_entry() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::ToolStart {
        id: "shell-1".into(),
        label: "Shell".into(),
        target: "Write-Output hi".into(),
    });
    app.on_agent(AgentEvent::ToolOutput {
        id: "shell-1".into(),
        output: "exit code: 0\nhi".into(),
    });
    assert!(matches!(
        app.entries.last(),
        Some(Entry::Tool(t)) if t.output.as_deref() == Some("exit code: 0\nhi")
    ));
}

#[test]
fn usage_sums_every_request_of_the_chat() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(20, 6, Some(5)),
        complete: true,
    });

    assert_eq!(app.token_usage.prompt_tokens, 30);
    assert_eq!(app.token_usage.completion_tokens, 10);
    assert_eq!(app.token_usage.total_tokens, 40);
    assert_eq!(app.token_usage.cached_tokens, 7);
    assert!(app.token_usage.is_complete());
    assert!(app.token_usage.cache_is_complete());
}

#[test]
fn request_without_usage_stays_unknown() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Done(Vec::new()));

    assert!(!app.token_usage.has_usage);
    assert!(!app.token_usage.is_complete());
    assert_eq!(app.token_usage.prompt_tokens, 0);
}

#[test]
fn incomplete_snapshot_keeps_known_sums_as_lower_bound() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: false,
    });

    assert!(app.token_usage.has_usage);
    assert!(!app.token_usage.is_complete());
    assert!(!app.token_usage.cache_is_complete());
}

#[test]
fn known_zero_cache_is_not_missing_data() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(0)),
        complete: true,
    });

    assert!(app.token_usage.has_cache);
    assert_eq!(app.token_usage.cached_tokens, 0);
    assert!(app.token_usage.cache_is_complete());
}

#[test]
fn cache_coverage_is_independent_from_token_coverage() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(3)),
        complete: true,
    });
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(5, 1, None),
        complete: true,
    });

    assert!(app.token_usage.is_complete());
    assert!(!app.token_usage.cache_is_complete());
    assert_eq!(app.token_usage.cached_tokens, 3);
}

#[test]
fn clear_resets_the_usage_counter() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });
    app.command("/clear");

    assert_eq!(app.token_usage, ChatUsage::default());
}

#[test]
fn clear_starts_a_new_conversation() {
    let (mut app, _rx) = app();
    app.history = vec![json!({"role":"user", "content":"hello"})];
    app.history_mark = 1;
    app.entries.push(Entry::User("hello".into()));
    app.command("/clear");

    assert!(app.history.is_empty());
    assert_eq!(app.history_mark, 0);
    assert_eq!(app.entries.len(), 1);
    assert!(matches!(app.entries[0], Entry::Banner));
}

#[test]
fn clear_drops_the_prompt_history_and_the_draft() {
    let (mut app, _rx) = app();
    app.input_history.push("first".into());
    app.hist_pos = Some(0);
    app.draft = "draft".into();
    app.last_submit = "first".into();
    app.command("/clear");

    assert!(app.input_history.is_empty());
    assert!(app.hist_pos.is_none());
    assert!(app.draft.is_empty());
    assert!(app.last_submit.is_empty());
}

#[test]
fn clear_resets_the_history_mark_for_an_interrupted_turn() {
    let (mut app, _rx) = app();
    app.history = vec![json!({"role":"user", "content":"old"})];
    app.history_mark = 1;
    app.command("/clear");

    // The first turn after the clear behaves like any other first turn: the
    // unanswered user message is dropped when the turn is interrupted.
    app.history = vec![json!({"role":"user", "content":"new"})];
    app.history_mark = 0;
    app.last_submit = "new".into();
    app.rollback_unanswered();
    assert!(app.history.is_empty());
}

#[tokio::test]
async fn cancel_consumes_synced_tool_history_without_rollback() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = App::new(config(), tx.clone()).unwrap();
    app.history = vec![json!({"role":"user", "content":"write files"})];
    app.history_mark = 0;
    app.last_submit = "write files".into();
    let history = vec![
        app.history[0].clone(),
        json!({"role":"assistant", "content":"", "tool_calls":[{"id":"first", "type":"function", "function":{"name":"write_file", "arguments":"{}"}}]}),
        json!({"role":"tool", "tool_call_id":"first", "content":"Wrote file"}),
    ];
    tx.send(AgentEvent::Sync(history.clone())).unwrap();
    app.cancel();
    app.finish_cancel(&mut rx).await;
    assert_eq!(app.history, history);
    assert!(app.input.is_empty());
}

#[tokio::test]
async fn cancel_preserves_synced_image_message_without_restoring_user_draft() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = App::new(config(), tx.clone()).unwrap();
    app.history = vec![json!({"role":"user", "content":"inspect image"})];
    app.history_mark = 0;
    app.last_submit = "inspect image".into();
    let history = vec![
        app.history[0].clone(),
        json!({"role":"assistant", "content":"", "tool_calls":[{"id":"image", "type":"function", "function":{"name":"read_images", "arguments":"{}"}}]}),
        json!({"role":"tool", "tool_call_id":"image", "content":"Read sample.png"}),
        json!({"role":"user", "content":[
            {"type":"text", "text":"Images returned by read_images (tool_call_id: image)"},
            {"type":"image_url", "image_url":{"url":"data:image/png;base64,test"}}
        ]}),
    ];
    tx.send(AgentEvent::Sync(history.clone())).unwrap();
    app.cancel();
    app.finish_cancel(&mut rx).await;
    assert_eq!(app.history, history);
    assert_eq!(app.history_mark, history.len());
    assert!(app.input.is_empty());
}

#[test]
fn cancelled_request_keeps_its_partial_usage() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = App::new(config(), tx.clone()).expect("app");
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(10, 4, Some(2)),
        complete: true,
    });
    app.on_agent(AgentEvent::UsageStart);
    app.cancel();
    tx.send(AgentEvent::Usage {
        usage: usage(7, 3, None),
        complete: false,
    })
    .unwrap();

    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(app.finish_cancel(&mut rx));

    assert_eq!(app.token_usage.prompt_tokens, 17);
    assert_eq!(app.token_usage.completion_tokens, 7);
    assert_eq!(app.token_usage.total_tokens, 24);
    assert_eq!(app.token_usage.cached_tokens, 2);
    assert!(!app.token_usage.is_complete());
    assert!(!app.token_usage.cache_is_complete());
}

#[test]
fn usage_overflow_marks_the_counter_as_partial() {
    let (mut app, _rx) = app();
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(u64::MAX, 0, None),
        complete: true,
    });
    app.on_agent(AgentEvent::UsageStart);
    app.on_agent(AgentEvent::Usage {
        usage: usage(1, 0, None),
        complete: true,
    });

    assert_eq!(app.token_usage.prompt_tokens, u64::MAX);
    assert!(!app.token_usage.is_complete());
}

// ─────────────────────────── command menu ───────────────────────────

/// Types text into the prompt one key at a time.
fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        app.on_event(Event::Key(KeyEvent::new(
            KeyCode::Char(c),
            KeyModifiers::NONE,
        )));
    }
}

fn press(app: &mut App, code: KeyCode) {
    app.on_event(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
}

/// Presses the shortcut that switches the reasoning level.
fn tab_key(app: &mut App) {
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Tab,
        KeyModifiers::NONE,
    )));
}

/// Names of the commands currently listed in the menu.
fn menu_names(app: &App) -> Vec<&'static str> {
    app.slash
        .as_ref()
        .map(|m| m.items.iter().map(|&i| COMMANDS[i].name).collect())
        .unwrap_or_default()
}

#[test]
fn a_leading_slash_opens_the_command_menu() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/");
    assert!(app.slash_menu_visible());
    assert_eq!(
        menu_names(&app),
        vec!["/model", "/auto", "/clear", "/help", "/exit"]
    );
}

#[test]
fn the_command_menu_filters_by_the_typed_prefix() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/mo");
    assert!(app.slash_menu_visible());
    assert_eq!(menu_names(&app), vec!["/model"]);
}

#[test]
fn a_matching_name_wins_over_a_matching_description() {
    let (mut app, _rx) = app();
    // "a" is a prefix of /auto and appears in several descriptions.
    type_text(&mut app, "/a");
    assert_eq!(menu_names(&app), vec!["/auto"]);
}

#[test]
fn the_description_is_searched_when_no_name_matches() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/show");
    assert!(app.slash_menu_visible());
    assert_eq!(menu_names(&app), vec!["/model", "/help"]);
}

#[test]
fn the_arrows_move_the_selection_and_wrap_around() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/");
    let count = app.slash.as_ref().unwrap().items.len();
    assert_eq!(app.slash.as_ref().unwrap().selected, 0);
    press(&mut app, KeyCode::Down);
    assert_eq!(app.slash.as_ref().unwrap().selected, 1);
    press(&mut app, KeyCode::Up);
    assert_eq!(app.slash.as_ref().unwrap().selected, 0);
    press(&mut app, KeyCode::Up);
    assert_eq!(app.slash.as_ref().unwrap().selected, count - 1);
    press(&mut app, KeyCode::Down);
    assert_eq!(app.slash.as_ref().unwrap().selected, 0);
}

#[test]
fn enter_completes_the_command_without_running_it() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/mo");
    press(&mut app, KeyCode::Enter);
    // /model takes an argument, so it is completed with a trailing space.
    assert_eq!(app.input.text(), "/model ");
    assert!(!app.slash_menu_visible());
    // The command was not executed.
    assert_eq!(app.entries.len(), 1);
    assert_eq!(app.cfg.model, "test-model");
}

#[test]
fn enter_completes_the_command_in_the_menu() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/cle");
    press(&mut app, KeyCode::Enter);
    assert_eq!(app.input.text(), "/clear");
    assert!(!app.slash_menu_visible());
    assert_eq!(app.entries.len(), 1);
}

#[test]
fn escape_hides_the_menu_until_the_request_changes() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/");
    assert!(app.slash_menu_visible());
    press(&mut app, KeyCode::Esc);
    assert!(!app.slash_menu_visible());
    // Typing more reopens the menu for the new query.
    type_text(&mut app, "m");
    assert!(app.slash_menu_visible());
    assert_eq!(menu_names(&app), vec!["/model"]);
}

#[test]
fn the_command_menu_yields_to_an_approval_prompt() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/");
    assert!(app.slash_menu_visible());
    let (reply, mut receiver) = oneshot::channel();
    app.on_agent(AgentEvent::ToolApproval {
        id: "shell-1".into(),
        prompt: "Run".into(),
        reply,
    });
    assert!(!app.slash_menu_visible());
    // Enter answers the approval instead of completing the command.
    app.pending.as_mut().unwrap().can_approve = true;
    press(&mut app, KeyCode::Enter);
    assert!(matches!(receiver.try_recv(), Ok(Approval::Yes)));
    assert_eq!(app.input.text(), "/");
    assert_eq!(app.entries.len(), 1);
}

#[test]
fn text_that_does_not_start_with_a_slash_has_no_menu() {
    let (mut app, _rx) = app();
    type_text(&mut app, "hello");
    assert!(!app.slash_menu_visible());
    assert!(app.slash.is_none());
}

#[test]
fn tab_cycles_the_reasoning_level() {
    let (mut app, _rx) = app();
    // The level that is active at startup is part of the cycle, so the first
    // press always switches to a different one.
    assert_eq!(app.cfg.reasoning_effort, None);
    // The level is only shown in the footer, no chat entry is added.
    let entries = app.entries.len();
    tab_key(&mut app);
    assert_eq!(app.cfg.reasoning_effort.as_deref(), Some("low"));
    assert_eq!(app.effort_label().as_deref(), Some("LOW"));
    assert_eq!(app.entries.len(), entries);

    tab_key(&mut app);
    assert_eq!(app.cfg.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(app.effort_label().as_deref(), Some("HIGH"));
    assert_eq!(app.entries.len(), entries);

    // The cycle wraps around to the level that was active at the start.
    tab_key(&mut app);
    assert_eq!(app.cfg.reasoning_effort, None);
    assert_eq!(app.effort_label(), None);
    assert_eq!(app.entries.len(), entries);
}

#[test]
fn tab_switches_the_level_instead_of_completing_the_command() {
    let (mut app, _rx) = app();
    type_text(&mut app, "/cle");
    tab_key(&mut app);
    assert_eq!(app.input.text(), "/cle");
    assert!(app.slash_menu_visible());
    assert_eq!(app.cfg.reasoning_effort.as_deref(), Some("low"));
}

#[test]
fn ctrl_tab_does_not_switch_the_reasoning_level() {
    let (mut app, _rx) = app();
    // Windows Terminal spends Ctrl+Tab on switching tabs, so the shortcut is
    // not bound to the reasoning level anymore.
    app.on_event(Event::Key(KeyEvent::new(
        KeyCode::Tab,
        KeyModifiers::CONTROL,
    )));
    assert_eq!(app.cfg.reasoning_effort, None);
    assert_eq!(app.effort_label(), None);
}

#[test]
fn a_single_configured_level_cannot_be_switched() {
    let (mut app, _rx) = app();
    app.effort_levels = vec![Some("high".into())];
    app.effort_index = 0;
    tab_key(&mut app);
    assert_eq!(app.cfg.reasoning_effort.as_deref(), Some("high"));
    assert!(matches!(
        app.entries.last(),
        Some(Entry::Notice(msg)) if msg.contains("REASONING_EFFORTS")
    ));
}
