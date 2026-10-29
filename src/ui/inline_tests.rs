use super::tests::app;
use super::*;
use crate::agent::AgentEvent;

fn refresh(app: &mut App, width: usize) {
    let mut cache = std::mem::take(&mut app.rendered);
    cache.update(app, width);
    app.rendered = cache;
}

#[test]
fn completed_history_is_not_reformatted_on_next_frame() {
    let mut app = app();
    app.entries = (0..1000)
        .map(|_| Entry::Assistant("words ".repeat(30)))
        .collect();
    refresh(&mut app, 60);
    assert_eq!(app.rendered.formatted_entries(), 1000);
    let lines = app.rendered.lines.clone();
    refresh(&mut app, 60);
    assert_eq!(app.rendered.formatted_entries(), 0);
    assert_eq!(app.rendered.lines, lines);
    refresh(&mut app, 20);
    assert_eq!(app.rendered.formatted_entries(), 1000);
    assert!(app.rendered.line_count() > lines.len());
}

#[test]
fn streaming_only_reformats_the_live_tail_and_discard_removes_it() {
    let mut app = app();
    app.entries = (0..100).map(|_| Entry::Notice("keep".into())).collect();
    refresh(&mut app, 60);
    app.running = true;
    app.on_agent(AgentEvent::CompletionStart);
    app.on_agent(AgentEvent::Text("partial".into()));
    refresh(&mut app, 60);
    assert_eq!(app.rendered.formatted_entries(), 1);
    app.on_agent(AgentEvent::Text(" tail".into()));
    refresh(&mut app, 60);
    assert_eq!(app.rendered.formatted_entries(), 1);
    app.on_agent(AgentEvent::CompletionDiscard);
    refresh(&mut app, 60);
    assert_eq!(app.rendered.lines, build_chat_lines(&app, 60));
    app.on_agent(AgentEvent::Text("replacement".into()));
    refresh(&mut app, 60);
    assert_eq!(app.rendered.lines, build_chat_lines(&app, 60));
}

#[test]
fn tool_completion_invalidates_cached_status() {
    let mut app = app();
    app.entries.clear();
    app.on_agent(AgentEvent::ToolStart {
        id: "t".into(),
        label: "Read".into(),
        target: "file".into(),
    });
    refresh(&mut app, 60);
    assert!(app.rendered.has_running_tool());
    app.on_agent(AgentEvent::ToolEnd {
        id: "t".into(),
        status: ToolStatus::Ok,
        summary: "finished".into(),
    });
    refresh(&mut app, 60);
    assert!(!app.rendered.has_running_tool());
    assert_eq!(app.rendered.lines, build_chat_lines(&app, 60));
    refresh(&mut app, 60);
    assert_eq!(app.rendered.formatted_entries(), 0);
}

#[test]
fn fullscreen_draw_is_bounded_by_visible_rows() {
    let mut app = app();
    app.entries = (0..1000)
        .map(|i| Entry::Notice(format!("entry {i}")))
        .collect();
    let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
    terminal.draw(|f| draw(f, &mut app)).unwrap();
    assert!(app.rendered.drawn_rows() <= 24);
    assert!(app.last_total > 1000);
    app.scroll = 20;
    terminal.draw(|f| draw(f, &mut app)).unwrap();
    assert_eq!(app.rendered.formatted_entries(), 0);
    assert_eq!(app.scroll, 20);
    app.entries.push(Entry::Notice("new entry".into()));
    terminal.draw(|f| draw(f, &mut app)).unwrap();
    assert!(app.scroll > 20);
}

#[test]
#[ignore = "manual frame benchmark; writes target/fullscreen-frame-benchmark.log"]
fn frame_cost_is_independent_of_completed_history() {
    use std::io::Write;
    let mut log = std::fs::File::create("target/fullscreen-frame-benchmark.log").unwrap();
    for count in [10, 10_000, 100_000] {
        let mut app = app();
        app.entries = (0..count)
            .map(|_| Entry::Assistant("completed response".into()))
            .collect();
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 40)).unwrap();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let start = std::time::Instant::now();
        for _ in 0..100 {
            terminal.draw(|f| draw(f, &mut app)).unwrap();
            assert_eq!(app.rendered.formatted_entries(), 0);
        }
        writeln!(
            log,
            "entries={count} mean_frame_us={}",
            start.elapsed().as_micros() / 100
        )
        .unwrap();
    }
}
