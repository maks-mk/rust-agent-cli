use super::*;
use crossterm::event::KeyEvent;
use futures_util::stream;

fn key(code: KeyCode) -> Event {
    Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
}

async fn collect(input: Vec<Event>) -> Vec<Event> {
    let mut events = stream::iter(input.into_iter().map(Ok));
    let first = events.next().await.unwrap().unwrap();
    collect_burst(&mut events, first).await.unwrap()
}

#[tokio::test]
async fn multiline_burst_keeps_blank_lines_tabs_unicode_and_trailing_enter() {
    let input = vec![
        key(KeyCode::Char('я')),
        key(KeyCode::Enter),
        key(KeyCode::Enter),
        key(KeyCode::Tab),
        key(KeyCode::Char('🦀')),
        key(KeyCode::Enter),
    ];
    assert_eq!(
        collect(input).await,
        vec![Event::Paste("я\n\n\t🦀\n".into())]
    );
}

#[tokio::test]
async fn release_records_do_not_split_a_paste() {
    let release = Event::Key(KeyEvent::new_with_kind(
        KeyCode::Enter,
        KeyModifiers::NONE,
        KeyEventKind::Release,
    ));
    assert_eq!(
        collect(vec![key(KeyCode::Char('a')), release, key(KeyCode::Enter)]).await,
        vec![Event::Paste("a\n".into())]
    );
}

#[tokio::test]
async fn isolated_keys_and_shortcuts_are_unchanged() {
    for event in [
        key(KeyCode::Enter),
        key(KeyCode::Char('y')),
        Event::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT)),
        Event::Key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Event::Paste("already bracketed\n".into()),
    ] {
        assert_eq!(collect(vec![event.clone()]).await, vec![event]);
    }
}

#[tokio::test]
async fn non_text_event_is_preserved_after_the_burst() {
    assert_eq!(
        collect(vec![
            key(KeyCode::Char('a')),
            key(KeyCode::Char('b')),
            key(KeyCode::Left)
        ])
        .await,
        vec![Event::Paste("ab".into()), key(KeyCode::Left)]
    );
}

#[tokio::test]
async fn idle_gap_leaves_later_enter_for_submission() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tx.send(Ok(key(KeyCode::Char('b')))).unwrap();
    let mut events = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (event, rx))
    })
    .boxed();
    assert_eq!(
        collect_burst(&mut events, key(KeyCode::Char('a')))
            .await
            .unwrap(),
        vec![Event::Paste("ab".into())]
    );
    tx.send(Ok(key(KeyCode::Enter))).unwrap();
    let enter = events.next().await.unwrap().unwrap();
    assert_eq!(
        collect_burst(&mut events, enter).await.unwrap(),
        vec![key(KeyCode::Enter)]
    );
}
