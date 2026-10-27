//! Windows ReadConsoleInput has no paste event. Coalesce rapid text-key bursts
//! before dispatch so pasted Enter keys cannot submit messages or approvals.

use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::{Stream, StreamExt};
use std::{io, time::Duration};

// A heuristic for consoles without paste delimiters: ordinary typing separated
// by this idle gap remains key input; a rapid text burst becomes Event::Paste.
const PASTE_IDLE: Duration = Duration::from_millis(20);

fn text_char(event: &Event) -> Option<char> {
    let Event::Key(key) = event else {
        return None;
    };
    if key.kind == KeyEventKind::Release {
        return None;
    }
    match key.code {
        // Preserve AltGr text, but never turn shortcuts into pasted characters.
        KeyCode::Char(c)
            if !key.modifiers.contains(KeyModifiers::CONTROL)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            Some(c)
        }
        KeyCode::Enter if key.modifiers.is_empty() => Some('\n'),
        KeyCode::Tab if key.modifiers.is_empty() => Some('\t'),
        _ => None,
    }
}

pub async fn collect_burst<S>(events: &mut S, first: Event) -> io::Result<Vec<Event>>
where
    S: Stream<Item = io::Result<Event>> + Unpin,
{
    let Some(first_char) = text_char(&first) else {
        return Ok(vec![first]);
    };
    let mut text = String::from(first_char);
    let mut multiple = false;
    let mut trailing = None;
    let mut deadline = tokio::time::Instant::now() + PASTE_IDLE;
    loop {
        match tokio::time::timeout_at(deadline, events.next()).await {
            Ok(Some(Ok(Event::Key(key)))) if key.kind == KeyEventKind::Release => {
                // Key-up records are interleaved with pasted characters on Windows.
                // They must neither split a burst nor extend the idle deadline.
            }
            Ok(Some(Ok(event))) => {
                if let Some(c) = text_char(&event) {
                    text.push(c);
                    multiple = true;
                    deadline = tokio::time::Instant::now() + PASTE_IDLE;
                } else {
                    trailing = Some(event);
                    break;
                }
            }
            Ok(Some(Err(error))) => return Err(error),
            Ok(None) | Err(_) => break,
        }
    }
    let mut result = vec![if multiple { Event::Paste(text) } else { first }];
    result.extend(trailing);
    Ok(result)
}

#[cfg(test)]
mod tests;
