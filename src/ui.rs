//! Fullscreen rendering with cached history and a pinned input area.

use crate::agent::{Config, DiffKind, ToolStatus};
use crate::app::{App, ChatUsage, Entry, COMMANDS};
use crate::session::format_timestamp;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph};
use ratatui::Frame;
use unicode_width::UnicodeWidthChar;

// Inherit the terminal palette outside deliberately shaded controls.
#[cfg(test)]
const BACKGROUND: Color = Color::Reset;
const SURFACE: Color = Color::Rgb(30, 30, 30);
const TEXT: Color = Color::Reset;
const BORDER: Color = Color::Rgb(80, 80, 80);
const ACCENT: Color = Color::Rgb(187, 154, 247);
const USER: Color = Color::Rgb(125, 190, 255);
const USER_BAR: Color = Color::Rgb(158, 206, 106);
/// Cooler and dimmer white for the messages the user sent, so they stand out
/// from the near-white text of the agent's replies.
const USER_TEXT: Color = Color::Rgb(190, 205, 230);
const READ: Color = Color::Rgb(100, 210, 195);
const OK: Color = Color::Rgb(158, 206, 106);
const ERR: Color = Color::Rgb(247, 118, 142);
const WARN: Color = Color::Rgb(224, 175, 104);
const MUTED: Color = Color::Rgb(151, 163, 184);
/// Menus sit in the background, so they use a quieter palette than chat content.
const MENU_BORDER: Color = Color::Rgb(70, 77, 88);
const MENU_TITLE: Color = Color::Rgb(174, 183, 196);
const MENU_COMMAND: Color = Color::Rgb(166, 180, 196);
const MENU_ACTIVE: Color = Color::Rgb(137, 160, 136);
const MENU_SELECTION: Color = Color::Rgb(47, 53, 61);
const CODE: Color = Color::Rgb(125, 207, 255);

const SPINNER: [&str; 4] = ["─", "╱", "│", "╲"];
const SPINNER_ASCII: [&str; 4] = ["-", "\\", "|", "/"];

/// Decorative glyphs are limited to characters present in Consolas so the UI
/// does not fall back to boxes when the terminal uses that font. Plain ASCII
/// is used only when explicitly requested via `RUST_AGENT_ASCII=1`.
struct Glyphs {
    user: &'static str,     // prompt gutter for user/assistant messages
    user_bar: &'static str, // green rule to the left of user messages
    think: &'static str,    // gutter for reasoning entries
    err: &'static str,      // gutter for error entries
    result: &'static str,   // tool result leader
    block: &'static str,    // streaming cursor block
    quote: &'static str,    // markdown quote bar
    rule: &'static str,     // markdown horizontal rule
    ellipsis: char,
    ascii: bool,
}

pub(crate) fn ascii_mode() -> bool {
    use std::sync::OnceLock;
    static ASCII: OnceLock<bool> = OnceLock::new();
    *ASCII.get_or_init(|| ascii_requested(std::env::var("RUST_AGENT_ASCII").ok().as_deref()))
}

/// Formats a filesystem path for display. Windows canonicalization adds a
/// verbatim prefix (`\\?\`), which is noise in the UI; the underlying path used
/// for file access keeps its original form.
pub(crate) fn display_path(path: impl AsRef<std::path::Path>) -> String {
    let raw = path.as_ref().to_string_lossy();
    if let Some(rest) = raw.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = raw.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        raw.into_owned()
    }
}

fn ascii_requested(value: Option<&str>) -> bool {
    value.map(|value| value.trim() == "1").unwrap_or(false)
}

fn glyphs() -> Glyphs {
    glyphs_for(ascii_mode())
}

fn glyphs_for(ascii: bool) -> Glyphs {
    if ascii {
        Glyphs {
            user: "> ",
            user_bar: "| ",
            think: "* ",
            err: "x ",
            result: "  `- ",
            block: "|",
            quote: "> ",
            rule: "-",
            ellipsis: '.',
            ascii: true,
        }
    } else {
        Glyphs {
            user: "› ",
            user_bar: "▌ ",
            think: "◊ ",
            err: "× ",
            result: "  └─ ",
            block: "▌",
            quote: "│ ",
            rule: "─",
            ellipsis: '…',
            ascii: false,
        }
    }
}

/// A logical line plus the hanging indent used when it wraps.
type RLine = (Line<'static>, usize);

fn fg(c: Color) -> Style {
    Style::default().fg(c)
}
fn muted() -> Style {
    fg(MUTED)
}

// ─────────────────────────────────── layout ─────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    f.render_widget(Block::default().style(fg(TEXT)), area);

    let text_w = (area.width as usize).saturating_sub(4).max(1);
    let (in_lines, cursor) = layout_input(&app.input.buf, app.input.cursor, text_w);
    let input_h: u16 = if let Some(p) = &app.pending {
        approval_lines(&p.prompt, area.width.saturating_sub(2) as usize)
            .len()
            .saturating_add(4)
            .min(area.height.saturating_sub(4).max(4) as usize) as u16
    } else {
        in_lines.len().clamp(1, 8) as u16 + 2
    };

    let usage_lines = token_usage_lines(
        &app.token_usage,
        area.width.saturating_sub(2) as usize,
        ascii_mode(),
    );
    // Keep the editor and its controls usable even in a very short terminal.
    let min_input = if app.pending.is_some() { 4 } else { 3 };
    let footer_h = u16::from(area.height > min_input);
    let usage_h = (usage_lines.len() as u16).min(area.height.saturating_sub(min_input + footer_h));
    let status_h = u16::from(area.height > min_input + footer_h + usage_h);
    let input_h = input_h.min(area.height.saturating_sub(footer_h + usage_h + status_h));

    // Command completion and the session selector share a menu band.
    let session_menu = app.session_menu_visible();
    let menu_items = if session_menu {
        app.session_menu
            .as_ref()
            .map(|m| m.items.len())
            .unwrap_or(0)
    } else {
        app.slash.as_ref().map(|m| m.items.len()).unwrap_or(0)
    };
    let menu_content_lines = if session_menu {
        menu_items.saturating_mul(2)
    } else {
        menu_items
    };
    let free = area
        .height
        .saturating_sub(status_h + input_h + usage_h + footer_h + 1);
    let menu_h = if (session_menu || app.slash_menu_visible()) && app.pending.is_none() && free >= 3
    {
        (menu_content_lines as u16 + 2).min(free).min(12)
    } else {
        0
    };
    let show_menu = menu_h >= 3;

    let mut constraints = vec![Constraint::Min(0)];
    if show_menu {
        constraints.push(Constraint::Length(menu_h));
    }
    constraints.push(Constraint::Length(status_h));
    constraints.push(Constraint::Length(input_h));
    constraints.push(Constraint::Length(usage_h));
    constraints.push(Constraint::Length(footer_h));
    let chunks = Layout::vertical(constraints).split(area);

    let mut idx = 0;
    let chat_chunk = chunks[idx];
    idx += 1;
    let menu_chunk = if show_menu {
        let r = chunks[idx];
        idx += 1;
        Some(r)
    } else {
        None
    };
    let status_chunk = chunks[idx];
    idx += 1;
    let input_chunk = chunks[idx];
    idx += 1;
    let usage_chunk = chunks[idx];
    idx += 1;
    let footer_chunk = chunks[idx];

    // Separate chat from the controls without hiding its only available row.
    let chat_gap = u16::from(chat_chunk.height > 1);
    let chat = Rect {
        x: chat_chunk.x + 1,
        y: chat_chunk.y,
        width: chat_chunk.width.saturating_sub(2),
        height: chat_chunk.height.saturating_sub(chat_gap),
    };
    draw_chat(f, app, chat);
    if let Some(rect) = menu_chunk {
        if session_menu {
            draw_session_menu(f, app, rect);
        } else {
            draw_slash_menu(f, app, rect);
        }
    }
    draw_status(f, app, pad_h(status_chunk));
    draw_input(f, app, input_chunk, &in_lines, cursor);
    f.render_widget(Paragraph::new(usage_lines), pad_h(usage_chunk));
    draw_footer(f, app, pad_h(footer_chunk));
}

/// The list of commands that match what the user typed after the slash.
fn draw_slash_menu(f: &mut Frame, app: &App, area: Rect) {
    let Some(m) = &app.slash else { return };
    if m.items.is_empty() {
        return;
    }
    let items: Vec<ListItem> = m
        .items
        .iter()
        .map(|&i| {
            let cmd = &COMMANDS[i];
            let mut spans = vec![Span::styled(
                cmd.name.to_string(),
                fg(MENU_COMMAND).add_modifier(Modifier::BOLD),
            )];
            if let Some(arg) = cmd.arg {
                spans.push(Span::styled(arg.to_string(), muted()));
            }
            spans.push(Span::raw("  "));
            spans.push(Span::styled(cmd.desc.to_string(), muted()));
            ListItem::new(Line::from(spans))
        })
        .collect();
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(Line::styled(
                    format!(" Commands /{} ", m.query),
                    fg(MENU_TITLE).add_modifier(Modifier::BOLD),
                ))
                .style(fg(USER_TEXT).bg(SURFACE))
                .border_style(fg(MENU_BORDER)),
        )
        .highlight_style(fg(USER_TEXT).bg(MENU_SELECTION))
        .highlight_symbol(glyphs().user);
    let mut state = ListState::default();
    state.select(Some(m.selected));
    f.render_stateful_widget(list, area, &mut state);
}

fn draw_session_menu(f: &mut Frame, app: &App, area: Rect) {
    let Some(menu) = &app.session_menu else {
        return;
    };
    let items: Vec<ListItem> = menu
        .items
        .iter()
        .map(|session| {
            let active = session.id == app.active_session_id();
            let marker = if active { "* " } else { "  " };
            let title_style = if active {
                fg(MENU_TITLE).add_modifier(Modifier::BOLD)
            } else {
                fg(TEXT).add_modifier(Modifier::BOLD)
            };
            let title = Line::from(vec![
                Span::styled(marker, if active { fg(MENU_ACTIVE) } else { muted() }),
                Span::styled(session.title.clone(), title_style),
                Span::styled(format!("  {}", session.id), muted()),
            ]);
            let details = Line::from(vec![
                Span::raw("    "),
                Span::styled(format_timestamp(&session.updated_at), muted()),
                Span::styled("  ·  ", muted()),
                Span::styled(display_path(&session.workspace), muted()),
            ]);
            ListItem::new(vec![title, details])
        })
        .collect();
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(Line::styled(
                    " Sessions  ·  ↑/↓ select  Enter switch  Esc close ",
                    fg(MENU_TITLE).add_modifier(Modifier::BOLD),
                ))
                .style(fg(USER_TEXT).bg(SURFACE))
                .border_style(fg(MENU_BORDER)),
        )
        .highlight_style(fg(USER_TEXT).bg(MENU_SELECTION))
        .highlight_symbol(glyphs().user);
    let mut state = ListState::default();
    state.select(Some(menu.selected));
    f.render_stateful_widget(list, area, &mut state);
}

fn pad_h(r: Rect) -> Rect {
    Rect {
        x: r.x + 1,
        y: r.y,
        width: r.width.saturating_sub(2),
        height: r.height,
    }
}

// ─────────────────────────────────── chat ───────────────────────────────────

fn draw_chat(f: &mut Frame, app: &mut App, area: Rect) {
    let mut cache = std::mem::take(&mut app.rendered);
    cache.update(app, area.width as usize);
    let total = cache.lines.len();
    let height = area.height as usize;
    if app.scroll > 0 && total > app.last_total {
        app.scroll = app.scroll.saturating_add(total - app.last_total);
    }
    app.last_total = total;
    app.chat_height = height;
    app.scroll = app.scroll.min(total.saturating_sub(height));
    let end = total.saturating_sub(app.scroll);
    let start = end.saturating_sub(height);
    cache.drawn_rows = end - start;
    for (row, line) in cache.lines[start..end].iter().enumerate() {
        f.render_widget(
            line.clone(),
            Rect {
                y: area.y + row as u16,
                height: 1,
                ..area
            },
        );
    }
    app.rendered = cache;
}

#[derive(Default)]
pub struct HistoryState {
    width: Option<usize>,
    lines: Vec<Line<'static>>,
    starts: Vec<usize>,
    dynamic: std::collections::BTreeSet<usize>,
    dirty_from: Option<usize>,
    running_tools: bool,
    drawn_rows: usize,
    formatted_entries: usize,
}

impl HistoryState {
    pub fn invalidate(&mut self, index: usize) {
        self.dirty_from = Some(self.dirty_from.map_or(index, |old| old.min(index)));
    }

    pub fn remove(&mut self, index: usize) {
        self.invalidate(index);
    }

    pub fn has_running_tool(&self) -> bool {
        self.running_tools
    }
    pub fn drawn_rows(&self) -> usize {
        self.drawn_rows
    }
    pub fn formatted_entries(&self) -> usize {
        self.formatted_entries
    }
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    fn update(&mut self, app: &App, width: usize) {
        let mut first = self.starts.len().min(app.entries.len());
        if self.width != Some(width) {
            first = 0;
        }
        if let Some(index) = self.dirty_from.take() {
            first = first.min(index);
        }
        if let Some(&index) = self.dynamic.first() {
            first = first.min(index);
        }
        let row = self.starts.get(first).copied().unwrap_or(self.lines.len());
        self.lines.truncate(row);
        self.starts.truncate(first);
        self.dynamic.retain(|&index| index < first);
        self.width = Some(width);
        self.formatted_entries = 0;
        let spin_set = if ascii_mode() {
            &SPINNER_ASCII
        } else {
            &SPINNER
        };
        let spin = spin_set[app.tick as usize % spin_set.len()];
        for index in first..app.entries.len() {
            let entry = &app.entries[index];
            self.starts.push(self.lines.len());
            let active = app.streaming_entry(index);
            if active
                || matches!(entry, Entry::Tool(t) if matches!(t.status, ToolStatus::Running | ToolStatus::Waiting))
            {
                self.dynamic.insert(index);
            }
            if needs_gap(index.checked_sub(1).map(|i| &app.entries[i]), entry) {
                self.lines.push(Line::default());
            }
            self.lines
                .extend(entry_lines(app, entry, active, spin, width, false));
            self.formatted_entries += 1;
        }
        self.running_tools = self
            .dynamic
            .iter()
            .any(|&i| matches!(&app.entries[i], Entry::Tool(t) if t.status == ToolStatus::Running));
    }
}

#[cfg(test)]
fn build_chat_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut cache = HistoryState::default();
    cache.update(app, width);
    cache.lines
}
fn entry_lines(
    app: &App,
    entry: &Entry,
    active: bool,
    spin: &str,
    width: usize,
    continuation: bool,
) -> Vec<Line<'static>> {
    let g = glyphs();
    let bar_width = g.user_bar.chars().map(cw).sum::<usize>();
    let show_bar = matches!(entry, Entry::User(_)) && width > bar_width;
    let content_width = if show_bar { width - bar_width } else { width };
    let mut out = Vec::new();
    for (row, (mut line, hang)) in render_entry(app, entry, active, spin, content_width)
        .into_iter()
        .enumerate()
    {
        if continuation && row == 0 && matches!(entry, Entry::Assistant(_)) {
            line.spans[0] = Span::raw("  ");
        }
        let mut wrapped = wrap_line(&line, content_width, hang);
        if show_bar {
            let bar = Span::styled(g.user_bar, fg(USER_BAR).bg(SURFACE));
            for line in &mut wrapped {
                line.spans.insert(0, bar.clone());
            }
        }
        out.extend(wrapped);
    }
    out
}

fn needs_gap(prev: Option<&Entry>, cur: &Entry) -> bool {
    !matches!(
        (prev, cur),
        (None, _) | (Some(Entry::Tool(_)), Entry::Tool(_)) | (Some(Entry::Thinking(_)), _)
    )
}

fn render_entry(app: &App, e: &Entry, is_last: bool, spin: &str, width: usize) -> Vec<RLine> {
    let mut v: Vec<RLine> = Vec::new();
    match e {
        Entry::Banner => return banner_lines(app, width),

        Entry::User(text) => {
            // The green bar already marks user messages, so no `You:` label.
            let lines: Vec<&str> = text.split('\n').map(|l| l.trim_end_matches('\r')).collect();
            let mut i = 0;
            while i < lines.len() {
                // Pasted markdown tables are boxed as well, otherwise word
                // wrapping would shear their columns apart.
                if let Some((table, next)) = render_table_block(&lines, i, width) {
                    for spans in table {
                        v.push((Line::from(spans).style(fg(USER_TEXT).bg(SURFACE)), 0));
                    }
                    i = next;
                    continue;
                }
                v.push((
                    Line::from(vec![Span::raw(lines[i].to_string())])
                        .style(fg(USER_TEXT).bg(SURFACE)),
                    0,
                ));
                i += 1;
            }
        }

        Entry::Assistant(text) => {
            let g = glyphs();
            let prefix = if g.ascii { "* " } else { "● " };
            let gutter_width = prefix.chars().map(cw).sum::<usize>();
            // Tables are laid out for the space that is left next to the gutter.
            let mut md = render_md(text.trim_end(), width.saturating_sub(gutter_width));
            if is_last {
                if let Some((spans, _)) = md.last_mut() {
                    spans.push(Span::styled(g.block, fg(ACCENT)));
                }
            }
            for (i, (mut spans, extra)) in md.into_iter().enumerate() {
                let gutter = if i == 0 {
                    Span::styled(prefix, fg(ACCENT).add_modifier(Modifier::BOLD))
                } else {
                    Span::raw(" ".repeat(gutter_width))
                };
                spans.insert(0, gutter);
                v.push((Line::from(spans), gutter_width + extra));
            }
        }

        Entry::Thinking(text) => {
            let g = glyphs();
            let style = muted().add_modifier(Modifier::ITALIC);
            if is_last {
                let ls: Vec<&str> = text.trim().lines().collect();
                let tail = &ls[ls.len().saturating_sub(3)..];
                for (i, l) in tail.iter().enumerate() {
                    let g = if i == 0 { g.think } else { "  " };
                    v.push((
                        Line::from(vec![
                            Span::styled(g, muted()),
                            Span::styled(l.to_string(), style),
                        ]),
                        2,
                    ));
                }
            } else {
                let words = text.split_whitespace().count();
                v.push((
                    Line::from(vec![
                        Span::styled(g.think, muted()),
                        Span::styled(format!("Reasoning · {words} words"), style),
                    ]),
                    2,
                ));
            }
        }

        Entry::Tool(t) => {
            let (icon, state, color) = match t.status {
                ToolStatus::Running => (spin, "running", ACCENT),
                ToolStatus::Waiting => ("?", "approval", WARN),
                ToolStatus::Ok => (if ascii_mode() { "+" } else { "√" }, "done", OK),
                ToolStatus::Failed => ("x", "failed", ERR),
                ToolStatus::Denied => ("!", "denied", WARN),
            };
            let tool_color = match t.label.as_str() {
                "Read" => READ,
                "Write" | "Edit" | "Shell" => WARN,
                _ => ACCENT,
            };
            let shell_preview = t.label == "Shell" && !t.diff.is_empty();
            let mut header = vec![
                Span::styled(format!("{icon} "), fg(color)),
                Span::styled(t.label.clone(), fg(tool_color).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" [{state}]"), fg(color)),
            ];
            // Shell already has a command preview below; do not echo it twice.
            if !shell_preview && !t.target.is_empty() {
                header.push(Span::styled(format!("  {}", t.target), muted()));
            }
            v.push((Line::from(header), 2));
            for d in &t.diff {
                let (text, style) = match d.kind {
                    DiffKind::Add => (format!("+ {}", d.text), fg(OK)),
                    DiffKind::Del => (format!("- {}", d.text), fg(ERR)),
                    DiffKind::Context if shell_preview => (d.text.clone(), muted()),
                    DiffKind::Context => (format!("  {}", d.text), muted()),
                    DiffKind::More => (d.text.clone(), muted().add_modifier(Modifier::ITALIC)),
                };
                v.push((
                    Line::from(vec![Span::raw("     "), Span::styled(text, style)]),
                    7,
                ));
            }
            if let Some(output) = &t.output {
                let output_style = if matches!(t.status, ToolStatus::Failed) {
                    fg(ERR)
                } else {
                    muted()
                };
                for (i, line) in output.split('\n').enumerate() {
                    let prefix = if i == 0 { glyphs().result } else { "     " };
                    v.push((
                        Line::from(vec![
                            Span::styled(prefix, output_style),
                            Span::styled(line.to_string(), output_style),
                        ]),
                        7,
                    ));
                }
            }
            if let Some(r) = &t.result {
                v.push((
                    Line::from(vec![
                        Span::styled(glyphs().result, fg(color)),
                        Span::styled(r.clone(), fg(color)),
                    ]),
                    5,
                ));
            }
        }

        Entry::Notice(text) | Entry::Warning(text) => {
            let (prefix, style) = if matches!(e, Entry::Warning(_)) {
                ("! ", fg(WARN))
            } else {
                ("i ", muted())
            };
            for (i, l) in text.split('\n').enumerate() {
                v.push((
                    Line::from(vec![
                        Span::styled(if i == 0 { prefix } else { "  " }, style),
                        Span::styled(l.to_string(), style),
                    ]),
                    2,
                ));
            }
        }

        Entry::Error(text) => {
            let g = glyphs();
            for (i, l) in text.split('\n').enumerate() {
                let g = if i == 0 { g.err } else { "  " };
                v.push((
                    Line::from(vec![
                        Span::styled(g, fg(ERR).add_modifier(Modifier::BOLD)),
                        Span::styled(l.to_string(), fg(ERR)),
                    ]),
                    2,
                ));
            }
        }
    }
    v
}

// ────────────────────────────────── banner ──────────────────────────────────

fn fit(s: &str, max: usize, ellipsis: char) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().map(cw).sum::<usize>() <= max {
        return s.to_string();
    }
    let mut used = 0;
    let mut text: String = s
        .chars()
        .take_while(|&c| {
            used += cw(c);
            used < max
        })
        .collect();
    text.push(ellipsis);
    text
}

fn fit_left(s: &str, max: usize, ellipsis: char) -> String {
    if max == 0 {
        return String::new();
    }
    if s.chars().map(cw).sum::<usize>() <= max {
        return s.to_string();
    }
    let mut used = 0;
    let tail: Vec<char> = s
        .chars()
        .rev()
        .take_while(|&c| {
            used += cw(c);
            used < max
        })
        .collect();
    format!("{}{}", ellipsis, tail.into_iter().rev().collect::<String>())
}

fn banner_lines(app: &App, width: usize) -> Vec<RLine> {
    banner_with_glyphs(&app.cfg, width, glyphs())
}

fn pixel_glyph(ch: char, width: usize) -> &'static [&'static str] {
    match width {
        5 => match ch {
            'R' => &[
                "11110", "10001", "10001", "11110", "10100", "10010", "10001",
            ],
            'U' => &[
                "10001", "10001", "10001", "10001", "10001", "10001", "01110",
            ],
            'S' => &[
                "01111", "10000", "10000", "01110", "00001", "00001", "11110",
            ],
            'T' => &[
                "11111", "00100", "00100", "00100", "00100", "00100", "00100",
            ],
            'A' => &[
                "01110", "10001", "10001", "11111", "10001", "10001", "10001",
            ],
            'G' => &[
                "01110", "10001", "10000", "10111", "10001", "10001", "01110",
            ],
            'E' => &[
                "11111", "10000", "10000", "11110", "10000", "10000", "11111",
            ],
            'N' => &[
                "10001", "11001", "10101", "10101", "10011", "10001", "10001",
            ],
            _ => &[
                "00000", "00000", "00000", "00000", "00000", "00000", "00000",
            ],
        },
        4 => match ch {
            'R' => &["1110", "1001", "1001", "1110", "1010", "1001", "1001"],
            'U' => &["1001", "1001", "1001", "1001", "1001", "1001", "1111"],
            'S' => &["1111", "1000", "1000", "1110", "0001", "0001", "1110"],
            'T' => &["1111", "0110", "0110", "0110", "0110", "0110", "0110"],
            'A' => &["0110", "1001", "1001", "1111", "1001", "1001", "1001"],
            'G' => &["0111", "1000", "1000", "1011", "1001", "1001", "0111"],
            'E' => &["1111", "1000", "1000", "1110", "1000", "1000", "1111"],
            'N' => &["1001", "1101", "1101", "1011", "1011", "1001", "1001"],
            _ => &["0000", "0000", "0000", "0000", "0000", "0000", "0000"],
        },
        _ => match ch {
            'R' => &["110", "101", "101", "110", "101", "101", "101"],
            'U' => &["101", "101", "101", "101", "101", "101", "111"],
            'S' => &["111", "100", "100", "110", "001", "001", "110"],
            'T' => &["111", "010", "010", "010", "010", "010", "010"],
            'A' => &["010", "101", "101", "111", "101", "101", "101"],
            'G' => &["011", "100", "100", "101", "101", "101", "011"],
            'E' => &["111", "100", "100", "110", "100", "100", "111"],
            'N' => &["101", "111", "111", "111", "111", "101", "101"],
            _ => &["000", "000", "000", "000", "000", "000", "000"],
        },
    }
}

/// Render a pixel wordmark. Two terminal cells per pixel keep it close to square;
/// the font scales between 5x7, 4x7, and 3x7 variants to fit the terminal.
fn pixel_wordmark(ascii: bool, available: usize) -> Option<Vec<Vec<Span<'static>>>> {
    let glyph_width = if available >= 108 {
        5
    } else if available >= 90 {
        4
    } else if available >= 72 {
        3
    } else {
        return None;
    };
    let chars: Vec<char> = "RUSTAGENT".chars().collect();
    let gap_cells = chars.len() - 1 + 1; // one extra cell between RUST and AGENT
    let total_cells = chars.len() * glyph_width + gap_cells;
    let mut output = Vec::with_capacity(7);

    for y in 0..7 {
        let mut spans = Vec::new();
        let mut x = 0usize;
        for (index, ch) in chars.iter().enumerate() {
            if index > 0 {
                let gap = if index == 4 { 2 } else { 1 };
                spans.push(Span::raw("  ".repeat(gap)));
                x += gap;
            }
            let rows = pixel_glyph(*ch, glyph_width);
            for bit in rows[y].chars() {
                if bit == '1' {
                    let shade = 82u16 + (x as u16 * 126 / total_cells.max(1) as u16);
                    let gray = shade.min(208) as u8;
                    let color = Color::Rgb(gray, gray, gray);
                    let pixel = if ascii {
                        if (x + y * 3).is_multiple_of(13) {
                            "::"
                        } else {
                            "##"
                        }
                    } else if (x + y * 3).is_multiple_of(13) {
                        "▒▒"
                    } else if (x * 2 + y).is_multiple_of(9) {
                        "▓▓"
                    } else {
                        "██"
                    };
                    spans.push(Span::styled(pixel, fg(color).add_modifier(Modifier::BOLD)));
                } else {
                    spans.push(Span::raw("  "));
                }
                x += 1;
            }
        }
        output.push(spans);
    }
    Some(output)
}

fn banner_with_glyphs(cfg: &Config, width: usize, g: Glyphs) -> Vec<RLine> {
    if width < 2 {
        return Vec::new();
    }
    let cwd = display_path(&cfg.root);

    let padding = width.saturating_sub(2).saturating_div(2).min(2);
    let available = if g.ascii {
        width.saturating_sub(padding)
    } else {
        width.saturating_sub(2 + padding * 2)
    };
    let label_width = 8.min(available);
    let max_val = available.saturating_sub(label_width).min(56);
    let rows = [("folder", fit_left(&cwd, max_val, g.ellipsis))];

    let mut art_rows = pixel_wordmark(g.ascii, available).unwrap_or_else(|| {
        vec![vec![Span::styled(
            fit("RUST AGENT", available, g.ellipsis),
            fg(Color::Rgb(180, 180, 180)).add_modifier(Modifier::BOLD),
        )]]
    });
    let art_width = art_rows
        .iter()
        .map(|spans| Line::from(spans.clone()).width())
        .max()
        .unwrap_or(0);
    let title = fit(
        &format!("rust-agent v{}", env!("CARGO_PKG_VERSION")),
        available,
        g.ellipsis,
    );
    let title_width = title.chars().map(cw).sum::<usize>();
    let metadata_width = rows
        .iter()
        .map(|(_, val)| label_width + val.chars().map(cw).sum::<usize>())
        .max()
        .unwrap_or(0);
    let inner = art_width.max(title_width).max(metadata_width);

    let border = fg(Color::Rgb(110, 110, 110));
    let mut v: Vec<RLine> = Vec::new();
    if !g.ascii {
        v.push((
            Line::from(Span::styled(
                format!("╭{}╮", "─".repeat(inner + padding * 2)),
                border,
            )),
            0,
        ));
    }

    let row = |spans: Vec<Span<'static>>| -> RLine {
        let content = Line::from(spans);
        let used = content.width();
        let mut all = if g.ascii {
            vec![Span::raw(" ".repeat(padding))]
        } else {
            vec![Span::styled(format!("│{}", " ".repeat(padding)), border)]
        };
        all.extend(content.spans);
        if !g.ascii {
            all.push(Span::raw(" ".repeat(inner.saturating_sub(used) + padding)));
            all.push(Span::styled("│", border));
        }
        (Line::from(all), 0)
    };

    for spans in art_rows.drain(..) {
        v.push(row(spans));
    }
    let title_pad = inner.saturating_sub(title_width) / 2;
    v.push(row(vec![
        Span::raw(" ".repeat(title_pad)),
        Span::styled(
            title,
            fg(Color::Rgb(170, 170, 170)).add_modifier(Modifier::BOLD),
        ),
    ]));
    v.push(row(vec![]));
    for (key, val) in rows {
        let label = fit(key, label_width, g.ellipsis);
        v.push(row(vec![
            Span::styled(
                format!("{label:<label_width$}"),
                fg(Color::Rgb(135, 135, 135)),
            ),
            Span::styled(val, fg(Color::Rgb(185, 185, 185))),
        ]));
    }
    if !g.ascii {
        v.push(row(vec![]));
        v.push((
            Line::from(Span::styled(
                format!("╰{}╯", "─".repeat(inner + padding * 2)),
                border,
            )),
            0,
        ));
    }
    v
}

#[cfg(test)]
mod banner_tests;

// ───────────────────────────── markdown (light) ─────────────────────────────

fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    fn flush(spans: &mut Vec<Span<'static>>, buf: &mut String, style: Style) {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), style));
        }
    }

    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut is_bold = false;
    let mut i = 0;

    while i < chars.len() {
        let style = if is_bold {
            base.add_modifier(Modifier::BOLD)
        } else {
            base
        };
        if chars[i] == '`' {
            if let Some(j) = chars[i + 1..].iter().position(|&c| c == '`') {
                flush(&mut spans, &mut buf, style);
                let code: String = chars[i + 1..i + 1 + j].iter().collect();
                spans.push(Span::styled(code, fg(CODE).bg(SURFACE)));
                i += j + 2;
                continue;
            }
        }
        if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
            flush(&mut spans, &mut buf, style);
            is_bold = !is_bold;
            i += 2;
            continue;
        }
        buf.push(chars[i]);
        i += 1;
    }
    let style = if is_bold {
        base.add_modifier(Modifier::BOLD)
    } else {
        base
    };
    flush(&mut spans, &mut buf, style);
    spans
}

/// Returns (spans without gutter, extra hanging indent) per logical line.
fn render_md(text: &str, width: usize) -> Vec<(Vec<Span<'static>>, usize)> {
    let mut out: Vec<(Vec<Span<'static>>, usize)> = Vec::new();
    let mut in_code = false;
    let lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim_end_matches('\r');
        let t = line.trim_start();
        i += 1;

        if let Some(rest) = t.strip_prefix("```") {
            let g = glyphs();
            let (open, close) = if g.ascii {
                ("+-", "+-")
            } else {
                ("╭─", "╰─")
            };
            if in_code {
                out.push((vec![Span::styled(close, muted())], 0));
            } else {
                let lang = rest.trim();
                let label = if lang.is_empty() {
                    open.to_string()
                } else {
                    format!("{open} {lang}")
                };
                out.push((vec![Span::styled(label, muted())], 0));
            }
            in_code = !in_code;
            continue;
        }

        if in_code {
            let bar = if glyphs().ascii { "| " } else { "│ " };
            out.push((
                vec![
                    Span::styled(bar, muted()),
                    Span::styled(line.to_string(), fg(CODE)),
                ],
                2,
            ));
            continue;
        }

        // A pipe row directly above a `|---|` separator starts a markdown table.
        if let Some((table, next)) = render_table_block(&lines, i - 1, width) {
            out.extend(table.into_iter().map(|spans| (spans, 0)));
            i = next;
            continue;
        }

        let level = t.chars().take_while(|&c| c == '#').count();
        if (1..=6).contains(&level) && t[level..].starts_with(' ') {
            let style = fg(ACCENT).add_modifier(Modifier::BOLD);
            out.push((inline(t[level..].trim(), style), 0));
            continue;
        }

        let indent = line.len() - t.len();
        if let Some(rest) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")) {
            let mut spans = vec![
                Span::raw(" ".repeat(indent)),
                Span::styled(if glyphs().ascii { "- " } else { "• " }, fg(ACCENT)),
            ];
            spans.extend(inline(rest, Style::default()));
            out.push((spans, indent + 2));
            continue;
        }

        let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
        if (1..=3).contains(&digits) && t[digits..].starts_with(". ") {
            let mut spans = vec![
                Span::raw(" ".repeat(indent)),
                Span::styled(t[..digits + 2].to_string(), fg(ACCENT)),
            ];
            spans.extend(inline(&t[digits + 2..], Style::default()));
            out.push((spans, indent + digits + 2));
            continue;
        }

        if let Some(rest) = t.strip_prefix("> ") {
            let mut spans = vec![Span::styled(glyphs().quote, muted())];
            spans.extend(inline(rest, muted().add_modifier(Modifier::ITALIC)));
            out.push((spans, 2));
            continue;
        }

        if t == "---" || t == "***" {
            out.push((vec![Span::styled(glyphs().rule.repeat(24), muted())], 0));
            continue;
        }

        out.push((inline(line, Style::default()), 0));
    }
    if out.is_empty() {
        out.push((vec![], 0));
    }
    out
}

/// Column alignment declared by a table's `:---:` separator row.
#[derive(Clone, Copy, PartialEq)]
enum ColAlign {
    Left,
    Center,
    Right,
}

fn is_table_row(t: &str) -> bool {
    t.len() >= 2 && t.contains('|')
}

fn is_table_separator(t: &str) -> bool {
    if !t.contains('|') {
        return false;
    }
    let cells = split_cells(t);
    !cells.is_empty()
        && cells
            .iter()
            .all(|c| c.contains('-') && c.chars().all(|ch| ch == '-' || ch == ':'))
}

/// Split a table row into trimmed cells, honouring `\|` escapes and dropping
/// the optional leading and trailing pipes.
fn split_cells(t: &str) -> Vec<String> {
    let mut s = t.trim();
    if let Some(rest) = s.strip_prefix('|') {
        s = rest;
    }
    if let Some(rest) = s.strip_suffix('|') {
        s = rest;
    }
    let mut cells = Vec::new();
    let mut cur = String::new();
    let mut escaped = false;
    for ch in s.chars() {
        match ch {
            '|' if !escaped => cells.push(std::mem::take(&mut cur)),
            '\\' if !escaped => escaped = true,
            _ => {
                escaped = false;
                cur.push(ch);
            }
        }
    }
    if escaped {
        cur.push('\\');
    }
    cells.push(cur);
    cells.into_iter().map(|c| c.trim().to_string()).collect()
}

fn table_aligns(separator: &str) -> Vec<ColAlign> {
    split_cells(separator)
        .iter()
        .map(|c| match (c.starts_with(':'), c.ends_with(':')) {
            (true, true) => ColAlign::Center,
            (false, true) => ColAlign::Right,
            _ => ColAlign::Left,
        })
        .collect()
}

fn display_width(s: &str) -> usize {
    s.chars().map(cw).sum()
}

/// Width of a cell as it will be shown, ignoring inline markdown markers.
fn cell_width(text: &str) -> usize {
    inline(text, Style::default())
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum()
}

/// Style a cell, trim it to `width` display columns and pad it according to
/// `align`. Working on styled spans keeps `**bold**` and `` `code` `` markers
/// from stealing columns.
fn fit_cell(text: &str, width: usize, align: ColAlign, base: Style) -> Vec<Span<'static>> {
    let width = width.max(1);
    let styled = inline(&text.replace('\t', " "), base);
    let mut chars: Vec<(char, Style)> = styled
        .iter()
        .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
        .collect();

    let mut used = 0;
    let mut cut = chars.len();
    for (i, (ch, _)) in chars.iter().enumerate() {
        let w = cw(*ch);
        if used + w > width {
            cut = i;
            break;
        }
        used += w;
    }
    let truncated = cut < chars.len();
    chars.truncate(cut);
    if truncated && width > 1 {
        while used > width - 1 {
            match chars.pop() {
                Some((ch, _)) => used -= cw(ch),
                None => break,
            }
        }
        chars.push((glyphs().ellipsis, muted()));
        used += cw(glyphs().ellipsis);
    }

    let pad = width.saturating_sub(used);
    let left = match align {
        ColAlign::Left => 0,
        ColAlign::Center => pad / 2,
        ColAlign::Right => pad,
    };
    let mut out: Vec<Span<'static>> = Vec::new();
    if left > 0 {
        out.push(Span::raw(" ".repeat(left)));
    }
    let mut buf = String::new();
    let mut cur: Option<Style> = None;
    for (ch, st) in chars {
        match cur {
            Some(s) if s == st => buf.push(ch),
            _ => {
                if let Some(s) = cur {
                    out.push(Span::styled(std::mem::take(&mut buf), s));
                }
                buf.push(ch);
                cur = Some(st);
            }
        }
    }
    if let Some(s) = cur {
        out.push(Span::styled(buf, s));
    }
    if pad > left {
        out.push(Span::raw(" ".repeat(pad - left)));
    }
    out
}

/// Lay a markdown table out as bordered, column-aligned lines. Returns `None`
/// when the table cannot fit into `width` columns and should stay plain text.
fn render_table(
    rows: &[Vec<String>],
    aligns: &[ColAlign],
    width: usize,
) -> Option<Vec<Vec<Span<'static>>>> {
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    if cols == 0 || rows.is_empty() {
        return None;
    }
    // Padding around every cell plus the borders between and around columns.
    let overhead = 3 * cols + 1;
    if width < overhead + cols {
        return None;
    }

    let mut widths = vec![1usize; cols];
    for row in rows {
        for (c, cell) in row.iter().enumerate() {
            widths[c] = widths[c].max(cell_width(cell).max(1));
        }
    }
    // Shrink the widest columns until the whole table fits the chat width.
    while widths.iter().sum::<usize>() + overhead > width {
        let (widest, size) = widths
            .iter()
            .enumerate()
            .max_by_key(|(_, size)| **size)
            .map(|(idx, size)| (idx, *size))?;
        if size <= 1 {
            return None;
        }
        widths[widest] = size - 1;
    }

    let g = glyphs();
    let (h, v) = if g.ascii { ('-', '|') } else { ('─', '│') };
    let (tl, tm, tr) = if g.ascii {
        ('+', '+', '+')
    } else {
        ('┌', '┬', '┐')
    };
    let (ml, mm, mr) = if g.ascii {
        ('+', '+', '+')
    } else {
        ('├', '┼', '┤')
    };
    let (bl, bm, br) = if g.ascii {
        ('+', '+', '+')
    } else {
        ('└', '┴', '┘')
    };

    let rule = |left: char, middle: char, right: char| -> Vec<Span<'static>> {
        let mut s = String::from(left);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                s.push(middle);
            }
            s.extend(std::iter::repeat_n(h, w + 2));
        }
        s.push(right);
        vec![Span::styled(s, muted())]
    };

    let mut out: Vec<Vec<Span<'static>>> = Vec::with_capacity(rows.len() + 3);
    out.push(rule(tl, tm, tr));
    for (r, row) in rows.iter().enumerate() {
        let header = r == 0;
        let style = if header {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        let mut spans = vec![Span::styled(v.to_string(), muted())];
        for (c, w) in widths.iter().enumerate() {
            let text = row.get(c).map(String::as_str).unwrap_or("");
            let align = aligns.get(c).copied().unwrap_or(ColAlign::Left);
            spans.push(Span::raw(" "));
            spans.extend(fit_cell(text, *w, align, style));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(v.to_string(), muted()));
        }
        out.push(spans);
        if header {
            out.push(rule(ml, mm, mr));
        }
    }
    out.push(rule(bl, bm, br));
    Some(out)
}

/// Render the table whose header row is `lines[header]` (the next line must be
/// a `|---|` separator). Returns the drawn lines and the index just past the
/// table, or `None` when the block is not a table or does not fit `width`.
fn render_table_block(
    lines: &[&str],
    header: usize,
    width: usize,
) -> Option<(Vec<Vec<Span<'static>>>, usize)> {
    let separator = lines.get(header + 1)?.trim();
    if !is_table_row(lines[header].trim()) || !is_table_separator(separator) {
        return None;
    }
    let aligns = table_aligns(separator);
    let mut rows = vec![split_cells(lines[header].trim())];
    let mut next = header + 2;
    while let Some(row) = lines.get(next) {
        let row = row.trim_end_matches('\r').trim();
        if !is_table_row(row) {
            break;
        }
        rows.push(split_cells(row));
        next += 1;
    }
    render_table(&rows, &aligns, width).map(|table| (table, next))
}

// ─────────────────────────────── word wrapping ──────────────────────────────

fn cw(c: char) -> usize {
    UnicodeWidthChar::width(c).unwrap_or(0)
}

fn pad(hang: usize, style: Style) -> Vec<(char, Style)> {
    vec![(' ', style); hang]
}

/// Style-preserving word wrap with a hanging indent for continuation lines.
pub fn wrap_line(line: &Line<'static>, width: usize, hang: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    // Leave space for a wide glyph even on narrow terminals.
    let hang = hang.min(width.saturating_sub(2));

    let mut chars: Vec<(char, Style)> = Vec::new();
    for s in &line.spans {
        let style = line.style.patch(s.style);
        for c in s.content.chars() {
            if c == '\t' {
                for _ in 0..4 {
                    chars.push((' ', style));
                }
            } else if !c.is_control() {
                chars.push((if cw(c) > width { '?' } else { c }, style));
            }
        }
    }

    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0usize;
    let mut i = 0usize;

    while i < chars.len() {
        let (c, st) = chars[i];
        let w = cw(c);

        if cur_w + w > width {
            if c == ' ' {
                rows.push(std::mem::take(&mut cur));
                cur = pad(hang, line.style);
                cur_w = hang;
                i += 1;
                continue;
            }
            let brk = cur
                .iter()
                .rposition(|(ch, _)| *ch == ' ')
                .filter(|&p| p > hang);
            if let Some(p) = brk {
                let rest: Vec<(char, Style)> = cur.split_off(p + 1);
                cur.pop(); // drop the space we broke on
                rows.push(std::mem::take(&mut cur));
                cur = pad(hang, line.style);
                cur.extend(rest);
                cur_w = cur.iter().map(|(ch, _)| cw(*ch)).sum();
            } else {
                rows.push(std::mem::take(&mut cur));
                cur = pad(hang, line.style);
                cur_w = hang;
            }
            continue; // re-process the same char on the new row
        }

        cur.push((c, st));
        cur_w += w;
        i += 1;
    }
    if !cur.is_empty() || rows.is_empty() {
        rows.push(cur);
    }

    rows.into_iter()
        .map(|row| {
            let mut spans: Vec<Span<'static>> = Vec::new();
            let mut buf = String::new();
            let mut cur_style: Option<Style> = None;
            for (c, st) in row {
                match cur_style {
                    Some(s) if s == st => buf.push(c),
                    _ => {
                        if let Some(s) = cur_style {
                            spans.push(Span::styled(std::mem::take(&mut buf), s));
                        }
                        buf.push(c);
                        cur_style = Some(st);
                    }
                }
            }
            if let Some(s) = cur_style {
                spans.push(Span::styled(buf, s));
            }
            Line::from(spans).style(line.style)
        })
        .collect()
}

// ─────────────────────────────── status & footer ────────────────────────────

fn fmt_elapsed(secs: u64) -> String {
    if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

/// Fill bar for the current context size relative to the configured limit.
fn context_bar(app: &App) -> Option<String> {
    let limit = app.context_limit;
    if limit == 0 {
        return None;
    }
    let percent = (app.context_tokens as f64 / limit as f64 * 100.0).min(100.0) as usize;
    let filled = percent / 10;
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(10 - filled));
    Some(format!(
        "Context [{bar}] {percent}%  ~{} / {} tokens",
        app.context_tokens, limit
    ))
}

/// Reserve separate columns instead of painting right-aligned text over the left side.
fn bar_areas(area: Rect, right_width: usize) -> (Rect, Rect) {
    let right_width = right_width.min(area.width.saturating_sub(16) as usize) as u16;
    let left_width = area
        .width
        .saturating_sub(right_width + u16::from(right_width > 0));
    (
        Rect {
            width: left_width,
            ..area
        },
        Rect {
            x: area.right().saturating_sub(right_width),
            width: right_width,
            ..area
        },
    )
}

fn context_color(app: &App) -> Color {
    let ratio = app.context_tokens as f64 / app.context_limit.max(1) as f64;
    if ratio >= 0.9 {
        ERR
    } else if ratio >= 0.75 {
        WARN
    } else {
        READ
    }
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let hint = if app.scroll > 0 {
        let down = if ascii_mode() { "v" } else { "↓" };
        if area.width >= 70 {
            format!("{down} {} lines below | Ctrl+End", app.scroll)
        } else {
            format!("{down}{} | ^End", app.scroll)
        }
    } else {
        String::new()
    };
    let (left, right) = bar_areas(area, hint.chars().map(cw).sum());
    let color = if app.pending.is_some() {
        WARN
    } else if app.running {
        ACCENT
    } else {
        OK
    };
    let label = if app.pending.is_some() {
        "Approval required"
    } else if app.compressing_context {
        "Compressing context"
    } else if app.running {
        match app.entries.last() {
            Some(Entry::Tool(t)) if matches!(t.status, ToolStatus::Running) => "Running tools",
            Some(Entry::Thinking(_)) => "Thinking",
            Some(Entry::Assistant(_)) => "Writing",
            _ => "Waiting for model",
        }
    } else {
        "Ready"
    };
    let mut spans = Vec::new();
    if app.running {
        let spin_set: &[&str] = if ascii_mode() {
            &SPINNER_ASCII
        } else {
            &SPINNER
        };
        spans.push(Span::styled(
            format!("{} ", spin_set[app.tick as usize % spin_set.len()]),
            fg(color),
        ));
    }
    spans.push(Span::styled(label, fg(color).add_modifier(Modifier::BOLD)));
    if app.running && left.width >= 32 {
        spans.push(Span::styled(
            format!("  {}", fmt_elapsed(app.run_started.elapsed().as_secs())),
            muted(),
        ));
    }
    if let Some(bar) = context_bar(app) {
        let used: usize = spans.iter().map(Span::width).sum();
        let compact = format!(
            "Context {}%",
            (app.context_tokens as f64 / app.context_limit as f64 * 100.0) as usize
        );
        let context = if used + bar.len() + 3 <= left.width as usize {
            bar
        } else {
            compact
        };
        if used + context.len() + 3 <= left.width as usize {
            spans.push(Span::styled(" | ", muted()));
            spans.push(Span::styled(context, fg(context_color(app))));
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)), left);
    f.render_widget(
        Paragraph::new(Span::styled(hint, fg(WARN))).alignment(Alignment::Right),
        right,
    );
}

fn token_usage_lines(usage: &ChatUsage, width: usize, ascii: bool) -> Vec<Line<'static>> {
    if width < 8 {
        return Vec::new();
    }
    let value = |tokens: u64, known: bool, complete: bool| {
        if !known {
            (if ascii { "-" } else { "—" }).to_string()
        } else if complete {
            tokens.to_string()
        } else {
            format!("{}{tokens}", if ascii { ">=" } else { "≥" })
        }
    };
    let input = value(usage.prompt_tokens, usage.has_usage, usage.is_complete());
    let output = value(
        usage.completion_tokens,
        usage.has_usage,
        usage.is_complete(),
    );
    let total = value(usage.total_tokens, usage.has_usage, usage.is_complete());
    let cache = value(
        usage.cached_tokens,
        usage.has_cache,
        usage.cache_is_complete(),
    );
    let line = Line::from(vec![
        Span::styled("Chat tokens: in ", muted()),
        Span::styled(input, fg(USER)),
        Span::styled(" | out ", muted()),
        Span::styled(output, fg(ACCENT)),
        Span::styled(" | total ", muted()),
        Span::styled(total, fg(TEXT)),
        Span::styled(" | cache hit ", muted()),
        Span::styled(cache, fg(READ)),
    ]);
    wrap_line(&line, width, 0)
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let auto = app.auto.load(std::sync::atomic::Ordering::Relaxed);
    let badge = match (auto, area.width >= 55) {
        (true, true) => " AUTO-APPROVE ON ",
        (true, false) => " AUTO ON ",
        (false, true) => " APPROVALS ON ",
        (false, false) => " ASK ",
    };
    let mut right = Vec::new();
    if area.width >= 100 {
        let tools = format!("Tools: {}  ", app.tool_count);
        let model_width = (area.width as usize - 70).saturating_sub(tools.len());
        right.push(Span::styled(tools, muted()));
        right.push(Span::styled(
            fit(&app.cfg.model, model_width.min(40), glyphs().ellipsis),
            muted(),
        ));
        right.push(Span::raw("  "));
    }
    // The reasoning level sits next to the model name.
    if let Some(level) = app.effort_label() {
        right.push(Span::styled(level, fg(ACCENT)));
        right.push(Span::raw("  "));
    }
    right.push(Span::styled(
        badge,
        if auto {
            fg(Color::Black).bg(WARN).add_modifier(Modifier::BOLD)
        } else {
            fg(READ).bg(SURFACE)
        },
    ));
    let right = Line::from(right);
    let (left_area, right_area) = bar_areas(area, right.width());
    let left = if app.session_menu_visible() {
        let up = if ascii_mode() { "Up" } else { "↑" };
        let down = if ascii_mode() { "Down" } else { "↓" };
        if left_area.width >= 30 {
            format!("{up}/{down} select | Enter switch | Esc close")
        } else {
            "Enter switch | Esc close".to_string()
        }
    } else if app.slash_menu_visible() {
        let up = if ascii_mode() { "Up" } else { "↑" };
        let down = if ascii_mode() { "Down" } else { "↓" };
        if left_area.width >= 42 {
            format!("{up}/{down} select | Enter complete | Esc dismiss")
        } else if left_area.width >= 24 {
            format!("{up}/{down} select | Esc dismiss")
        } else {
            "Esc dismiss".to_string()
        }
    } else if let Some(p) = &app.pending {
        if p.can_approve && !p.session_trust && left_area.width >= 23 {
            "y yes | a always | n no".to_string()
        } else if p.can_approve && left_area.width >= 12 {
            "y yes | n no".to_string()
        } else if !p.can_approve && left_area.width >= 18 {
            "n no | PgDn review".to_string()
        } else {
            "n no".to_string()
        }
    } else if app.running {
        if left_area.width >= 37 {
            "Esc stop | PgUp/PgDn scroll | F2 mouse".to_string()
        } else {
            "Esc stop".to_string()
        }
    } else if left_area.width >= 53 {
        "enter send | Shift+Enter newline | /help | F2 select".to_string()
    } else if left_area.width >= 29 {
        "enter send | /help | F2 select".to_string()
    } else if left_area.width >= 19 {
        "enter send | /help".to_string()
    } else {
        "enter send".to_string()
    };
    f.render_widget(Paragraph::new(Span::styled(left, muted())), left_area);
    f.render_widget(
        Paragraph::new(right).alignment(Alignment::Right),
        right_area,
    );
}

// ──────────────────────────────────── input ─────────────────────────────────

/// Hard-wrap the input buffer to `width` columns and locate the cursor.
fn layout_input(buf: &[char], cursor: usize, width: usize) -> (Vec<String>, (usize, usize)) {
    let mut lines: Vec<String> = vec![String::new()];
    let mut col = 0usize;
    let (mut cx, mut cy) = (0usize, 0usize);
    let mut recorded = false;

    for (i, &ch) in buf.iter().enumerate() {
        if ch == '\n' {
            if i == cursor {
                cx = col;
                cy = lines.len() - 1;
                recorded = true;
            }
            lines.push(String::new());
            col = 0;
            continue;
        }
        let w = cw(ch);
        if col + w > width {
            lines.push(String::new());
            col = 0;
        }
        if i == cursor {
            cx = col;
            cy = lines.len() - 1;
            recorded = true;
        }
        lines.last_mut().unwrap().push(ch);
        col += w;
    }
    if !recorded {
        if col >= width {
            lines.push(String::new());
            col = 0;
        }
        cx = col;
        cy = lines.len() - 1;
    }
    (lines, (cx, cy))
}

fn approval_lines(prompt: &str, width: usize) -> Vec<String> {
    // Make control characters visible rather than allowing them to conceal text.
    let visible: Vec<char> = prompt
        .chars()
        .flat_map(|c| {
            if (c.is_control() && c != '\n')
                || matches!(c, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
            {
                c.escape_default().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect();
    layout_input(&visible, 0, width.max(1)).0
}

fn draw_input(f: &mut Frame, app: &mut App, area: Rect, lines: &[String], cursor: (usize, usize)) {
    let pending_approvals = app.pending_approval_count();
    if let Some(p) = &mut app.pending {
        let approval_title = if p.session_trust {
            " Trust MCP server — this session ".to_string()
        } else if pending_approvals > 1 {
            format!(" Approval required (+{} queued) ", pending_approvals - 1)
        } else {
            " Approval required ".to_string()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(Line::styled(
                approval_title,
                fg(WARN).add_modifier(Modifier::BOLD),
            ))
            .style(fg(USER_TEXT).bg(SURFACE))
            .border_style(fg(WARN));
        let lines = approval_lines(&p.prompt, area.width.saturating_sub(2) as usize);
        let rows = area.height.saturating_sub(4) as usize;
        p.page_rows = rows;
        p.scroll = p.scroll.min(lines.len().saturating_sub(rows.max(1)));
        let end = (p.scroll + rows).min(lines.len());
        p.can_approve = rows > 0 && area.width >= 24 && end == lines.len();
        let mut text: Vec<Line> = lines[p.scroll..end]
            .iter()
            .cloned()
            .map(Line::from)
            .collect();
        text.push(Line::styled(
            format!(
                "Lines {}-{end}/{} | PgUp/PgDn review",
                p.scroll + 1,
                lines.len()
            ),
            muted(),
        ));
        text.push(Line::styled(
            if p.can_approve && p.session_trust {
                "y trust for session | n skip server"
            } else if p.can_approve {
                "y yes | a always | n no"
            } else {
                "Review remaining text first | n no"
            },
            fg(WARN),
        ));
        f.render_widget(Paragraph::new(text).block(block), area);
        return;
    }

    let border = fg(if app.running { BORDER } else { USER });
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(Line::styled(
            if app.running {
                " Draft (send after completion) "
            } else {
                " Message "
            },
            if app.running { muted() } else { fg(USER) },
        ))
        .style(fg(USER_TEXT).bg(SURFACE))
        .border_style(border);

    if area.height < 3 || area.width < 5 {
        f.render_widget(block, area);
        return;
    }
    let rows = area.height.saturating_sub(2) as usize;
    let (cx, cy) = cursor;
    let start = (cy + 1).saturating_sub(rows);
    let end = (start + rows).min(lines.len());

    let gutter_style = if app.running {
        muted()
    } else {
        fg(USER).add_modifier(Modifier::BOLD)
    };
    let mut text: Vec<Line> = Vec::new();

    if app.input.is_empty() {
        text.push(Line::from(vec![
            Span::styled(glyphs().user, gutter_style),
            Span::styled(
                if app.running {
                    "You can draft the next message here..."
                } else {
                    "Ask anything...  (/help for commands)"
                },
                muted(),
            ),
        ]));
    } else {
        for (i, l) in lines[start..end].iter().enumerate() {
            let g = if start + i == 0 { glyphs().user } else { "  " };
            text.push(Line::from(vec![
                Span::styled(g, gutter_style),
                Span::raw(l.clone()),
            ]));
        }
    }
    f.render_widget(Paragraph::new(text).block(block), area);

    let x = area.x + 1 + 2 + cx as u16;
    let y = area.y + 1 + (cy - start) as u16;
    f.set_cursor_position((x.min(area.x + area.width.saturating_sub(2)), y));
}

#[cfg(test)]
mod inline_tests;
#[cfg(test)]
mod tests;
