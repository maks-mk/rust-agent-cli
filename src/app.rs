//! Application state: chat entries, input editor, key handling, and the glue
//! between the UI and the background agent task.

use crate::agent::{self, AgentEvent, Approval, Config, DiffLine, TokenUsage, ToolStatus, Tx};
use crate::mcp::McpManager;
use crate::session::{self, Session, SessionStore, SessionSummary, UNTITLED};
use crate::ui::{ascii_mode, display_path};
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

// ─────────────────────────────── chat entries ───────────────────────────────

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolEntry {
    pub id: String,
    pub label: String,
    pub target: String,
    pub status: ToolStatus,
    pub diff: Vec<DiffLine>,
    pub result: Option<String>,
    pub output: Option<String>,
}

pub enum Entry {
    Banner,
    User(String),
    Assistant(String),
    Thinking(String),
    Tool(ToolEntry),
    Notice(String),
    Warning(String),
    Error(String),
}

pub struct Pending {
    pub id: String,
    pub prompt: String,
    pub scroll: usize,
    pub page_rows: usize,
    pub can_approve: bool,
    pub session_trust: bool,
    reply: oneshot::Sender<Approval>,
}

// ─────────────────────────────── input editor ───────────────────────────────

#[derive(Default)]
pub struct Input {
    pub buf: Vec<char>,
    pub cursor: usize,
}

impl Input {
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    pub fn text(&self) -> String {
        self.buf.iter().collect()
    }
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
    }
    pub fn take(&mut self) -> String {
        let s = self.text();
        self.clear();
        s
    }
    pub fn set(&mut self, s: &str) {
        self.buf = s.chars().collect();
        self.cursor = self.buf.len();
    }
    pub fn insert_char(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += 1;
    }
    pub fn insert_str(&mut self, s: &str) {
        for c in s.replace('\t', "    ").chars() {
            if c == '\n' || !c.is_control() {
                self.insert_char(c);
            }
        }
    }
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.buf.remove(self.cursor);
        }
    }
    pub fn delete(&mut self) {
        if self.cursor < self.buf.len() {
            self.buf.remove(self.cursor);
        }
    }
    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.buf.len());
    }
    fn line_start(&self, pos: usize) -> usize {
        let mut i = pos;
        while i > 0 && self.buf[i - 1] != '\n' {
            i -= 1;
        }
        i
    }
    fn line_end(&self, pos: usize) -> usize {
        let mut i = pos;
        while i < self.buf.len() && self.buf[i] != '\n' {
            i += 1;
        }
        i
    }
    pub fn home(&mut self) {
        self.cursor = self.line_start(self.cursor);
    }
    pub fn end(&mut self) {
        self.cursor = self.line_end(self.cursor);
    }
    pub fn kill_to_line_start(&mut self) {
        let ls = self.line_start(self.cursor);
        self.buf.drain(ls..self.cursor);
        self.cursor = ls;
    }
    pub fn delete_word_back(&mut self) {
        let mut i = self.cursor;
        while i > 0 && self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        self.buf.drain(i..self.cursor);
        self.cursor = i;
    }
    /// Move the cursor one line up. Returns false if already on the first line.
    pub fn up(&mut self) -> bool {
        let ls = self.line_start(self.cursor);
        if ls == 0 {
            return false;
        }
        let col = self.cursor - ls;
        let prev_end = ls - 1;
        let prev_start = self.line_start(prev_end);
        self.cursor = (prev_start + col).min(prev_end);
        true
    }
    /// Move the cursor one line down. Returns false if already on the last line.
    pub fn down(&mut self) -> bool {
        let le = self.line_end(self.cursor);
        if le >= self.buf.len() {
            return false;
        }
        let col = self.cursor - self.line_start(self.cursor);
        let next_start = le + 1;
        let next_end = self.line_end(next_start);
        self.cursor = (next_start + col).min(next_end);
        true
    }
}

// ────────────────────────────── slash commands ──────────────────────────────

/// A command that can be typed at the prompt and picked from the command menu.
pub struct CommandSpec {
    /// Canonical name, including the leading slash.
    pub name: &'static str,
    /// Short description shown next to the name in the menu.
    pub desc: &'static str,
    /// Placeholder of the command argument, if the command takes one.
    pub arg: Option<&'static str>,
    /// Alternative names that run the same command.
    pub aliases: &'static [&'static str],
}

/// Every command the prompt understands.
pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/model",
        desc: "show or change the model",
        arg: Some("[name]"),
        aliases: &[],
    },
    CommandSpec {
        name: "/auto",
        desc: "toggle auto-approve for writes and shell commands",
        arg: None,
        aliases: &[],
    },
    CommandSpec {
        name: "/clear",
        desc: "start a new conversation",
        arg: None,
        aliases: &[],
    },
    CommandSpec {
        name: "/session",
        desc: "list or switch saved sessions",
        arg: Some("[id]"),
        aliases: &[],
    },
    CommandSpec {
        name: "/workspace",
        desc: "change workspace directory",
        arg: Some("[path]"),
        aliases: &[],
    },
    CommandSpec {
        name: "/help",
        desc: "show commands and key bindings",
        arg: None,
        aliases: &[],
    },
    CommandSpec {
        name: "/exit",
        desc: "quit",
        arg: None,
        aliases: &["/quit", "/q"],
    },
];

impl CommandSpec {
    /// Whether the command name or one of its aliases starts with the query.
    fn name_matches(&self, query: &str) -> bool {
        let q = query.trim_start_matches('/').to_lowercase();
        if q.is_empty() {
            return true;
        }
        let starts = |name: &str| name.trim_start_matches('/').to_lowercase().starts_with(&q);
        starts(self.name) || self.aliases.iter().any(|a| starts(a))
    }

    /// Whether the description contains the query.
    fn desc_matches(&self, query: &str) -> bool {
        let q = query.trim_start_matches('/').to_lowercase();
        if q.is_empty() {
            return true;
        }
        self.desc.to_lowercase().contains(&q)
    }
}

/// The commands that match what the user typed after the slash.
///
/// Names and aliases win; the description is only used when no name matches,
/// so a single letter does not flood the menu with unrelated commands.
fn matching_commands(query: &str) -> Vec<usize> {
    let by_name: Vec<usize> = COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, c)| c.name_matches(query))
        .map(|(i, _)| i)
        .collect();
    if !by_name.is_empty() {
        return by_name;
    }
    COMMANDS
        .iter()
        .enumerate()
        .filter(|(_, c)| c.desc_matches(query))
        .map(|(i, _)| i)
        .collect()
}

/// The command menu shown while a command name is being typed.
pub struct SlashMenu {
    /// Text typed after the slash.
    pub query: String,
    /// Indices into `COMMANDS` of the matching commands.
    pub items: Vec<usize>,
    /// Index of the highlighted command inside `items`.
    pub selected: usize,
    /// Set when the user closes the menu with Esc.
    pub dismissed: bool,
}

/// Interactive selector opened by `/session`.
pub struct SessionMenu {
    pub items: Vec<SessionSummary>,
    pub selected: usize,
}

// ──────────────────────────────────── app ───────────────────────────────────

// Measured usage for this conversation. Missing/unfinished requests keep known
// sums as lower bounds; cache coverage is independent of token coverage.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub has_usage: bool,
    pub has_cache: bool,
    pending: bool,
    incomplete: bool,
    cache_incomplete: bool,
}

impl ChatUsage {
    fn start_request(&mut self) {
        self.finish_turn();
        self.pending = true;
    }

    fn record(&mut self, usage: TokenUsage, complete: bool) {
        fn add(sum: &mut u64, value: u64) -> bool {
            let result = sum.checked_add(value);
            *sum = result.unwrap_or(u64::MAX);
            result.is_some()
        }

        self.has_usage = true;
        self.incomplete |= !complete;
        self.incomplete |= !add(&mut self.prompt_tokens, usage.prompt_tokens);
        self.incomplete |= !add(&mut self.completion_tokens, usage.completion_tokens);
        self.incomplete |= !add(&mut self.total_tokens, usage.total_tokens);
        self.cache_incomplete |= !complete;
        match usage.cached_tokens {
            Some(cached) => {
                self.has_cache = true;
                self.cache_incomplete |= !add(&mut self.cached_tokens, cached);
            }
            None => self.cache_incomplete = true,
        }
        self.pending = false;
    }

    fn finish_turn(&mut self) {
        if self.pending {
            self.incomplete = true;
            self.cache_incomplete = true;
            self.pending = false;
        }
    }

    pub fn is_complete(&self) -> bool {
        self.has_usage && !self.incomplete && !self.pending
    }

    pub fn cache_is_complete(&self) -> bool {
        self.has_cache && !self.cache_incomplete && !self.pending
    }
}

/// Builds the reasoning-effort cycle and the index of the level that is
/// active right now. The active level is always part of the cycle, so
/// Tab always moves to a different level. The token `off` marks the
/// level that is not sent to the API.
fn effort_cycle(configured: &[String], active: Option<&str>) -> (Vec<Option<String>>, usize) {
    let same = |a: &Option<String>, b: &Option<String>| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
        _ => false,
    };
    let mut levels: Vec<Option<String>> = Vec::new();
    for raw in configured {
        let level = raw.trim();
        if level.is_empty() {
            continue;
        }
        let entry = if level.eq_ignore_ascii_case("off") {
            None
        } else {
            Some(level.to_string())
        };
        if !levels.iter().any(|l| same(l, &entry)) {
            levels.push(entry);
        }
    }
    let active = active
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty());
    let found = levels.iter().position(|l| same(l, &active));
    let index = match found {
        Some(i) => i,
        None => {
            levels.insert(0, active);
            0
        }
    };
    (levels, index)
}

pub struct App {
    pub cfg: Config,
    /// Reasoning-effort levels cycled with Tab. `None` is the level that
    /// is not sent to the API at all.
    pub effort_levels: Vec<Option<String>>,
    /// Index of the active level inside `effort_levels`.
    pub effort_index: usize,
    client: reqwest::Client,
    tx: Tx,
    mcp: Arc<tokio::sync::Mutex<McpManager>>,

    pub entries: Vec<Entry>,
    completion_text: Option<usize>,
    completion_reasoning: Option<usize>,
    history: Vec<Value>,
    history_mark: usize,
    session_store: SessionStore,
    session: Session,
    persistence_error: Option<String>,

    pub input: Input,
    input_history: Vec<String>,
    hist_pos: Option<usize>,
    draft: String,
    last_submit: String,

    pub running: bool,
    pub run_started: Instant,
    pub streamed_chars: usize,
    pub token_usage: ChatUsage,
    pub context_tokens: usize,
    pub context_limit: usize,
    pub compressing_context: bool,
    task: Option<JoinHandle<()>>,
    cancelling: bool,
    pub pending: Option<Pending>,
    pending_queue: VecDeque<Pending>,
    pub slash: Option<SlashMenu>,
    pub session_menu: Option<SessionMenu>,
    pub auto: Arc<AtomicBool>,

    pub scroll: usize,
    pub last_total: usize,
    pub chat_height: usize,
    pub tick: u64,
    pub should_quit: bool,
    last_ctrl_c: Option<Instant>,
}

impl App {
    pub fn new(mut cfg: Config, tx: Tx) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .build()?;
        let context_limit = cfg.context_tokens;
        let auto_approve = cfg.auto_approve;
        let (session_store, mut session, restored) = SessionStore::open(
            &cfg.config_dir,
            &cfg.root,
            &cfg.model,
            cfg.reasoning_effort.as_deref(),
        )?;
        if restored {
            cfg.model = session.model.clone();
            cfg.reasoning_effort = session.reasoning_effort.clone();
            let stored_workspace = std::path::PathBuf::from(&session.workspace);
            if stored_workspace.is_dir() {
                cfg.root = stored_workspace;
            } else {
                session.workspace = cfg.root.to_string_lossy().into_owned();
            }
        }
        let (effort_levels, effort_index) =
            effort_cycle(&cfg.reasoning_efforts, cfg.reasoning_effort.as_deref());
        // The active level is the cycle entry it belongs to.
        cfg.reasoning_effort = effort_levels.get(effort_index).cloned().flatten();
        session.model = cfg.model.clone();
        session.reasoning_effort = cfg.reasoning_effort.clone();
        session.workspace = cfg.root.to_string_lossy().into_owned();
        session_store.save(&mut session)?;
        session_store.activate(&session.id)?;
        let history = session.history.clone();
        let mut entries = entries_from_history(&history, &session.tool_entries);
        if restored {
            entries.push(Entry::Notice(format!(
                "Restored session: {}",
                session.title
            )));
        }
        Ok(Self {
            cfg,
            effort_levels,
            effort_index,
            client,
            tx,
            mcp: Arc::new(tokio::sync::Mutex::new(McpManager::default())),
            entries,
            completion_text: None,
            completion_reasoning: None,
            history,
            history_mark: 0,
            session_store,
            session,
            persistence_error: None,
            input: Input::default(),
            input_history: Vec::new(),
            hist_pos: None,
            draft: String::new(),
            last_submit: String::new(),
            running: false,
            run_started: Instant::now(),
            streamed_chars: 0,
            token_usage: ChatUsage::default(),
            context_tokens: 0,
            context_limit,
            compressing_context: false,
            task: None,
            cancelling: false,
            pending: None,
            pending_queue: VecDeque::new(),
            slash: None,
            session_menu: None,
            auto: Arc::new(AtomicBool::new(auto_approve)),
            scroll: 0,
            last_total: 0,
            chat_height: 10,
            tick: 0,
            should_quit: false,
            last_ctrl_c: None,
        })
    }

    // ───────────────────────────── terminal events ──────────────────────────

    pub fn on_event(&mut self, ev: Event) {
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => self.on_key(k),
            Event::Paste(s) => {
                if self.pending.is_none() {
                    let s = s.replace("\r\n", "\n").replace('\r', "\n");
                    self.input.insert_str(&s);
                    self.sync_slash();
                }
            }
            Event::Resize(_, _) => {
                if let Some(p) = &mut self.pending {
                    p.can_approve = false;
                }
            }
            Event::Mouse(m) => match m.kind {
                MouseEventKind::ScrollUp => self.scroll = self.scroll.saturating_add(3),
                MouseEventKind::ScrollDown => self.scroll = self.scroll.saturating_sub(3),
                _ => {}
            },
            _ => {}
        }
    }

    fn on_key(&mut self, k: KeyEvent) {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);

        // Tab switches the reasoning level. Ctrl+Tab is avoided on purpose:
        // Windows Terminal uses it to switch tabs, so the key never reaches
        // the app. The shortcut is handled before the approval prompt and the
        // command menu, which capture their own keys.
        if !ctrl && !alt && matches!(k.code, KeyCode::Tab | KeyCode::BackTab) {
            self.cycle_reasoning_effort();
            return;
        }

        // Approval prompt captures the keyboard.
        if let Some(p) = &mut self.pending {
            match k.code {
                KeyCode::PageDown | KeyCode::Down => {
                    p.scroll = p
                        .scroll
                        .saturating_add(p.page_rows.saturating_sub(1).max(1));
                    p.can_approve = false;
                    return;
                }
                KeyCode::PageUp | KeyCode::Up => {
                    p.scroll = p
                        .scroll
                        .saturating_sub(p.page_rows.saturating_sub(1).max(1));
                    p.can_approve = false;
                    return;
                }
                KeyCode::Home => {
                    p.scroll = 0;
                    p.can_approve = false;
                    return;
                }
                _ => {}
            }
            let can_approve = p.can_approve;
            let decision = match k.code {
                KeyCode::Char('c') if ctrl => {
                    self.cancel();
                    return;
                }
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter if can_approve => {
                    Some(Approval::Yes)
                }
                KeyCode::Char('a') | KeyCode::Char('A') if can_approve && !p.session_trust => {
                    Some(Approval::Always)
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => Some(Approval::No),
                _ => None,
            };
            if let Some(d) = decision {
                if let Some(p) = self.pending.take() {
                    let id = p.id.clone();
                    let _ = p.reply.send(d);
                    if matches!(d, Approval::Yes | Approval::Always) {
                        if let Some(t) = self.tool_mut(&id) {
                            if t.status == ToolStatus::Waiting {
                                t.status = ToolStatus::Running;
                            }
                        }
                    }
                }
                self.pending = self.pending_queue.pop_front();
            }
            return;
        }

        // The session selector captures navigation keys while it is open.
        if self.session_menu_visible() {
            match k.code {
                KeyCode::Up => {
                    self.session_move(-1);
                    return;
                }
                KeyCode::Down => {
                    self.session_move(1);
                    return;
                }
                KeyCode::Enter => {
                    self.session_accept();
                    return;
                }
                KeyCode::Esc => {
                    self.session_menu = None;
                    return;
                }
                _ => self.session_menu = None,
            }
        }

        // The command menu captures the navigation keys while it is open.
        if self.slash_menu_visible() {
            match k.code {
                KeyCode::Up => {
                    self.slash_move(-1);
                    return;
                }
                KeyCode::Down => {
                    self.slash_move(1);
                    return;
                }
                KeyCode::Enter => {
                    // A fully typed command runs right away; only a prefix is completed.
                    if self.exact_command() {
                        self.submit();
                    } else {
                        self.slash_accept();
                    }
                    return;
                }
                KeyCode::Esc => {
                    self.slash_dismiss();
                    return;
                }
                _ => {}
            }
        }

        match k.code {
            KeyCode::Char('c') if ctrl && !alt => self.on_ctrl_c(),
            KeyCode::Char('d') if ctrl => {
                if self.input.is_empty() {
                    self.should_quit = true;
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_line_start(),
            KeyCode::Char('w') if ctrl => self.input.delete_word_back(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            // AltGr on Windows is reported as Ctrl+Alt, so let that through.
            KeyCode::Char(c) if !ctrl || alt => {
                self.input.insert_char(c);
                self.hist_pos = None;
            }
            KeyCode::Enter if shift || alt => self.input.insert_char('\n'),
            KeyCode::Enter => {
                // "\" + Enter inserts a newline (works in every terminal).
                if self.input.cursor > 0 && self.input.buf[self.input.cursor - 1] == '\\' {
                    self.input.backspace();
                    self.input.insert_char('\n');
                } else {
                    self.submit();
                }
            }
            KeyCode::Esc => {
                if self.running {
                    self.cancel();
                }
            }
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End if ctrl => self.scroll = 0,
            KeyCode::End => self.input.end(),
            KeyCode::Up => {
                if !self.input.up() && !self.input.buf.contains(&'\n') {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if !self.input.down() && !self.input.buf.contains(&'\n') {
                    self.history_next();
                }
            }
            KeyCode::PageUp => {
                self.scroll = self
                    .scroll
                    .saturating_add(self.chat_height.saturating_sub(2).max(1));
            }
            KeyCode::PageDown => {
                self.scroll = self
                    .scroll
                    .saturating_sub(self.chat_height.saturating_sub(2).max(1));
            }
            _ => {}
        }
        self.sync_slash();
    }

    // ─────────────────────────── reasoning effort ────────────────────────────

    /// Moves to the next reasoning-effort level. The new level is sent with
    /// the next request; a request that is already running keeps its own.
    /// The level is shown in the footer, no chat entry is added.
    fn cycle_reasoning_effort(&mut self) {
        if self.effort_levels.len() < 2 {
            // The configuration always mirrors the active level, even when there
            // is nothing to switch to.
            self.cfg.reasoning_effort =
                self.effort_levels.get(self.effort_index).cloned().flatten();
            self.entries.push(Entry::Notice(
                "No other reasoning level configured (REASONING_EFFORTS)".into(),
            ));
            return;
        }
        self.effort_index = (self.effort_index + 1) % self.effort_levels.len();
        self.cfg.reasoning_effort = self.effort_levels[self.effort_index].clone();
        self.persist_current_session();
    }

    /// The reasoning level shown in the footer, if one is active.
    pub fn effort_label(&self) -> Option<String> {
        self.effort_levels
            .get(self.effort_index)
            .cloned()
            .flatten()
            .map(|level| level.to_uppercase())
    }

    // ────────────────────────────── command menu ────────────────────────────

    /// The command name typed at the prompt, if the prompt starts a command.
    fn slash_query(&self) -> Option<String> {
        let cursor = self.input.cursor.min(self.input.buf.len());
        // Nothing typed in front of the cursor means no command is being typed.
        if cursor == 0 {
            return None;
        }
        let upto: String = self.input.buf[..cursor].iter().collect();
        if !upto.starts_with('/') || upto.contains('\n') {
            return None;
        }
        // Skip the leading slash; the cursor may still sit right before it.
        let query: String = upto.chars().skip(1).collect();
        if query.contains(' ') {
            return None;
        }
        Some(query)
    }

    /// Recomputes the menu after the prompt changed.
    pub(crate) fn sync_slash(&mut self) {
        let Some(query) = self.slash_query() else {
            self.slash = None;
            return;
        };
        let items = matching_commands(&query);
        match &mut self.slash {
            Some(m) if m.query == query => {
                m.items = items;
                m.selected = m.selected.min(m.items.len().saturating_sub(1));
            }
            Some(m) => {
                m.query = query;
                m.items = items;
                m.selected = 0;
                m.dismissed = false;
            }
            None => {
                self.slash = Some(SlashMenu {
                    query,
                    items,
                    selected: 0,
                    dismissed: false,
                });
            }
        }
    }

    /// Whether the command menu is currently shown.
    pub fn slash_menu_visible(&self) -> bool {
        // Session selection and approval prompts take precedence over command completion.
        if self.pending.is_some() || self.session_menu.is_some() {
            return false;
        }
        self.slash
            .as_ref()
            .is_some_and(|m| !m.dismissed && !m.items.is_empty())
    }

    pub fn session_menu_visible(&self) -> bool {
        self.pending.is_none()
            && self
                .session_menu
                .as_ref()
                .is_some_and(|m| !m.items.is_empty())
    }

    pub fn active_session_id(&self) -> &str {
        &self.session.id
    }

    fn session_move(&mut self, delta: i32) {
        let Some(menu) = &mut self.session_menu else {
            return;
        };
        if menu.items.is_empty() {
            return;
        }
        let len = menu.items.len() as i32;
        menu.selected = ((menu.selected as i32 + delta).rem_euclid(len)) as usize;
    }

    fn session_accept(&mut self) {
        let Some(menu) = self.session_menu.take() else {
            return;
        };
        let Some(item) = menu.items.get(menu.selected) else {
            return;
        };
        let id = item.id.clone();
        match self.switch_session(&id) {
            Ok(true) => {}
            Ok(false) => self
                .entries
                .push(Entry::Error(format!("Session not found: {id}"))),
            Err(error) => self
                .entries
                .push(Entry::Error(format!("Cannot switch session: {error:#}"))),
        }
    }

    fn persist_current_session(&mut self) {
        self.session.history = self.history.clone();
        self.save_current_session();
    }

    fn save_current_session(&mut self) {
        self.session.tool_entries = self
            .entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::Tool(tool) => Some(tool.clone()),
                _ => None,
            })
            .collect();
        self.session.model = self.cfg.model.clone();
        self.session.reasoning_effort = self.cfg.reasoning_effort.clone();
        self.session.workspace = self.cfg.root.to_string_lossy().into_owned();
        match self.session_store.save(&mut self.session) {
            Ok(()) => self.persistence_error = None,
            Err(error) => {
                let message = format!("Session save failed: {error:#}");
                if self.persistence_error.as_deref() != Some(message.as_str()) {
                    self.entries.push(Entry::Warning(message.clone()));
                }
                self.persistence_error = Some(message);
            }
        }
    }

    fn create_fresh_session(&mut self, workspace: std::path::PathBuf, notice: Option<&str>) {
        self.persist_current_session();
        let previous_save_error = self.persistence_error.clone();
        match self.session_store.new_session(
            &workspace,
            &self.cfg.model,
            self.cfg.reasoning_effort.as_deref(),
        ) {
            Ok(mut session) => {
                let mut notices = Vec::new();
                let save_error = self.session_store.save(&mut session).err();
                if let Some(error) = &save_error {
                    notices.push(format!("Session save failed: {error:#}"));
                }
                if let Err(error) = self.session_store.activate(&session.id) {
                    notices.push(format!("Cannot activate session: {error:#}"));
                }
                self.cfg.root = workspace;
                self.session = session;
                self.reset_conversation_state();
                self.persistence_error =
                    save_error.map(|error| format!("Session save failed: {error:#}"));
                if let Some(message) = previous_save_error {
                    self.entries.push(Entry::Warning(message));
                }
                for message in notices {
                    self.entries.push(Entry::Warning(message));
                }
                if let Some(message) = notice {
                    self.entries.push(Entry::Notice(message.into()));
                }
            }
            Err(error) => self
                .entries
                .push(Entry::Error(format!("Cannot create session: {error:#}"))),
        }
    }

    fn reset_conversation_state(&mut self) {
        self.history.clear();
        self.history_mark = 0;
        self.token_usage = ChatUsage::default();
        self.context_tokens = 0;
        self.compressing_context = false;
        self.completion_text = None;
        self.completion_reasoning = None;
        self.input_history.clear();
        self.hist_pos = None;
        self.draft.clear();
        self.last_submit.clear();
        self.entries.clear();
        self.entries.push(Entry::Banner);
        self.scroll = 0;
        self.session_menu = None;
        self.slash = None;
        self.persistence_error = None;
    }

    fn switch_session(&mut self, id: &str) -> Result<bool> {
        if id == self.session.id {
            self.session_menu = None;
            self.entries.push(Entry::Notice(format!(
                "Already in session: {}",
                self.session.title
            )));
            return Ok(true);
        }

        self.persist_current_session();
        let previous_save_error = self.persistence_error.clone();
        let Some(mut session) = self.session_store.load(id)? else {
            return Ok(false);
        };

        let workspace = std::path::PathBuf::from(&session.workspace);
        let workspace_warning = if workspace.is_dir() {
            self.cfg.root = workspace;
            None
        } else {
            let warning = format!(
                "Workspace is missing: {}. Keeping current workspace {}.",
                display_path(&session.workspace),
                display_path(&self.cfg.root)
            );
            session.workspace = self.cfg.root.to_string_lossy().into_owned();
            Some(warning)
        };

        self.session_store.activate(&session.id)?;
        self.cfg.model = session.model.clone();
        self.cfg.reasoning_effort = session.reasoning_effort.clone();
        let (levels, index) = effort_cycle(
            &self.cfg.reasoning_efforts,
            self.cfg.reasoning_effort.as_deref(),
        );
        self.effort_levels = levels;
        self.effort_index = index;
        self.cfg.reasoning_effort = self.effort_levels.get(self.effort_index).cloned().flatten();
        self.session = session;
        self.reset_conversation_state();
        self.history = self.session.history.clone();
        self.entries = entries_from_history(&self.history, &self.session.tool_entries);
        if let Some(message) = workspace_warning {
            self.entries.push(Entry::Warning(message));
        }
        if let Some(message) = previous_save_error {
            self.entries.push(Entry::Warning(message));
        }
        self.entries.push(Entry::Notice(format!(
            "Switched to session: {}",
            self.session.title
        )));
        self.token_usage = ChatUsage::default();
        self.context_tokens = 0;
        self.persist_current_session();
        Ok(true)
    }

    /// Whether the prompt contains exactly a command name or alias.
    fn exact_command(&self) -> bool {
        let text = self.input.text();
        let text = text.trim();
        COMMANDS
            .iter()
            .any(|command| command.name == text || command.aliases.contains(&text))
    }

    /// Moves the highlight inside the command menu.
    fn slash_move(&mut self, delta: i32) {
        let Some(m) = &mut self.slash else { return };
        if m.items.is_empty() {
            return;
        }
        let len = m.items.len() as i32;
        m.selected = ((m.selected as i32 + delta).rem_euclid(len)) as usize;
    }

    /// Puts the highlighted command into the prompt.
    fn slash_accept(&mut self) {
        let Some(m) = &self.slash else { return };
        let Some(&idx) = m.items.get(m.selected) else {
            return;
        };
        let cmd = &COMMANDS[idx];
        let cursor = self.input.cursor.min(self.input.buf.len());
        // Replace the whole command word, not just the part before the cursor.
        let word_end = self.input.buf[cursor..]
            .iter()
            .position(|c| c.is_whitespace())
            .map(|i| cursor + i)
            .unwrap_or(self.input.buf.len());
        let tail: Vec<char> = self.input.buf[word_end..].to_vec();
        let mut text = cmd.name.to_string();
        // A command that takes an argument needs a space to type it into.
        if cmd.arg.is_some() && tail.is_empty() {
            text.push(' ');
        }
        let mut buf: Vec<char> = text.chars().collect();
        buf.extend_from_slice(&tail);
        self.input.buf = buf;
        self.input.cursor = text.chars().count();
        self.hist_pos = None;
        // Keep the menu closed until the command text changes again.
        let query = self.slash_query().unwrap_or_default();
        self.slash = Some(SlashMenu {
            query,
            items: Vec::new(),
            selected: 0,
            dismissed: true,
        });
    }

    /// Hides the menu until the typed command changes.
    fn slash_dismiss(&mut self) {
        if let Some(m) = &mut self.slash {
            m.dismissed = true;
        }
    }

    fn on_ctrl_c(&mut self) {
        if self.running {
            self.cancel();
            return;
        }
        if !self.input.is_empty() {
            self.input.clear();
            return;
        }
        if let Some(t) = self.last_ctrl_c {
            if t.elapsed() < Duration::from_secs(2) {
                self.should_quit = true;
                return;
            }
        }
        self.last_ctrl_c = Some(Instant::now());
        self.entries
            .push(Entry::Notice("Press Ctrl+C again to exit".into()));
    }

    fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let pos = match self.hist_pos {
            None => {
                self.draft = self.input.text();
                self.input_history.len() - 1
            }
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.hist_pos = Some(pos);
        let s = self.input_history[pos].clone();
        self.input.set(&s);
    }

    fn history_next(&mut self) {
        let Some(p) = self.hist_pos else { return };
        if p + 1 < self.input_history.len() {
            self.hist_pos = Some(p + 1);
            let s = self.input_history[p + 1].clone();
            self.input.set(&s);
        } else {
            self.hist_pos = None;
            let d = std::mem::take(&mut self.draft);
            self.input.set(&d);
        }
    }

    // ───────────────────────────── submit & commands ────────────────────────

    fn submit(&mut self) {
        if self.running {
            return;
        }
        let text = self.input.take().trim().to_string();
        if text.is_empty() {
            return;
        }
        self.hist_pos = None;

        if text.starts_with('/') {
            self.command(&text);
            return;
        }

        self.input_history.push(text.clone());
        self.entries.push(Entry::User(text.clone()));
        self.history_mark = self.history.len();
        if self.session.title == UNTITLED {
            self.session.title = session::title_from_prompt(&text);
        }
        self.history
            .push(json!({ "role": "user", "content": text }));
        self.last_submit = text;
        self.persist_current_session();
        self.start_run();
    }

    fn start_run(&mut self) {
        self.running = true;
        self.compressing_context = false;
        self.run_started = Instant::now();
        self.streamed_chars = 0;
        self.completion_text = None;
        self.completion_reasoning = None;
        self.scroll = 0;

        let client = self.client.clone();
        let cfg = self.cfg.clone();
        let hist = self.history.clone();
        let tx = self.tx.clone();
        let auto = self.auto.clone();
        let mcp = self.mcp.clone();

        self.task = Some(tokio::spawn(async move {
            match agent::run_turn(&client, &cfg, hist, &tx, &auto, &mcp).await {
                Ok(h) => {
                    let _ = tx.send(AgentEvent::Done(h));
                }
                Err(e) => {
                    let _ = tx.send(AgentEvent::Error(format!("{e:#}")));
                }
            }
        }));
    }

    fn command(&mut self, text: &str) {
        let mut parts = text.splitn(2, ' ');
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match cmd {
            "/exit" | "/quit" | "/q" => self.should_quit = true,
            "/clear" => self.create_fresh_session(self.cfg.root.clone(), None),
            "/session" => {
                if arg.is_empty() || arg.eq_ignore_ascii_case("list") {
                    match self.session_store.list() {
                        Ok(items) if !items.is_empty() => {
                            self.session_menu = Some(SessionMenu { items, selected: 0 });
                            self.slash = None;
                        }
                        Ok(_) => self.entries.push(Entry::Notice("No saved sessions".into())),
                        Err(error) => self
                            .entries
                            .push(Entry::Error(format!("Cannot list sessions: {error:#}"))),
                    }
                } else if arg.eq_ignore_ascii_case("new") {
                    self.create_fresh_session(self.cfg.root.clone(), None);
                } else {
                    let id = arg.trim();
                    match self.switch_session(id) {
                        Ok(true) => {}
                        Ok(false) => self
                            .entries
                            .push(Entry::Error(format!("Session not found: {id}"))),
                        Err(error) => self
                            .entries
                            .push(Entry::Error(format!("Cannot switch session: {error:#}"))),
                    }
                }
            }
            "/workspace" => {
                if arg.is_empty() {
                    self.entries.push(Entry::Notice(format!(
                        "Workspace: {}\nUsage: /workspace <directory>",
                        display_path(&self.cfg.root)
                    )));
                } else {
                    let unquoted = arg
                        .strip_prefix('"')
                        .and_then(|value| value.strip_suffix('"'))
                        .or_else(|| {
                            arg.strip_prefix('\'')
                                .and_then(|value| value.strip_suffix('\''))
                        })
                        .unwrap_or(arg);
                    let requested = std::path::PathBuf::from(unquoted);
                    let candidate = if requested.is_absolute() {
                        requested
                    } else {
                        self.cfg.root.join(requested)
                    };
                    match std::fs::canonicalize(&candidate) {
                        Ok(path) if path.is_dir() => self.create_fresh_session(
                            path,
                            Some("Workspace changed; started a new session"),
                        ),
                        Ok(_) => self.entries.push(Entry::Error(format!(
                            "Workspace is not a directory: {}",
                            display_path(&candidate)
                        ))),
                        Err(error) => self.entries.push(Entry::Error(format!(
                            "Cannot access workspace {}: {error}",
                            display_path(&candidate)
                        ))),
                    }
                }
            }
            "/model" => {
                if arg.is_empty() {
                    self.entries
                        .push(Entry::Notice(format!("Current model: {}", self.cfg.model)));
                } else {
                    self.cfg.model = arg.to_string();
                    self.entries
                        .push(Entry::Notice(format!("Model set to {arg}")));
                    self.persist_current_session();
                }
            }
            "/auto" => {
                let v = !self.auto.load(Ordering::Relaxed);
                self.auto.store(v, Ordering::Relaxed);
                let msg = if v {
                    "Auto-approve is ON: writes and shell commands no longer ask for confirmation"
                } else {
                    "Auto-approve is OFF: every write and shell command needs your approval"
                };
                self.entries.push(if v {
                    Entry::Warning(msg.into())
                } else {
                    Entry::Notice(msg.into())
                });
            }
            "/help" => {
                let sep = if ascii_mode() { "|" } else { "·" };
                let arrows = if ascii_mode() { "Up/Down" } else { "↑/↓" };
                self.entries.push(Entry::Notice(format!(
                    "Commands\n\
                     /model [name]  show or change the model\n\
                     /auto          toggle auto-approve for writes and shell commands\n\
                     /clear         start a new session in this workspace\n\
                     /session       list or switch saved sessions\n\
                     /workspace     show or change the workspace directory\n\
                     /exit          quit\n\
                     \n\
                     Type / to open the command menu {sep} {arrows} select, Enter complete, Esc dismiss\n\
                     \n\
                     Keys\n\
                     Enter send {sep} \\ then Enter, Alt+Enter or Shift+Enter: new line\n\
                     Esc / Ctrl+C interrupt {sep} Ctrl+C twice or Ctrl+D quit\n\
                     {arrows} prompt history {sep} PgUp/PgDn / mouse wheel scroll\n\
                     Ctrl+End jump to latest message\n\
                     Tab switch the reasoning level\n\
                     F2 toggle mouse scrolling / native text selection"
                )));
            }
            _ => {
                let dash = if ascii_mode() { "-" } else { "—" };
                self.entries.push(Entry::Error(format!(
                    "Unknown command {cmd} {dash} try /help"
                )));
            }
        }
    }

    // ───────────────────────────── cancel & rollback ────────────────────────

    /// If the agent never completed a step, drop the unanswered user message
    /// so the history stays valid, and put the prompt back for editing.
    fn rollback_unanswered(&mut self) {
        if self.history.len() == self.history_mark + 1 {
            self.history.truncate(self.history_mark);
            if self.input.is_empty() {
                let s = self.last_submit.clone();
                self.input.set(&s);
            }
        }
    }

    fn fail_open_tools(&mut self, why: &str) {
        for e in self.entries.iter_mut() {
            if let Entry::Tool(t) = e {
                if matches!(t.status, ToolStatus::Running | ToolStatus::Waiting) {
                    t.status = ToolStatus::Failed;
                    t.result = Some(why.to_string());
                }
            }
        }
    }

    pub async fn shutdown(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
        self.mcp.lock().await.shutdown().await;
    }

    fn cancel(&mut self) {
        if let Some(t) = &self.task {
            t.abort();
        }
        self.cancelling = true;
    }

    pub async fn finish_cancel(
        &mut self,
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<AgentEvent>,
    ) {
        if !self.cancelling {
            return;
        }
        // Abort drops the stream's usage guard. Await it and consume all queued
        // events before allowing another submit or /clear to change the chat.
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
        while let Ok(event) = rx.try_recv() {
            self.on_agent(event);
        }
        self.token_usage.finish_turn();
        self.pending = None;
        self.pending_queue.clear();
        self.running = false;
        self.compressing_context = false;
        self.cancelling = false;
        self.fail_open_tools("Interrupted");
        self.rollback_unanswered();
        self.persist_current_session();
        self.entries.push(Entry::Notice("Interrupted".into()));
    }

    // ───────────────────────────── agent events ─────────────────────────────

    pub fn pending_approval_count(&self) -> usize {
        (if self.pending.is_some() { 1 } else { 0 }) + self.pending_queue.len()
    }

    fn tool_mut(&mut self, id: &str) -> Option<&mut ToolEntry> {
        self.entries.iter_mut().rev().find_map(|e| match e {
            Entry::Tool(t) if t.id == id => Some(t),
            _ => None,
        })
    }

    pub fn on_agent(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::CompletionStart => {
                self.completion_text = None;
                self.completion_reasoning = None;
            }
            AgentEvent::CompletionDiscard => {
                let mut indices: Vec<_> = [
                    self.completion_text.take(),
                    self.completion_reasoning.take(),
                ]
                .into_iter()
                .flatten()
                .collect();
                indices.sort_unstable();
                for index in indices.into_iter().rev() {
                    if let Entry::Assistant(text) | Entry::Thinking(text) =
                        self.entries.remove(index)
                    {
                        self.streamed_chars =
                            self.streamed_chars.saturating_sub(text.chars().count());
                    }
                }
            }
            AgentEvent::Text(t) => {
                self.streamed_chars += t.chars().count();
                if let Some(index) = self.completion_text {
                    if let Entry::Assistant(s) = &mut self.entries[index] {
                        s.push_str(&t);
                    }
                } else if !t.is_empty() {
                    self.completion_text = Some(self.entries.len());
                    self.entries.push(Entry::Assistant(t));
                }
            }
            AgentEvent::Reasoning(t) => {
                self.streamed_chars += t.chars().count();
                if let Some(index) = self.completion_reasoning {
                    if let Entry::Thinking(s) = &mut self.entries[index] {
                        s.push_str(&t);
                    }
                } else if !t.is_empty() {
                    self.completion_reasoning = Some(self.entries.len());
                    self.entries.push(Entry::Thinking(t));
                }
            }
            AgentEvent::ToolStart { id, label, target } => {
                self.entries.push(Entry::Tool(ToolEntry {
                    id,
                    label,
                    target,
                    status: ToolStatus::Running,
                    diff: Vec::new(),
                    result: None,
                    output: None,
                }));
            }
            AgentEvent::ToolDiff { id, diff } => {
                if let Some(t) = self.tool_mut(&id) {
                    t.diff = diff;
                }
            }
            AgentEvent::ToolOutput { id, output } => {
                if let Some(t) = self.tool_mut(&id) {
                    t.output = Some(output);
                }
            }
            AgentEvent::McpApproval { prompt, reply } => {
                let pending = Pending {
                    id: String::new(),
                    prompt,
                    reply,
                    scroll: 0,
                    page_rows: 1,
                    can_approve: false,
                    session_trust: true,
                };
                if self.pending.is_none() {
                    self.pending = Some(pending);
                } else {
                    self.pending_queue.push_back(pending);
                }
            }
            AgentEvent::ToolApproval { id, prompt, reply } => {
                if let Some(t) = self.tool_mut(&id) {
                    t.status = ToolStatus::Waiting;
                }
                let pending = Pending {
                    id,
                    prompt,
                    reply,
                    scroll: 0,
                    page_rows: 1,
                    can_approve: false,
                    session_trust: false,
                };
                if self.pending.is_none() {
                    self.pending = Some(pending);
                } else {
                    self.pending_queue.push_back(pending);
                }
            }
            AgentEvent::ToolEnd {
                id,
                status,
                summary,
            } => {
                if let Some(t) = self.tool_mut(&id) {
                    t.status = status;
                    t.result = Some(summary);
                }
            }
            AgentEvent::ContextUsage { tokens, limit } => {
                self.context_tokens = tokens;
                self.context_limit = limit;
            }
            AgentEvent::ContextCompression { active } => self.compressing_context = active,
            AgentEvent::UsageStart => self.token_usage.start_request(),
            AgentEvent::Usage { usage, complete } => self.token_usage.record(usage, complete),
            AgentEvent::HistoryPatch {
                start,
                end,
                messages,
            } => {
                // Events from one turn are ordered, including ToolBatch::drop on cancellation.
                self.history.splice(start..end, messages.iter().cloned());
                self.session.history.splice(start..end, messages);
                self.history_mark = self.history.len();
                self.save_current_session();
            }
            AgentEvent::Sync(h) => {
                self.history_mark = if h.last().is_some_and(agent::is_user_request) {
                    h.len().saturating_sub(1)
                } else {
                    h.len()
                };
                self.history = h;
                self.persist_current_session();
            }
            AgentEvent::Done(h) => {
                self.compressing_context = false;
                self.history = h;
                self.persist_current_session();
                self.token_usage.finish_turn();
                if !self.cancelling {
                    self.running = false;
                    self.task = None;
                    self.pending = None;
                    self.pending_queue.clear();
                }
            }
            AgentEvent::Notice(message) => {
                self.entries.push(Entry::Notice(message));
            }
            AgentEvent::Error(e) => {
                self.compressing_context = false;
                self.token_usage.finish_turn();
                if !self.cancelling {
                    self.running = false;
                    self.task = None;
                    self.pending = None;
                    self.pending_queue.clear();
                    self.fail_open_tools("Failed");
                    self.entries.push(Entry::Error(e));
                    self.rollback_unanswered();
                    self.persist_current_session();
                }
            }
        }
    }
}

fn entries_from_history(history: &[Value], saved_tools: &[ToolEntry]) -> Vec<Entry> {
    let mut entries = vec![Entry::Banner];
    for message in history {
        if agent::is_context_summary(message) {
            continue;
        }
        match message.get("role").and_then(Value::as_str).unwrap_or("") {
            "user" => {
                let text = history_content_text(message.get("content").unwrap_or(&Value::Null));
                if !text.is_empty() {
                    entries.push(Entry::User(text));
                }
            }
            "assistant" => {
                if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
                    if !reasoning.is_empty() {
                        entries.push(Entry::Thinking(reasoning.to_string()));
                    }
                }
                if let Some(content) = message.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        entries.push(Entry::Assistant(content.to_string()));
                    }
                }
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let id = call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("restored-tool")
                            .to_string();
                        if let Some(tool) = saved_tools.iter().find(|tool| tool.id == id) {
                            entries.push(Entry::Tool(tool.clone()));
                            continue;
                        }
                        let function = call.get("function").unwrap_or(&Value::Null);
                        let name = function
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let label = match name {
                            "read_file" => "Read",
                            "read_images" => "Read images",
                            "write_file" => "Write",
                            "edit_file" => "Edit",
                            "shell" => "Shell",
                            other => other,
                        }
                        .to_string();
                        let args = match function.get("arguments") {
                            Some(Value::String(value)) => {
                                serde_json::from_str(value).unwrap_or(Value::Null)
                            }
                            Some(value) => value.clone(),
                            None => Value::Null,
                        };
                        let target = args
                            .get("path")
                            .or_else(|| args.get("command"))
                            .or_else(|| args.get("query"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .chars()
                            .take(180)
                            .collect();
                        entries.push(Entry::Tool(ToolEntry {
                            id,
                            label,
                            target,
                            status: ToolStatus::Running,
                            diff: Vec::new(),
                            result: None,
                            output: None,
                        }));
                    }
                }
            }
            "tool" => {
                let id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if let Some(tool) = entries.iter_mut().rev().find_map(|entry| match entry {
                    Entry::Tool(tool) if tool.id == id => Some(tool),
                    _ => None,
                }) {
                    if !saved_tools.iter().any(|saved| saved.id == id) {
                        tool.status = ToolStatus::Ok;
                    }
                }
            }
            _ => {}
        }
    }
    entries
}

fn history_content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                if let Some(text) = part.as_str() {
                    Some(text.to_string())
                } else if part.get("type").and_then(Value::as_str) == Some("text") {
                    part.get("text").and_then(Value::as_str).map(str::to_string)
                } else if part
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.contains("image"))
                {
                    Some("[image]".to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests;
