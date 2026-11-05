//! Bounded live preview, independent of the head/tail capture sent to the model.
use super::{CapturedStream, SHELL_STREAM_HEAD_BYTES, SHELL_STREAM_TAIL_BYTES};
use std::collections::VecDeque;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::watch;

pub(crate) const PREVIEW_LINES: usize = 5;
// Also bound unterminated lines, including zero-width Unicode and control strings.
pub(crate) const PREVIEW_LINE_BYTES: usize = 2048;
pub(crate) const UPDATE_INTERVAL: Duration = Duration::from_millis(34);

pub(super) struct Capture {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    total: usize,
    head_limit: usize,
    tail_limit: usize,
}

impl Capture {
    pub(super) fn new(head_limit: usize, tail_limit: usize) -> Self {
        Self {
            head: Vec::new(),
            tail: VecDeque::with_capacity(tail_limit),
            total: 0,
            head_limit,
            tail_limit,
        }
    }

    pub(super) fn push(&mut self, chunk: &[u8]) {
        self.total = self.total.saturating_add(chunk.len());
        let take = self
            .head_limit
            .saturating_sub(self.head.len())
            .min(chunk.len());
        self.head.extend_from_slice(&chunk[..take]);
        let rest = &chunk[take..];
        if self.tail_limit > 0 && !rest.is_empty() {
            if rest.len() >= self.tail_limit {
                self.tail.clear();
                self.tail.extend(&rest[rest.len() - self.tail_limit..]);
            } else {
                let discard = (self.tail.len() + rest.len()).saturating_sub(self.tail_limit);
                self.tail.drain(..discard);
                self.tail.extend(rest);
            }
        }
    }

    pub(super) fn finish(self) -> CapturedStream {
        CapturedStream {
            head: self.head,
            tail: self.tail.into(),
            total: self.total,
        }
    }
}

#[derive(Clone, Copy, Default)]
enum Escape {
    #[default]
    Text,
    Start,
    Intermediate,
    Csi,
    String {
        osc: bool,
        esc: bool,
    },
}

#[derive(Default)]
struct Decoder {
    utf8: [u8; 4],
    pending: usize,
    expected: usize,
    escape: Escape,
    carriage_return: bool,
    line: Option<u64>,
}

struct PreviewLine {
    id: u64,
    text: String,
    truncated: bool,
}

#[derive(Default)]
pub(super) struct Preview {
    lines: VecDeque<PreviewLine>,
    decoders: [Decoder; 2],
    sequence: u64,
}

impl Preview {
    pub(super) fn push(&mut self, stream: usize, mut bytes: &[u8]) {
        // Move the tiny decoder out while it updates the shared five-line ring.
        let mut decoder = std::mem::take(&mut self.decoders[stream]);
        while !bytes.is_empty() {
            if decoder.pending == 0 && matches!(decoder.escape, Escape::Text) {
                let count = bytes
                    .iter()
                    .position(|b| !(0x20..0x7f).contains(b))
                    .unwrap_or(bytes.len());
                if count > 0 {
                    decoder.ascii(&bytes[..count], self);
                    bytes = &bytes[count..];
                    continue;
                }
            }
            decoder.byte(bytes[0], self);
            bytes = &bytes[1..];
        }
        self.decoders[stream] = decoder;
    }

    fn finish(&mut self, stream: usize) {
        let mut decoder = std::mem::take(&mut self.decoders[stream]);
        if decoder.pending > 0 {
            decoder.character('\u{fffd}', self);
        }
        self.decoders[stream] = decoder;
    }

    fn line(&mut self, cursor: &mut Option<u64>) -> Option<&mut PreviewLine> {
        if let Some(id) = cursor {
            // The other pipe may have evicted this unfinished line. Ignore its
            // continuation until a newline/CR instead of inventing a new row.
            return self.lines.iter_mut().find(|line| line.id == *id);
        }
        // Reuse evicted allocations even under continuous, high-volume output.
        let mut line = if self.lines.len() == PREVIEW_LINES {
            self.lines.pop_front().unwrap()
        } else {
            PreviewLine {
                id: 0,
                text: String::new(),
                truncated: false,
            }
        };
        line.text.clear();
        line.truncated = false;
        line.id = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        *cursor = Some(line.id);
        self.lines.push_back(line);
        self.lines.back_mut()
    }

    pub(super) fn snapshot(&self) -> String {
        let size = self.lines.iter().map(|line| line.text.len() + 1).sum();
        let mut output = String::with_capacity(size);
        for (i, line) in self.lines.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            output.push_str(&line.text);
        }
        output
    }
}

impl Decoder {
    fn output_line<'a>(&mut self, preview: &'a mut Preview) -> Option<&'a mut PreviewLine> {
        if self.carriage_return
            && self
                .line
                .is_some_and(|id| !preview.lines.iter().any(|line| line.id == id))
        {
            self.line = None;
        }
        let line = preview.line(&mut self.line)?;
        if self.carriage_return {
            line.text.clear();
            line.truncated = false;
            self.carriage_return = false;
        }
        Some(line)
    }

    fn ascii(&mut self, bytes: &[u8], preview: &mut Preview) {
        let Some(line) = self.output_line(preview) else {
            return;
        };
        if line.truncated {
            return;
        }
        let take = bytes
            .len()
            .min((PREVIEW_LINE_BYTES - 3).saturating_sub(line.text.len()));
        line.text
            .push_str(std::str::from_utf8(&bytes[..take]).expect("ASCII run"));
        if take < bytes.len() {
            line.text.push('…');
            line.truncated = true;
        }
    }

    fn byte(&mut self, byte: u8, preview: &mut Preview) {
        if self.pending > 0 {
            if byte & 0xc0 == 0x80 {
                self.utf8[self.pending] = byte;
                self.pending += 1;
                if self.pending == self.expected {
                    let c = std::str::from_utf8(&self.utf8[..self.pending])
                        .ok()
                        .and_then(|s| s.chars().next())
                        .unwrap_or('\u{fffd}');
                    self.pending = 0;
                    self.character(c, preview);
                }
                return;
            }
            self.pending = 0;
            self.character('\u{fffd}', preview);
        }
        if byte < 0x80 {
            self.character(byte as char, preview);
        } else {
            self.expected = match byte {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => 0,
            };
            if self.expected == 0 {
                self.character('\u{fffd}', preview);
            } else {
                self.utf8[0] = byte;
                self.pending = 1;
            }
        }
    }

    fn character(&mut self, c: char, preview: &mut Preview) {
        // A newline recovers from malformed/truncated control sequences.
        if c == '\n' || c == '\r' {
            self.escape = Escape::Text;
        }
        match self.escape {
            Escape::Start => {
                self.escape = match c {
                    '[' => Escape::Csi,
                    ']' | 'P' | 'X' | '^' | '_' => Escape::String {
                        osc: c == ']',
                        esc: false,
                    },
                    '\u{20}'..='\u{2f}' => Escape::Intermediate,
                    _ => Escape::Text,
                };
                return;
            }
            Escape::Intermediate => {
                if !('\u{20}'..='\u{2f}').contains(&c) {
                    self.escape = Escape::Text;
                }
                return;
            }
            Escape::Csi => {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    self.escape = Escape::Text;
                }
                return;
            }
            Escape::String { osc, esc } => {
                self.escape = if c == '\u{9c}' || (osc && c == '\x07') || (esc && c == '\\') {
                    Escape::Text
                } else {
                    Escape::String {
                        osc,
                        esc: c == '\x1b',
                    }
                };
                return;
            }
            Escape::Text => {}
        }
        match c {
            '\x1b' => self.escape = Escape::Start,
            '\u{9b}' => self.escape = Escape::Csi,
            '\u{9d}' | '\u{90}' | '\u{98}' | '\u{9e}' | '\u{9f}' => {
                self.escape = Escape::String {
                    osc: c == '\u{9d}',
                    esc: false,
                };
            }
            '\r' => self.carriage_return = true,
            '\n' => {
                let _ = preview.line(&mut self.line);
                self.line = None;
                self.carriage_return = false;
            }
            c if c == '\t' || !c.is_control() => {
                let Some(line) = self.output_line(preview) else {
                    return;
                };
                if !line.truncated {
                    let extra = if c == '\t' { 4 } else { c.len_utf8() };
                    if line.text.len() + extra <= PREVIEW_LINE_BYTES - 3 {
                        if c == '\t' {
                            line.text.push_str("    ");
                        } else {
                            line.text.push(c);
                        }
                    } else {
                        line.text.push('…');
                        line.truncated = true;
                    }
                }
            }
            _ => {}
        }
    }
}

pub(super) async fn drain(
    mut stdout: impl AsyncRead + Unpin,
    mut stderr: impl AsyncRead + Unpin,
    updates: Option<watch::Sender<String>>,
) -> std::io::Result<(CapturedStream, CapturedStream)> {
    let mut captures = [
        Capture::new(SHELL_STREAM_HEAD_BYTES, SHELL_STREAM_TAIL_BYTES),
        Capture::new(SHELL_STREAM_HEAD_BYTES, SHELL_STREAM_TAIL_BYTES),
    ];
    let mut preview = Preview::default();
    let (mut out_buf, mut err_buf) = ([0; 8192], [0; 8192]);
    let mut open = [true, true];
    let mut dirty = false;
    let mut bytes_since_yield = 0;
    let mut tick = tokio::time::interval(UPDATE_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    while open[0] || open[1] {
        let (stream, bytes) = tokio::select! {
            n = stdout.read(&mut out_buf), if open[0] => (0, &out_buf[..n?]),
            n = stderr.read(&mut err_buf), if open[1] => (1, &err_buf[..n?]),
            _ = tick.tick(), if updates.is_some() && dirty => {
                if let Some(tx) = &updates { tx.send_replace(preview.snapshot()); }
                dirty = false;
                continue;
            }
        };
        if bytes.is_empty() {
            open[stream] = false;
            if updates.is_some() {
                preview.finish(stream);
                dirty = true;
            }
        } else {
            captures[stream].push(bytes);
            if updates.is_some() {
                preview.push(stream, bytes);
                dirty = true;
            }
            bytes_since_yield += bytes.len();
            // Tokio's operation budget can otherwise drain megabytes before
            // another task runs. Bound work per turn even when both pipes stay ready.
            if bytes_since_yield >= 64 * 1024 {
                bytes_since_yield = 0;
                tokio::task::yield_now().await;
            }
        }
    }
    if let Some(tx) = updates {
        tx.send_replace(preview.snapshot());
    }
    let [stdout, stderr] = captures;
    Ok((stdout.finish(), stderr.finish()))
}

#[cfg(test)]
mod tests;
