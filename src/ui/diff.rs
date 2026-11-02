//! Numbered edit previews with independent wrapping in the old/new columns.

use super::{cw, fg, glyphs, Color, DiffKind, Line, Modifier, RLine, Span, Style, MUTED};
use crate::agent::DiffLine;
use similar::{ChangeTag, TextDiff};
use std::time::Duration;

const SURFACE: Color = Color::Rgb(24, 24, 24);
const FOLD: Color = Color::Rgb(46, 46, 46);
const TEXT: Color = Color::Rgb(215, 221, 231);
const ADD: Color = Color::Rgb(29, 49, 34);
const DEL: Color = Color::Rgb(57, 30, 27);
const ADD_EMPHASIS: Color = Color::Rgb(43, 82, 52);
const DEL_EMPHASIS: Color = Color::Rgb(101, 40, 32);
const GREEN: Color = Color::Rgb(73, 211, 132);
const RED: Color = Color::Rgb(255, 104, 85);

pub(super) fn render(diff: &[DiffLine], width: usize) -> Vec<RLine> {
    if diff.is_empty() {
        return Vec::new();
    }
    let indent = 5.min(width.saturating_sub(1));
    let available = width.saturating_sub(indent).max(1);
    let digits = diff
        .iter()
        .flat_map(|line| [line.old_line, line.new_line])
        .flatten()
        .max()
        .unwrap_or(1)
        .to_string()
        .len();
    let split = available > 2 * (digits + 3 + 24);
    let left_width = (available - usize::from(split)) / 2;
    let right_width = available - 1 - left_width;
    let separator = if glyphs().ascii { "|" } else { "│" };
    let mut out = Vec::new();
    let mut i = 0;
    while i < diff.len() {
        let line = &diff[i];
        if line.kind == DiffKind::More {
            let style = fg(MUTED).bg(FOLD).add_modifier(Modifier::ITALIC);
            for spans in hard_wrap(&[Span::styled(line.text.clone(), style)], available) {
                out.push((padded_line(spans, available, indent, style), 0));
            }
            i += 1;
        } else if split && line.kind == DiffKind::Context {
            out.extend(split_row(
                Some(line),
                Some(line),
                left_width,
                right_width,
                digits,
                indent,
                separator,
            ));
            i += 1;
        } else if split {
            // A replacement block can have unequal old/new line counts. Pair
            // corresponding rows and leave the missing side shaded and empty.
            let end = diff[i..]
                .iter()
                .position(|line| !matches!(line.kind, DiffKind::Add | DiffKind::Del))
                .map_or(diff.len(), |offset| i + offset);
            let old: Vec<_> = diff[i..end]
                .iter()
                .filter(|l| l.kind == DiffKind::Del)
                .collect();
            let new: Vec<_> = diff[i..end]
                .iter()
                .filter(|l| l.kind == DiffKind::Add)
                .collect();
            for row in 0..old.len().max(new.len()) {
                out.extend(split_row(
                    old.get(row).copied(),
                    new.get(row).copied(),
                    left_width,
                    right_width,
                    digits,
                    indent,
                    separator,
                ));
            }
            i = end;
        } else {
            out.extend(unified_row(line, available, digits, indent));
            i += 1;
        }
    }
    out
}

fn style(kind: DiffKind) -> Style {
    fg(TEXT).bg(match kind {
        DiffKind::Add => ADD,
        DiffKind::Del => DEL,
        _ => SURFACE,
    })
}

fn highlighted(
    old: Option<&DiffLine>,
    new: Option<&DiffLine>,
) -> (Vec<Span<'static>>, Vec<Span<'static>>) {
    let plain = |line: Option<&DiffLine>| {
        line.map(|line| vec![Span::styled(line.text.clone(), style(line.kind))])
            .unwrap_or_default()
    };
    let (Some(old), Some(new)) = (old, new) else {
        return (plain(old), plain(new));
    };
    // Bound inline matching independently of the line diff. Very long rows
    // still wrap in full, with the ordinary added/removed background.
    if old.kind != DiffKind::Del
        || new.kind != DiffKind::Add
        || old.text.len() + new.text.len() > 4_000
    {
        return (plain(Some(old)), plain(Some(new)));
    }
    let diff = TextDiff::configure()
        .timeout(Duration::from_millis(10))
        .diff_chars(&old.text, &new.text);
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for change in diff.iter_all_changes() {
        let text = change.value().to_string();
        match change.tag() {
            ChangeTag::Equal => {
                left.push(Span::styled(text.clone(), style(DiffKind::Del)));
                right.push(Span::styled(text, style(DiffKind::Add)));
            }
            ChangeTag::Delete => {
                left.push(Span::styled(text, style(DiffKind::Del).bg(DEL_EMPHASIS)))
            }
            ChangeTag::Insert => {
                right.push(Span::styled(text, style(DiffKind::Add).bg(ADD_EMPHASIS)))
            }
        }
    }
    (left, right)
}

fn split_row(
    old: Option<&DiffLine>,
    new: Option<&DiffLine>,
    left_width: usize,
    right_width: usize,
    digits: usize,
    indent: usize,
    separator: &str,
) -> Vec<RLine> {
    let (left, right) = highlighted(old, new);
    let left = cell(old, left, left_width, digits, true);
    let right = cell(new, right, right_width, digits, false);
    (0..left.len().max(right.len()))
        .map(|row| {
            let mut spans = vec![Span::raw(" ".repeat(indent))];
            spans.extend(left.get(row).cloned().unwrap_or_else(|| blank(left_width)));
            spans.push(Span::styled(separator.to_string(), fg(FOLD).bg(SURFACE)));
            spans.extend(
                right
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| blank(right_width)),
            );
            (Line::from(spans), 0)
        })
        .collect()
}

fn blank(width: usize) -> Vec<Span<'static>> {
    vec![Span::styled(" ".repeat(width), style(DiffKind::Context))]
}

fn cell(
    line: Option<&DiffLine>,
    spans: Vec<Span<'static>>,
    width: usize,
    digits: usize,
    old: bool,
) -> Vec<Vec<Span<'static>>> {
    let Some(line) = line else {
        return vec![blank(width)];
    };
    let gutter = digits + 3;
    let row_style = style(line.kind);
    let color = match line.kind {
        DiffKind::Add => GREEN,
        DiffKind::Del => RED,
        _ => MUTED,
    };
    hard_wrap(&spans, width - gutter)
        .into_iter()
        .enumerate()
        .map(|(row, body)| {
            let number = if row == 0 {
                if old {
                    line.old_line
                } else {
                    line.new_line
                }
            } else {
                None
            };
            let number = number.map(|n| n.to_string()).unwrap_or_default();
            let marker = if matches!(line.kind, DiffKind::Add | DiffKind::Del) {
                if glyphs().ascii {
                    if line.kind == DiffKind::Add {
                        "+"
                    } else {
                        "-"
                    }
                } else {
                    "▌"
                }
            } else {
                " "
            };
            let mut spans = vec![Span::styled(
                format!("{marker} {number:>digits$} "),
                fg(color).bg(row_style.bg.unwrap()),
            )];
            spans.extend(body);
            let used: usize = spans.iter().map(Span::width).sum();
            spans.push(Span::styled(" ".repeat(width - used), row_style));
            spans
        })
        .collect()
}

fn unified_row(line: &DiffLine, width: usize, digits: usize, indent: usize) -> Vec<RLine> {
    let marker = match line.kind {
        DiffKind::Add => "+",
        DiffKind::Del => "-",
        _ => " ",
    };
    let old = line.old_line.map(|n| n.to_string()).unwrap_or_default();
    let new = line.new_line.map(|n| n.to_string()).unwrap_or_default();
    let prefix = if width > 2 * digits + 4 {
        format!("{old:>digits$} {new:>digits$} {marker} ")
    } else if width > digits + 3 {
        let number = if line.kind == DiffKind::Add {
            &new
        } else {
            &old
        };
        format!("{number:>digits$} {marker} ")
    } else if width > 1 {
        marker.to_string()
    } else {
        String::new()
    };
    let color = match line.kind {
        DiffKind::Add => GREEN,
        DiffKind::Del => RED,
        _ => MUTED,
    };
    let row_style = style(line.kind);
    hard_wrap(
        &[Span::styled(line.text.clone(), row_style)],
        width - prefix.len(),
    )
    .into_iter()
    .enumerate()
    .map(|(row, body)| {
        let mut spans = vec![Span::styled(
            if row == 0 {
                prefix.clone()
            } else {
                " ".repeat(prefix.len())
            },
            fg(color).bg(row_style.bg.unwrap()),
        )];
        spans.extend(body);
        (padded_line(spans, width, indent, row_style), 0)
    })
    .collect()
}

fn padded_line(
    mut spans: Vec<Span<'static>>,
    width: usize,
    indent: usize,
    style: Style,
) -> Line<'static> {
    let used: usize = spans.iter().map(Span::width).sum();
    spans.push(Span::styled(" ".repeat(width.saturating_sub(used)), style));
    spans.insert(0, Span::raw(" ".repeat(indent)));
    Line::from(spans)
}

/// Hard wrap without dropping source whitespace or letting a wide glyph cross
/// the separator. Expand tabs consistently with the rest of the chat renderer.
fn hard_wrap(spans: &[Span<'static>], width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut rows = vec![Vec::new()];
    let mut used = 0;
    for span in spans {
        for ch in span.content.chars().flat_map(|c| {
            let count = if c == '\t' {
                4
            } else {
                usize::from(!c.is_control())
            };
            std::iter::repeat_n(if c == '\t' { ' ' } else { c }, count)
        }) {
            let ch = if cw(ch) > width { '?' } else { ch };
            let size = cw(ch);
            if used + size > width {
                rows.push(Vec::new());
                used = 0;
            }
            let row = rows.last_mut().unwrap();
            if let Some(last) = row
                .last_mut()
                .filter(|s: &&mut Span<'static>| s.style == span.style)
            {
                last.content.to_mut().push(ch);
            } else {
                row.push(Span::styled(ch.to_string(), span.style));
            }
            used += size;
        }
    }
    rows
}

#[cfg(test)]
mod tests;
