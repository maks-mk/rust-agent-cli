use super::*;
use std::collections::HashSet;
use std::path::PathBuf;

fn config(model: &str, folder: &str) -> Config {
    Config {
        api_mode: crate::agent::ApiMode::Chat,
        base_url: "https://api.example.com/v1".to_string(),
        api_key: String::new(),
        model: model.to_string(),
        reasoning_effort: None,
        reasoning_efforts: Vec::new(),
        config_dir: PathBuf::from("."),
        root: PathBuf::from(folder),
        max_steps: crate::agent::DEFAULT_MAX_STEPS,
        context_tokens: crate::agent::DEFAULT_CONTEXT_TOKENS,
        auto_approve: false,
    }
}

fn text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

fn has_pixel_art(width: usize, ascii: bool) -> bool {
    let padding = width.saturating_sub(2).saturating_div(2).min(2);
    let available = if ascii {
        width.saturating_sub(padding)
    } else {
        width.saturating_sub(2 + padding * 2)
    };
    available >= 72
}

#[test]
fn truncation_respects_display_width() {
    for ellipsis in ['.', '…'] {
        for value in ["", "short", "模型模型模型", "a模型bc", "cafe\u{301}"] {
            for width in 0..=16 {
                for result in [
                    fit(value, width, ellipsis),
                    fit_left(value, width, ellipsis),
                ] {
                    assert!(Line::from(result.clone()).width() <= width);
                    if value.chars().map(cw).sum::<usize>() <= width {
                        assert_eq!(result, value);
                    }
                }
            }
        }
        assert_eq!(fit("abcdef", 4, ellipsis), format!("abc{ellipsis}"));
        assert_eq!(fit_left("abcdef", 4, ellipsis), format!("{ellipsis}def"));
        assert_eq!(fit("模型", 2, ellipsis), ellipsis.to_string());
        assert_eq!(fit_left("模型", 2, ellipsis), ellipsis.to_string());
    }
}

#[test]
fn pixel_wordmark_scales_and_uses_grayscale() {
    for (available, expected_width) in [(72, 72), (90, 90), (108, 108)] {
        let rows = pixel_wordmark(false, available).expect("wordmark should fit");
        assert_eq!(rows.len(), 7);
        for row in &rows {
            assert_eq!(Line::from(row.clone()).width(), expected_width);
        }
        let shades: HashSet<(u8, u8, u8)> = rows
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|span| match span.style.fg {
                Some(Color::Rgb(r, g, b)) => Some((r, g, b)),
                _ => None,
            })
            .collect();
        assert!(shades.len() > 5, "wordmark should use a grayscale gradient");
        assert!(shades.iter().all(|(r, g, b)| r == g && g == b));
    }
    assert!(pixel_wordmark(false, 71).is_none());
}

#[test]
fn banners_fit_without_wrapping() {
    let long_model = "模型-model-".repeat(20);
    let long_folder = format!("{}rust-agent", "目录/".repeat(40));
    let configs = [
        config("demo", "rust-agent"),
        config(&long_model, &long_folder),
        config("cafe\u{301}", "目录/cafe\u{301}"),
        config("", ""),
    ];
    for ascii in [false, true] {
        for cfg in &configs {
            for width in 0..=120 {
                let lines = banner_with_glyphs(cfg, width, glyphs_for(ascii));
                if width < 2 {
                    assert!(lines.is_empty());
                    continue;
                }
                let art = has_pixel_art(width, ascii);
                let expected_lines = match (ascii, art) {
                    (true, true) => 12,
                    (true, false) => 6,
                    (false, true) => 15,
                    (false, false) => 9,
                };
                assert_eq!(lines.len(), expected_lines, "width={width}, ascii={ascii}");
                for (line, hang) in &lines {
                    assert!(line.width() <= width, "width={width}, ascii={ascii}");
                    let wrapped = wrap_line(line, width, *hang);
                    assert_eq!(wrapped.len(), 1, "width={width}, ascii={ascii}");
                    assert_eq!(text(&wrapped[0]), text(line));
                }
                if ascii {
                    assert!(text(&lines[0].0).is_ascii());
                } else {
                    let frame_width = lines[0].0.width();
                    for (line, _) in &lines {
                        assert_eq!(line.width(), frame_width, "width={width}");
                    }
                    assert!(text(&lines[0].0).starts_with('╭'));
                    assert!(text(&lines.last().unwrap().0).starts_with('╰'));
                    for (line, _) in &lines[1..lines.len() - 1] {
                        let value = text(line);
                        assert!(value.starts_with('│') && value.ends_with('│'));
                    }
                }
            }
        }
    }
}

#[test]
fn unicode_banner_uses_pixel_wordmark_and_gray_metadata() {
    let cfg = config("demo", "rust-agent");
    let lines = banner_with_glyphs(&cfg, 100, glyphs_for(false));
    assert_eq!(lines.len(), 15);
    assert!(text(&lines[1].0).contains("██") || text(&lines[1].0).contains("▓▓"));
    assert!(text(&lines[7].0).contains("██") || text(&lines[7].0).contains("▓▓"));
    assert!(text(&lines[8].0).contains("rust-agent v"));
    assert!(text(&lines[10].0).contains("model"));
    assert!(text(&lines[11].0).contains("server"));
    assert!(text(&lines[12].0).contains("folder"));
    for (line, _) in &lines[1..8] {
        assert!(text(line).starts_with("│  "));
    }
    let blank = text(&lines[9].0);
    assert!(blank.trim_matches('│').chars().all(|c| c == ' '));
}

#[test]
fn ascii_banner_is_borderless_and_has_pixel_wordmark() {
    let cfg = config("demo", "rust-agent");
    let lines = banner_with_glyphs(&cfg, 80, glyphs_for(true));
    assert_eq!(lines.len(), 12);
    assert!(text(&lines[0].0).contains("##") || text(&lines[0].0).contains("::"));
    assert!(text(&lines[7].0).contains(&format!("rust-agent v{}", env!("CARGO_PKG_VERSION"))));
    assert_eq!(text(&lines[9].0), "  model   demo");
    assert_eq!(text(&lines[10].0), "  server  api.example.com");
    assert_eq!(text(&lines[11].0), "  folder  rust-agent");
    for (line, _) in &lines {
        let value = text(line);
        assert!(value.is_ascii());
        assert!(!value.contains('|'));
        assert!(!value.contains("+-") && !value.contains("-+"));
    }
    for index in [7, 9, 10, 11] {
        assert!(!text(&lines[index].0).ends_with(' '));
    }
}

#[test]
fn banner_hides_the_verbatim_folder_prefix() {
    let cfg = config("demo", r"\\?\D:\t1");
    let lines = banner_with_glyphs(&cfg, 80, glyphs_for(true));
    let folder = lines
        .iter()
        .map(|(line, _)| text(line))
        .find(|row| row.contains("folder"))
        .expect("folder row");

    assert!(folder.contains(r"D:\t1"), "unexpected folder row: {folder:?}");
    assert!(!folder.contains(r"\\?\"), "verbatim prefix leaked: {folder:?}");
}

#[test]
fn long_folder_keeps_its_tail() {
    let folder = format!("{}rust-agent", "目录/".repeat(40));
    let cfg = config(&"model-".repeat(20), &folder);
    for ascii in [false, true] {
        let lines = banner_with_glyphs(&cfg, 40, glyphs_for(ascii));
        let ellipsis = glyphs_for(ascii).ellipsis;
        let model_index = if ascii { 3 } else { 4 };
        let folder_index = if ascii { 5 } else { 6 };
        assert!(text(&lines[model_index].0).contains(ellipsis));
        let folder_row = text(&lines[folder_index].0);
        assert!(folder_row.contains(ellipsis));
        assert!(folder_row.contains("rust-agent"));
    }
}
