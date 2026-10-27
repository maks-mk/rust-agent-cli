use super::*;
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
                assert_eq!(lines.len(), if ascii { 7 } else { 10 });
                let content_lines = if ascii { 5 } else { 8 };
                for (line, hang) in &lines[..content_lines] {
                    assert!(line.width() <= width, "width={width}, ascii={ascii}");
                    let wrapped = wrap_line(line, width, *hang);
                    assert_eq!(wrapped.len(), 1, "width={width}, ascii={ascii}");
                    assert_eq!(text(&wrapped[0]), text(line));
                }
                if ascii {
                    assert!(text(&lines[1].0).trim().is_empty());
                    assert!(text(&lines[5].0).is_empty());
                } else {
                    let frame_width = lines[0].0.width();
                    for (line, _) in &lines[..8] {
                        assert_eq!(line.width(), frame_width, "width={width}");
                    }
                    assert!(text(&lines[0].0).starts_with('╭'));
                    assert!(text(&lines[7].0).starts_with('╰'));
                    for (line, _) in &lines[1..7] {
                        let value = text(line);
                        assert!(value.starts_with('│') && value.ends_with('│'));
                    }
                    assert_eq!(text(&lines[2].0), text(&lines[6].0));
                    assert!(text(&lines[8].0).is_empty());
                }
            }
        }
    }
}

#[test]
fn unicode_banner_has_balanced_padding() {
    let cfg = config("demo", "rust-agent");
    let lines = banner_with_glyphs(&cfg, 80, glyphs_for(false));
    assert!(text(&lines[1].0).starts_with("│  ⌘ rust-agent v"));
    assert!(text(&lines[3].0).starts_with("│  model   demo"));
    for (line, _) in &lines[1..7] {
        let value = text(line);
        assert!(value.starts_with("│  "));
        assert!(value.ends_with("  │"));
    }
    let blank = text(&lines[2].0);
    assert!(blank.trim_matches('│').chars().all(|c| c == ' '));
}

#[test]
fn ascii_banner_is_borderless_with_aligned_labels() {
    let cfg = config("demo", "rust-agent");
    let lines = banner_with_glyphs(&cfg, 80, glyphs_for(true));
    assert_eq!(lines.len(), 7);
    assert_eq!(
        text(&lines[0].0),
        format!("  >_ rust-agent v{}", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(text(&lines[2].0), "  model   demo");
    assert_eq!(text(&lines[3].0), "  server  api.example.com");
    assert_eq!(text(&lines[4].0), "  folder  rust-agent");
    assert!(text(&lines[1].0).trim().is_empty());
    assert!(text(&lines[5].0).is_empty());
    assert!(text(&lines[6].0).contains("/help commands"));
    for (line, _) in &lines {
        let value = text(line);
        assert!(value.is_ascii());
        assert!(!value.contains('|'));
        assert!(!value.contains("+-") && !value.contains("-+"));
    }
    for index in [0, 2, 3, 4] {
        assert!(!text(&lines[index].0).ends_with(' '));
    }
    assert_eq!(lines[0].0.spans[1].style.fg, Some(ACCENT));
    assert!(lines[0].0.spans[2]
        .style
        .add_modifier
        .contains(Modifier::BOLD));
}

#[test]
fn long_folder_keeps_its_tail() {
    let folder = format!("{}rust-agent", "目录/".repeat(40));
    let cfg = config(&"model-".repeat(20), &folder);
    for ascii in [false, true] {
        let lines = banner_with_glyphs(&cfg, 40, glyphs_for(ascii));
        let ellipsis = glyphs_for(ascii).ellipsis;
        let model_index = if ascii { 2 } else { 3 };
        let folder_index = if ascii { 4 } else { 5 };
        assert!(text(&lines[model_index].0).contains(ellipsis));
        let folder_row = text(&lines[folder_index].0);
        assert!(folder_row.contains(ellipsis));
        assert!(folder_row.contains("rust-agent"));
    }
}
