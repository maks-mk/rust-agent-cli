use super::*;

fn line(kind: DiffKind, text: &str, old_line: Option<usize>, new_line: Option<usize>) -> DiffLine {
    DiffLine {
        kind,
        text: text.into(),
        old_line,
        new_line,
    }
}

#[test]
fn replacements_align_numbers_and_highlight_only_changed_fragments() {
    let diff = vec![
        line(
            DiffKind::Context,
            "background: model_dialog_bg;",
            Some(294),
            Some(295),
        ),
        line(DiffKind::Del, "border: none;", Some(295), None),
        line(DiffKind::Add, "border: 1px solid BORDER;", None, Some(296)),
    ];
    let rows = render(&diff, 110);
    assert_eq!(rows.len(), 2);
    let replacement = &rows[1].0;
    let text = replacement.to_string();
    let (old, new) = text
        .split_once(if glyphs().ascii { '|' } else { '│' })
        .unwrap();
    assert!(old.contains("295 border: none;"), "{text}");
    assert!(new.contains("296 border: 1px solid BORDER;"), "{text}");
    assert!(replacement
        .spans
        .iter()
        .any(|s| s.style.bg == Some(DEL_EMPHASIS)));
    assert!(replacement
        .spans
        .iter()
        .any(|s| s.style.bg == Some(ADD_EMPHASIS)));
    let (old, new) = highlighted(Some(&diff[1]), Some(&diff[2]));
    assert_eq!(
        old.iter().map(|s| s.content.as_ref()).collect::<String>(),
        diff[1].text
    );
    assert_eq!(
        new.iter().map(|s| s.content.as_ref()).collect::<String>(),
        diff[2].text
    );
    assert!(old.iter().take(7).all(|s| s.style.bg == Some(DEL)));
}

#[test]
fn insertions_deletions_and_folds_keep_the_other_column_empty() {
    let diff = vec![
        line(DiffKind::More, "16 unmodified lines", Some(16), Some(16)),
        line(
            DiffKind::Add,
            "SURFACE_SETTINGS = \"#1C1C1C\"",
            None,
            Some(18),
        ),
        line(DiffKind::More, "68 unmodified lines", Some(86), Some(87)),
        line(DiffKind::Del, "removed only", Some(88), None),
    ];
    let rows = render(&diff, 110);
    assert_eq!(rows.len(), 4);
    assert!(rows[0].0.spans.iter().any(|s| s.style.bg == Some(FOLD)));
    let sep = if glyphs().ascii { '|' } else { '│' };
    let insertion = rows[1].0.to_string();
    let (old, new) = insertion.split_once(sep).unwrap();
    assert!(old.trim().is_empty());
    assert!(new.contains("18 SURFACE_SETTINGS"));
    let deletion = rows[3].0.to_string();
    let (old, new) = deletion.split_once(sep).unwrap();
    assert!(old.contains("88 removed only"));
    assert!(new.trim().is_empty());
}

#[test]
fn wrapping_preserves_source_spaces_tabs_and_unicode_with_a_fixed_separator() {
    let source = format!("\tначало  {} e\u{301}  конец", "中".repeat(80));
    let diff = vec![
        line(DiffKind::Del, &source, Some(100), None),
        line(DiffKind::Add, "short new row", None, Some(101)),
    ];
    let wrapped = hard_wrap(&[Span::raw(source.clone())], 20);
    let rebuilt: String = wrapped
        .iter()
        .flatten()
        .map(|s| s.content.as_ref())
        .collect();
    assert_eq!(rebuilt, source.replace('\t', "    "));
    for width in [1, 2, 6, 16, 40, 80, 110, 160] {
        let rows = render(&diff, width);
        assert!(
            rows.iter().all(|(line, _)| line.width() <= width),
            "width={width}"
        );
        if width >= 80 {
            let positions: Vec<_> = rows
                .iter()
                .map(|(line, _)| {
                    line.to_string()
                        .chars()
                        .take_while(|&c| c != '│' && c != '|')
                        .map(cw)
                        .sum::<usize>()
                })
                .collect();
            assert!(
                positions.iter().all(|&p| p == positions[0]),
                "{positions:?}"
            );
            assert_eq!(
                rows.iter()
                    .filter(|(line, _)| line.to_string().contains("100"))
                    .count(),
                1
            );
        } else if width == 40 {
            let text = rows
                .iter()
                .map(|(line, _)| line.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(text.contains("100     -"), "{text}");
            assert!(text.contains("101 +"), "{text}");
        }
    }
}

#[test]
fn legacy_diff_lines_deserialize_and_new_numbers_survive_session_round_trip() {
    let legacy: DiffLine = serde_json::from_str(r#"{"kind":"Del","text":"old"}"#).unwrap();
    assert_eq!(legacy.old_line, None);
    assert_eq!(legacy.new_line, None);
    assert!(!render(&[legacy], 90).is_empty());
    let original = line(DiffKind::Context, "unchanged", Some(294), Some(295));
    let restored: DiffLine =
        serde_json::from_value(serde_json::to_value(&original).unwrap()).unwrap();
    assert_eq!(restored.old_line, original.old_line);
    assert_eq!(restored.new_line, original.new_line);
}
