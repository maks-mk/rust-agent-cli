//! Child-process output policy shared by shell, background jobs and MCP.
use tokio::process::Command;

/// Reduce terminal formatting without overriding explicit user/server settings.
pub(crate) fn apply_output_env(command: &mut Command) {
    for (name, value) in [("NO_COLOR", "1"), ("FORCE_COLOR", "0")] {
        let explicit = command.as_std().get_envs().any(|(key, _)| {
            if cfg!(windows) {
                key.to_string_lossy().eq_ignore_ascii_case(name)
            } else {
                key == name
            }
        });
        if !explicit && std::env::var_os(name).is_none() {
            command.env(name, value);
        }
    }
}

#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// No shared console and no flashing window; DETACHED_PROCESS breaks PowerShell.
#[cfg(windows)]
pub(crate) fn apply_windows_flags(command: &mut Command, extra_flags: u32) {
    command.creation_flags(CREATE_NO_WINDOW | extra_flags);
}

/// Remove terminal commands, retaining the last state of CR-updated lines.
/// This is text normalization, not a terminal emulator; ordinary text is unchanged.
pub(crate) fn terminal_safe_text(output: &str) -> String {
    let mut safe = String::with_capacity(output.len());
    let mut chars = output.chars().peekable();
    let mut line_start = 0;
    let mut carriage_return = false;
    while let Some(ch) = chars.next() {
        let sequence = match ch {
            '\x1b' => chars.next_if(|c| ('\u{0020}'..='\u{007e}').contains(c)),
            '\u{009b}' => Some('['),
            '\u{009d}' => Some(']'),
            '\u{0090}' => Some('P'),
            '\u{0098}' => Some('X'),
            '\u{009e}' => Some('^'),
            '\u{009f}' => Some('_'),
            _ => None,
        };
        match sequence {
            Some('[') => {
                while chars
                    .next_if(|c| ('\u{0020}'..='\u{003f}').contains(c))
                    .is_some()
                {}
                chars.next_if(|c| ('\u{0040}'..='\u{007e}').contains(c));
                continue;
            }
            Some(kind @ (']' | 'P' | 'X' | '^' | '_')) => {
                // OSC ends with BEL or ST; other control strings end with ST.
                // Stop at a newline too, so truncated sequences cannot eat log lines.
                while let Some(&c) = chars.peek() {
                    if c == '\n' || c == '\r' {
                        break;
                    }
                    chars.next();
                    if c == '\u{009c}' || (kind == ']' && c == '\x07') {
                        break;
                    }
                    if c == '\x1b' && chars.next_if_eq(&'\\').is_some() {
                        break;
                    }
                }
                continue;
            }
            Some(c) => {
                // Other ESC sequences have optional intermediates and a final byte.
                if ('\u{0020}'..='\u{002f}').contains(&c) {
                    while chars
                        .next_if(|c| ('\u{0020}'..='\u{002f}').contains(c))
                        .is_some()
                    {}
                    chars.next_if(|c| ('\u{0030}'..='\u{007e}').contains(c));
                }
                continue;
            }
            None => {}
        }
        match ch {
            '\r' => carriage_return = true,
            '\n' => {
                safe.push('\n');
                line_start = safe.len();
                carriage_return = false;
            }
            c if c == '\t' || !c.is_control() => {
                if carriage_return {
                    safe.truncate(line_start);
                    carriage_return = false;
                }
                safe.push(c);
            }
            _ => {}
        }
    }
    safe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_terminal_output() {
        for (input, expected) in [
            (
                "plain [32;1m text\nЮникод 🦀\tend",
                "plain [32;1m text\nЮникод 🦀\tend",
            ),
            ("\x1b[31mred\x1b[0m", "red"),
            ("\x1b[2J\x1b[1;1Htext\x1b[2K", "text"),
            ("\x1b]0;title\x07text", "text"),
            (
                "\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\",
                "link",
            ),
            ("\u{009b}31mred\u{009b}0m", "red"),
            ("\u{009d}title\u{009c}text", "text"),
            ("one\r\ntwo\r\n", "one\ntwo\n"),
            ("0%\r50%\r100%\nnext", "100%\nnext"),
            ("long progress\rOK\r", "OK"),
            ("one\nold\r\x1b[32mновое\x1b[0m", "one\nновое"),
            ("a\x00\x07\x08\x0b\x7f\u{0085}b\t", "ab\t"),
            ("text\x1b", "text"),
            ("\x1b[31\nnext", "\nnext"),
            ("\x1b]unfinished\nnext", "\nnext"),
            ("\x1bPpayload\x1b\\text", "text"),
            ("\x1b(Btext\x1b7", "text"),
        ] {
            let safe = terminal_safe_text(input);
            assert_eq!(safe, expected, "{input:?}");
            assert_eq!(
                terminal_safe_text(&safe),
                safe,
                "normalization is idempotent"
            );
        }
    }

    #[test]
    fn color_defaults_are_added_only_when_not_inherited() {
        let mut command = Command::new("unused");
        apply_output_env(&mut command);
        for (name, default) in [("NO_COLOR", "1"), ("FORCE_COLOR", "0")] {
            let setting = command.as_std().get_envs().find(|(key, _)| *key == name);
            if std::env::var_os(name).is_some() {
                assert!(setting.is_none(), "inherited {name} must not be overridden");
            } else {
                assert_eq!(setting.unwrap().1, Some(std::ffi::OsStr::new(default)));
            }
        }
    }

    #[test]
    fn explicit_color_settings_are_preserved() {
        let mut command = Command::new("unused");
        command.env("NO_COLOR", "custom").env("FORCE_COLOR", "3");
        apply_output_env(&mut command);
        let env: Vec<_> = command.as_std().get_envs().collect();
        assert!(env.iter().any(
            |(key, value)| *key == "NO_COLOR" && *value == Some(std::ffi::OsStr::new("custom"))
        ));
        assert!(
            env.iter()
                .any(|(key, value)| *key == "FORCE_COLOR"
                    && *value == Some(std::ffi::OsStr::new("3")))
        );
    }

    #[test]
    fn explicitly_removed_color_settings_stay_removed() {
        let mut command = Command::new("unused");
        command.env_remove("NO_COLOR").env_remove("FORCE_COLOR");
        apply_output_env(&mut command);
        let env: Vec<_> = command.as_std().get_envs().collect();
        assert!(env
            .iter()
            .any(|(key, value)| *key == "NO_COLOR" && value.is_none()));
        assert!(env
            .iter()
            .any(|(key, value)| *key == "FORCE_COLOR" && value.is_none()));
    }
}
