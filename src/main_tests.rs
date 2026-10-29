use super::{config_dir, load_env_file, pick_config_dir};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn temp_config_dir(name: &str) -> PathBuf {
    let dir = std::env::current_dir()
        .unwrap()
        .join("target")
        .join(format!("config-dir-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn config_next_to_the_executable_wins() {
    let exe = temp_config_dir("exe");
    let manifest = temp_config_dir("manifest");
    std::fs::write(exe.join(".env"), "OPENAI_MODEL=exe").unwrap();
    std::fs::write(manifest.join("mcp.json"), "{}").unwrap();
    assert_eq!(pick_config_dir(Some(&exe)).unwrap(), exe);
    let _ = std::fs::remove_dir_all(&exe);
    let _ = std::fs::remove_dir_all(&manifest);
}

#[test]
fn source_checkout_is_ignored_when_the_executable_has_no_config() {
    let exe = temp_config_dir("empty-exe");
    let manifest = temp_config_dir("source");
    std::fs::write(manifest.join("mcp.json"), "{}").unwrap();
    assert_eq!(pick_config_dir(Some(&exe)).unwrap(), exe);
    std::fs::write(manifest.join(".env"), "OPENAI_MODEL=source").unwrap();
    assert_eq!(pick_config_dir(Some(&exe)).unwrap(), exe);
    let _ = std::fs::remove_dir_all(&exe);
    let _ = std::fs::remove_dir_all(&manifest);
}

#[test]
fn env_file_overrides_inherited_variables() {
    let dir = temp_config_dir("env-load");
    std::fs::write(dir.join(".env"), "RUST_AGENT_CONFIG_PROBE=loaded\n").unwrap();
    let previous = std::env::var_os("RUST_AGENT_CONFIG_PROBE");
    std::env::set_var("RUST_AGENT_CONFIG_PROBE", "inherited");
    load_env_file(&dir).unwrap();
    assert_eq!(
        std::env::var("RUST_AGENT_CONFIG_PROBE").as_deref(),
        Ok("loaded")
    );
    if let Some(previous) = previous {
        std::env::set_var("RUST_AGENT_CONFIG_PROBE", previous);
    } else {
        std::env::remove_var("RUST_AGENT_CONFIG_PROBE");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn executable_directory_is_used_without_config_files() {
    let exe = temp_config_dir("empty-fallback-exe");
    assert_eq!(pick_config_dir(Some(&exe)).unwrap(), exe);
    assert!(pick_config_dir(None).is_err());
    assert_eq!(
        config_dir().unwrap(),
        std::env::current_exe().unwrap().parent().unwrap()
    );
    let _ = std::fs::remove_dir_all(&exe);
}

#[test]
fn missing_env_file_does_not_search_workspace_or_parents() {
    const CONFIG_DIR_PROBE: &str = "RUST_AGENT_CONFIG_TEST_DIR";
    const SEARCH_PROBE: &str = "RUST_AGENT_CONFIG_SEARCH_PROBE";
    if let Some(dir) = std::env::var_os(CONFIG_DIR_PROBE) {
        load_env_file(Path::new(&dir)).unwrap();
        assert!(std::env::var_os(SEARCH_PROBE).is_none());
        return;
    }

    let dir = temp_config_dir("env-no-fallback");
    let exe = dir.join("exe");
    let source = dir.join("source");
    let workspace = source.join("workspace");
    std::fs::create_dir(&exe).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::write(source.join(".env"), format!("{SEARCH_PROBE}=parent\n")).unwrap();
    std::fs::write(
        workspace.join(".env"),
        format!("{SEARCH_PROBE}=workspace\n"),
    )
    .unwrap();
    for _ in 0..2 {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::missing_env_file_does_not_search_workspace_or_parents",
            ])
            .env(CONFIG_DIR_PROBE, &exe)
            .env_remove(SEARCH_PROBE)
            .current_dir(&workspace)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated env probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        if workspace.join(".env").exists() {
            std::fs::remove_file(workspace.join(".env")).unwrap();
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_env_file_reports_an_error_without_secret_contents() {
    let dir = temp_config_dir("invalid-env");
    std::fs::write(
        dir.join(".env"),
        "SECRET_TEST_VALUE=\"private-test-marker\n",
    )
    .unwrap();
    let error = load_env_file(&dir).unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("invalid .env file"));
    assert!(message.contains(&dir.display().to_string()));
    assert!(!message.contains("private-test-marker"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Variables that come from the shell or the operating system instead of the
/// project configuration, plus the probe used by the `.env` loading test.
const EXTERNAL_ENV_VARS: &[&str] = &[
    "TERM",
    "TERM_PROGRAM",
    "WT_SESSION",
    "PATH",
    "RUST_AGENT_CONFIG_PROBE",
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().map(|ext| ext == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn env_var_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for pattern in ["env::var(\"", "env::var_os(\""] {
        for (index, _) in text.match_indices(pattern) {
            let rest = &text[index + pattern.len()..];
            if let Some(end) = rest.find('"') {
                names.push(rest[..end].to_string());
            }
        }
    }
    names
}

#[test]
fn env_example_documents_every_configured_variable() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let example = std::fs::read_to_string(root.join(".env.example")).expect(".env.example exists");

    let mut sources = Vec::new();
    rust_sources(&root.join("src"), &mut sources);
    let mut used: BTreeSet<String> = BTreeSet::new();
    for source in &sources {
        let text = std::fs::read_to_string(source).unwrap();
        used.extend(env_var_names(&text));
    }
    used.retain(|name| !EXTERNAL_ENV_VARS.contains(&name.as_str()));
    assert!(
        !used.is_empty(),
        "no environment variables found in the sources"
    );

    // Every variable the code reads has to be mentioned in `.env.example`.
    for name in &used {
        let mentioned = example.lines().any(|line| {
            let line = line.trim().trim_start_matches('#').trim();
            line.starts_with(&format!("{name}="))
        });
        assert!(
            mentioned,
            "{name} is read by the code but missing in .env.example"
        );
    }

    // And every active entry of `.env.example` has to be read by the code.
    for line in example.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let name = line.split('=').next().unwrap().trim();
        assert!(
            used.contains(name),
            "{name} is listed in .env.example but never read"
        );
    }
}
