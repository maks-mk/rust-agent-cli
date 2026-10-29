use super::{load_env_file, pick_config_dir};
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
    assert_eq!(pick_config_dir(Some(&exe), &manifest), exe);
    let _ = std::fs::remove_dir_all(&exe);
    let _ = std::fs::remove_dir_all(&manifest);
}

#[test]
fn source_checkout_is_used_when_the_executable_has_no_config() {
    let exe = temp_config_dir("empty-exe");
    let manifest = temp_config_dir("source");
    std::fs::write(manifest.join("mcp.json"), "{}").unwrap();
    assert_eq!(pick_config_dir(Some(&exe), &manifest), manifest);
    std::fs::write(manifest.join(".env"), "OPENAI_MODEL=source").unwrap();
    assert_eq!(pick_config_dir(Some(&exe), &manifest), manifest);
    let _ = std::fs::remove_dir_all(&exe);
    let _ = std::fs::remove_dir_all(&manifest);
}

#[test]
fn env_file_is_loaded_from_the_config_directory() {
    let dir = temp_config_dir("env-load");
    std::fs::write(dir.join(".env"), "RUST_AGENT_CONFIG_PROBE=loaded\n").unwrap();
    load_env_file(&dir);
    assert_eq!(
        std::env::var("RUST_AGENT_CONFIG_PROBE").as_deref(),
        Ok("loaded")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn executable_directory_is_the_fallback() {
    let exe = temp_config_dir("fallback-exe");
    let manifest = temp_config_dir("fallback-manifest");
    assert_eq!(pick_config_dir(Some(&exe), &manifest), exe);
    assert_eq!(pick_config_dir(None, &manifest), manifest);
    let _ = std::fs::remove_dir_all(&exe);
    let _ = std::fs::remove_dir_all(&manifest);
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
