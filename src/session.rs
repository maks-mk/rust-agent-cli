//! Local persistence for conversation sessions.
//!
//! Each session is one JSON file. Updates are written through a temporary file
//! and a backup so interrupted writes can be recovered on the next launch.

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub const UNTITLED: &str = "Новая сессия";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub id: String,
    pub title: String,
    pub workspace: String,
    pub model: String,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub history: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub workspace: String,
    pub model: String,
    pub updated_at: String,
}

pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn open(
        config_dir: &Path,
        workspace: &Path,
        model: &str,
        reasoning_effort: Option<&str>,
    ) -> Result<(Self, Session, bool)> {
        let dir = config_dir.join("sessions");
        fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create session directory {}", dir.display()))?;
        let store = Self { dir };

        let mut restored_session = store
            .read_active_id()
            .and_then(|id| store.load(&id).ok().flatten());
        // If the active pointer is damaged, recover the most recently updated session.
        if restored_session.is_none() {
            if let Ok(sessions) = store.list() {
                if let Some(latest) = sessions.first() {
                    restored_session = store.load(&latest.id).ok().flatten();
                }
            }
        }
        if let Some(mut session) = restored_session {
            if !Path::new(&session.workspace).is_dir() {
                session.workspace = workspace.to_string_lossy().into_owned();
            }
            return Ok((store, session, true));
        }

        let mut session = store.new_session(workspace, model, reasoning_effort)?;
        store.save(&mut session)?;
        store.activate(&session.id)?;
        Ok((store, session, false))
    }

    pub fn new_session(
        &self,
        workspace: &Path,
        model: &str,
        reasoning_effort: Option<&str>,
    ) -> Result<Session> {
        let now = timestamp();
        let mut id = format!("s{}", Utc::now().timestamp_micros());
        let mut suffix = 1u32;
        while self.session_path(&id).exists() {
            id = format!("s{}-{suffix}", Utc::now().timestamp_micros());
            suffix += 1;
        }
        Ok(Session {
            version: 1,
            id,
            title: UNTITLED.into(),
            workspace: workspace.to_string_lossy().into_owned(),
            model: model.into(),
            reasoning_effort: reasoning_effort.map(str::to_string),
            created_at: now.clone(),
            updated_at: now,
            history: Vec::new(),
        })
    }

    pub fn save(&self, session: &mut Session) -> Result<()> {
        session.updated_at = timestamp();
        write_json_recoverable(&self.session_path(&session.id), session)?;
        Ok(())
    }

    pub fn activate(&self, id: &str) -> Result<()> {
        write_json_recoverable(&self.dir.join("active.json"), &serde_json::json!({"id": id}))?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Option<Session>> {
        // IDs are generated internally and validated to avoid path traversal.
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            return Ok(None);
        }
        let path = self.session_path(id);
        if !path.is_file() && !backup_path(&path).is_file() {
            return Ok(None);
        }
        let session: Session = read_json_recoverable(&path)?;
        if session.id != id {
            return Ok(None);
        }
        Ok(Some(session))
    }

    pub fn list(&self) -> Result<Vec<SessionSummary>> {
        let mut sessions = Vec::new();
        let entries = fs::read_dir(&self.dir)
            .with_context(|| format!("cannot read session directory {}", self.dir.display()))?;
        let mut candidates = Vec::new();
        for entry in entries {
            let path = entry?.path();
            let file_name = path.file_name().and_then(|name| name.to_str()).unwrap_or_default();
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("json") if file_name != "active.json" => candidates.push(path),
                // Recover a session interrupted while its current JSON was being replaced.
                Some("bak") if !path.with_extension("json").exists() => candidates.push(path),
                _ => {}
            }
        }
        for path in candidates {
            let parsed = if path.extension().and_then(|ext| ext.to_str()) == Some("bak") {
                fs::read(&path)
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<Session>(&bytes).ok())
            } else {
                read_json_recoverable::<Session>(&path).ok()
            };
            let Some(session) = parsed else { continue };
            sessions.push(SessionSummary {
                id: session.id,
                title: session.title,
                workspace: session.workspace,
                model: session.model,
                updated_at: session.updated_at,
            });
        }
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    fn read_active_id(&self) -> Option<String> {
        let value: Value = read_json_recoverable(&self.dir.join("active.json")).ok()?;
        value.get("id")?.as_str().map(str::to_string)
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}

pub fn title_from_prompt(prompt: &str) -> String {
    let first_line = prompt.lines().next().unwrap_or_default();
    let normalized = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() {
        return UNTITLED.into();
    }
    let mut title: String = normalized.chars().take(56).collect();
    if normalized.chars().count() > 56 {
        title.push('…');
    }
    title
}

pub fn format_timestamp(value: &str) -> String {
    // The RFC3339 form is kept on disk; this compact form is for the selector.
    value
        .get(0..16)
        .map(|s| s.replace('T', " "))
        .unwrap_or_else(|| value.to_string())
}

fn timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn backup_path(path: &Path) -> PathBuf {
    path.with_extension("bak")
}

fn write_json_recoverable(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    let tmp = path.with_extension("tmp");
    let backup = backup_path(path);
    fs::write(&tmp, bytes).with_context(|| format!("cannot write {}", tmp.display()))?;

    if path.exists() {
        let _ = fs::remove_file(&backup);
        fs::rename(path, &backup)
            .with_context(|| format!("cannot back up {}", path.display()))?;
    }
    if let Err(error) = fs::rename(&tmp, path) {
        if backup.exists() && !path.exists() {
            let _ = fs::rename(&backup, path);
        }
        let _ = fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("cannot replace {}", path.display()));
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

fn read_json_recoverable<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    match fs::read(path) {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(value) => Ok(value),
            Err(primary) => {
                let backup = backup_path(path);
                let bytes = fs::read(&backup)
                    .with_context(|| format!("invalid JSON in {}: {primary}", path.display()))?;
                serde_json::from_slice(&bytes)
                    .with_context(|| format!("invalid backup {}", backup.display()))
            }
        },
        Err(primary) => {
            let backup = backup_path(path);
            let bytes = fs::read(&backup)
                .with_context(|| format!("cannot read {}: {primary}", path.display()))?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("invalid backup {}", backup.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn saves_and_restores_active_session() {
        let dir = tempdir().unwrap();
        let workspace = dir.path().join("project");
        fs::create_dir(&workspace).unwrap();
        let (store, mut session, restored) =
            SessionStore::open(dir.path(), &workspace, "model-a", Some("high")).unwrap();
        assert!(!restored);
        session.title = "Test session".into();
        session.history.push(serde_json::json!({"role":"user", "content":"hello"}));
        store.save(&mut session).unwrap();

        let (store2, restored_session, restored) =
            SessionStore::open(dir.path(), &workspace, "model-b", None).unwrap();
        assert!(restored);
        assert_eq!(restored_session.id, session.id);
        assert_eq!(restored_session.title, "Test session");
        assert_eq!(restored_session.model, "model-a");
        assert_eq!(restored_session.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(restored_session.history.len(), 1);
        assert_eq!(store2.list().unwrap().len(), 1);
    }

    #[test]
    fn title_uses_first_line_and_is_limited() {
        assert_eq!(title_from_prompt("Fix the bug\nand test"), "Fix the bug");
        assert_eq!(title_from_prompt("  one   two "), "one two");
        assert!(title_from_prompt(&"я".repeat(100)).chars().count() <= 57);
    }
}
