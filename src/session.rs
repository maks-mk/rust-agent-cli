//! Local persistence for conversation sessions.
//!
//! Each session is one JSON file. Updates are written through a temporary file
//! and a backup so interrupted writes can be recovered on the next launch.

use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
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
    #[serde(default)]
    pub tool_entries: Vec<crate::app::ToolEntry>,
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
    // Fields drop in declaration order: join the writer before releasing the lock.
    writer: SessionWriter,
    // Keep the OS lock alive; the file must not be removed on close.
    _lock: File,
}

#[derive(Default)]
struct WriteState {
    pending: Option<(PathBuf, Session)>,
    writing: bool,
    stopping: bool,
    result: Option<std::result::Result<(), String>>,
}

type SharedWriteState = std::sync::Arc<(std::sync::Mutex<WriteState>, std::sync::Condvar)>;

struct SessionWriter {
    shared: SharedWriteState,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SessionWriter {
    fn new() -> Result<Self> {
        let shared = SharedWriteState::default();
        let state = shared.clone();
        let thread = std::thread::Builder::new()
            .name("session-writer".into())
            .spawn(move || {
                let (mutex, ready) = &*state;
                let mut state = mutex.lock().unwrap();
                loop {
                    if let Some((path, session)) = state.pending.take() {
                        state.writing = true;
                        drop(state);
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            write_json_recoverable(&path, &session)
                        }))
                        .map_err(|_| "session writer panicked".to_string())
                        .and_then(|result| result.map_err(|error| format!("{error:#}")));
                        state = mutex.lock().unwrap();
                        state.writing = false;
                        state.result = Some(result);
                        ready.notify_all();
                    } else if state.stopping {
                        break;
                    } else {
                        state = ready.wait(state).unwrap();
                    }
                }
            })
            .context("cannot start session writer")?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    fn flush(shared: &SharedWriteState) -> Result<()> {
        let (mutex, ready) = &**shared;
        let mut state = mutex.lock().unwrap();
        while state.pending.is_some() || state.writing {
            state = ready.wait(state).unwrap();
        }
        match state.result.take() {
            Some(Err(error)) => anyhow::bail!(error),
            _ => Ok(()),
        }
    }
}

impl Drop for SessionWriter {
    fn drop(&mut self) {
        let (mutex, ready) = &*self.shared;
        mutex.lock().unwrap().stopping = true;
        ready.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
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
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("store.lock"))
            .context("cannot open session store lock")?;
        lock.try_lock().with_context(|| {
            format!(
                "cannot lock session store {}; another agent may already be using it",
                dir.display()
            )
        })?;
        let store = Self {
            dir,
            _lock: lock,
            writer: SessionWriter::new()?,
        };

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
            tool_entries: Vec::new(),
        })
    }

    pub fn queue_save(&self, session: &mut Session) {
        session.updated_at = timestamp();
        let snapshot = (self.session_path(&session.id), session.clone());
        let (mutex, ready) = &*self.writer.shared;
        let previous = mutex.lock().unwrap().pending.replace(snapshot);
        // Release the mutex before freeing a potentially large old history.
        drop(previous);
        ready.notify_one();
    }

    pub fn take_save_result(&self) -> Option<std::result::Result<(), String>> {
        self.writer.shared.0.lock().unwrap().result.take()
    }

    pub async fn flush(&self) -> Result<()> {
        let shared = self.writer.shared.clone();
        tokio::task::spawn_blocking(move || SessionWriter::flush(&shared))
            .await
            .context("session flush task failed")?
    }

    pub fn save(&self, session: &mut Session) -> Result<()> {
        // A fresh explicit save retries a failed autosave, after draining it.
        let _ = SessionWriter::flush(&self.writer.shared);
        session.updated_at = timestamp();
        write_json_recoverable(&self.session_path(&session.id), session)?;
        Ok(())
    }

    pub fn activate(&self, id: &str) -> Result<()> {
        write_json_recoverable(
            &self.dir.join("active.json"),
            &serde_json::json!({"id": id}),
        )?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Option<Session>> {
        SessionWriter::flush(&self.writer.shared)?;
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
        SessionWriter::flush(&self.writer.shared)?;
        let mut sessions = Vec::new();
        let entries = fs::read_dir(&self.dir)
            .with_context(|| format!("cannot read session directory {}", self.dir.display()))?;
        let mut candidates = Vec::new();
        for entry in entries {
            let path = entry?.path();
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
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
    let tmp = path.with_extension("tmp");
    let backup = backup_path(path);
    let write_result = (|| -> Result<()> {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        // Drop would discard flush errors. Publish only fully written JSON.
        writer.flush()?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("cannot write {}", tmp.display()));
    }

    if path.exists() {
        let _ = fs::remove_file(&backup);
        fs::rename(path, &backup).with_context(|| format!("cannot back up {}", path.display()))?;
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

    const LOCK_PROBE_DIR: &str = "RUST_AGENT_TEST_LOCK_DIR";

    #[test]
    fn saves_and_restores_active_session() {
        let dir = tempdir().unwrap();
        let workspace = dir.path().join("project");
        fs::create_dir(&workspace).unwrap();
        let (store, mut session, restored) =
            SessionStore::open(dir.path(), &workspace, "model-a", Some("high")).unwrap();
        assert!(!restored);
        session.title = "Test session".into();
        session
            .history
            .push(serde_json::json!({"role":"user", "content":"hello"}));
        store.save(&mut session).unwrap();

        drop(store);
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
    fn store_lock_is_exclusive_and_released_on_drop() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        let error = SessionStore::open(dir.path(), dir.path(), "model", None)
            .err()
            .expect("a second store must not acquire the lock");
        assert!(error.to_string().contains("cannot lock session store"));
        drop(store);
        let (_store, restored, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        assert_eq!(restored.id, session.id);
    }

    #[test]
    fn serialization_failure_preserves_previous_json_and_removes_temp() {
        struct Broken;
        impl Serialize for Broken {
            fn serialize<S: serde::Serializer>(
                &self,
                _serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("injected serialization failure"))
            }
        }
        let dir = tempfile::tempdir_in("target").unwrap();
        let path = dir.path().join("session.json");
        let original = serde_json::json!({"history": ["keep me"]});
        write_json_recoverable(&path, &original).unwrap();
        assert!(write_json_recoverable(&path, &Broken).is_err());
        assert_eq!(read_json_recoverable::<Value>(&path).unwrap(), original);
        assert!(!path.with_extension("tmp").exists());
    }

    #[tokio::test]
    async fn queued_saves_keep_latest_snapshot_and_report_completion() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, mut session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        for revision in 0..100 {
            session.history = vec![serde_json::json!({"revision": revision})];
            store.queue_save(&mut session);
        }
        store.flush().await.unwrap();
        assert_eq!(
            store.load(&session.id).unwrap().unwrap().history,
            session.history
        );
        assert!(store.take_save_result().is_none());
    }

    #[test]
    fn explicit_save_follows_queued_snapshot() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, mut session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        session.title = "queued".into();
        store.queue_save(&mut session);
        session.title = "explicit".into();
        store.save(&mut session).unwrap();
        assert_eq!(store.load(&session.id).unwrap().unwrap().title, "explicit");
    }

    #[test]
    fn drop_drains_queue_before_releasing_store_lock() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, mut session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        session.title = "saved on drop".into();
        store.queue_save(&mut session);
        drop(store);
        let (_store, restored, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        assert_eq!(restored.title, "saved on drop");
    }

    #[tokio::test]
    async fn failed_queued_save_preserves_previous_session_and_can_be_retried() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, mut session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        let original = session.title.clone();
        let temp = store.session_path(&session.id).with_extension("tmp");
        fs::create_dir(&temp).unwrap();
        session.title = "retry me".into();
        store.queue_save(&mut session);
        assert!(store.flush().await.is_err());
        assert_eq!(store.load(&session.id).unwrap().unwrap().title, original);
        fs::remove_dir(temp).unwrap();
        store.queue_save(&mut session);
        store.flush().await.unwrap();
        assert_eq!(store.load(&session.id).unwrap().unwrap().title, "retry me");
    }

    #[test]
    fn damaged_current_and_active_files_recover_backup() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (store, session, _) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        let path = store.session_path(&session.id);
        fs::copy(&path, backup_path(&path)).unwrap();
        fs::write(&path, b"broken JSON").unwrap();
        fs::write(store.dir.join("active.json"), b"broken pointer").unwrap();
        drop(store);
        let (_store, restored, recovered) =
            SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        assert!(recovered);
        assert_eq!(restored.id, session.id);
    }

    #[test]
    fn store_lock_child_probe() {
        let Some(dir) = std::env::var_os(LOCK_PROBE_DIR) else {
            return;
        };
        let dir = PathBuf::from(dir);
        let error = SessionStore::open(&dir, &dir, "model", None)
            .err()
            .expect("another process must not acquire the store lock");
        assert!(error.to_string().contains("cannot lock session store"));
    }

    #[test]
    fn store_lock_excludes_another_process() {
        let dir = tempfile::tempdir_in("target").unwrap();
        let (_store, _, _) = SessionStore::open(dir.path(), dir.path(), "model", None).unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "session::tests::store_lock_child_probe"])
            .env(LOCK_PROBE_DIR, dir.path().canonicalize().unwrap())
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn title_uses_first_line_and_is_limited() {
        assert_eq!(title_from_prompt("Fix the bug\nand test"), "Fix the bug");
        assert_eq!(title_from_prompt("  one   two "), "one two");
        assert!(title_from_prompt(&"я".repeat(100)).chars().count() <= 57);
    }
}
