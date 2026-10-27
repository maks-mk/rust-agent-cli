//! Background jobs started by the `shell` tool.
//!
//! A background job runs detached from the agent: it survives the agent
//! process, writes its output to a log file inside the workspace and can be
//! inspected or stopped later by its job id. The registry lives in the
//! process, so a job started in one turn is still known in the next one.

#[cfg(test)]
mod tests;

use crate::agent::{executable_dir, powershell_command, powershell_command_fallback, shell_path};
use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Local};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Workspace directory that holds the logs of background jobs.
const LOG_DIR: &str = ".agent_state/background";

/// How long a stop request waits for the process to disappear.
const STOP_WAIT: Duration = Duration::from_secs(3);

/// State of a background job.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JobState {
    /// The process is still running.
    Running,
    /// The process finished on its own with the given exit code.
    Exited(i32),
    /// The process was stopped on request.
    Stopped,
}

impl JobState {
    /// Human readable state used in the tool output.
    pub fn label(self) -> String {
        match self {
            JobState::Running => "running".to_string(),
            JobState::Exited(code) => format!("exited with code {code}"),
            JobState::Stopped => "stopped".to_string(),
        }
    }
}

/// A snapshot of one background job.
#[derive(Clone, Debug)]
pub struct JobInfo {
    /// Identifier used by the `background_id` argument.
    pub id: String,
    /// The command line the job was started with.
    pub command: String,
    /// Process id of the detached process.
    pub pid: u32,
    /// Absolute path of the log file.
    pub log_path: PathBuf,
    /// Log path relative to the workspace, the form `read_file` expects.
    pub log_rel: String,
    /// When the job was started.
    pub started_at: DateTime<Local>,
    /// Current state of the job.
    pub state: JobState,
}

impl JobInfo {
    /// Seconds since the job started.
    pub fn elapsed_secs(&self) -> i64 {
        Local::now()
            .signed_duration_since(self.started_at)
            .num_seconds()
            .max(0)
    }

    /// A description of the job for the model and the TUI.
    pub fn describe(&self) -> String {
        format!(
            "background job {id}\nstate: {state}\npid: {pid}\ncommand: {command}\nlog: {log}\nstarted: {started} ({elapsed}s ago)",
            id = self.id,
            state = self.state.label(),
            pid = self.pid,
            command = self.command,
            log = self.log_rel,
            started = self.started_at.format("%Y-%m-%d %H:%M:%S"),
            elapsed = self.elapsed_secs(),
        )
    }
}

/// A registered job and the handle needed to observe or stop it.
struct Job {
    info: JobInfo,
    child: Option<Child>,
    /// Set when the job was stopped on request; the state stays `Stopped`
    /// afterwards, because the exit code of a killed process means nothing.
    stopped: bool,
}

static JOBS: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();

/// The registry of background jobs of this agent process.
fn jobs() -> &'static Mutex<HashMap<String, Job>> {
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Counter that keeps the generated ids unique inside one run.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A short id that is unique inside one agent run.
fn new_id() -> String {
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let stamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("bg-{stamp:x}-{seq:x}")
}

/// Start `command` detached from the agent and register it as a background job.
///
/// The process keeps running after the agent exits, so its output goes to a
/// log file instead of a pipe. The log lives inside the workspace, which lets
/// the model read it with `read_file`.
pub async fn start(command: &str, root: &Path) -> Result<JobInfo> {
    let id = new_id();
    let (log_path, log_rel) = prepare_log(root, &id)?;
    write_log_header(&log_path, &id, command)?;

    let stdout = open_append(&log_path)?;
    let stderr = open_append(&log_path)?;
    let child = spawn_detached(command, root, stdout, stderr)?;

    let info = JobInfo {
        id: id.clone(),
        command: command.to_string(),
        pid: child.id().unwrap_or(0),
        log_path,
        log_rel,
        started_at: Local::now(),
        state: JobState::Running,
    };
    jobs().lock().await.insert(
        id,
        Job {
            info: info.clone(),
            child: Some(child),
            stopped: false,
        },
    );
    Ok(info)
}

/// Return the current state of a background job.
pub async fn status(id: &str) -> Result<JobInfo> {
    let mut map = jobs().lock().await;
    let job = map
        .get_mut(id)
        .ok_or_else(|| anyhow!("unknown background job '{id}'"))?;
    refresh(job).await;
    Ok(job.info.clone())
}

/// Stop a background job and report its final state.
pub async fn stop(id: &str) -> Result<JobInfo> {
    let mut map = jobs().lock().await;
    let job = map
        .get_mut(id)
        .ok_or_else(|| anyhow!("unknown background job '{id}'"))?;
    refresh(job).await;
    if job.info.state == JobState::Running {
        kill_tree(job.info.pid);
        if let Some(child) = job.child.as_mut() {
            let _ = child.start_kill();
        }
        // The job was stopped on request, so the state reports that and not
        // the exit code of the killed process.
        job.stopped = true;
        job.info.state = JobState::Stopped;
        let deadline = tokio::time::Instant::now() + STOP_WAIT;
        while tokio::time::Instant::now() < deadline {
            let Some(child) = job.child.as_mut() else {
                break;
            };
            match child.try_wait() {
                Ok(Some(_)) | Err(_) => break,
                Ok(None) => {}
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    Ok(job.info.clone())
}

/// Return every known background job, oldest first.
pub async fn list() -> Vec<JobInfo> {
    let mut map = jobs().lock().await;
    let mut out = Vec::with_capacity(map.len());
    for job in map.values_mut() {
        refresh(job).await;
        out.push(job.info.clone());
    }
    out.sort_by_key(|job| job.started_at);
    out
}

/// Read the tail of a log file so the model can see the recent output.
///
/// Only the last `max_bytes` are read, which keeps the answer small even when
/// the job produced a lot of output.
pub async fn log_tail(path: &Path, max_bytes: usize) -> Option<String> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};

    let mut file = tokio::fs::File::open(path).await.ok()?;
    let len = file.metadata().await.ok()?.len();
    let start = len.saturating_sub(max_bytes as u64);
    file.seek(std::io::SeekFrom::Start(start)).await.ok()?;
    let mut buf = vec![0u8; (len - start) as usize];
    file.read_exact(&mut buf).await.ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Update the state of a job from its process handle.
async fn refresh(job: &mut Job) {
    if job.stopped {
        return;
    }
    let Some(child) = job.child.as_mut() else {
        return;
    };
    if let Ok(Some(status)) = child.try_wait() {
        job.info.state = JobState::Exited(status.code().unwrap_or(-1));
    }
}

/// Choose the log file of a job and make sure its directory exists.
///
/// The log stays inside the workspace so the model can read it with
/// `read_file`. A workspace that cannot hold it falls back to the temp
/// directory, which stays readable from a shell command.
fn prepare_log(root: &Path, id: &str) -> Result<(PathBuf, String)> {
    let relative = Path::new(LOG_DIR).join(format!("{id}.log"));
    let inside = root.join(&relative);
    if let Some(parent) = inside.parent() {
        if std::fs::create_dir_all(parent).is_ok() {
            return Ok((inside, relative.to_string_lossy().replace('\\', "/")));
        }
    }
    let fallback = std::env::temp_dir().join("rust-agent").join("background");
    std::fs::create_dir_all(&fallback)
        .with_context(|| format!("cannot create {}", fallback.display()))?;
    let path = fallback.join(format!("{id}.log"));
    let shown = path.to_string_lossy().replace('\\', "/");
    Ok((path, shown))
}

/// Write the header that documents the job in its log file.
fn write_log_header(path: &Path, id: &str, command: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    writeln!(
        file,
        "# rust-agent background job {id} ({}): {command}",
        Local::now().format("%Y-%m-%d %H:%M:%S")
    )
    .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Open the log file for appending.
fn open_append(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))
}

/// Start the command in a detached process that outlives the agent.
fn spawn_detached(command: &str, root: &Path, stdout: File, stderr: File) -> Result<Child> {
    let path = shell_path(executable_dir().as_deref(), std::env::var_os("PATH"));
    let spawn = |program: &str, args: Vec<String>, stdout: File, stderr: File| {
        spawn_with(program, args, root, stdout, stderr, path.as_deref())
    };
    if cfg!(windows) {
        let (program, args) = powershell_command(command);
        match spawn(program, args, stdout.try_clone()?, stderr.try_clone()?) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let (program, args) = powershell_command_fallback(command);
                spawn(program, args, stdout, stderr)
            }
            other => other,
        }
    } else {
        spawn("sh", vec!["-c".into(), command.into()], stdout, stderr)
    }
    .context("cannot start the background command")
}

/// Spawn one process without a controlling terminal and without a kill on drop.
fn spawn_with(
    program: &str,
    args: Vec<String>,
    root: &Path,
    stdout: File,
    stderr: File,
    path: Option<&OsStr>,
) -> std::io::Result<Child> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if let Some(path) = path {
        command.env("PATH", path);
    }
    #[cfg(windows)]
    {
        // The job gets no console of its own, so it neither flashes a window
        // nor receives the interrupts of the terminal the agent runs in.
        // `DETACHED_PROCESS` is not used: PowerShell refuses to run a script
        // with it, while `CREATE_NO_WINDOW` starts it just fine.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(unix)]
    {
        // The child leads its own process group, so it survives the agent and
        // the whole group can be signalled at once.
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
    }
    // `kill_on_drop` stays off: a background job must outlive the agent.
    command.spawn()
}

/// Terminate a background job together with the processes it started.
fn kill_tree(pid: u32) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(unix)]
    {
        // The job leads its own process group, so the negative pid signals
        // every process the job started.
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
