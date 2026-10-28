use super::*;
use std::time::Duration;

/// A workspace directory for one test.
fn workspace() -> tempfile::TempDir {
    std::fs::create_dir_all("target").unwrap();
    tempfile::tempdir_in("target").unwrap()
}

/// Wait until the job left the running state.
async fn wait_for_exit(id: &str) -> JobInfo {
    for _ in 0..200 {
        let info = status(id).await.unwrap();
        if info.state != JobState::Running {
            return info;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the background job did not finish in time");
}

/// Whether a process with that id still exists.
fn process_alive(pid: u32) -> bool {
    #[cfg(windows)]
    {
        let output = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output();
        match output {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout);
                text.contains(&format!("\"{pid}\""))
            }
            Err(_) => true,
        }
    }
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(true)
    }
}

#[tokio::test]
async fn background_job_writes_its_log_and_reports_the_exit_code() {
    let dir = workspace();
    let job = start("echo background-marker; exit 3", dir.path())
        .await
        .unwrap();
    assert!(job.id.starts_with("bg-"));
    assert_eq!(job.state, JobState::Running);
    // The log lives inside the workspace, so the model can read it.
    assert!(job.log_rel.starts_with(".agent_state/background/"));
    assert!(job.log_rel.ends_with(".log"));

    let info = wait_for_exit(&job.id).await;
    assert_eq!(info.state, JobState::Exited(3));
    let log = std::fs::read_to_string(&info.log_path).unwrap();
    // The marker is a line of its own; the header mentions the command.
    assert!(
        log.lines().any(|line| line.trim() == "background-marker"),
        "log was: {log}"
    );
}

#[tokio::test]
async fn background_jobs_are_listed_and_unknown_ids_fail() {
    let dir = workspace();
    let job = start("echo listed", dir.path()).await.unwrap();
    let jobs = list().await;
    // The registry is shared by the whole test process, so only the
    // membership of the job is checked.
    assert!(jobs.iter().any(|entry| entry.id == job.id));
    assert!(status("bg-unknown").await.is_err());
    assert!(stop("bg-unknown").await.is_err());
}

#[tokio::test]
async fn stopping_a_background_job_ends_the_process() {
    let dir = workspace();
    let command = if cfg!(windows) {
        "Start-Sleep -Seconds 60"
    } else {
        "sleep 60"
    };
    let job = start(command, dir.path()).await.unwrap();
    assert_eq!(status(&job.id).await.unwrap().state, JobState::Running);

    let stopped = stop(&job.id).await.unwrap();
    assert_eq!(stopped.state, JobState::Stopped);
    // The state of a stopped job stays "stopped" and not the exit code of
    // the killed process.
    assert_eq!(status(&job.id).await.unwrap().state, JobState::Stopped);

    for _ in 0..60 {
        if !process_alive(job.pid) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(!process_alive(job.pid), "the process should be gone");
}

#[tokio::test]
async fn failed_stop_command_returns_an_error() {
    let dir = workspace();
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("cmd.exe");
        command.args(["/D", "/C", "exit 7"]);
        command
    };
    #[cfg(unix)]
    let mut command = {
        let mut command = Command::new("sh");
        command.args(["-c", "exit 7"]);
        command
    };
    command.current_dir(dir.path());
    let error = run_stop_command(&mut command).await.unwrap_err();
    assert!(error.to_string().contains("tree termination failed"));
    assert!(kill_tree(0).await.is_err());
}

#[tokio::test]
async fn locked_job_does_not_block_another_jobs_status() {
    let make_job = || {
        let id = new_id();
        let handle = Arc::new(Mutex::new(Job {
            info: JobInfo {
                id: id.clone(),
                command: String::new(),
                pid: 0,
                log_path: PathBuf::new(),
                log_rel: String::new(),
                started_at: Local::now(),
                state: JobState::Stopped,
            },
            child: None,
            stopped: true,
        }));
        (id, handle)
    };
    let (blocked_id, blocked) = make_job();
    let (other_id, other) = make_job();
    jobs()
        .lock()
        .await
        .insert(blocked_id.clone(), blocked.clone());
    jobs().lock().await.insert(other_id.clone(), other);
    let guard = blocked.lock().await;
    let waiting = status(&blocked_id);
    tokio::pin!(waiting);
    assert!(
        tokio::time::timeout(Duration::from_millis(20), &mut waiting)
            .await
            .is_err()
    );
    let result = tokio::time::timeout(Duration::from_secs(1), status(&other_id)).await;
    drop(guard);
    assert_eq!(waiting.await.unwrap().state, JobState::Stopped);
    let mut registry = jobs().lock().await;
    registry.remove(&blocked_id);
    registry.remove(&other_id);
    assert_eq!(result.unwrap().unwrap().state, JobState::Stopped);
}

#[tokio::test]
async fn log_tail_returns_the_end_of_the_log() {
    let dir = workspace();
    let job = start("echo tail-marker", dir.path()).await.unwrap();
    wait_for_exit(&job.id).await;
    let tail = log_tail(&job.log_path, 4096).await.unwrap();
    assert!(tail.contains("tail-marker"), "tail was: {tail}");
    // A small budget keeps only the end of the log.
    let short = log_tail(&job.log_path, 10).await.unwrap();
    assert!(short.len() <= 10, "tail was: {short}");
}
