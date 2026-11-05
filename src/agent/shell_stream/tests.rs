use super::*;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;

use crate::agent::shell_output_streaming;

#[tokio::test]
async fn real_parallel_shells_stream_and_cancel_independently() {
    let dir = tempfile::tempdir().unwrap();
    let mut runs = Vec::new();
    for id in 0..3 {
        let command = if cfg!(windows) {
            format!("[Console]::Error.Write('job-{id}'); Start-Sleep -Seconds 2; [Console]::Out.Write('done-{id}'); exit {id}")
        } else {
            format!("printf job-{id} >&2; sleep 2; printf done-{id}; exit {id}")
        };
        let root = dir.path().to_path_buf();
        let (tx, rx) = watch::channel(String::new());
        let task = tokio::spawn(async move {
            shell_output_streaming(&command, &root, Duration::from_secs(15), Some(tx)).await
        });
        runs.push((task, rx));
    }
    for (id, (task, rx)) in runs.iter_mut().enumerate() {
        timeout(Duration::from_secs(10), async {
            loop {
                rx.changed().await.unwrap();
                if rx.borrow_and_update().contains(&format!("job-{id}")) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert!(!task.is_finished(), "output must arrive before completion");
    }
    let (cancelled, mut updates) = runs.remove(0);
    cancelled.abort();
    assert!(cancelled.await.unwrap_err().is_cancelled());
    while updates.changed().await.is_ok() {}
    for (index, (task, rx)) in runs.into_iter().enumerate() {
        let id = index + 1;
        let run = task.await.unwrap().unwrap();
        assert_eq!(run.code, Some(id as i32));
        assert_eq!(run.stdout.text(), format!("done-{id}"));
        assert_eq!(run.stderr.text(), format!("job-{id}"));
        assert_eq!(&*rx.borrow(), &format!("job-{id}\ndone-{id}"));
    }
}

#[tokio::test]
async fn spawn_failure_closes_the_preview_channel() {
    let dir = tempfile::tempdir().unwrap();
    let (tx, mut rx) = watch::channel(String::new());
    let error = shell_output_streaming(
        "echo test",
        &dir.path().join("missing"),
        Duration::from_secs(2),
        Some(tx),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("cannot start"));
    assert!(rx.changed().await.is_err());
}

/// Run explicitly in release mode: measures four real processes and UI polling.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual shell throughput measurement"]
async fn shell_stream_stress() {
    let dir = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let mut runs = Vec::new();
    for id in 0..4 {
        let command = if cfg!(windows) {
            format!("$b = [Text.Encoding]::UTF8.GetBytes(('job-{id} ' + ('x' * 56) + [char]10) * 128); $o = [Console]::OpenStandardOutput(); $e = [Console]::OpenStandardError(); for ($i=0; $i -lt 1024; $i++) {{ $o.Write($b,0,$b.Length); $e.Write($b,0,$b.Length) }}")
        } else {
            format!("awk 'BEGIN {{ for (i=0;i<131072;i++) {{ print \"job-{id} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"; print \"job-{id} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\" > \"/dev/stderr\" }} }}'")
        };
        let root = dir.path().to_path_buf();
        let (tx, rx) = watch::channel(String::new());
        runs.push((
            tokio::spawn(async move {
                shell_output_streaming(&command, &root, Duration::from_secs(60), Some(tx)).await
            }),
            rx,
        ));
    }
    let mut polls = 0;
    let mut snapshots = 0;
    let mut worst_gap = Duration::ZERO;
    let mut streaming_gap = Duration::ZERO;
    let mut seen_output = [false; 4];
    let mut previous = std::time::Instant::now();
    while runs.iter().any(|(task, _)| !task.is_finished()) {
        tokio::time::sleep(UPDATE_INTERVAL).await;
        let now = std::time::Instant::now();
        worst_gap = worst_gap.max(now.duration_since(previous));
        if seen_output.iter().all(|seen| *seen) {
            streaming_gap = streaming_gap.max(now.duration_since(previous));
        }
        previous = now;
        polls += 1;
        for (id, (_, rx)) in runs.iter_mut().enumerate() {
            if rx.has_changed().unwrap_or(true) {
                let snapshot = rx.borrow_and_update();
                seen_output[id] |= !snapshot.is_empty();
                assert!(snapshot.lines().count() <= PREVIEW_LINES);
                assert!(snapshot.len() <= PREVIEW_LINES * (PREVIEW_LINE_BYTES + 1));
                let prefix = format!("job-{id}");
                assert!(snapshot
                    .lines()
                    .all(|line| { line.starts_with(&prefix) || prefix.starts_with(line) }));
                snapshots += 1;
            }
        }
    }
    let mut bytes = 0;
    for (task, _) in runs {
        let run = task.await.unwrap().unwrap();
        assert_eq!(run.code, Some(0));
        for stream in [run.stdout, run.stderr] {
            bytes += stream.total;
            assert!(
                stream.head.len() + stream.tail.len()
                    <= SHELL_STREAM_HEAD_BYTES + SHELL_STREAM_TAIL_BYTES
            );
            assert!(stream.text().contains("bytes omitted"));
        }
    }
    let elapsed = started.elapsed();
    eprintln!("shell stress: bytes={bytes}, elapsed={elapsed:?}, MiB/s={:.1}, polls={polls}, snapshots={snapshots}, max_poll_gap={worst_gap:?}, max_streaming_gap={streaming_gap:?}", bytes as f64 / 1048576.0 / elapsed.as_secs_f64());
    assert!(polls > 1);
    assert!(worst_gap < Duration::from_secs(1), "runtime was starved");
}

#[test]
fn preview_handles_every_utf8_and_ansi_chunk_boundary() {
    for (input, expected) in [
        ("Привет 🦀 e\u{301}\n末尾", "Привет 🦀 e\u{301}\n末尾"),
        ("one\r\ntwo\r\n", "one\ntwo"),
        ("old progress\r\x1b[32mготово\x1b[0m\r", "готово"),
        (
            "\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x07!",
            "link!",
        ),
        ("\u{9b}31mred\u{9b}0m\u{9d}title\u{9c}!", "red!"),
        ("\x1bPpayload\x1b\\\x1b(Btext\x1b7", "text"),
        ("\x1b[31\nnext\x1b]unfinished\nend", "\nnext\nend"),
        ("\n\n\n", "\n\n"),
        ("a\x00\x07\x08\x0b\x7fb\t", "ab    "),
    ] {
        for chunk_size in 1..=input.len().max(1) {
            let mut preview = Preview::default();
            for chunk in input.as_bytes().chunks(chunk_size) {
                preview.push(0, chunk);
            }
            preview.finish(0);
            assert_eq!(
                preview.snapshot(),
                expected,
                "chunk {chunk_size}: {input:?}"
            );
        }
    }
}

#[test]
fn independent_decoders_preserve_interleaved_partial_lines() {
    let mut preview = Preview::default();
    preview.push(0, &[0xd0]);
    preview.push(1, b"\x1b[");
    preview.push(0, &[0x90]);
    preview.push(1, b"31merror\x1b[0m");
    preview.push(0, b"\rOK\n");
    preview.push(1, b"-tail\n");
    assert_eq!(preview.snapshot(), "OK\nerror-tail");
    preview.push(0, b"1\n2\n3\n4\n5\n6\n");
    assert_eq!(preview.snapshot(), "2\n3\n4\n5\n6");
}

#[test]
fn evicted_partial_lines_do_not_reappear_as_spurious_rows() {
    let mut preview = Preview::default();
    preview.push(0, b"partial");
    preview.push(1, b"1\n2\n3\n4\n5\n6\n");
    preview.push(0, b"continuation\nnew");
    assert_eq!(preview.snapshot(), "3\n4\n5\n6\nnew");
    preview.push(1, b"7\n8\n9\n10\n11\n");
    preview.push(0, b"\rprogress");
    assert_eq!(preview.snapshot(), "8\n9\n10\n11\nprogress");
}

#[test]
fn invalid_utf8_and_incomplete_eof_do_not_hide_following_text() {
    let mut preview = Preview::default();
    preview.push(0, b"\xffa\xd0b\xf0\x9f");
    preview.finish(0);
    assert_eq!(preview.snapshot(), "\u{fffd}a\u{fffd}b\u{fffd}");
}

#[test]
fn preview_memory_is_bounded_even_without_newlines() {
    let mut preview = Preview::default();
    let chunk = "я\u{301}".repeat(10_000);
    for _ in 0..10 {
        preview.push(0, chunk.as_bytes());
        assert!(preview.snapshot().len() <= PREVIEW_LINE_BYTES);
    }
    assert!(preview.snapshot().ends_with('…'));
    preview.push(0, b"\rOK\n");
    assert_eq!(preview.snapshot(), "OK");
    for _ in 0..100 {
        preview.push(0, chunk.as_bytes());
        preview.push(0, b"\n");
        assert_eq!(preview.lines.len().min(PREVIEW_LINES), preview.lines.len());
        assert!(preview.snapshot().len() <= PREVIEW_LINES * (PREVIEW_LINE_BYTES + 1));
    }
    preview.push(1, b"\x1b]");
    preview.push(1, chunk.as_bytes());
    assert!(preview.lines.len() <= PREVIEW_LINES);
    preview.push(1, b"\x07visible");
    assert!(preview.snapshot().ends_with("visible"));
}

#[tokio::test]
async fn both_streams_publish_partial_output_before_eof_without_blocking_each_other() {
    let (mut out, stdout) = tokio::io::duplex(32);
    let (mut err, stderr) = tokio::io::duplex(32);
    let (tx, mut rx) = watch::channel(String::new());
    let reader = tokio::spawn(drain(stdout, stderr, Some(tx)));
    out.write_all("первый".as_bytes()).await.unwrap();
    timeout(Duration::from_secs(1), async {
        loop {
            rx.changed().await.unwrap();
            if rx.borrow_and_update().contains("первый") {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert!(!reader.is_finished());
    err.write_all(b"stderr-only").await.unwrap();
    timeout(Duration::from_secs(1), async {
        loop {
            rx.changed().await.unwrap();
            if rx.borrow_and_update().contains("stderr-only") {
                break;
            }
        }
    })
    .await
    .unwrap();
    assert!(!reader.is_finished());
    drop(out);
    drop(err);
    let (stdout, stderr) = reader.await.unwrap().unwrap();
    assert_eq!(stdout.text(), "первый");
    assert_eq!(stderr.text(), "stderr-only");
    assert_eq!(&*rx.borrow(), "первый\nstderr-only");
}

#[tokio::test]
async fn slow_ui_gets_latest_five_lines_while_capture_keeps_head_and_tail() {
    let input: String = (0..20_000).map(|n| format!("line-{n}\n")).collect();
    let (tx, rx) = watch::channel(String::new());
    let (stdout, stderr) = drain(input.as_bytes(), tokio::io::empty(), Some(tx))
        .await
        .unwrap();
    assert_eq!(stdout.total, input.len());
    assert_eq!(stdout.head.len(), SHELL_STREAM_HEAD_BYTES);
    assert_eq!(stdout.tail.len(), SHELL_STREAM_TAIL_BYTES);
    assert!(stdout.text().starts_with("line-0\n"));
    assert!(stdout.text().ends_with("line-19999\n"));
    assert!(stdout.text().contains("bytes omitted"));
    assert_eq!(stderr.total, 0);
    assert_eq!(
        &*rx.borrow(),
        "line-19995\nline-19996\nline-19997\nline-19998\nline-19999"
    );
}
