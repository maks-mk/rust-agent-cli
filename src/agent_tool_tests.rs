use super::*;

#[tokio::test]
async fn file_approval_contains_full_diff_not_just_preview() {
    for tool in ["create", "write_file", "edit_file"] {
        let api = MockApi::start(vec![]).await;
        let cfg = api.cfg.clone();
        let path = cfg.root.join("review.txt");
        let old = format!("{}OLD_TAIL\n", "old line\n".repeat(45));
        let new = format!("{}NEW_TAIL\n", "new line\n".repeat(45));
        if tool != "create" {
            std::fs::write(&path, &old).unwrap();
        }
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let args = json!({"path":"review.txt", "content":new, "old_string":old, "new_string":new});
        let task = tokio::spawn(async move {
            let auto = Arc::new(AtomicBool::new(false));
            if tool == "edit_file" {
                edit_file("review", &args, &cfg, &tx, &auto).await
            } else {
                write_file("review", &args, &cfg, &tx, &auto).await
            }
        });
        loop {
            match timeout(Duration::from_secs(5), rx.recv())
                .await
                .unwrap()
                .unwrap()
            {
                AgentEvent::ToolDiff { diff, .. } => assert!(diff.len() <= MAX_DIFF_LINES + 1),
                AgentEvent::ToolApproval { prompt, reply, .. } => {
                    assert!(prompt.contains("+ NEW_TAIL"));
                    if tool != "create" {
                        assert!(prompt.contains("- OLD_TAIL"));
                    }
                    assert!(reply.send(Approval::No).is_ok());
                    break;
                }
                _ => {}
            }
        }
        let (status, _, content) = task.await.unwrap();
        assert_eq!(status, ToolStatus::Denied);
        assert!(!content.contains("Applied diff"));
        if tool == "create" {
            assert!(!path.exists());
        } else {
            assert_eq!(std::fs::read_to_string(&path).unwrap(), old);
        }
    }
}

#[tokio::test]
async fn approval_rejects_concurrent_writes_edits_deletion_and_creation() {
    for tool in ["write_file", "edit_file"] {
        for delete in [false, true] {
            let api = MockApi::start(vec![]).await;
            let cfg = api.cfg.clone();
            let path = cfg.root.join("note.txt");
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let task = tokio::spawn(async move {
                let args = json!({"path":"note.txt", "content":"new", "old_string":"test content", "new_string":"new"});
                let auto = Arc::new(AtomicBool::new(false));
                if tool == "write_file" {
                    write_file("test", &args, &cfg, &tx, &auto).await
                } else {
                    edit_file("test", &args, &cfg, &tx, &auto).await
                }
            });
            loop {
                if let AgentEvent::ToolApproval { reply, .. } =
                    timeout(Duration::from_secs(5), rx.recv())
                        .await
                        .unwrap()
                        .unwrap()
                {
                    if delete {
                        std::fs::remove_file(&path).unwrap();
                    } else {
                        std::fs::write(&path, "user edit").unwrap();
                    }
                    reply.send(Approval::Yes).ok().unwrap();
                    break;
                }
            }
            let (status, _, content) = task.await.unwrap();
            assert_eq!(status, ToolStatus::Failed);
            assert!(!content.contains("Applied diff"));
            if delete {
                assert!(!path.exists());
            } else {
                assert_eq!(std::fs::read_to_string(&path).unwrap(), "user edit");
            }
        }
    }
    // An empty file that appeared while approval waited is still a conflict.
    let api = MockApi::start(vec![]).await;
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        write_file(
            "new",
            &json!({"path":"new.txt", "content":"new"}),
            &cfg,
            &tx,
            &Arc::new(AtomicBool::new(false)),
        )
        .await
    });
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ToolApproval { reply, .. } = event {
            std::fs::write(api.cfg.root.join("new.txt"), "").unwrap();
            reply.send(Approval::Yes).ok().unwrap();
            break;
        }
    }
    assert_eq!(task.await.unwrap().0, ToolStatus::Failed);
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("new.txt")).unwrap(),
        ""
    );
}

#[tokio::test]
async fn edit_file_returns_applied_diff_with_context_and_line_numbers() {
    let api = MockApi::start(vec![]).await;
    let old = "first\nsecond\nold value\nfourth\nfifth\nsixth\n";
    let new = old.replace("old value", "new value\nextra value");
    std::fs::write(api.cfg.root.join("note.txt"), old).unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (status, summary, content) = edit_file(
        "edit",
        &json!({"path":"note.txt", "old_string":"old value", "new_string":"new value\nextra value"}),
        &api.cfg,
        &tx,
        &Arc::new(AtomicBool::new(true)),
    )
    .await;
    assert_eq!(status, ToolStatus::Ok);
    assert_eq!(summary, "Edited · 1 occurrence · +2 −1");
    assert!(content.starts_with("Edited "));
    assert!(
        content.contains("note.txt: replaced 1 occurrence (+2 −1 lines)."),
        "{content}"
    );
    assert!(content.contains("@@ -1,5 +1,6 @@\n first\n second\n-old value\n+new value\n+extra value\n fourth\n fifth\n"), "{content}");
    assert!(!content.contains("sixth"));
    assert!(!content.contains("Diff truncated"));
    assert!(content.contains("final newline: old=true, new=true"));
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
        new
    );
    match rx.try_recv().unwrap() {
        AgentEvent::ToolDiff { diff, .. } => assert!(diff.len() <= MAX_DIFF_LINES + 1),
        _ => panic!("expected the existing UI diff preview"),
    }
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn edit_file_replace_all_returns_separate_hunks_and_accurate_totals() {
    let api = MockApi::start(vec![]).await;
    let middle = (0..20).map(|i| format!("keep {i}\n")).collect::<String>();
    let old = format!("before\nold\n{middle}old\nafter\n");
    let new = old.replace("old", "new\nextra");
    std::fs::write(api.cfg.root.join("note.txt"), &old).unwrap();
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    let (status, summary, content) = edit_file(
        "edit",
        &json!({"path":"note.txt", "old_string":"old", "new_string":"new\nextra", "replace_all":true}),
        &api.cfg,
        &tx,
        &Arc::new(AtomicBool::new(true)),
    )
    .await;
    assert_eq!(status, ToolStatus::Ok);
    assert_eq!(summary, "Edited · 2 occurrences · +4 −2");
    assert_eq!(
        content
            .lines()
            .filter(|line| line.starts_with("@@ "))
            .count(),
        2
    );
    assert!(content.contains("@@ -1,4 +1,5 @@"), "{content}");
    assert!(content.contains("@@ -21,4 +22,5 @@"), "{content}");
    assert!(!content.contains("keep 10"));
    assert_eq!(content.matches("-old\n").count(), 2);
    assert_eq!(content.matches("+new\n+extra\n").count(), 2);
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
        new
    );
}

#[tokio::test]
async fn edit_file_diff_handles_deletion_same_line_matches_unicode_crlf_and_eof() {
    let api = MockApi::start(vec![]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    for (old, find, replacement, expected_diff, metadata) in [
        ("remove\n", "remove\n", "", "@@ -1 +0,0 @@\n-remove\n", "final newline: old=true, new=false"),
        ("first\nremove\nlast\n", "remove\n", "", "-remove\n last\n", "final newline: old=true, new=true"),
        ("foo and foo", "foo", "bar", "-foo and foo\n\\ No newline at end of file\n+bar and bar\n\\ No newline at end of file", "final newline: old=false, new=false"),
        ("привет\nмир", "мир", "world", "-мир\n\\ No newline at end of file\n+world\n", "final newline: old=false, new=false"),
        ("one\r\ntwo\r\n", "two\r\n", "three\r\n", "-two\r\n+three\r\n", "old CRLF=2, new CRLF=2"),
        ("one\r\n", "\r\n", "\n", "-one\r\n+one\n", "old CRLF=1, new CRLF=0"),
        ("one", "one", "one\n", "-one\n\\ No newline at end of file\n+one\n", "final newline: old=false, new=true"),
        ("one\n", "\n", "", "-one\n+one\n\\ No newline at end of file", "final newline: old=true, new=false"),
        ("one\r\ntwo\nthree\r", "two", "changed", "-two\n+changed\n", "old CRLF=1, new CRLF=1"),
    ] {
        std::fs::write(api.cfg.root.join("note.txt"), old).unwrap();
        let (status, _, content) = edit_file(
            "edit",
            &json!({"path":"note.txt", "old_string":find, "new_string":replacement, "replace_all":true}),
            &api.cfg,
            &tx,
            &Arc::new(AtomicBool::new(true)),
        )
        .await;
        assert_eq!(status, ToolStatus::Ok, "{content}");
        assert!(content.contains(expected_diff), "{old:?}: {content}");
        assert!(content.contains(metadata), "{old:?}: {content}");
        assert!(!content.contains("Diff truncated"));
        assert_eq!(std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(), old.replace(find, replacement));
    }
}

#[tokio::test]
async fn edit_file_truncates_diff_without_truncating_the_actual_edit() {
    let api = MockApi::start(vec![]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    for (old, new) in [
        ("old\n".repeat(300), "new\n".repeat(300)),
        (
            format!("{}\n", "я".repeat(MAX_EDIT_DIFF_BYTES)),
            "new\n".to_string(),
        ),
    ] {
        std::fs::write(api.cfg.root.join("note.txt"), &old).unwrap();
        let (status, _, content) = edit_file(
            "edit",
            &json!({"path":"note.txt", "old_string":old, "new_string":new}),
            &api.cfg,
            &tx,
            &Arc::new(AtomicBool::new(true)),
        )
        .await;
        assert_eq!(status, ToolStatus::Ok);
        assert!(content.contains(EDIT_DIFF_TRUNCATION));
        assert!(content.contains(EDIT_DIFF_PARTIAL_HUNK));
        assert!(content.contains(&format!(
            "(+{} −{} lines)",
            new.lines().count(),
            old.lines().count()
        )));
        let start = content.find("Applied diff").unwrap();
        let end = content.find("Line endings:").unwrap();
        let diff = &content[start..end];
        assert!(diff.len() <= MAX_EDIT_DIFF_BYTES, "{} bytes", diff.len());
        assert!(diff.lines().count() <= MAX_EDIT_DIFF_LINES);
        assert!(!diff.lines().any(|line| line.starts_with('+')), "{diff}");
        if old.lines().count() > MAX_EDIT_DIFF_LINES {
            assert!(diff.lines().any(|line| line.starts_with('-')), "{diff}");
        }
        assert_eq!(
            std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
            new
        );
    }
}

#[tokio::test]
async fn edit_file_failures_do_not_claim_an_applied_diff() {
    let api = MockApi::start(vec![]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    for args in [
        json!({"path":"missing.txt", "old_string":"a", "new_string":"b"}),
        json!({"path":"note.txt", "old_string":"missing", "new_string":"b"}),
        json!({"path":"note.txt", "old_string":"test content", "new_string":"test content"}),
        json!({"path":"note.txt", "old_string":"t", "new_string":"b"}),
    ] {
        let (status, _, content) = edit_file(
            "edit",
            &args,
            &api.cfg,
            &tx,
            &Arc::new(AtomicBool::new(true)),
        )
        .await;
        assert_eq!(status, ToolStatus::Failed);
        assert!(!content.contains("Applied diff"));
        assert_eq!(
            std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
            "test content"
        );
    }
}

#[tokio::test]
async fn edit_file_diff_reaches_the_next_model_request_without_an_extra_read() {
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"tool_calls":[{"index":0, "id":"edit", "function":{"name":"edit_file", "arguments":"{\"path\":\"note.txt\",\"old_string\":\"test content\",\"new_string\":\"changed content\"}"}}]}))]),
        sse(vec![delta(json!({"content":"Done"}))]),
    ]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    let history = run_turn(
        &reqwest::Client::new(),
        &api.cfg,
        vec![json!({"role":"user", "content":"Edit note.txt"})],
        &tx,
        &Arc::new(AtomicBool::new(true)),
        &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
    )
    .await
    .unwrap();
    let requests = api.requests().await;
    assert_eq!(requests.len(), 2);
    let result = requests[1]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap();
    let content = result["content"].as_str().unwrap();
    assert!(
        content.contains("-test content\n\\ No newline at end of file\n+changed content\n"),
        "{content}"
    );
    assert_eq!(result["tool_call_id"], "edit");
    assert!(history.contains(result));
    let responses = responses_input(&history);
    assert!(responses
        .iter()
        .any(|item| item["type"] == "function_call_output"
            && item["call_id"] == "edit"
            && item["output"] == content));
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
        "changed content"
    );
}

#[tokio::test]
async fn read_then_exact_edit_preserves_line_endings_and_eof() {
    let api = MockApi::start(vec![]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    for content in [
        "one\r\ntwo\r\n",
        "one\ntwo",
        "one\r\ntwo\nthree\r",
        "привет\nмир",
    ] {
        std::fs::write(api.cfg.root.join("note.txt"), content).unwrap();
        let (status, _, read) = read_file(&json!({"path":"note.txt"}), &api.cfg).await;
        assert_eq!(status, ToolStatus::Ok);
        assert_eq!(read, content);
        let result = edit_file(
            "edit",
            &json!({"path":"note.txt", "old_string":read, "new_string":"changed"}),
            &api.cfg,
            &tx,
            &Arc::new(AtomicBool::new(true)),
        )
        .await;
        assert_eq!(result.0, ToolStatus::Ok);
        assert_eq!(
            std::fs::read_to_string(api.cfg.root.join("note.txt")).unwrap(),
            "changed"
        );
    }
}

#[tokio::test]
async fn read_file_validates_pagination_before_reading() {
    let api = MockApi::start(vec![]).await;
    for path in ["note.txt", "missing.txt"] {
        for name in ["offset", "limit"] {
            for value in [
                json!(0),
                json!(-1),
                json!(1.5),
                json!("2"),
                json!(true),
                json!([]),
                json!({}),
                json!(1e30),
            ] {
                let mut args = json!({"path":path});
                args[name] = value;
                let (status, _, content) = read_file(&args, &api.cfg).await;
                assert_eq!(status, ToolStatus::Failed, "{args}: {content}");
                assert!(content.contains(name), "{content}");
            }
        }
    }
    std::fs::write(api.cfg.root.join("note.txt"), "one\r\ntwo\nthree").unwrap();
    for args in [
        json!({"path":"note.txt"}),
        json!({"path":"note.txt", "offset":null, "limit":null}),
        json!({"path":"note.txt", "limit":usize::MAX}),
    ] {
        let (status, _, content) = read_file(&args, &api.cfg).await;
        assert_eq!(status, ToolStatus::Ok);
        assert_eq!(content, "one\r\ntwo\nthree");
    }
    let (status, _, content) =
        read_file(&json!({"path":"note.txt", "offset":2, "limit":2}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Ok);
    assert_eq!(content, "two\nthree");
    let (status, _, content) =
        read_file(&json!({"path":"note.txt", "offset":usize::MAX}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Ok);
    assert!(content.is_empty());
}

#[tokio::test]
async fn bounded_file_read_handles_long_lines_pagination_and_utf8_boundary() {
    let (text, _) = read_text_window(tokio::io::repeat(b'x'), 1, 2000)
        .await
        .unwrap();
    assert!(text.len() < MAX_READ_BYTES + 200);
    assert!(text.contains("truncated within line 1"));
    let input = format!("{}\r\nnext\r\nlast", "x".repeat(MAX_READ_BYTES * 3));
    assert_eq!(
        read_text_window(input.as_bytes(), 2, 10).await.unwrap().0,
        "next\r\nlast"
    );
    assert_eq!(
        read_text_window(b"one\r\ntwo".as_slice(), 1, 1)
            .await
            .unwrap()
            .0,
        "one\r\n\n[showing lines 1-1; use offset/limit to read more]"
    );
    assert_eq!(
        read_text_window(b"one".as_slice(), 99, 10).await.unwrap().0,
        ""
    );
    assert_eq!(read_text_window(b"".as_slice(), 1, 10).await.unwrap().0, "");
    assert!(read_text_window(b"\xff".as_slice(), 1, 10).await.is_err());
    let input = format!("{}€tail", "x".repeat(MAX_READ_BYTES - 1));
    let output = read_text_window(input.as_bytes(), 1, 10).await.unwrap().0;
    assert!(output.starts_with(&"x".repeat(MAX_READ_BYTES - 1)));
    assert!(!output.contains('\u{fffd}'));
}

#[tokio::test]
async fn writes_commit_only_on_approval_and_clean_up_temporary_files() {
    let api = MockApi::start(vec![]).await;
    commit_file(&api.cfg, "sub/new.txt", None, "initial").unwrap();
    commit_file(&api.cfg, "sub/new.txt", Some("initial"), "changed").unwrap();
    assert!(commit_file(&api.cfg, "sub/new.txt", Some("stale"), "lost").is_err());
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("sub/new.txt")).unwrap(),
        "changed"
    );
    assert_eq!(
        std::fs::read_dir(api.cfg.root.join("sub")).unwrap().count(),
        1
    );
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        write_file(
            "deny",
            &json!({"path":"sub/new.txt", "content":"denied"}),
            &cfg,
            &tx,
            &Arc::new(AtomicBool::new(false)),
        )
        .await
    });
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ToolApproval { reply, .. } = event {
            reply.send(Approval::No).ok().unwrap();
            break;
        }
    }
    assert_eq!(task.await.unwrap().0, ToolStatus::Denied);
    assert_eq!(
        std::fs::read_to_string(api.cfg.root.join("sub/new.txt")).unwrap(),
        "changed"
    );
}

#[tokio::test]
async fn cancelling_tool_batch_preserves_completed_and_pairs_unfinished_calls() {
    let calls: Vec<_> = (0..3).map(|i| json!({"index":i, "id":format!("call_{i}"), "function":{"name":"write_file", "arguments":json!({"path":format!("file_{i}"), "content":"completed"}).to_string()}})).collect();
    let api = MockApi::start(vec![sse(vec![delta(json!({"tool_calls":calls}))])]).await;
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        run_turn(
            &reqwest::Client::new(),
            &cfg,
            vec![json!({"role":"user", "content":"write files"})],
            &tx,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
        )
        .await
    });
    let mut history = vec![json!({"role":"user", "content":"write files"})];
    loop {
        match timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap()
        {
            AgentEvent::ToolApproval { id, reply, .. } if id == "call_0" => {
                reply.send(Approval::Yes).ok().unwrap();
            }
            AgentEvent::ToolApproval { id, .. } if id == "call_1" => {
                task.abort();
                break;
            }
            event => apply_history_event(&mut history, event),
        }
    }
    assert!(task.await.unwrap_err().is_cancelled());
    while let Ok(event) = rx.try_recv() {
        apply_history_event(&mut history, event);
    }
    assert_eq!(history.len(), 5);
    assert_eq!(history[1]["tool_calls"].as_array().unwrap().len(), 3);
    for i in 0..3 {
        assert_eq!(history[2 + i]["tool_call_id"], format!("call_{i}"));
    }
    assert!(history[2]["content"].as_str().unwrap().contains("Wrote"));
    assert!(history[3]["content"]
        .as_str()
        .unwrap()
        .contains("Interrupted"));
    assert!(history[4]["content"]
        .as_str()
        .unwrap()
        .contains("Not executed"));
    assert!(api.cfg.root.join("file_0").exists());
    assert!(!api.cfg.root.join("file_1").exists());
    assert!(!api.cfg.root.join("file_2").exists());
}

#[tokio::test]
async fn run_turn_executes_independent_shell_commands_concurrently() {
    // Each command writes its own marker and waits for its peer to start.
    // Sequential execution fails at the barrier instead of passing by chance
    // on a fast machine; no elapsed-time comparison is needed.
    let calls: Vec<_> = (0..2)
        .map(|index| {
            let peer = 1 - index;
            let command = if cfg!(windows) {
                format!(
                    r#"
$ErrorActionPreference = 'Stop'
[IO.File]::WriteAllText('started_{index}', '')
$wait = [Diagnostics.Stopwatch]::StartNew()
while (-not [IO.File]::Exists('started_{peer}')) {{
    if ($wait.Elapsed.TotalSeconds -ge 10) {{ throw 'peer did not start concurrently' }}
    Start-Sleep -Milliseconds 20
}}
[Console]::WriteLine('shell-{index}-completed')
"#
                )
            } else {
                format!(
                    r#"
: > started_{index}
attempts=0
while [ ! -f started_{peer} ]; do
    if [ "$attempts" -ge 200 ]; then
        printf '%s\n' 'peer did not start concurrently' >&2
        exit 1
    fi
    attempts=$((attempts + 1))
    sleep 0.05
done
printf '%s\n' 'shell-{index}-completed'
"#
                )
            };
            json!({"index":index, "id":format!("shell_{index}"), "function":{
                "name":"shell", "arguments":json!({"command":command, "timeout":15}).to_string()
            }})
        })
        .collect();
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"tool_calls":calls}))]),
        sse(vec![delta(json!({"content":"Done"}))]),
    ])
    .await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = timeout(
        Duration::from_secs(40),
        run_turn(
            &reqwest::Client::new(),
            &api.cfg,
            vec![json!({"role":"user", "content":"Run two independent commands"})],
            &tx,
            &Arc::new(AtomicBool::new(true)),
            &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
        ),
    )
    .await
    .expect("shell orchestration must finish within its deadline")
    .unwrap();

    assert_eq!(history.len(), 5);
    assert_eq!(history[1]["tool_calls"].as_array().unwrap().len(), 2);
    for index in 0..2 {
        let result = &history[2 + index];
        assert_eq!(result["role"], "tool");
        assert_eq!(result["tool_call_id"], format!("shell_{index}"));
        let content = result["content"].as_str().unwrap();
        assert!(content.contains("\nexit code: 0\n"), "{content}");
        assert_eq!(
            content.lines().last().unwrap(),
            format!("shell-{index}-completed"),
            "{content}"
        );
        assert!(api.cfg.root.join(format!("started_{index}")).exists());
    }
    let mut completed = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolEnd { id, status, .. } = event {
            assert_eq!(status, ToolStatus::Ok, "{id}");
            completed.push(id);
        }
    }
    completed.sort();
    assert_eq!(completed, ["shell_0", "shell_1"]);

    let requests = api.requests().await;
    assert_eq!(requests.len(), 2);
    // Results reach the next model request in the original call order.
    assert_eq!(requests[1]["messages"][3], history[2]);
    assert_eq!(requests[1]["messages"][4], history[3]);
    assert_eq!(history[4]["content"], "Done");
}

#[tokio::test]
async fn shell_exposes_the_executable_dir_on_path() {
    let api = MockApi::start(vec![]).await;
    let command = if cfg!(windows) {
        "$env:Path"
    } else {
        r#"printf '%s' "$PATH""#
    };
    let run = shell_output(command, &api.cfg.root, Duration::from_secs(15))
        .await
        .unwrap();
    let printed = run.stdout.text();
    let entries: Vec<PathBuf> =
        std::env::split_paths(std::ffi::OsStr::new(printed.trim())).collect();
    let exe_dir = std::env::current_exe().unwrap();
    let exe_dir = exe_dir.parent().unwrap();
    // pwsh 7 moves its own directory to the front, so only presence matters.
    assert!(
        entries.iter().any(|entry| entry == exe_dir),
        "shell PATH should include the executable dir: {entries:?}"
    );
    assert!(
        entries.len() > 1,
        "the inherited PATH should be preserved: {entries:?}"
    );
}

#[tokio::test]
async fn shell_capture_preserves_utf8_across_head_tail_boundary() {
    for input in [
        "start\r\nПривет € 🦀 end\r\n".as_bytes(),
        b"\xffstart\xe2\x82\xactail\xff",
    ] {
        for head_limit in 0..=input.len() {
            let captured = drain_output(input, head_limit, input.len()).await.unwrap();
            assert_eq!(
                captured.text(),
                String::from_utf8_lossy(input).replace("\r\n", "\n"),
                "head={head_limit}"
            );
        }
    }
}

#[tokio::test]
async fn shell_capture_truncates_at_utf8_boundaries_and_marks_all_omissions() {
    let input = "a€0123456789🦀z";
    let captured = drain_output(input.as_bytes(), 2, 3).await.unwrap();
    assert_eq!(captured.text(), "a\n[... 17 bytes omitted ...]\nz");
    let head_only = drain_output(b"abcdef".as_slice(), 3, 0).await.unwrap();
    assert_eq!(head_only.text(), "abc\n[... 3 bytes omitted ...]\n");
    let empty = drain_output(b"abcdef".as_slice(), 0, 0).await.unwrap();
    assert_eq!(empty.text(), "\n[... 6 bytes omitted ...]\n");
}

#[tokio::test]
async fn shell_capture_sanitizes_truncated_windows_without_losing_gap_marker() {
    let head = "head\x1b]unfinished";
    let middle = "dropped middle";
    let tail = "\x1b[31mold\rnew\x1b[0m";
    let input = format!("{head}{middle}{tail}");
    let captured = drain_output(input.as_bytes(), head.len(), tail.len())
        .await
        .unwrap();
    assert_eq!(captured.total, input.len());
    assert_eq!(
        captured.text(),
        format!("head\n[... {} bytes omitted ...]\nnew", middle.len())
    );
}

#[tokio::test]
async fn shell_capture_preserves_utf8_for_all_truncation_windows() {
    let input = "a€🦀Жz";
    for head_limit in 0..=input.len() {
        for tail_limit in 0..=input.len() {
            let captured = drain_output(input.as_bytes(), head_limit, tail_limit)
                .await
                .unwrap();
            let expected = if head_limit + tail_limit >= input.len() {
                input.to_string()
            } else {
                let mut head_end = head_limit;
                while !input.is_char_boundary(head_end) {
                    head_end -= 1;
                }
                let mut tail_start = input.len() - tail_limit;
                while !input.is_char_boundary(tail_start) {
                    tail_start += 1;
                }
                format!(
                    "{}\n[... {} bytes omitted ...]\n{}",
                    &input[..head_end],
                    tail_start - head_end,
                    &input[tail_start..]
                )
            };
            assert_eq!(
                captured.text(),
                expected,
                "head={head_limit} tail={tail_limit}"
            );
        }
    }
    let input = b"\xffa\xe2\x82\xac0123456789\xf0\x9f\xa6\x80z\xff";
    let captured = drain_output(input.as_slice(), 3, 4).await.unwrap();
    assert_eq!(
        captured.text(),
        "\u{fffd}a\n[... 17 bytes omitted ...]\nz\u{fffd}"
    );
}

#[tokio::test]
async fn shell_capture_keeps_exact_byte_windows_for_varied_budgets() {
    let input: Vec<u8> = (0..50_000).map(|i| (i % 251) as u8).collect();
    for head_limit in [0, 1, 31, 8191, 8192, 8193, 20_000, 60_000] {
        for tail_limit in [0, 1, 17, 8191, 8192, 8193, 10_000, 60_000] {
            let captured = drain_output(input.as_slice(), head_limit, tail_limit)
                .await
                .unwrap();
            let head_end = head_limit.min(input.len());
            let tail_start = head_end.max(input.len().saturating_sub(tail_limit));
            assert_eq!(captured.total, input.len());
            assert_eq!(captured.head, input[..head_end]);
            assert_eq!(captured.tail, input[tail_start..]);
        }
    }
}

#[tokio::test]
async fn shell_drains_both_streams_with_bounded_capture() {
    let api = MockApi::start(vec![]).await;
    let command = if cfg!(windows) {
        "[Console]::Out.Write(('o' * 300000)); [Console]::Error.Write(('e' * 300000)); exit 7"
    } else {
        "head -c 300000 /dev/zero; head -c 300000 /dev/zero >&2; exit 7"
    };
    let run = shell_output(command, &api.cfg.root, Duration::from_secs(15))
        .await
        .unwrap();
    assert_eq!(run.code, 7);
    // Every stream keeps a head and a tail inside the byte budget.
    assert_eq!(run.stdout.head.len(), SHELL_STREAM_HEAD_BYTES);
    assert_eq!(run.stdout.tail.len(), SHELL_STREAM_TAIL_BYTES);
    assert_eq!(run.stdout.total, 300000);
    assert_eq!(run.stderr.head.len(), SHELL_STREAM_HEAD_BYTES);
    assert_eq!(run.stderr.tail.len(), SHELL_STREAM_TAIL_BYTES);
    assert_eq!(run.stderr.total, 300000);
    // The dropped middle is marked in the rendered text.
    assert!(run.stdout.text().contains("bytes omitted"));
    let short = drain_output(
        b"short".as_slice(),
        SHELL_STREAM_HEAD_BYTES,
        SHELL_STREAM_TAIL_BYTES,
    )
    .await
    .unwrap();
    assert_eq!(short.text(), "short");
    assert_eq!(short.total, 5);
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    let result = shell(
        "out",
        &json!({"command":command}),
        &api.cfg,
        &tx,
        &Arc::new(AtomicBool::new(true)),
    )
    .await;
    assert_eq!(result.0, ToolStatus::Failed);
    assert_eq!(result.1, "Failed · exit 7");
    assert!(result.2.len() < MAX_SHELL_OUTPUT_BYTES + command.len() + 150);
    assert!(result.2.contains("bytes omitted"));
}

#[tokio::test]
async fn shell_exit_status_preserves_both_streams_for_the_model() {
    let api = MockApi::start(vec![]).await;
    let (tx, _) = tokio::sync::mpsc::unbounded_channel();
    let auto = Arc::new(AtomicBool::new(true));
    for (code, expected, summary) in [
        (0, ToolStatus::Ok, "Done · exit 0"),
        (7, ToolStatus::Failed, "Failed · exit 7"),
    ] {
        let command = if cfg!(windows) {
            format!("[Console]::Out.Write('stdout-marker'); [Console]::Error.Write('stderr-marker'); exit {code}")
        } else {
            format!("printf stdout-marker; printf stderr-marker >&2; exit {code}")
        };
        let (status, result, content) =
            shell("output", &json!({"command": command}), &api.cfg, &tx, &auto).await;
        assert_eq!(status, expected);
        assert_eq!(result, summary);
        assert!(content.contains(&format!("exit code: {code}\nstdout-marker\nstderr-marker")));
    }
}

#[tokio::test]
async fn shell_tool_formats_empty_and_ansi_output_with_exit_code() {
    let api = MockApi::start(vec![]).await;
    let auto = Arc::new(AtomicBool::new(true));
    for (code, ansi, label) in [
        (0, false, ""),
        (7, false, ""),
        (0, true, ""),
        (7, true, "Path"),
    ] {
        let command = if !ansi {
            format!("exit {code}")
        } else if cfg!(windows) {
            format!("[Console]::Out.Write(([string][char]27) + '[32;1m{label}' + ([string][char]27) + '[0m'); exit {code}")
        } else {
            format!("printf '\\033[32;1m{label}\\033[0m'; exit {code}")
        };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let (status, _, content) =
            shell("format", &json!({"command": command}), &api.cfg, &tx, &auto).await;
        assert_eq!(
            status,
            if code == 0 {
                ToolStatus::Ok
            } else {
                ToolStatus::Failed
            }
        );
        let expected = if label.is_empty() {
            "(no output)"
        } else {
            label
        };
        let mut outputs = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::ToolOutput { output, .. } = event {
                outputs.push(output);
            }
        }
        assert_eq!(outputs, [format!("exit code: {code}\n{expected}")]);
        assert!(content.ends_with(&format!("exit code: {code}\n{expected}")));
        assert!(!content.contains('\x1b'));
    }
}

#[test]
fn shell_ui_summary_shows_the_end_of_the_output() {
    let stdout = (1..=25)
        .map(|n| format!("line-{n}"))
        .collect::<Vec<_>>()
        .join("\n");
    let summary = shell_ui_summary(&stdout, "");
    let lines: Vec<&str> = summary.lines().collect();
    // The header marks the omitted head, the tail keeps the last lines.
    assert_eq!(lines.len(), MAX_SHELL_UI_TAIL_LINES + 1);
    assert!(lines[0].contains(&format!("last {MAX_SHELL_UI_TAIL_LINES} of 25 lines")));
    assert_eq!(lines[1], format!("line-{}", 26 - MAX_SHELL_UI_TAIL_LINES));
    assert_eq!(lines[MAX_SHELL_UI_TAIL_LINES], "line-25");
    assert!(!summary.contains(&format!("line-{}", 25 - MAX_SHELL_UI_TAIL_LINES)));
    // A short output is shown as it is.
    let short = "first\nsecond";
    assert_eq!(shell_ui_summary(short, ""), short);
}

#[test]
fn shell_ui_summary_pulls_errors_and_warnings_out_of_the_omitted_part() {
    let mut lines: Vec<String> = (1..=20).map(|n| format!("line-{n}")).collect();
    lines[2] = "ERROR: boom".into();
    lines[3] = "WARN: careful".into();
    let summary = shell_ui_summary(&lines.join("\n"), "");
    assert!(summary.contains("[errors: 1 of 1]"));
    assert!(summary.contains("ERROR: boom"));
    assert!(summary.contains("[warnings: 1 of 1]"));
    assert!(summary.contains("WARN: careful"));
    // The tail still shows how the command ended.
    assert!(summary.contains("line-20"));
}

#[test]
fn shell_ui_summary_keeps_the_end_of_both_streams() {
    let stdout: String = (1..=15).map(|n| format!("out-{n}\n")).collect();
    let stderr: String = (1..=15).map(|n| format!("err-{n}\n")).collect();
    let summary = shell_ui_summary(&stdout, &stderr);
    assert!(summary.contains(&format!(
        "[stdout: last {MAX_SHELL_UI_TAIL_LINES} of 15 lines]"
    )));
    assert!(summary.contains(&format!(
        "[stderr: last {MAX_SHELL_UI_TAIL_LINES} of 15 lines]"
    )));
    assert!(summary.contains("out-15"));
    assert!(summary.contains("err-15"));
    assert!(!summary.contains("out-1\n"));
    assert!(!summary.contains("err-1\n"));
}

#[test]
fn shell_ui_summary_does_not_repeat_lines_that_are_already_shown() {
    let mut lines: Vec<String> = (1..=20).map(|n| format!("line-{n}")).collect();
    lines[19] = "ERROR: boom".into();
    let summary = shell_ui_summary(&lines.join("\n"), "");
    assert!(!summary.contains("[errors:"));
    assert!(summary.contains("ERROR: boom"));
}

#[tokio::test]
async fn shell_tool_gives_the_model_the_full_output_and_the_user_the_tail() {
    let api = MockApi::start(vec![]).await;
    let command = if cfg!(windows) {
        "1..25 | ForEach-Object { \"line-$_\" }"
    } else {
        "for i in $(seq 1 25); do echo line-$i; done"
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (status, _, content) = shell(
        "output",
        &json!({"command": command}),
        &api.cfg,
        &tx,
        &Arc::new(AtomicBool::new(true)),
    )
    .await;
    assert_eq!(status, ToolStatus::Ok);
    // The model receives every line of the output.
    assert!(content.contains("line-1\n"));
    assert!(content.contains("line-25"));
    assert!(!content.contains("kept the first"));

    // The user sees the end of the output only.
    let mut ui = String::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolOutput { output, .. } = event {
            ui = output;
        }
    }
    assert!(ui.contains("exit code: 0"));
    assert!(ui.contains("line-25"));
    assert!(!ui.contains("line-1\n"));
    assert!(ui.contains(&format!("last {MAX_SHELL_UI_TAIL_LINES} of 25 lines")));
}

#[tokio::test]
async fn shell_preview_is_cut_while_the_approval_prompt_keeps_the_full_command() {
    let api = MockApi::start(vec![]).await;
    let cfg = api.cfg.clone();
    let tail = "x".repeat(MAX_SHELL_UI_ARG_CHARS + 40);
    let command = format!("echo {tail}");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        let auto = Arc::new(AtomicBool::new(false));
        shell("preview", &json!({"command": command}), &cfg, &tx, &auto).await
    });
    let mut preview = String::new();
    loop {
        match timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap()
        {
            AgentEvent::ToolDiff { diff, .. } => {
                preview = diff
                    .iter()
                    .map(|d| d.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            AgentEvent::ToolApproval { prompt, reply, .. } => {
                // The user approves the command that will really run.
                assert!(prompt.contains(&tail));
                assert!(reply.send(Approval::No).is_ok());
                break;
            }
            _ => {}
        }
    }
    assert_eq!(task.await.unwrap().0, ToolStatus::Denied);
    assert!(preview.starts_with("$ echo x"));
    assert!(preview.ends_with(SHELL_UI_CUT_MARKER));
    assert!(!preview.contains(&tail));
}

#[tokio::test]
async fn shell_cancellation_and_timeout_stop_descendants() {
    for cancel in [true, false] {
        let api = MockApi::start(vec![]).await;
        let root = api.cfg.root.clone();
        let script = "[IO.File]::WriteAllText('started', 'yes'); Start-Sleep -Seconds 5; [IO.File]::WriteAllText('done', 'bad')";
        let command = if cfg!(windows) {
            format!("& powershell {}", powershell_args(script).join(" "))
        } else {
            "sh -c 'echo yes > started; sleep 5; echo bad > done'; :".into()
        };
        let deadline = if cancel {
            Duration::from_secs(20)
        } else {
            Duration::from_secs(3)
        };
        let task = tokio::spawn(async move { shell_output(&command, &root, deadline).await });
        timeout(Duration::from_secs(10), async {
            while !api.cfg.root.join("started").exists() {
                assert!(
                    !task.is_finished(),
                    "shell exited before descendant started"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap();
        if cancel {
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else {
            assert!(task
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("timed out"));
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
        assert!(
            !api.cfg.root.join("done").exists(),
            "descendant survived termination"
        );
    }
}

#[tokio::test]
async fn background_start_needs_approval() {
    let api = MockApi::start(vec![]).await;
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let args = json!({"command": "echo background-approval", "background": true});
    let task = tokio::spawn(async move {
        let auto = Arc::new(AtomicBool::new(false));
        shell("bg", &args, &cfg, &tx, &auto).await
    });
    let mut approvals = 0usize;
    loop {
        let event = timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        if let AgentEvent::ToolApproval { prompt, reply, .. } = event {
            assert!(prompt.contains("background"), "prompt was: {prompt}");
            assert!(reply.send(Approval::No).is_ok());
            approvals += 1;
            break;
        }
    }
    assert_eq!(approvals, 1, "the background start asked for approval");
    let (status, _, _) = task.await.unwrap();
    assert_eq!(status, ToolStatus::Denied);
}

#[tokio::test]
async fn shell_sanitizes_captured_output_for_model_and_ui() {
    let api = MockApi::start(vec![]).await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let command = if cfg!(windows) {
        "[Console]::Write(([char]27) + '[2J' + ([char]27) + '[1;1HANSI-SPAM' + ([char]13) + ([char]27) + '[31mred' + ([char]27) + '[0m' + ([char]10)); [Console]::Error.Write(([char]27) + ']0;title' + ([char]7) + 'stderr-ok' + ([char]10))"
    } else {
        "printf '\\033[2J\\033[1;1HANSI-SPAM\\r\\033[31mred\\033[0m\\n'; printf '\\033]0;title\\007stderr-ok\\n' >&2"
    };
    let auto = Arc::new(AtomicBool::new(true));
    let (status, _, content) = shell(
        "safe-output",
        &json!({"command": command}),
        &api.cfg,
        &tx,
        &auto,
    )
    .await;
    assert_eq!(status, ToolStatus::Ok);
    assert!(content.ends_with("\nred\nstderr-ok\n"), "{content:?}");
    assert!(!content.contains('\x1b'), "{content:?}");
    let ui = std::iter::from_fn(|| rx.try_recv().ok())
        .find_map(|event| match event {
            AgentEvent::ToolOutput { output, .. } => Some(output),
            _ => None,
        })
        .unwrap();
    assert_eq!(ui, "exit code: 0\nred\nstderr-ok");
}

#[cfg(windows)]
#[tokio::test]
async fn shell_children_cannot_write_directly_to_console() {
    let api = MockApi::start(vec![]).await;
    // CREATE_NO_WINDOW may still permit CONOUT$ in the child's own hidden
    // console. The important guarantee is that it cannot write to our console.
    let command = format!(
        r#"$ErrorActionPreference = 'Stop'; Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class ConsoleProbe {{ [DllImport("kernel32.dll")] public static extern uint GetConsoleProcessList([Out] uint[] p, uint n); }}'; $pids = [uint32[]]::new(256); $count = [ConsoleProbe]::GetConsoleProcessList($pids, 256); if ($count -gt 256 -or $pids -contains {parent}) {{ [Console]::WriteLine('console-shared'); exit 1 }}; try {{ $f = [IO.File]::Open('\\.\CONOUT$', 'Open', 'Write', 'ReadWrite'); $bytes = [Text.Encoding]::UTF8.GetBytes('DIRECT-WRITE'); $f.Write($bytes, 0, $bytes.Length); $f.Dispose() }} catch {{ }}; [Console]::WriteLine('console-isolated')"#,
        parent = std::process::id()
    );
    let run = shell_output(&command, &api.cfg.root, Duration::from_secs(15))
        .await
        .unwrap();
    assert_eq!(run.code, 0, "{}", run.stderr.text());
    assert_eq!(run.stdout.text().trim(), "console-isolated");
}

#[tokio::test]
async fn background_actions_do_not_need_approval() {
    let api = MockApi::start(vec![]).await;
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let args = json!({"background_action": "list"});
    let task = tokio::spawn(async move {
        let auto = Arc::new(AtomicBool::new(false));
        shell("bg", &args, &cfg, &tx, &auto).await
    });
    let (status, _, content) = task.await.unwrap();
    assert_eq!(status, ToolStatus::Ok);
    // The action only looks at the jobs of this process, so it runs without an
    // approval.
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolApproval { .. } = event {
            panic!("a background action must not ask for approval");
        }
    }
    assert!(
        content.contains("background job") || content.contains("no background jobs"),
        "content was: {content}"
    );
}

#[tokio::test]
async fn background_start_writes_a_log_file_inside_the_workspace() {
    let api = MockApi::start(vec![]).await;
    let cfg = api.cfg.clone();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let args = json!({"command": "echo background-started", "background": true});
    let auto = Arc::new(AtomicBool::new(true));
    let (status, summary, content) = shell("bg", &args, &cfg, &tx, &auto).await;
    assert_eq!(status, ToolStatus::Ok);
    assert!(summary.contains("Started"), "summary was: {summary}");
    // The answer names the job and the log file inside the workspace.
    assert!(content.contains("background job"), "content was: {content}");
    let logs = cfg.root.join(".agent_state").join("background");
    let logs = std::fs::read_dir(&logs).unwrap().count();
    assert!(logs >= 1, "the job log is missing in the workspace");
    // The detached job reports its state without an approval.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let args = json!({"background_action": "list"});
    let auto = Arc::new(AtomicBool::new(false));
    let (status, _, _) = shell("bg", &args, &cfg, &tx, &auto).await;
    assert_eq!(status, ToolStatus::Ok);
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::ToolApproval { .. } = event {
            panic!("a background action must not ask for approval");
        }
    }
}
