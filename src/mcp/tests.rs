use super::*;

#[test]
fn model_tool_schema_includes_loaded_mcp_tools() {
    let mut mcp = McpManager::default();
    let builtin_count = crate::agent::tools_schema(&mcp).as_array().unwrap().len();
    mcp.tools.push(ToolBinding {
        api_name: "mcp_test_lookup".into(),
        server_key: "test".into(),
        server_index: 0,
        original_name: "lookup".into(),
        description: "Test lookup".into(),
        input_schema: json!({"type": "object"}),
        read_only: true,
    });
    let schemas = crate::agent::tools_schema(&mcp);
    let tools = schemas.as_array().unwrap();
    assert_eq!(tools.len(), builtin_count + 1);
    assert_eq!(tools.last().unwrap()["function"]["name"], "mcp_test_lookup");
}

#[test]
fn only_boolean_read_only_grants_trust_and_prompt_hides_credentials() {
    for policy in [
        Value::Null,
        json!({}),
        json!({"read_only":false}),
        json!({"read_only":"true"}),
        json!({"read_only":1}),
    ] {
        assert!(!policy_read_only(&json!({"policy":policy})));
    }
    assert!(policy_read_only(&json!({"policy":{"read_only":true}})));
    let prompt = server_approval_prompt(
        Path::new("mcp.json"),
        Path::new("."),
        "remote",
        &json!({
            "url":"https://user:password@example.com/private?token=secret",
            "headers":{"Authorization":"hidden-header"}, "env":{"TOKEN":"hidden-env"}
        }),
    );
    assert!(prompt.contains("https://example.com"));
    for secret in [
        "password",
        "private",
        "secret",
        "hidden-header",
        "hidden-env",
    ] {
        assert!(!prompt.contains(secret));
    }
}

#[tokio::test]
async fn startup_trust_is_snapshot_bound_and_denial_does_not_spawn() {
    std::fs::create_dir_all("target").unwrap();
    let dir = tempfile::tempdir_in("target").unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let path = root.join("mcp.json");
    let config =
        json!({"server":{"command":"missing-mcp-test-command", "policy":{"read_only":false}}});
    std::fs::write(&path, config.to_string()).unwrap();
    let mut manager = McpManager::default();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let answer = async {
        let AgentEvent::McpApproval { prompt, reply } = rx.recv().await.unwrap() else {
            panic!("expected trust request")
        };
        assert!(prompt.contains("missing-mcp-test-command"));
        assert!(reply.send(Approval::No).is_ok());
    };
    tokio::join!(
        manager.ensure_loaded_with_approval(&root, &root, &tx),
        answer
    );
    assert!(manager.servers.is_empty());
    assert!(manager
        .take_warnings()
        .iter()
        .all(|warning| warning.contains("skipped by user")));
    manager.ensure_loaded_with_approval(&root, &root, &tx).await;
    assert!(rx.try_recv().is_err());

    // A changed file requires fresh trust. Changing it again during review must
    // not replace the snapshot being approved with an unreviewed command.
    std::fs::write(&path, format!("{config}\n")).unwrap();
    let answer = async {
        let AgentEvent::McpApproval { reply, .. } = rx.recv().await.unwrap() else {
            panic!("expected fresh trust")
        };
        std::fs::write(&path, "{}").unwrap();
        assert!(reply.send(Approval::Yes).is_ok());
    };
    tokio::join!(
        manager.ensure_loaded_with_approval(&root, &root, &tx),
        answer
    );
    assert!(manager
        .take_warnings()
        .iter()
        .any(|warning| warning.contains("missing-mcp-test-command")));
    manager.ensure_loaded_with_approval(&root, &root, &tx).await;
    assert!(manager.decisions.is_empty());

    std::fs::write(&path, config.to_string()).unwrap();
    let cancel = async {
        let AgentEvent::McpApproval { reply, .. } = rx.recv().await.unwrap() else {
            panic!("expected trust")
        };
        drop(reply);
    };
    tokio::join!(
        manager.ensure_loaded_with_approval(&root, &root, &tx),
        cancel
    );
    assert!(!manager.loaded);
    assert!(manager.decisions.is_empty());
    let answer = async {
        let AgentEvent::McpApproval { reply, .. } = rx.recv().await.unwrap() else {
            panic!("expected retry after cancellation")
        };
        assert!(reply.send(Approval::No).is_ok());
    };
    tokio::join!(
        manager.ensure_loaded_with_approval(&root, &root, &tx),
        answer
    );
    assert!(manager.loaded);
}

#[cfg(windows)]
#[tokio::test]
async fn stdio_interruption_closes_connection_without_retry() {
    std::fs::create_dir_all("target").unwrap();
    let dir = tempfile::tempdir_in("target").unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    for mode in [
        "blocked_rpc",
        "blocked_notify",
        "partial_timeout",
        "partial_cancel",
    ] {
        let partial = if mode.starts_with("partial") {
            "$null = [Console]::ReadLine(); [Console]::Write('{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":'); [Console]::Out.Flush();"
        } else {
            ""
        };
        let script = format!(
            r#"
$null = [Console]::ReadLine()
[Console]::WriteLine('{{"jsonrpc":"2.0","id":1,"result":{{}}}}')
$null = [Console]::ReadLine()
{partial}
Start-Sleep -Seconds 30
"#
        );
        let config = json!({"command":"powershell", "args":["-NoProfile", "-NonInteractive", "-Command", script]});
        let mut server = timeout(
            Duration::from_secs(10),
            StdioServer::spawn("test", &config, &root),
        )
        .await
        .unwrap()
        .unwrap();
        server.request_timeout = Duration::from_millis(200);
        let error = if mode == "partial_cancel" {
            server.request_timeout = Duration::from_secs(30);
            assert!(timeout(
                Duration::from_millis(200),
                server.rpc("tools/call", json!({}))
            )
            .await
            .is_err());
            None
        } else if mode == "blocked_notify" {
            Some(
                server
                    .notify(
                        "notifications/test",
                        json!({"text":"x".repeat(2 * 1024 * 1024)}),
                    )
                    .await
                    .unwrap_err(),
            )
        } else {
            let params = if mode == "blocked_rpc" {
                json!({"text":"x".repeat(2 * 1024 * 1024)})
            } else {
                json!({})
            };
            Some(server.rpc("tools/call", params).await.unwrap_err())
        };
        if let Some(error) = error {
            assert!(error.to_string().contains("timed out"), "{error}");
        }
        assert!(!server.usable, "{mode}");
        let next_id = server.next_id;
        assert!(server
            .rpc("tools/call", json!({}))
            .await
            .unwrap_err()
            .to_string()
            .contains("not retried"));
        assert_eq!(server.next_id, next_id);
        timeout(Duration::from_secs(5), server._child.wait())
            .await
            .unwrap()
            .unwrap();
    }
}

#[tokio::test]
async fn stdio_line_limit_preserves_exact_boundary_and_following_message() {
    for ending in ["\n", "\r\n"] {
        let mut input = "x".repeat(MAX_MESSAGE_BYTES - ending.len());
        input.push_str(ending);
        input.push_str("next\n");
        let mut reader = input.as_bytes();
        let line = read_stdio_line(&mut reader).await.unwrap();
        assert_eq!(line.len(), MAX_MESSAGE_BYTES);
        assert!(line.ends_with(ending));
        assert_eq!(read_stdio_line(&mut reader).await.unwrap(), "next\n");
        assert!(read_stdio_line(&mut reader).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn stdio_line_limit_rejects_oversized_input_without_reading_the_rest() {
    for ending in ["", "\n"] {
        let input = format!("{}{ending}", "x".repeat(MAX_MESSAGE_BYTES + 100));
        let mut reader = input.as_bytes();
        let error = read_stdio_line(&mut reader).await.unwrap_err();
        assert!(error.to_string().contains("stdio MCP message exceeds"));
        assert_eq!(reader.len(), input.len() - MAX_MESSAGE_BYTES - 1);
    }
    // No newline or EOF will ever arrive; the byte limit must stop the read.
    let mut reader = BufReader::new(tokio::io::repeat(b'x'));
    let error = timeout(Duration::from_secs(5), read_stdio_line(&mut reader))
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("stdio MCP message exceeds"));
}

#[tokio::test]
async fn stdio_line_limit_preserves_fragmented_utf8() {
    let input = "Привет\r\n世界\n";
    let mut reader = BufReader::with_capacity(1, input.as_bytes());
    assert_eq!(read_stdio_line(&mut reader).await.unwrap(), "Привет\r\n");
    assert_eq!(read_stdio_line(&mut reader).await.unwrap(), "世界\n");
    let mut invalid = &b"\xff\n"[..];
    assert!(read_stdio_line(&mut invalid).await.is_err());
}

#[test]
fn rpc_rejects_missing_or_ambiguous_results() {
    for response in [json!({"id":1}), json!({"id":1,"result":{},"error":{}})] {
        assert!(rpc_result(&response).is_err());
    }
    assert_eq!(
        rpc_result(&json!({"id":1,"result":null})).unwrap(),
        Value::Null
    );
    assert!(server_request_reply(&json!({"method":"ping"})).is_none());
    assert_eq!(
        server_request_reply(&json!({"method":"ping", "id":0})).unwrap()["result"],
        json!({})
    );
    assert_eq!(
        server_request_reply(&json!({"method":"unsupported", "id":"server"})).unwrap()["error"]
            ["code"],
        -32601
    );
}

#[tokio::test]
async fn stdio_ping_with_colliding_id_does_not_consume_tool_response() {
    std::fs::create_dir_all("target").unwrap();
    let root = tempfile::tempdir_in("target").unwrap();
    let root = std::fs::canonicalize(root.path()).unwrap();
    // Keep the directory alive explicitly until the stdio process has stopped.
    let script = if cfg!(windows) {
        r#"
$ErrorActionPreference = 'Stop'
$null = [Console]::ReadLine()
[Console]::WriteLine('{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"mock","version":"1"}}}')
$null = [Console]::ReadLine()
$null = [Console]::ReadLine()
[Console]::WriteLine('{"jsonrpc":"2.0","id":2,"method":"ping"}')
$reply = [Console]::ReadLine() | ConvertFrom-Json
if ($reply.id -ne 2 -or $null -eq $reply.result) { exit 7 }
[Console]::WriteLine('{"jsonrpc":"2.0","id":"server","method":"unsupported"}')
$reply = [Console]::ReadLine() | ConvertFrom-Json
if ($reply.id -ne 'server' -or $reply.error.code -ne -32601) { exit 8 }
[Console]::WriteLine('{"jsonrpc":"2.0","method":"notifications/test"}')
[Console]::WriteLine('{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"visible","inputSchema":{"type":"object"}}]}}')
$null = [Console]::ReadLine()
"#
    } else {
        r#"
read -r line
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"mock","version":"1"}}}'
read -r line
read -r line
printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"ping"}'
read -r reply
case "$reply" in *'"result":{}'*) ;; *) exit 7;; esac
printf '%s\n' '{"jsonrpc":"2.0","id":"server","method":"unsupported"}'
read -r reply
case "$reply" in *'-32601'*) ;; *) exit 8;; esac
printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/test"}' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"visible","inputSchema":{"type":"object"}}]}}'
read -r line
"#
    };
    let config = if cfg!(windows) {
        json!({"command":"powershell", "args":["-NoProfile", "-NonInteractive", "-Command", script]})
    } else {
        json!({"command":"sh", "args":["-c", script]})
    };
    let mut server = timeout(
        Duration::from_secs(10),
        StdioServer::spawn("test", &config, &root),
    )
    .await
    .unwrap()
    .unwrap();
    let result = timeout(Duration::from_secs(10), server.rpc("tools/list", json!({})))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result["tools"][0]["name"], "visible");
    server._child.kill().await.unwrap();
}

#[test]
fn stderr_tail_keeps_only_recent_bounded_lines() {
    let tail = StderrTail::default();
    for line in 0..50 {
        tail.push(format!("line {line}").as_bytes());
    }
    tail.push(b"   \r\n");
    let summary = tail.summary().expect("recent stderr lines");
    assert!(summary.contains("line 49"), "{summary}");
    assert!(!summary.contains("line 0 |"), "{summary}");
    assert!(summary.len() <= STDERR_SUMMARY_CHARS, "{summary}");

    let tail = StderrTail::default();
    tail.push(&[b'y'; 10_000]);
    let summary = tail.summary().expect("long single line");
    assert!(summary.len() <= STDERR_SUMMARY_CHARS);
}

#[tokio::test]
async fn startup_failure_reports_server_stderr_tail() {
    std::fs::create_dir_all("target").unwrap();
    let dir = tempfile::tempdir_in("target").unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    // Exits before answering initialize and explains the reason on stderr only.
    let script = if cfg!(windows) {
        r#"
[Console]::Error.WriteLine('old line: ' + ('x' * 500))
1..20 | ForEach-Object { [Console]::Error.WriteLine("noise $_") }
[Console]::Error.WriteLine('no solution found for duckgo-mcp-server[browser]')
exit 1
"#
    } else {
        r#"
printf 'old line: %s\n' "$(printf 'x%.0s' $(seq 1 500))" >&2
i=1
while [ $i -le 20 ]; do printf 'noise %s\n' "$i" >&2; i=$((i + 1)); done
printf '%s\n' 'no solution found for duckgo-mcp-server[browser]' >&2
exit 1
"#
    };
    let config = if cfg!(windows) {
        json!({"command":"powershell", "args":["-NoProfile", "-NonInteractive", "-Command", script]})
    } else {
        json!({"command":"sh", "args":["-c", script]})
    };
    let error = match timeout(
        Duration::from_secs(20),
        StdioServer::spawn("broken", &config, &root),
    )
    .await
    .unwrap()
    {
        Ok(_) => panic!("server exits before answering initialize"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("MCP initialization failed for 'broken'"),
        "{message}"
    );
    assert!(
        message.contains("closed stdout while waiting for 'initialize'"),
        "{message}"
    );
    assert!(
        message.contains("no solution found for duckgo-mcp-server[browser]"),
        "{message}"
    );
    // Only the tail is kept, so a long log cannot flood the UI.
    assert!(!message.contains("noise 1 |"), "{message}");
    assert!(message.len() < 1_000, "{message}");
}
