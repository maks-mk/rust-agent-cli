use super::*;

#[path = "agent_image_tests.rs"]
mod image_tests;
#[path = "agent_tool_tests.rs"]
mod tool_tests;
use std::sync::atomic::AtomicUsize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::timeout;

fn apply_history_event(history: &mut Vec<Value>, event: AgentEvent) {
    match event {
        AgentEvent::Sync(messages) => *history = messages,
        AgentEvent::HistoryPatch {
            start,
            end,
            messages,
        } => {
            history.splice(start..end, messages);
        }
        _ => {}
    }
}

#[test]
fn tool_results_send_only_changed_messages() {
    let prefix = json!({"role": "user", "content": "large prefix ".repeat(10_000)});
    let mut history = vec![
        prefix,
        json!({"role": "assistant", "tool_calls": []}),
        json!({"role": "tool", "tool_call_id": "first", "content": "pending"}),
        json!({"role": "tool", "tool_call_id": "second", "content": "pending"}),
    ];
    let mut mirrored = history.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    {
        let mut batch = ToolBatch {
            history: &mut history,
            tx: &tx,
            first_result: 2,
            image_messages: vec![None, None],
        };
        for index in [1, 0] {
            let call = ToolCall {
                id: if index == 0 { "first" } else { "second" }.into(),
                name: "read_file".into(),
                args: "{}".into(),
            };
            batch.record(
                index,
                &call,
                ToolOutput {
                    content: format!("result {index}"),
                    images: vec![],
                },
            );
        }
    }
    for index in [1, 0] {
        let event = rx.try_recv().unwrap();
        match &event {
            AgentEvent::HistoryPatch {
                start,
                end,
                messages,
            } => {
                assert_eq!((*start, *end), (2 + index, 3 + index));
                assert_eq!(messages.len(), 1);
                assert!(serde_json::to_vec(messages).unwrap().len() < 200);
            }
            _ => panic!("expected a result delta"),
        }
        apply_history_event(&mut mirrored, event);
    }
    assert!(rx.try_recv().is_err());
    assert_eq!(mirrored, history);
}

struct MockApi {
    cfg: Config,
    task: Option<JoinHandle<Vec<Value>>>,
}

impl MockApi {
    async fn start(responses: Vec<String>) -> Self {
        Self::start_with_options(responses, 200, 3, true).await
    }

    async fn start_with_options(
        responses: Vec<String>,
        status: u16,
        chunk_size: usize,
        content_length: bool,
    ) -> Self {
        Self::start_scripted(
            responses
                .into_iter()
                .map(|body| (status, body, String::new()))
                .collect(),
            chunk_size,
            content_length,
        )
        .await
    }

    async fn start_scripted(
        responses: Vec<(u16, String, String)>,
        chunk_size: usize,
        content_length: bool,
    ) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "agent-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("note.txt"), "test content").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, response, extra_headers) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let header_end = loop {
                    let mut chunk = [0; 1024];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                    if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let headers = std::str::from_utf8(&bytes[..header_end]).unwrap();
                let is_responses = headers.starts_with("POST /v1/responses HTTP/1.1");
                assert!(
                    is_responses || headers.starts_with("POST /v1/chat/completions HTTP/1.1"),
                    "unexpected completion endpoint: {headers}"
                );
                let length: usize = headers
                    .lines()
                    .filter_map(|line| line.split_once(':'))
                    .find(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                    .unwrap()
                    .1
                    .trim()
                    .parse()
                    .unwrap();
                while bytes.len() < header_end + length {
                    let mut chunk = [0; 1024];
                    let count = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(count, 0);
                    bytes.extend_from_slice(&chunk[..count]);
                }
                let request_body: Value =
                    serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap();
                if is_responses {
                    assert!(request_body.get("input").is_some());
                    assert!(request_body.get("messages").is_none());
                } else {
                    assert!(request_body.get("messages").is_some());
                }
                requests.push(request_body);
                // Status 0 simulates a connection lost before response headers.
                if status == 0 {
                    continue;
                }
                let length_header = if content_length {
                    format!("Content-Length: {}\r\n", response.len())
                } else {
                    String::new()
                };
                let headers = format!("HTTP/1.1 {status} Mock\r\nContent-Type: text/event-stream\r\n{length_header}{extra_headers}Connection: close\r\n\r\n");
                socket.write_all(headers.as_bytes()).await.unwrap();
                // Split UTF-8 and JSON across transport chunks, independent of SSE lines.
                for chunk in response.as_bytes().chunks(chunk_size) {
                    if socket.write_all(chunk).await.is_err() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                let _ = socket.shutdown().await;
            }
            requests
        });
        Self {
            cfg: Config {
                api_mode: crate::agent::ApiMode::Chat,
                base_url,
                api_key: String::new(),
                model: "mock".into(),
                reasoning_effort: Some("high".into()),
                reasoning_efforts: Vec::new(),
                config_dir: root.clone(),
                root,
                max_steps: crate::agent::DEFAULT_MAX_STEPS,
                context_tokens: crate::agent::DEFAULT_CONTEXT_TOKENS,
                auto_approve: false,
            },
            task: Some(task),
        }
    }

    async fn requests(&mut self) -> Vec<Value> {
        timeout(Duration::from_secs(10), self.task.as_mut().unwrap())
            .await
            .unwrap()
            .unwrap()
    }
}

impl Drop for MockApi {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
        let _ = std::fs::remove_dir_all(&self.cfg.root);
    }
}

#[test]
fn retry_policy_respects_backoff_and_retry_after() {
    let error = transient_completion_error("temporary");
    for (attempt, seconds) in [2, 4, 8].into_iter().enumerate() {
        assert_eq!(
            retry_delay(&error, attempt),
            Some(Duration::from_secs(seconds))
        );
    }
    assert_eq!(retry_delay(&error, 3), None);
    assert_eq!(retry_delay(&anyhow!("invalid response"), 0), None);
    assert_eq!(parse_retry_after("12"), Some(Duration::from_secs(12)));
    assert_eq!(
        parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
        Some(Duration::ZERO)
    );
    assert_eq!(parse_retry_after("invalid"), None);
    for (seconds, expected) in [(10, Some(Duration::from_secs(10))), (61, None)] {
        let error = RetryableCompletionError {
            message: "busy".into(),
            retry_after: Some(Duration::from_secs(seconds)),
        }
        .into();
        assert_eq!(retry_delay(&error, 0), expected);
    }
}

#[tokio::test]
async fn retries_transient_http_and_disconnected_requests() {
    for status in [0, 408, 429, 500, 502, 503, 504] {
        let mut api = MockApi::start_scripted(
            vec![
                (status, "temporary".into(), "Retry-After: 0\r\n".into()),
                (
                    200,
                    sse(vec![delta(json!({"content":"success"}))]),
                    String::new(),
                ),
            ],
            1024,
            true,
        )
        .await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let result = timeout(
            Duration::from_secs(10),
            completion_with_retries(
                &reqwest::Client::new(),
                &api.cfg,
                &[],
                Some(&json!([])),
                &tx,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.text, "success");
        let requests = api.requests().await;
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0], requests[1]);
        let mut starts = 0;
        let mut discards = 0;
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::CompletionStart => starts += 1,
                AgentEvent::CompletionDiscard => discards += 1,
                _ => {}
            }
        }
        assert_eq!((starts, discards), (2, 1));
    }
}

#[tokio::test]
async fn permanent_http_errors_and_limits_are_not_retried() {
    let oversized = format!("data: {}\n\n", "x".repeat(MAX_SSE_LINE_BYTES));
    for (status, body, headers) in [
        (401, "unauthorized".into(), String::new()),
        (400, "invalid request".into(), String::new()),
        (503, "busy".into(), "Retry-After: 61\r\n".into()),
        (200, oversized, String::new()),
    ] {
        let mut api = MockApi::start_scripted(vec![(status, body, headers)], 16 * 1024, true).await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(
            completion_with_retries(&reqwest::Client::new(), &api.cfg, &[], None, &tx)
                .await
                .is_err()
        );
        assert_eq!(api.requests().await.len(), 1);
        let mut starts = 0;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AgentEvent::UsageStart) {
                starts += 1;
            }
            assert!(!matches!(event, AgentEvent::Notice(_)));
        }
        assert_eq!(starts, 1);
    }
}

#[tokio::test]
async fn retries_stop_after_four_attempts() {
    let mut api = MockApi::start_with_options(vec!["unavailable".into(); 4], 503, 1024, true).await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let error = timeout(
        Duration::from_secs(20),
        completion_with_retries(&reqwest::Client::new(), &api.cfg, &[], None, &tx),
    )
    .await
    .unwrap()
    .err()
    .unwrap();
    assert!(error.to_string().starts_with("HTTP 503:"));
    assert_eq!(api.requests().await.len(), 4);
    let mut starts = 0;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::UsageStart) {
            starts += 1;
        }
    }
    assert_eq!(starts, 4);
}

#[tokio::test]
async fn broken_stream_retries_without_replaying_tools_and_keeps_usage() {
    let incomplete = format!(
        "data: {}\n\ndata: {}\n\n",
        usage(10, 2),
        delta(
            json!({"content":"partial", "tool_calls":[{"index":0,"id":"discard","function":{"name":"read_file","arguments":"{}"}}]})
        )
    );
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"tool_calls":[{"index":0,"id":"read","function":{"name":"read_file","arguments":"{\"path\":\"note.txt\"}"}}]}))]),
        incomplete,
        sse(vec![delta(json!({"content":"complete"})), usage(20, 3)]),
    ]).await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = run_turn(
        &reqwest::Client::new(),
        &api.cfg,
        vec![json!({"role":"user","content":"read"})],
        &tx,
        &Arc::new(AtomicBool::new(true)),
        &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
    )
    .await
    .unwrap();
    let requests = api.requests().await;
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1], requests[2]);
    assert_eq!(history.last().unwrap()["content"], "complete");
    assert_eq!(history.iter().filter(|m| m["role"] == "tool").count(), 1);
    let mut tools = 0;
    let mut reports = Vec::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolStart { .. } => tools += 1,
            AgentEvent::Usage { usage, complete } => reports.push((usage.total_tokens, complete)),
            _ => {}
        }
    }
    assert_eq!(tools, 1);
    assert_eq!(reports, [(12, false), (23, true)]);
}

async fn limited_completion(response: String) -> Result<Completion> {
    let mut api = MockApi::start_with_options(vec![response], 200, 16 * 1024, false).await;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let result = timeout(
        Duration::from_secs(10),
        stream_completion(&reqwest::Client::new(), &api.cfg, &[], None, &tx),
    )
    .await
    .expect("completion must stop within the test timeout");
    api.requests().await;
    result
}

#[tokio::test]
async fn completion_bounds_sse_lines_including_unterminated_input() {
    let mut exact = format!(":{}\n", "x".repeat(MAX_SSE_LINE_BYTES - 2));
    exact.push_str(&sse(vec![delta(json!({"content":"Привет"}))]));
    assert_eq!(limited_completion(exact).await.unwrap().text, "Привет");

    for ending in ["", "\n"] {
        let oversized = format!(":{}{ending}", "x".repeat(MAX_SSE_LINE_BYTES));
        let error = limited_completion(oversized).await.err().unwrap();
        assert!(error.to_string().contains("SSE line exceeds"), "{error}");
    }
}

#[tokio::test]
async fn completion_bounds_total_wire_bytes_even_for_ignored_lines() {
    for size in [
        MAX_COMPLETION_BYTES - 1,
        MAX_COMPLETION_BYTES,
        MAX_COMPLETION_BYTES + 1,
    ] {
        let done = "data: [DONE]\n";
        let mut response = String::with_capacity(size);
        let mut remaining = size - done.len();
        while remaining > 0 {
            let count = remaining.min(MAX_SSE_LINE_BYTES);
            response.push_str(&" ".repeat(count - 1));
            response.push('\n');
            remaining -= count;
        }
        response.push_str(done);
        assert_eq!(response.len(), size);
        let result = limited_completion(response).await;
        if size <= MAX_COMPLETION_BYTES {
            assert!(result.is_ok());
        } else {
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("model response exceeds"));
        }
    }
}

#[tokio::test]
async fn completion_bounds_accumulated_text_reasoning_and_tool_arguments() {
    let part = "x".repeat(16 * 1024);
    for value in [
        json!({"content":part}),
        json!({"reasoning_content":part}),
        json!({"tool_calls":[{"index":0,"id":"call","function":{"name":"read_file","arguments":part}}]}),
    ] {
        let line = format!("data: {}\n\n", delta(value));
        let response = line.repeat(MAX_COMPLETION_BYTES / line.len() + 1);
        let error = limited_completion(response).await.err().unwrap();
        assert!(
            error.to_string().contains("model response exceeds"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn completion_rejects_invalid_tool_indices_before_allocating() {
    for index in [
        json!(u64::MAX),
        json!(MAX_COMPLETION_TOOL_CALLS),
        json!(-1),
        json!(1.5),
        json!("0"),
        Value::Null,
    ] {
        let error = limited_completion(sse(vec![delta(json!({"tool_calls":[{
            "index":index, "id":"bad", "function":{"name":"read_file","arguments":"{}"}
        }]}))]))
        .await
        .err()
        .unwrap();
        assert!(error.to_string().contains("tool call index"), "{error}");
    }
    let completion = limited_completion(sse(vec![delta(json!({"tool_calls":[{
        "index":MAX_COMPLETION_TOOL_CALLS - 1, "id":"last", "function":{"name":"read_file","arguments":"{}"}
    }]}))])).await.unwrap();
    assert_eq!(completion.calls.len(), 1);
    assert_eq!(completion.calls[0].id, "last");
}

#[tokio::test]
async fn completion_bounds_tool_calls_without_indices() {
    let calls: Vec<Value> = (0..MAX_COMPLETION_TOOL_CALLS)
        .map(|i| {
            json!({
                "id":format!("call_{i}"), "function":{"name":"read_file","arguments":"{}"}
            })
        })
        .collect();
    let response = sse(vec![
        delta(json!({"tool_calls":calls})),
        delta(json!({
            "tool_calls":[{"id":"call_0","function":{"arguments":" "}}]
        })),
    ]);
    let completion = limited_completion(response).await.unwrap();
    assert_eq!(completion.calls.len(), MAX_COMPLETION_TOOL_CALLS);
    assert_eq!(completion.calls[0].args, "{} ");

    let response = sse(vec![
        delta(json!({"tool_calls":calls})),
        delta(json!({
            "tool_calls":[{"id":"overflow","function":{"name":"read_file","arguments":"{}"}}]
        })),
    ]);
    let error = limited_completion(response).await.err().unwrap();
    assert!(error.to_string().contains("128 tool calls"), "{error}");
}

#[tokio::test]
async fn http_error_body_is_bounded_with_or_without_content_length() {
    for content_length in [false, true] {
        for size in [
            MAX_HTTP_ERROR_BYTES - 1,
            MAX_HTTP_ERROR_BYTES,
            MAX_HTTP_ERROR_BYTES + 1,
        ] {
            let mut api =
                MockApi::start_with_options(vec!["x".repeat(size)], 429, 1024, content_length)
                    .await;
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let error = timeout(
                Duration::from_secs(10),
                stream_completion(&reqwest::Client::new(), &api.cfg, &[], None, &tx),
            )
            .await
            .unwrap()
            .err()
            .unwrap()
            .to_string();
            api.requests().await;
            assert!(error.starts_with("HTTP 429:"));
            assert!(error.contains("rate limit or quota"));
            assert_eq!(
                error.contains("body truncated"),
                size > MAX_HTTP_ERROR_BYTES
            );
            assert!(error.len() < 500);
        }
    }
}

#[tokio::test]
async fn http_error_limit_stops_before_eof_and_handles_split_utf8() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 1024];
        assert_ne!(socket.read(&mut request).await.unwrap(), 0);
        socket
            .write_all(b"HTTP/1.1 500 Mock\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        // The UTF-8 character straddles the retained prefix boundary.
        let body = format!("{}я", "x".repeat(MAX_HTTP_ERROR_BYTES - 1));
        socket.write_all(body.as_bytes()).await.unwrap();
        std::future::pending::<()>().await;
    });
    let result = timeout(Duration::from_secs(5), async {
        let response = reqwest::Client::new().get(url).send().await.unwrap();
        describe_http_response_error(response).await
    })
    .await;
    task.abort();
    let error = result.expect("HTTP error reading must not wait for EOF");
    assert!(error.starts_with("HTTP 500:"));
    assert!(error.contains("body truncated"));
    assert!(error.len() < 500);
}

#[test]
fn context_threshold_parsing_and_estimation() {
    for raw in [
        None,
        Some(""),
        Some("bad"),
        Some("-1"),
        Some("999999999999999999999999999999"),
    ] {
        assert_eq!(parse_context_tokens(raw), DEFAULT_CONTEXT_TOKENS);
    }
    assert_eq!(parse_context_tokens(Some("0")), 0);
    assert_eq!(parse_context_tokens(Some("100000")), 100_000);
    assert_eq!(parse_context_tokens(Some(" 12000 ")), 12_000);
    let history = vec![json!({"role":"user", "content":"Привет"})];
    let estimated = estimate_history(&history);
    assert_eq!(context_size(&history, 10, None), estimated + 10);
    assert_eq!(context_size(&history, 10, Some((estimated, 200))), 200);
}

fn long_history() -> Vec<Value> {
    vec![
        json!({"role":"user", "content":"old request ".repeat(500)}),
        json!({"role":"assistant", "content":"old result ".repeat(500)}),
        json!({"role":"user", "content":"Continue the current task"}),
    ]
}

#[tokio::test]
async fn compression_is_private_syncs_history_and_preserves_parallel_tool_pairs() {
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"reasoning_content":"private reasoning", "content":"Summary of earlier work"})), usage(6000, 20)]),
        sse(vec![delta(json!({"content":"Final answer"})), usage(1000, 3)]),
    ]).await;
    api.cfg.context_tokens = 1;
    let mut original = long_history();
    original.insert(
        0,
        json!({"role":"system", "content":"Preserve these project constraints"}),
    );
    original.extend([
        json!({"role":"assistant", "content":"Reading files", "tool_calls":[
            {"id":"a", "type":"function", "function":{"name":"read_file", "arguments":"{}"}},
            {"id":"b", "type":"function", "function":{"name":"read_file", "arguments":"{}"}}
        ]}),
        json!({"role":"tool", "tool_call_id":"a", "content":"First result"}),
        json!({"role":"tool", "tool_call_id":"b", "content":"Second result"}),
    ]);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = run_turn(
        &reqwest::Client::new(),
        &api.cfg,
        original.clone(),
        &tx,
        &Arc::new(AtomicBool::new(false)),
        &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
    )
    .await
    .unwrap();
    assert_eq!(history[0], original[0]);
    assert!(history[1]["content"]
        .as_str()
        .unwrap()
        .contains("Summary of earlier work"));
    assert_eq!(&history[2..6], &original[3..7]);
    assert_eq!(history[6]["content"], "Final answer");
    let requests = api.requests().await;
    assert_eq!(
        requests[0]["messages"][0]["content"],
        CONTEXT_COMPRESSION_PROMPT
    );
    assert!(requests[0].get("tools").is_none());
    assert!(requests[1]["tools"].is_array());
    assert_eq!(
        &requests[1]["messages"].as_array().unwrap()[1..],
        &history[..6]
    );
    let mut active = Vec::new();
    let mut texts = Vec::new();
    let mut usages = 0;
    let mut sizes = Vec::new();
    let mut synced = false;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ContextCompression { active: running } => active.push(running),
            AgentEvent::ContextUsage { tokens, .. } => sizes.push(tokens),
            AgentEvent::Text(text) => texts.push(text),
            AgentEvent::Reasoning(_) => panic!("private reasoning leaked"),
            AgentEvent::Usage { complete, .. } => {
                assert!(complete);
                usages += 1;
            }
            AgentEvent::Sync(messages) => {
                assert_eq!(messages, history[..6]);
                synced = true;
            }
            _ => {}
        }
    }
    assert_eq!(active, [true, false]);
    assert_eq!(texts, ["Final answer"]);
    assert_eq!(usages, 2);
    assert!(synced);
    assert!(sizes[1] < sizes[0]);
}

#[tokio::test]
async fn compression_marks_lossy_boundaries_and_preserves_labels_on_recompression() {
    let memories = [
        "Recorded state and evidence\n[reported] Earlier results; checks unknown.",
        "Recorded state and evidence\n[reported/prior-summary] [reported] Earlier results; checks unknown.",
    ];
    let mut api = MockApi::start(
        memories
            .iter()
            .map(|text| sse(vec![delta(json!({"content":text}))]))
            .collect(),
    )
    .await;
    let mut original = long_history();
    let latest_user = original.last().unwrap().clone();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();

    for expected in memories {
        let compressed = compress_context(&reqwest::Client::new(), &api.cfg, &original, &tx)
            .await
            .unwrap()
            .unwrap();
        assert!(is_context_summary(&compressed[0]));
        assert!(!is_user_request(&compressed[0]));
        assert_eq!(compressed[1], latest_user);
        let content = compressed[0]["content"].as_str().unwrap();
        assert!(content.starts_with(CONTEXT_SUMMARY_PREFIX));
        assert!(content
            .contains("Context compressed: 2 history messages replaced by this lossy summary."));
        assert!(content.contains("earlier messages, not fresh verification"));
        assert!(content.contains("retained verbatim and may predate compression"));
        assert!(content.ends_with(&format!("{expected}{CONTEXT_SUMMARY_END}")));
        assert_eq!(content.matches(CONTEXT_SUMMARY_PREFIX).count(), 1);
        assert_eq!(content.matches(CONTEXT_SUMMARY_END).count(), 1);

        let responses = responses_input(&compressed);
        assert_eq!(responses[0]["role"], "user");
        assert_eq!(responses[0]["content"], content);
        original = compressed;
        original.extend([
            json!({"role":"assistant", "content":"Further work; verification pending. ".repeat(100)}),
            json!({"role":"assistant", "content":"Awaiting verification"}),
        ]);
    }

    let requests = api.requests().await;
    assert_eq!(requests.len(), 2);
    for (index, request) in requests.iter().enumerate() {
        assert_eq!(
            request["messages"][0]["content"],
            CONTEXT_COMPRESSION_PROMPT
        );
        assert!(request.get("tools").is_none());
        let input: Value =
            serde_json::from_str(request["messages"][1]["content"].as_str().unwrap()).unwrap();
        assert_eq!(input["latest_user_request"], latest_user["content"]);
        assert_eq!(
            is_context_summary(&input["messages_to_summarize"][0]),
            index == 1
        );
        if index == 1 {
            assert!(input["messages_to_summarize"][0]["content"]
                .as_str()
                .unwrap()
                .contains(memories[0]));
        }
    }
}

#[tokio::test]
async fn repeated_compression_uses_latest_request_as_reference_and_merges_prior_memory() {
    let mut api = MockApi::start(vec![sse(vec![delta(json!({
        "content":"Changes made; verification pending."
    }))])])
    .await;
    let original = vec![
        json!({"role":"user", "content":format!("{CONTEXT_SUMMARY_PREFIX}{}", "Prior decisions; tests not run. ".repeat(100))}),
        json!({"role":"user", "content":"Improve compression without bloating the prompt"}),
        json!({"role":"assistant", "content":"Implemented changes; verification pending. ".repeat(100), "responses_output_items": [{"type":"message"}]}),
        json!({"role":"assistant", "content":"Next: run the tests"}),
    ];
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let compressed = compress_context(&reqwest::Client::new(), &api.cfg, &original, &tx)
        .await
        .unwrap()
        .unwrap();
    assert!(is_context_summary(&compressed[0]));
    assert_eq!(compressed[1..], [original[1].clone(), original[3].clone()]);

    let requests = api.requests().await;
    let input: Value =
        serde_json::from_str(requests[0]["messages"][1]["content"].as_str().unwrap()).unwrap();
    assert_eq!(input["latest_user_request"], original[1]["content"]);
    let mut updated = original[2].clone();
    updated
        .as_object_mut()
        .unwrap()
        .remove("responses_output_items");
    assert_eq!(
        input["messages_to_summarize"],
        json!([original[0], updated])
    );
}

#[tokio::test]
async fn compression_without_user_request_keeps_reference_empty() {
    let mut api =
        MockApi::start(vec![sse(vec![delta(json!({"content":"Earlier results"}))])]).await;
    let original = vec![
        json!({"role":"assistant", "content":"Earlier results ".repeat(500)}),
        json!({"role":"assistant", "content":"Current state"}),
    ];
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    let compressed = compress_context(&reqwest::Client::new(), &api.cfg, &original, &tx)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(compressed[1], original[1]);
    let requests = api.requests().await;
    let input: Value =
        serde_json::from_str(requests[0]["messages"][1]["content"].as_str().unwrap()).unwrap();
    assert!(input["latest_user_request"].is_null());
    assert_eq!(input["messages_to_summarize"], json!([original[0]]));
}

#[tokio::test]
async fn measured_prompt_growth_triggers_compression_after_tools() {
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"tool_calls":[{"index":0, "id":"read", "function":{"name":"read_file", "arguments":"{\"path\":\"note.txt\"}"}}]})), usage(20_000, 5)]),
        sse(vec![delta(json!({"content":"Earlier work summary"})), usage(4000, 10)]),
        sse(vec![delta(json!({"content":"Done"})), usage(1500, 2)]),
    ]).await;
    api.cfg.context_tokens = 20_000;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = run_turn(
        &reqwest::Client::new(),
        &api.cfg,
        long_history(),
        &tx,
        &Arc::new(AtomicBool::new(false)),
        &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
    )
    .await
    .unwrap();
    let requests = api.requests().await;
    assert!(requests[0].get("tools").is_some());
    assert!(requests[1].get("tools").is_none());
    assert_eq!(history[2]["tool_calls"][0]["id"], "read");
    assert_eq!(history[3]["tool_call_id"], "read");
    assert_eq!(history[4]["content"], "Done");
    let mut starts = 0;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::ContextCompression { active: true }) {
            starts += 1;
        }
    }
    assert_eq!(starts, 1);
}

#[tokio::test]
async fn disabled_below_threshold_and_indivisible_context_skip_compression() {
    for (limit, original) in [
        (0, long_history()),
        (usize::MAX, long_history()),
        (
            1,
            vec![json!({"role":"user", "content":"large request".repeat(500)})],
        ),
    ] {
        let mut api = MockApi::start(vec![sse(vec![delta(json!({"content":"Done"}))])]).await;
        api.cfg.context_tokens = limit;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let history = run_turn(
            &reqwest::Client::new(),
            &api.cfg,
            original.clone(),
            &tx,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
        )
        .await
        .unwrap();
        assert_eq!(&history[..original.len()], original.as_slice());
        assert_eq!(api.requests().await.len(), 1);
        while let Ok(event) = rx.try_recv() {
            assert!(!matches!(event, AgentEvent::ContextCompression { .. }));
        }
    }
}

#[tokio::test]
async fn failed_compression_preserves_history_and_clears_status() {
    for response in [
        sse(vec![
            usage(500, 1),
            json!({"error":{"message":"summary failure"}}),
        ]),
        sse(vec![delta(json!({"content":" "}))]),
        sse(vec![delta(json!({"content":"not shorter ".repeat(2000)}))]),
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n".into(),
    ] {
        let mut api = MockApi::start(vec![response]).await;
        let original = long_history();
        let snapshot = original.clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let result = compress_context(&reqwest::Client::new(), &api.cfg, &original, &tx).await;
        assert!(result.is_err());
        assert_eq!(original, snapshot);
        api.requests().await;
        let mut active = Vec::new();
        while let Ok(event) = rx.try_recv() {
            match event {
                AgentEvent::ContextCompression { active: running } => active.push(running),
                AgentEvent::Text(_) | AgentEvent::Reasoning(_) | AgentEvent::Sync(_) => {
                    panic!("private or failed summary leaked")
                }
                _ => {}
            }
        }
        assert_eq!(active, [true, false]);
    }
}

#[tokio::test]
async fn cancelling_compression_clears_status_without_syncing_history() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api = MockApi::start(vec![]).await;
    let mut cfg = api.cfg.clone();
    cfg.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        compress_context(&reqwest::Client::new(), &cfg, &long_history(), &tx).await
    });
    assert!(matches!(
        timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap(),
        AgentEvent::ContextCompression { active: true }
    ));
    assert!(matches!(rx.recv().await.unwrap(), AgentEvent::UsageStart));
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(matches!(
        rx.recv().await.unwrap(),
        AgentEvent::ContextCompression { active: false }
    ));
    assert!(rx.try_recv().is_err());
}

fn sse(chunks: Vec<Value>) -> String {
    let mut body = String::from(": keepalive\r\n\r\n");
    for chunk in chunks {
        body.push_str(&format!("data: {chunk}\r\n\r\n"));
    }
    body.push_str("data: [DONE]\r\n\r\n");
    body
}

fn delta(value: Value) -> Value {
    json!({"choices": [{"delta": value}], "usage": null})
}

fn usage(input: u64, output: u64) -> Value {
    json!({"choices": [], "usage": {"prompt_tokens": input, "completion_tokens": output, "total_tokens": input + output}})
}

fn cached_usage(input: u64, output: u64, cached: u64) -> Value {
    let mut chunk = usage(input, output);
    chunk["usage"]["prompt_tokens_details"] = json!({"cached_tokens": cached});
    chunk
}

#[tokio::test]
async fn responses_mode_streams_text_tools_usage_and_reuses_output_items() {
    let response = sse(vec![
        json!({"type":"response.output_text.delta", "delta":"Checking."}),
        json!({"type":"response.completed", "response":{
            "status":"completed",
            "output":[
                {"id":"msg_1", "type":"message", "role":"assistant", "status":"completed", "content":[
                    {"type":"output_text", "text":"Checking.", "annotations":[], "logprobs":[]}
                ]},
                {"id":"fc_1", "type":"function_call", "status":"completed", "call_id":"call_read",
                    "name":"read_file", "arguments":"{\"path\":\"note.txt\"}"}
            ],
            "usage":{"input_tokens":40, "output_tokens":8, "total_tokens":48,
                "input_tokens_details":{"cached_tokens":12}, "output_tokens_details":{"reasoning_tokens":2}}
        }}),
    ]);
    let mut api = MockApi::start(vec![response]).await;
    api.cfg.api_mode = crate::agent::ApiMode::Responses;
    let messages = vec![json!({"role":"user", "content":"read note.txt"})];
    let tools = json!([{"type":"function", "function":{
        "name":"read_file", "description":"Read a file",
        "parameters":{"type":"object", "properties":{"path":{"type":"string"}}, "required":["path"]}
    }}]);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let completion = stream_completion(
        &reqwest::Client::new(),
        &api.cfg,
        &messages,
        Some(&tools),
        &tx,
    )
    .await
    .unwrap();

    assert_eq!(completion.text, "Checking.");
    assert_eq!(completion.calls.len(), 1);
    assert_eq!(completion.calls[0].id, "call_read");
    assert_eq!(completion.calls[0].name, "read_file");
    assert_eq!(completion.calls[0].args, r#"{"path":"note.txt"}"#);
    let usage = completion.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 40);
    assert_eq!(usage.completion_tokens, 8);
    assert_eq!(usage.cached_tokens, Some(12));

    let assistant_message = completion.assistant_message();
    assert_eq!(
        assistant_message["responses_output_items"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(responses_input(&[assistant_message])[0]["type"], "message");
    assert_eq!(
        responses_input(&[completion.assistant_message()])[1]["call_id"],
        "call_read"
    );
    let mut text_events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Text(text) = event {
            text_events.push(text);
        }
    }
    assert_eq!(text_events.concat(), "Checking.");

    let requests = api.requests().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["input"][0]["role"], "user");
    assert_eq!(requests[0]["tools"][0]["name"], "read_file");
    assert!(requests[0].get("messages").is_none());
    // Responses mode carries the thinking level as reasoning.effort; the flat
    // chat-style reasoning_effort field must not be sent there.
    assert_eq!(requests[0]["reasoning"]["effort"], "high");
    assert!(requests[0].get("reasoning_effort").is_none());
}

#[tokio::test]
async fn reasoning_survives_tool_rounds_and_reloaded_history_without_duplicate_usage() {
    let mut api = MockApi::start(vec![
        sse(vec![
            delta(json!({"reasoning_content": "  Проверю"})),
            delta(json!({"reasoning_content": " файл.\n", "content": "Читаю."})),
            delta(json!({"tool_calls": [{"index": 0, "id": "read_1", "function": {"name": "read_file", "arguments": "{\"pa"}}]})),
            delta(json!({"tool_calls": [{"index": 0, "function": {"arguments": "th\":\"note.txt\"}"}}]})),
            cached_usage(10, 2, 0), cached_usage(20, 5, 9), cached_usage(20, 5, 9),
            json!({"choices": [], "usage": null}),
        ]),
        sse(vec![
            delta(json!({"reasoning": "Ещё проверка."})),
            delta(json!({"tool_calls": [{"index": 0, "id": "read_2", "function": {"name": "read_file", "arguments": {"path": "note.txt"}}}]})),
            cached_usage(30, 7, 12),
        ]),
        sse(vec![
            delta(json!({"reasoning_content": "Итог."})),
            json!({"choices": [{"delta": {"content": "Готово."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}}),
        ]),
        sse(vec![delta(json!({"content": "Next answer"})), cached_usage(50, 11, 25)]),
    ]).await;
    let client = reqwest::Client::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let auto = Arc::new(AtomicBool::new(false));
    let mcp = Arc::new(tokio::sync::Mutex::new(McpManager::default()));
    let history = timeout(
        Duration::from_secs(10),
        run_turn(
            &client,
            &api.cfg,
            vec![json!({"role": "user", "content": "Check"})],
            &tx,
            &auto,
            &mcp,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(history.len(), 6);
    assert_eq!(history[1]["reasoning_content"], "  Проверю файл.\n");
    assert_eq!(history[1]["content"], "Читаю.");
    assert_eq!(history[3]["reasoning_content"], "Ещё проверка.");
    assert_eq!(history[5]["reasoning_content"], "Итог.");
    for index in [2, 4] {
        assert!(history[index]["content"]
            .as_str()
            .unwrap()
            .contains("test content"));
    }
    // Emulate a saved desktop conversation being restored for the next user turn.
    let mut restored: Vec<Value> =
        serde_json::from_str(&serde_json::to_string(&history).unwrap()).unwrap();
    restored.push(json!({"role": "user", "content": "Continue"}));
    let final_history = timeout(
        Duration::from_secs(10),
        run_turn(&client, &api.cfg, restored.clone(), &tx, &auto, &mcp),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(final_history
        .last()
        .unwrap()
        .get("reasoning_content")
        .is_none());
    assert!(final_history.last().unwrap().get("tool_calls").is_none());

    let requests = api.requests().await;
    assert_eq!(requests.len(), 4);
    for request in &requests {
        assert_eq!(request["stream_options"]["include_usage"], true);
        assert_eq!(request["reasoning_effort"], "high");
        assert!(request.get("usage").is_none());
    }
    assert_eq!(requests[1]["messages"][2], history[1]);
    assert_eq!(requests[2]["messages"][4], history[3]);
    assert_eq!(
        &requests[3]["messages"].as_array().unwrap()[1..],
        restored.as_slice()
    );

    let mut starts = 0;
    let mut totals = Vec::new();
    let mut cached = Vec::new();
    let mut reasoning = String::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::UsageStart => starts += 1,
            AgentEvent::Usage { usage, complete } => {
                assert!(complete);
                totals.push(usage.total_tokens);
                cached.push(usage.cached_tokens);
            }
            AgentEvent::Reasoning(text) => reasoning.push_str(&text),
            _ => {}
        }
    }
    assert_eq!(starts, 4);
    assert_eq!(totals, vec![25, 37, 49, 61]);
    assert_eq!(cached, vec![Some(9), Some(12), None, Some(25)]);
    assert_eq!(reasoning, "  Проверю файл.\nЕщё проверка.Итог.");
}

#[tokio::test]
async fn missing_usage_and_reasoning_only_answers_are_preserved() {
    let mut api = MockApi::start(vec![sse(vec![delta(
        json!({"reasoning_content": "Thinking only"}),
    )])])
    .await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = timeout(
        Duration::from_secs(10),
        run_turn(
            &reqwest::Client::new(),
            &api.cfg,
            vec![],
            &tx,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(history[0]["reasoning_content"], "Thinking only");
    assert_eq!(history[0]["content"], "");
    api.requests().await;
    let mut starts = 0;
    while let Ok(event) = rx.try_recv() {
        assert!(
            !matches!(event, AgentEvent::Usage { .. }),
            "unknown usage must not become zero usage"
        );
        if matches!(event, AgentEvent::UsageStart) {
            starts += 1;
        }
    }
    assert_eq!(starts, 1);
}

#[tokio::test]
async fn stream_error_retains_measured_usage_as_incomplete() {
    let mut api = MockApi::start(vec![sse(vec![
        cached_usage(12, 3, 8),
        json!({"error": {"message": "mock failure"}}),
    ])])
    .await;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let result = timeout(
        Duration::from_secs(10),
        stream_completion(
            &reqwest::Client::new(),
            &api.cfg,
            &[],
            Some(&tools_schema(&McpManager::default())),
            &tx,
        ),
    )
    .await
    .unwrap();
    assert_eq!(result.err().unwrap().to_string(), "mock failure");
    api.requests().await;
    assert!(matches!(rx.try_recv().unwrap(), AgentEvent::UsageStart));
    match rx.try_recv().unwrap() {
        AgentEvent::Usage { usage, complete } => {
            assert_eq!(usage.total_tokens, 15);
            assert_eq!(usage.cached_tokens, Some(8));
            assert!(!complete);
        }
        _ => panic!("expected usage"),
    }
    assert!(rx.try_recv().is_err());
}

#[tokio::test]
async fn eof_requires_a_completion_marker_and_keeps_partial_usage() {
    for finished in [false, true] {
        let end = if finished {
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        } else {
            delta(json!({"content": "unfinished"}))
        };
        let response = sse(vec![usage(10, 2), end]);
        let response = response
            .strip_suffix("data: [DONE]\r\n\r\n")
            .unwrap()
            .to_string();
        let mut api = MockApi::start(vec![response]).await;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let result = timeout(
            Duration::from_secs(10),
            stream_completion(
                &reqwest::Client::new(),
                &api.cfg,
                &[],
                Some(&tools_schema(&McpManager::default())),
                &tx,
            ),
        )
        .await
        .unwrap();
        assert_eq!(result.is_ok(), finished);
        if !finished {
            assert_eq!(
                result.err().unwrap().to_string(),
                "stream ended before completion was confirmed"
            );
        }
        api.requests().await;
        let mut reports = 0;
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::Usage { usage, complete } = event {
                assert_eq!(usage.total_tokens, 12);
                assert_eq!(complete, finished);
                reports += 1;
            }
        }
        assert_eq!(reports, 1);
    }
}

#[tokio::test]
async fn cancellation_reports_latest_usage_only_once_as_incomplete() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (ready_tx, ready_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let _report = UsageReport {
            tx: &tx,
            latest: Some(TokenUsage {
                prompt_tokens: 12,
                completion_tokens: 3,
                total_tokens: 15,
                cached_tokens: Some(8),
            }),
            complete: false,
        };
        let _ = ready_tx.send(());
        std::future::pending::<()>().await;
    });
    ready_rx.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    match rx.try_recv().unwrap() {
        AgentEvent::Usage { usage, complete } => {
            assert_eq!(usage.total_tokens, 15);
            assert_eq!(usage.cached_tokens, Some(8));
            assert!(!complete);
        }
        _ => panic!("expected usage"),
    }
    assert!(rx.try_recv().is_err());
}

#[test]
fn cached_usage_is_optional_validated_and_already_included_in_totals() {
    for cached in [0, 800, 1200] {
        let parsed = TokenUsage::from_value(&cached_usage(1200, 300, cached)["usage"]).unwrap();
        assert_eq!(parsed.cached_tokens, Some(cached));
        assert_eq!(parsed.prompt_tokens, 1200);
        assert_eq!(parsed.total_tokens, 1500);
    }
    for details in [
        Value::Null,
        json!({}),
        json!("invalid"),
        json!({"cached_tokens": null}),
        json!({"cached_tokens": -1}),
        json!({"cached_tokens": 1.5}),
        json!({"cached_tokens": "800"}),
        json!({"cached_tokens": 1201}),
        json!({"cached_tokens": u64::MAX}),
    ] {
        let mut value = usage(1200, 300)["usage"].clone();
        value["prompt_tokens_details"] = details;
        let parsed = TokenUsage::from_value(&value).unwrap();
        assert_eq!(parsed.cached_tokens, None);
        assert_eq!(parsed.total_tokens, 1500);
    }
    assert_eq!(
        TokenUsage::from_value(&usage(1200, 300)["usage"])
            .unwrap()
            .cached_tokens,
        None
    );
    assert_eq!(
        TokenUsage::from_value(&cached_usage(u64::MAX, 0, u64::MAX)["usage"])
            .unwrap()
            .cached_tokens,
        Some(u64::MAX)
    );
}

#[test]
fn usage_parsing_rejects_unknown_values_and_keeps_api_total() {
    for value in [
        Value::Null,
        json!({}),
        json!({"prompt_tokens": 1}),
        json!({"prompt_tokens": -1, "completion_tokens": 2}),
        json!({"prompt_tokens": "1", "completion_tokens": 2}),
        json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": null}),
    ] {
        assert_eq!(TokenUsage::from_value(&value), None);
    }
    assert_eq!(
        TokenUsage::from_value(&json!({"prompt_tokens": 2, "completion_tokens": 3}))
            .unwrap()
            .total_tokens,
        5
    );
    assert_eq!(
        TokenUsage::from_value(
            &json!({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 7})
        )
        .unwrap()
        .total_tokens,
        7
    );
    assert_eq!(
        TokenUsage::from_value(
            &json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        )
        .unwrap()
        .total_tokens,
        0
    );
}
