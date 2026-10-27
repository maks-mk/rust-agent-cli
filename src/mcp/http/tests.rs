use super::*;
use crate::mcp::{server_transport, McpManager, McpServer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

struct Request {
    method: String,
    headers: HashMap<String, String>,
    body: Value,
}

struct Reply {
    status: &'static str,
    content_type: &'static str,
    session: Option<&'static str>,
    chunks: Vec<Vec<u8>>,
}

impl Reply {
    fn json(value: Value) -> Self {
        Self {
            status: "200 OK",
            content_type: "application/json; charset=utf-8",
            session: None,
            chunks: vec![value.to_string().into_bytes()],
        }
    }

    fn status(status: &'static str) -> Self {
        Self {
            status,
            content_type: "",
            session: None,
            chunks: Vec::new(),
        }
    }

    fn initialize(id: u64, session: Option<&'static str>) -> Self {
        let mut reply = Self::json(json!({ "jsonrpc": "2.0", "id": id, "result": {
            "protocolVersion": "2025-03-26", "capabilities": { "tools": {} },
            "serverInfo": { "name": "mock", "version": "1" }
        } }));
        reply.session = session;
        reply
    }

    fn sse(text: &str) -> Self {
        Self {
            status: "200 OK",
            content_type: "text/event-stream; charset=utf-8",
            session: None,
            chunks: text.as_bytes().chunks(7).map(<[u8]>::to_vec).collect(),
        }
    }
}

struct Mock {
    url: String,
    task: Option<JoinHandle<Vec<Request>>>,
}

impl Mock {
    async fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        Self::start_on(listener, replies)
    }

    fn start_on(listener: TcpListener, replies: Vec<Reply>) -> Self {
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                requests.push(read_request(&mut socket).await);
                let length: usize = reply.chunks.iter().map(Vec::len).sum();
                let mut headers = format!(
                    "HTTP/1.1 {}\r\nContent-Length: {length}\r\nConnection: close\r\n",
                    reply.status
                );
                if !reply.content_type.is_empty() {
                    headers.push_str(&format!("Content-Type: {}\r\n", reply.content_type));
                }
                if let Some(session) = reply.session {
                    headers.push_str(&format!("Mcp-Session-Id: {session}\r\n"));
                }
                headers.push_str("\r\n");
                socket.write_all(headers.as_bytes()).await.unwrap();
                for chunk in reply.chunks {
                    if socket.write_all(&chunk).await.is_err() {
                        break; // The client may stop reading as soon as its response arrives.
                    }
                    tokio::task::yield_now().await;
                }
                let _ = socket.shutdown().await;
            }
            requests
        });
        Self {
            url,
            task: Some(task),
        }
    }

    async fn finish(mut self) -> Vec<Request> {
        timeout(Duration::from_secs(5), self.task.take().unwrap())
            .await
            .unwrap()
            .unwrap()
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

async fn read_request(socket: &mut TcpStream) -> Request {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0; 1024];
        let count = socket.read(&mut chunk).await.unwrap();
        assert_ne!(count, 0, "client closed before request headers");
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = std::str::from_utf8(&bytes[..header_end]).unwrap();
    let method = head.split_whitespace().next().unwrap().to_string();
    let headers: HashMap<String, String> = head
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let length: usize = headers
        .get("content-length")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    while bytes.len() < header_end + length {
        let mut chunk = [0; 1024];
        let count = socket.read(&mut chunk).await.unwrap();
        assert_ne!(count, 0, "client closed before request body");
        bytes.extend_from_slice(&chunk[..count]);
    }
    let body = if length == 0 {
        Value::Null
    } else {
        serde_json::from_slice(&bytes[header_end..header_end + length]).unwrap()
    };
    Request {
        method,
        headers,
        body,
    }
}

struct TestConfig(PathBuf);

impl TestConfig {
    fn new(value: &Value) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::current_dir()
            .unwrap()
            .join("target")
            .join(format!(
                "mcp-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("mcp.json"), value.to_string()).unwrap();
        Self(root)
    }
}

impl Drop for TestConfig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn cancelled_loading_retries_only_unfinished_servers() {
    for keep_first in [false, true] {
        let tools_reply = || {
            Reply::json(
                json!({"jsonrpc":"2.0", "id":2, "result":{"tools":[{"name":"tool", "inputSchema":{"type":"object"}}]}}),
            )
        };
        let first = Mock::start(vec![
            Reply::initialize(1, None),
            Reply::status("202 Accepted"),
            tools_reply(),
        ])
        .await;
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let blocked_url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let mut config =
            json!({"b":{"transport":"http", "url":blocked_url, "policy":{"read_only":true}}});
        if keep_first {
            config["a"] = json!({"transport":"http", "url":first.url, "policy":{"read_only":true}});
        }
        let dir = TestConfig::new(&config);
        let mut manager = McpManager::default();
        {
            let loading = manager.ensure_loaded(&dir.0, &dir.0);
            tokio::pin!(loading);
            tokio::select! {
                _ = &mut loading => panic!("loading did not wait for the blocked server"),
                request = async {
                    let (mut socket, _) = listener.accept().await.unwrap();
                    read_request(&mut socket).await
                } => assert_eq!(request.body["method"], "initialize"),
                _ = tokio::time::sleep(Duration::from_secs(5)) => panic!("no initialize request"),
            }
            // Drop the in-flight load just as aborting an agent task would.
        }
        assert!(!manager.loaded);
        assert_eq!(manager.servers.len(), usize::from(keep_first));
        if keep_first {
            assert_eq!(first.finish().await.len(), 3);
        }
        let second = Mock::start_on(
            listener,
            vec![
                Reply::initialize(1, None),
                Reply::status("202 Accepted"),
                tools_reply(),
            ],
        );
        timeout(
            Duration::from_secs(5),
            manager.ensure_loaded(&dir.0, &dir.0),
        )
        .await
        .unwrap();
        assert!(manager.loaded);
        assert_eq!(manager.servers.len(), 1 + usize::from(keep_first));
        assert_eq!(
            manager.openai_tool_schemas().len(),
            1 + usize::from(keep_first)
        );
        assert!(manager.has_tool("mcp_b_tool"));
        if keep_first {
            assert!(manager.has_tool("mcp_a_tool"));
        }
        assert!(manager.take_warnings().is_empty());
        assert_eq!(second.finish().await.len(), 3);
        manager.ensure_loaded(&dir.0, &dir.0).await;
        manager.shutdown().await;
    }
}

#[test]
fn sse_handles_split_utf8_crlf_and_multiline_events() {
    let text = "\u{feff}: keep-alive\r\nid: 9\r\nevent: message\r\ndata: {\"jsonrpc\":\"2.0\",\r\ndata: \"id\":7,\"result\":{\"text\":\"Привет\"}}\r\n\r\n";
    let mut decoder = SseDecoder::default();
    let mut messages = Vec::new();
    for byte in text.as_bytes() {
        decoder.feed(&[*byte]).unwrap();
        while let Some(message) = decoder.next_message().unwrap() {
            messages.push(message);
        }
    }
    assert_eq!(
        messages,
        vec![json!({ "jsonrpc": "2.0", "id": 7, "result": { "text": "Привет" } })]
    );
}

#[test]
fn sse_handles_lf_cr_comments_and_multiple_events() {
    let mut decoder = SseDecoder::default();
    decoder.feed(b": comment\n\nid: priming-event\ndata:\n\ndata:{\"id\":1}\n\ndata: {\"id\":2}\r\rX\r\n").unwrap();
    assert_eq!(decoder.next_message().unwrap(), Some(json!({ "id": 1 })));
    assert_eq!(decoder.next_message().unwrap(), Some(json!({ "id": 2 })));
    assert!(decoder.next_message().unwrap().is_none());
}

#[test]
fn sse_rejects_invalid_and_oversized_messages() {
    for bytes in [b"data: nope\n\n".as_slice(), b"data: \xff\n\n".as_slice()] {
        let mut decoder = SseDecoder::default();
        decoder.feed(bytes).unwrap();
        assert!(decoder.next_message().is_err());
    }
    let mut decoder = SseDecoder::default();
    assert!(decoder.feed(&vec![b'x'; MAX_MESSAGE_BYTES + 1]).is_err());
}

#[test]
fn transport_defaults_and_aliases() {
    assert_eq!(server_transport(&json!({ "command": "server" })), "stdio");
    assert_eq!(server_transport(&json!({ "type": "remote" })), "http");
    assert_eq!(
        server_transport(&json!({ "url": "https://example.com/mcp" })),
        "http"
    );
    assert_eq!(
        server_transport(&json!({ "url": "https://example.com/mcp", "command": "server" })),
        "stdio"
    );
    assert_eq!(
        server_transport(&json!({ "type": "remote", "transport": "stdio" })),
        "stdio"
    );
}

#[tokio::test]
async fn rejects_invalid_configuration_without_exposing_values() {
    for config in [
        json!({}),
        json!({ "url": "invalid-secret" }),
        json!({ "url": "file:///secret" }),
        json!({ "url": "https://user:secret@example.com/mcp" }),
        json!({ "url": "http://127.0.0.1:1/mcp", "headers": [] }),
        json!({ "url": "http://127.0.0.1:1/mcp", "headers": { "bad name": "secret" } }),
        json!({ "url": "http://127.0.0.1:1/mcp", "headers": { "Authorization": "secret\n" } }),
        json!({ "url": "http://127.0.0.1:1/mcp", "headers": { "Authorization": 123 } }),
        json!({ "url": "http://127.0.0.1:1/mcp", "headers": { "MCP-Session-Id": "secret" } }),
    ] {
        let error = HttpServer::connect(&config)
            .await
            .err()
            .unwrap()
            .to_string();
        assert!(
            !error.contains("secret"),
            "configuration values leaked: {error}"
        );
    }
    assert!(McpServer::connect(
        "test",
        &json!({ "transport": "sse" }),
        std::path::Path::new(".")
    )
    .await
    .is_err());
}

#[tokio::test]
async fn manager_supports_flat_and_wrapped_json_pagination_and_calls() {
    for wrapped in [false, true] {
        let mock = Mock::start(vec![
            Reply::initialize(1, Some("session-one")), Reply::status("202 Accepted"),
            Reply::json(json!({ "jsonrpc": "2.0", "id": 2, "result": {
                "tools": [{ "name": "echo", "description": "Echo", "inputSchema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}
                } }], "nextCursor": "page-two"
            } })),
            Reply::json(json!({ "jsonrpc": "2.0", "id": 3, "result": {
                "tools": [{ "name": "other", "inputSchema": [] }]
            } })),
            Reply::json(json!({ "jsonrpc": "2.0", "id": 4, "result": {
                "content": [{ "type": "text", "text": "works" }]
            } })), Reply::status("405 Method Not Allowed"),
        ]).await;
        let entries = json!({ "context7": { "type": "remote", "url": mock.url, "transport": "http",
            "headers": { "CONTEXT7_API_KEY": "test-only-key" }, "policy": { "read_only": true } },
            "disabled": { "enabled": false, "transport": "unsupported" }
        });
        let config = TestConfig::new(&if wrapped {
            json!({ "mcpServers": entries })
        } else {
            entries
        });
        let mut manager = McpManager::default();
        manager.ensure_loaded(&config.0, &config.0).await;
        manager.ensure_loaded(&config.0, &config.0).await;
        assert!(manager.take_warnings().is_empty());
        assert!(manager.has_tool("mcp_context7_echo"));
        assert!(manager.is_read_only("mcp_context7_echo"));
        assert!(!manager.is_read_only("missing_tool"));
        let schemas = manager.openai_tool_schemas();
        assert_eq!(schemas.len(), 2);
        assert!(schemas[0]["function"]["parameters"]
            .get("$schema")
            .is_none());
        assert_eq!(schemas[1]["function"]["parameters"]["type"], "object");
        assert_eq!(
            manager
                .call_tool("mcp_context7_echo", json!({ "text": "hello" }))
                .await
                .unwrap(),
            "works"
        );
        manager.shutdown().await;
        assert!(!manager.has_tool("mcp_context7_echo"));
        let requests = mock.finish().await;
        assert_eq!(requests.len(), 6);
        assert_eq!(
            requests[0].body["params"]["protocolVersion"],
            HTTP_PROTOCOL_VERSION
        );
        assert!(!requests[0].headers.contains_key("mcp-session-id"));
        assert!(!requests[0].headers.contains_key("mcp-protocol-version"));
        assert_eq!(requests[1].body["method"], "notifications/initialized");
        assert!(requests[1].body.get("id").is_none());
        assert_eq!(requests[3].body["params"]["cursor"], "page-two");
        assert_eq!(
            requests[4].body["params"],
            json!({ "name": "echo", "arguments": { "text": "hello" } })
        );
        assert_eq!(requests[5].method, "DELETE");
        for request in &requests {
            assert_eq!(request.headers["context7_api_key"], "test-only-key");
            assert_eq!(
                request.headers["accept"],
                "application/json, text/event-stream"
            );
        }
        for request in &requests[1..] {
            assert_eq!(request.headers["mcp-session-id"], "session-one");
            assert_eq!(request.headers["mcp-protocol-version"], "2025-03-26");
        }
    }
}

#[tokio::test]
async fn sse_initialization_notifications_server_requests_and_matching_id() {
    let init = Reply::initialize(1, Some("stream-session"));
    let mut init_sse = Reply::sse(&format!(
        "event: message\r\ndata: {}\r\n\r\n",
        String::from_utf8(init.chunks.concat()).unwrap()
    ));
    init_sse.session = init.session;
    let mock = Mock::start(vec![init_sse, Reply::status("202 Accepted"), Reply::sse(
        ": keepalive\n\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n\
         data: {\"jsonrpc\":\"2.0\",\"id\":\"ping-id\",\"method\":\"ping\"}\n\n\
         data: {\"jsonrpc\":\"2.0\",\"id\":\"unknown-id\",\"method\":\"sampling/createMessage\"}\n\n\
         data: {\"jsonrpc\":\"2.0\",\"id\":999,\"result\":{}}\n\n\
         data: {\"jsonrpc\":\"2.0\",\n\
         data: \"id\":2,\"result\":{\"text\":\"Привет\"}}\n\n\
         data: ignored-invalid-trailing-event\n\n"
    ), Reply::status("202 Accepted"), Reply::status("202 Accepted"), Reply::status("200 OK")]).await;
    let mut server = HttpServer::connect(&json!({ "url": mock.url }))
        .await
        .unwrap();
    assert_eq!(
        server.rpc("tools/call", json!({})).await.unwrap(),
        json!({ "text": "Привет" })
    );
    server.close().await.unwrap();
    let requests = mock.finish().await;
    assert_eq!(
        requests[3].body,
        json!({ "jsonrpc": "2.0", "id": "ping-id", "result": {} })
    );
    assert_eq!(requests[4].body["error"]["code"], -32601);
    assert_eq!(requests[4].body["id"], "unknown-id");
    for request in &requests[1..] {
        assert_eq!(request.headers["mcp-session-id"], "stream-session");
    }
}

#[tokio::test]
async fn expired_session_reinitializes_and_retries_once() {
    for retry_fails in [false, true] {
        let retry = if retry_fails {
            Reply::status("404 Not Found")
        } else {
            Reply::json(json!({ "jsonrpc": "2.0", "id": 4, "result": { "ok": true } }))
        };
        let mock = Mock::start(vec![
            Reply::initialize(1, Some("old-session")),
            Reply::status("202 Accepted"),
            Reply::status("404 Not Found"),
            Reply::initialize(3, Some("new-session")),
            Reply::status("202 Accepted"),
            retry,
            Reply::status("200 OK"),
        ])
        .await;
        let mut server = HttpServer::connect(&json!({ "url": mock.url }))
            .await
            .unwrap();
        let result = server
            .rpc("tools/call", json!({ "name": "echo", "arguments": {} }))
            .await;
        if retry_fails {
            assert!(result.is_err());
        } else {
            assert_eq!(result.unwrap(), json!({ "ok": true }));
        }
        server.close().await.unwrap();
        let requests = mock.finish().await;
        assert_eq!(requests.len(), 7);
        assert_eq!(requests[2].headers["mcp-session-id"], "old-session");
        assert_eq!(requests[3].body["method"], "initialize");
        assert!(!requests[3].headers.contains_key("mcp-session-id"));
        assert!(!requests[3].headers.contains_key("mcp-protocol-version"));
        assert_eq!(requests[5].headers["mcp-session-id"], "new-session");
        assert_eq!(requests[2].body["params"], requests[5].body["params"]);
        assert_ne!(requests[2].body["id"], requests[5].body["id"]);
    }
}

#[tokio::test]
async fn failed_server_reply_does_not_repeat_an_accepted_tool_call() {
    let mock = Mock::start(vec![
        Reply::initialize(1, Some("session-one")),
        Reply::status("202 Accepted"),
        Reply::sse("data: {\"jsonrpc\":\"2.0\",\"id\":\"ping-id\",\"method\":\"ping\"}\n\n"),
        Reply::status("404 Not Found"),
        Reply::status("200 OK"),
    ])
    .await;
    let mut server = HttpServer::connect(&json!({ "url": mock.url }))
        .await
        .unwrap();
    let error = server
        .rpc("tools/call", json!({ "name": "echo" }))
        .await
        .unwrap_err();
    assert!(!error.is::<SessionExpired>());
    assert_eq!(
        error.downcast_ref::<reqwest::Error>().unwrap().status(),
        Some(StatusCode::NOT_FOUND)
    );
    server.close().await.unwrap();
    let requests = mock.finish().await;
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[2].body["method"], "tools/call");
    assert_eq!(requests[3].body["id"], "ping-id");
    assert_eq!(requests[4].method, "DELETE");
}

#[tokio::test]
async fn rejects_bad_responses_without_retrying_stateless_requests() {
    let replies = vec![
        (Reply::status("404 Not Found"), "404"),
        (Reply::status("500 Internal Server Error"), "500"),
        (Reply::status("302 Found"), "redirected"),
        (Reply::status("202 Accepted"), "no JSON/SSE"),
        (
            Reply::json(json!({ "jsonrpc": "2.0", "id": 99, "result": {} })),
            "request ID",
        ),
        (
            Reply::json(json!({ "jsonrpc": "1.0", "id": 2, "result": {} })),
            "JSON-RPC",
        ),
        (
            Reply::json(json!({ "jsonrpc": "2.0", "id": 2 })),
            "result or error",
        ),
        (
            Reply::json(
                json!({ "jsonrpc": "2.0", "id": 2, "error": { "code": -32602, "message": "bad arguments" } }),
            ),
            "-32602",
        ),
        (
            Reply::sse("data: {\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}\n\n"),
            "closed",
        ),
        (Reply::sse("data: invalid-json\n\n"), "invalid JSON"),
    ];
    for (reply, expected) in replies {
        let mock = Mock::start(vec![
            Reply::initialize(1, None),
            Reply::status("202 Accepted"),
            reply,
        ])
        .await;
        let mut server =
            HttpServer::connect(&json!({ "url": format!("{}?token=do-not-log", mock.url) }))
                .await
                .unwrap();
        let error = server
            .rpc("tools/list", json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
        assert!(!error.contains("do-not-log"));
        server.close().await.unwrap();
        assert_eq!(mock.finish().await.len(), 3);
    }
}

#[tokio::test]
async fn streamable_http_aliases_connect() {
    for transport in ["streamable-http", "streamable_http"] {
        let mock = Mock::start(vec![
            Reply::initialize(1, None),
            Reply::status("202 Accepted"),
        ])
        .await;
        let mut server = McpServer::connect(
            "test",
            &json!({ "url": mock.url, "transport": transport }),
            std::path::Path::new("."),
        )
        .await
        .unwrap();
        server.close().await;
        assert_eq!(mock.finish().await.len(), 2);
    }
}

#[tokio::test]
async fn failed_initialization_closes_assigned_sessions() {
    for notification_fails in [false, true] {
        let mut init = Reply::initialize(1, Some("rejected-session"));
        let mut replies = if notification_fails {
            vec![init, Reply::status("400 Bad Request")]
        } else {
            init.chunks = vec![json!({ "jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersion": "unsupported"
            } })
            .to_string()
            .into_bytes()];
            vec![init]
        };
        replies.push(Reply::status("200 OK"));
        let mock = Mock::start(replies).await;
        assert!(HttpServer::connect(&json!({ "url": mock.url }))
            .await
            .is_err());
        let requests = mock.finish().await;
        let last = requests.last().unwrap();
        assert_eq!(last.method, "DELETE");
        assert_eq!(last.headers["mcp-session-id"], "rejected-session");
    }
}

#[tokio::test]
#[ignore = "requires internet access to the real Context7 MCP endpoint"]
async fn context7_live() {
    let example: Value = serde_json::from_str(include_str!("../../../mcp.json.example")).unwrap();
    let config = example
        .get("context7")
        .expect("Context7 example is missing");
    let mut server = McpServer::connect("context7", config, std::path::Path::new("."))
        .await
        .unwrap();
    let tools = server.list_tools("context7").await.unwrap();
    assert!(!tools.is_empty());
    let (name, _, _) = tools
        .iter()
        .find(|(name, _, _)| name.contains("resolve-library-id"))
        .expect("Context7 resolver tool is missing");
    let result = server
        .rpc(
            "tools/call",
            json!({ "name": name, "arguments": {
        "libraryName": "tokio", "query": "Rust Tokio TCP listener example"
    } }),
        )
        .await
        .unwrap();
    assert_ne!(
        result.get("isError").and_then(Value::as_bool),
        Some(true),
        "Context7 tool failed"
    );
    assert!(result
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|content| !content.is_empty()));
    server.close().await;
    println!(
        "Context7: discovered {} tools; resolver call succeeded",
        tools.len()
    );
}

#[test]
fn mcp_config_prefers_the_executable_directory() {
    let config = TestConfig::new(&json!({ "server": {} }));
    let empty = TestConfig::new(&json!({}));
    std::fs::remove_file(empty.0.join("mcp.json")).unwrap();
    assert_eq!(
        crate::mcp::config_path(&config.0, &empty.0),
        config.0.join("mcp.json")
    );
    assert_eq!(
        crate::mcp::config_path(&empty.0, &config.0),
        config.0.join("mcp.json")
    );
}
