//! MCP client for stdio and Streamable HTTP servers.
//!
//! Implements initialization, tools/list (including cursors), and tools/call.

mod http;

#[cfg(test)]
mod tests;

use crate::agent::{AgentEvent, Approval, Tx};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

const MCP_PROTOCOL_VERSION: &str = "2025-03-26";
const MCP_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_TOOL_RESULT_CHARS: usize = 50_000;
const MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;
const STDERR_TAIL_LINES: usize = 8;
const STDERR_TAIL_BYTES: usize = 4_000;
const STDERR_LINE_CHARS: usize = 400;
const STDERR_SUMMARY_CHARS: usize = 300;
const STDERR_DRAIN_TIMEOUT: Duration = Duration::from_secs(1);

async fn read_stdio_line(reader: &mut (impl AsyncBufRead + Unpin)) -> Result<String> {
    let mut line = String::new();
    reader
        .take((MAX_MESSAGE_BYTES + 1) as u64)
        .read_line(&mut line)
        .await?;
    if line.len() > MAX_MESSAGE_BYTES {
        bail!("stdio MCP message exceeds {MAX_MESSAGE_BYTES} bytes");
    }
    Ok(line)
}

#[derive(Default)]
struct StderrBuffer {
    lines: VecDeque<String>,
    bytes: usize,
}

/// Keeps the tail of a stdio server's stderr so a failed start can explain itself.
#[derive(Clone, Default)]
struct StderrTail {
    buffer: Arc<Mutex<StderrBuffer>>,
}

impl StderrTail {
    fn push(&self, line: &[u8]) {
        let text: String =
            crate::process_output::terminal_safe_text(&String::from_utf8_lossy(line))
                .trim()
                .chars()
                .take(STDERR_LINE_CHARS)
                .collect();
        if text.is_empty() {
            return;
        }
        let Ok(mut buffer) = self.buffer.lock() else {
            return;
        };
        buffer.bytes += text.len() + 1;
        buffer.lines.push_back(text);
        while buffer.lines.len() > STDERR_TAIL_LINES || buffer.bytes > STDERR_TAIL_BYTES {
            let Some(dropped) = buffer.lines.pop_front() else {
                break;
            };
            buffer.bytes = buffer.bytes.saturating_sub(dropped.len() + 1);
        }
    }

    /// One compact line: the newest non-empty messages, oldest first.
    fn summary(&self) -> Option<String> {
        let buffer = self.buffer.lock().ok()?;
        let mut parts: Vec<&str> = Vec::new();
        let mut total = 0;
        for line in buffer.lines.iter().rev() {
            let extra = line.len() + if parts.is_empty() { 0 } else { 3 };
            if !parts.is_empty() && total + extra > STDERR_SUMMARY_CHARS {
                break;
            }
            total += extra;
            parts.push(line);
        }
        parts.reverse();
        let joined = parts.join(" | ");
        (!joined.is_empty()).then(|| joined.chars().take(STDERR_SUMMARY_CHARS).collect())
    }
}

fn drain_stderr(mut stderr: ChildStderr, tail: StderrTail) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut chunk = [0u8; 4096];
        let mut pending: Vec<u8> = Vec::new();
        loop {
            match stderr.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    for &byte in &chunk[..read] {
                        if byte == b'\n' {
                            tail.push(&pending);
                            pending.clear();
                        } else if pending.len() < STDERR_LINE_CHARS * 4 {
                            pending.push(byte);
                        }
                    }
                }
            }
        }
        tail.push(&pending);
    })
}

#[derive(Clone)]
struct ToolBinding {
    api_name: String,
    server_key: String,
    server_index: usize,
    original_name: String,
    description: String,
    input_schema: Value,
    read_only: bool,
}

struct StdioServer {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    usable: bool,
    request_timeout: Duration,
}

/// Dropping an in-flight operation must never leave a partial frame reusable.
struct StdioOperation<'a> {
    child: &'a mut Child,
    usable: &'a mut bool,
    complete: bool,
}

impl Drop for StdioOperation<'_> {
    fn drop(&mut self) {
        if !self.complete {
            *self.usable = false;
            let _ = self.child.start_kill();
        }
    }
}

impl StdioServer {
    async fn spawn(key: &str, config: &Value, root: &Path) -> Result<Self> {
        let command = config
            .get("command")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| anyhow!("MCP server '{key}' has no command"))?;

        let mut cmd = Command::new(command);
        if let Some(args) = config.get("args").and_then(Value::as_array) {
            for arg in args {
                if let Some(s) = arg.as_str() {
                    cmd.arg(s);
                } else {
                    bail!("MCP server '{key}' has a non-string argument");
                }
            }
        }

        if let Some(env) = config.get("env").and_then(Value::as_object) {
            for (name, value) in env {
                if let Some(value) = value.as_str() {
                    cmd.env(name, value);
                }
            }
        }

        let cwd = config
            .get("cwd")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .map(|p| if p.is_absolute() { p } else { root.join(p) })
            .unwrap_or_else(|| root.to_path_buf());
        cmd.current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // MCP uses stdout for protocol messages, so server logs are drained
            // into a small tail instead of reaching the full-screen terminal UI.
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        crate::process_output::apply_output_env(&mut cmd);
        #[cfg(windows)]
        crate::process_output::apply_windows_flags(&mut cmd, 0);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("could not start MCP server '{key}' ({command})"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP server '{key}' has no stdin pipe"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP server '{key}' has no stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("MCP server '{key}' has no stderr pipe"))?;
        let stderr_tail = StderrTail::default();
        let stderr_drain = drain_stderr(stderr, stderr_tail.clone());

        let mut server = Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
            usable: true,
            request_timeout: MCP_REQUEST_TIMEOUT,
        };

        let initialized = server
            .rpc(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "rust-agent", "version": env!("CARGO_PKG_VERSION") }
                }),
            )
            .await
            .with_context(|| format!("MCP initialization failed for '{key}'"));
        if let Err(error) = initialized {
            // A server that dies before answering usually says why on stderr, but
            // that text never reaches the terminal UI on its own.
            let _ = timeout(STDERR_DRAIN_TIMEOUT, stderr_drain).await;
            return Err(match stderr_tail.summary() {
                Some(detail) => anyhow!("{error:#} (server stderr: {detail})"),
                None => error,
            });
        }
        server
            .notify("notifications/initialized", json!({}))
            .await
            .with_context(|| format!("MCP initialized notification failed for '{key}'"))?;

        Ok(server)
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let mut message = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }))?;
        message.push(b'\n');
        if !self.usable {
            bail!("MCP connection interrupted; restart the session to reconnect (request not retried)");
        }
        let mut operation = StdioOperation {
            child: &mut self._child,
            usable: &mut self.usable,
            complete: false,
        };
        let result = timeout(self.request_timeout, async {
            self.stdin.write_all(&message).await?;
            self.stdin.flush().await?;
            Ok(())
        })
        .await
        .map_err(|_| anyhow!("MCP notification '{method}' timed out"))?;
        operation.complete = result.is_ok();
        result
    }

    async fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        if !self.usable {
            bail!("MCP connection interrupted; restart the session to reconnect (request not retried)");
        }
        self.next_id += 1;
        let request_id = self.next_id;
        let id = json!(request_id);
        let mut message = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": id.clone(),
            "method": method,
            "params": params
        }))?;
        message.push(b'\n');
        let mut operation = StdioOperation {
            child: &mut self._child,
            usable: &mut self.usable,
            complete: false,
        };
        let result = timeout(self.request_timeout, async {
            self.stdin.write_all(&message).await?;
            self.stdin.flush().await?;
            loop {
                let line = read_stdio_line(&mut self.stdout).await?;
                if line.is_empty() {
                    bail!("MCP server closed stdout while waiting for '{method}'");
                }
                if line.trim().is_empty() {
                    continue;
                }
                let response: Value =
                    serde_json::from_str(&line).context("MCP server returned invalid JSON")?;

                if response.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
                    bail!("invalid MCP JSON-RPC message");
                }
                if response.get("method").is_some() {
                    if let Some(reply) = server_request_reply(&response) {
                        let mut message = serde_json::to_vec(&reply)?;
                        message.push(b'\n');
                        self.stdin.write_all(&message).await?;
                        self.stdin.flush().await?;
                    }
                    continue;
                }
                if response.get("id") != Some(&id) {
                    continue;
                }
                // A complete matching error response also preserves framing.
                operation.complete = true;
                return rpc_result(&response);
            }
        })
        .await
        .map_err(|_| {
            anyhow!("MCP request '{method}' timed out; connection closed, request not retried")
        })?;
        result
    }
}

enum McpServer {
    Stdio(StdioServer),
    Http(http::HttpServer),
}

impl McpServer {
    async fn connect(key: &str, config: &Value, root: &Path) -> Result<Self> {
        match server_transport(config) {
            "stdio" => Ok(Self::Stdio(StdioServer::spawn(key, config, root).await?)),
            "http" | "streamable-http" | "streamable_http" => {
                Ok(Self::Http(http::HttpServer::connect(config).await?))
            }
            transport => bail!("unsupported MCP transport '{transport}' (use stdio or http)"),
        }
    }

    async fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            Self::Stdio(server) => server.rpc(method, params).await,
            Self::Http(server) => server.rpc(method, params).await,
        }
    }

    async fn close(&mut self) {
        if let Self::Http(server) = self {
            // Best effort: session cleanup must not prevent the application exiting.
            let _ = server.close().await;
        }
    }

    async fn list_tools(&mut self, server_key: &str) -> Result<Vec<(String, String, Value)>> {
        let mut output = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self
                .rpc("tools/list", params)
                .await
                .with_context(|| format!("could not list tools from MCP server '{server_key}'"))?;

            if let Some(tools) = result.get("tools").and_then(Value::as_array) {
                for tool in tools {
                    let Some(name) = tool.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let description = tool
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("MCP tool")
                        .to_string();
                    let mut schema = tool
                        .get("inputSchema")
                        .cloned()
                        .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
                    if !schema.is_object() {
                        schema = json!({ "type": "object", "properties": {} });
                    }
                    if let Some(object) = schema.as_object_mut() {
                        // Some OpenAI-compatible gateways reject this JSON Schema metadata.
                        object.remove("$schema");
                    }
                    output.push((name.to_string(), description, schema));
                }
            }

            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        Ok(output)
    }
}

/// Picks `mcp.json` from the config directory, falling back to the workspace.
fn config_path(config_dir: &Path, root: &Path) -> PathBuf {
    let next_to_exe = config_dir.join("mcp.json");
    if next_to_exe.is_file() {
        next_to_exe
    } else {
        root.join("mcp.json")
    }
}

#[derive(Default)]
pub struct McpManager {
    loaded: bool,
    processed_servers: HashSet<String>,
    servers: Vec<McpServer>,
    tools: Vec<ToolBinding>,
    warnings: Vec<String>,
    snapshot: Option<(PathBuf, String, PathBuf)>,
    decisions: HashMap<String, bool>,
}

impl McpManager {
    /// Reuse sessions only while the exact configuration remains unchanged.
    pub async fn ensure_loaded_with_approval(&mut self, config_dir: &Path, root: &Path, tx: &Tx) {
        self.load_pending(config_dir, root, tx).await;
    }

    #[cfg(test)]
    async fn ensure_loaded(&mut self, config_dir: &Path, root: &Path) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(rx);
        self.load_pending(config_dir, root, &tx).await;
    }

    async fn load_pending(&mut self, config_dir: &Path, root: &Path, tx: &Tx) {
        let path = config_path(config_dir, root);
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                self.shutdown().await;
                return;
            }
            Err(e) => {
                self.shutdown().await;
                self.warnings
                    .push(format!("could not read {}: {e}", path.display()));
                return;
            }
        };
        let snapshot = (path.clone(), text.clone(), root.to_path_buf());
        if self.snapshot.as_ref() != Some(&snapshot) {
            self.shutdown().await;
            self.snapshot = Some(snapshot);
        } else if self.loaded {
            return;
        }
        let config: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                self.warnings
                    .push(format!("invalid {}: {e}", path.display()));
                return;
            }
        };

        // Accept both this project's flat format and the common {"mcpServers":{...}} shape.
        let entries = config
            .get("mcpServers")
            .and_then(Value::as_object)
            .or_else(|| config.as_object());
        let Some(entries) = entries else {
            self.warnings
                .push("mcp.json must contain an object of server configurations".into());
            return;
        };

        let mut used_names = self
            .tools
            .iter()
            .map(|tool| tool.api_name.clone())
            .collect();
        for (key, entry) in entries {
            if self.processed_servers.contains(key) || key.starts_with('_') || !entry.is_object() {
                continue;
            }
            if !entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true)
            {
                continue;
            }

            let read_only = policy_read_only(entry);
            if !read_only {
                let allowed = if let Some(allowed) = self.decisions.get(key) {
                    *allowed
                } else {
                    let (reply, response) = tokio::sync::oneshot::channel();
                    let _ = tx.send(AgentEvent::McpApproval {
                        prompt: server_approval_prompt(&path, root, key, entry),
                        reply,
                    });
                    let Ok(answer) = response.await else {
                        return;
                    };
                    let allowed = matches!(answer, Approval::Yes);
                    self.decisions.insert(key.clone(), allowed);
                    allowed
                };
                if !allowed {
                    self.processed_servers.insert(key.clone());
                    self.warnings.push(format!(
                        "server '{key}' skipped by user; edit mcp.json or restart to reconsider"
                    ));
                    continue;
                }
            }

            let mut server = match McpServer::connect(key, entry, root).await {
                Ok(server) => server,
                Err(e) => {
                    self.processed_servers.insert(key.clone());
                    self.warnings
                        .push(format!("server '{key}' failed to start: {e:#}"));
                    continue;
                }
            };
            let definitions = match server.list_tools(key).await {
                Ok(tools) => tools,
                Err(e) => {
                    server.close().await;
                    self.processed_servers.insert(key.clone());
                    self.warnings
                        .push(format!("server '{key}' tool discovery failed: {e:#}"));
                    continue;
                }
            };

            self.processed_servers.insert(key.clone());
            let server_index = self.servers.len();
            self.servers.push(server);
            for (original_name, description, input_schema) in definitions {
                let api_name = unique_api_name(key, &original_name, &mut used_names);
                self.tools.push(ToolBinding {
                    api_name,
                    server_key: key.clone(),
                    server_index,
                    original_name,
                    description,
                    input_schema,
                    read_only,
                });
            }
        }
        self.loaded = true;
    }

    pub async fn shutdown(&mut self) {
        futures_util::future::join_all(self.servers.iter_mut().map(McpServer::close)).await;
        self.servers.clear();
        self.tools.clear();
        self.processed_servers.clear();
        self.loaded = false;
        self.snapshot = None;
        self.decisions.clear();
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    pub fn is_read_only(&self, api_name: &str) -> bool {
        self.tools
            .iter()
            .any(|tool| tool.api_name == api_name && tool.read_only)
    }

    pub fn has_tool(&self, api_name: &str) -> bool {
        self.tools.iter().any(|t| t.api_name == api_name)
    }

    pub fn openai_tool_schemas(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.api_name.clone(),
                        "description": format!("[MCP server: {}] {}", tool.server_key, tool.description),
                        "parameters": tool.input_schema.clone()
                    }
                })
            })
            .collect()
    }

    pub async fn call_tool(&mut self, api_name: &str, args: Value) -> Result<String> {
        let binding = self
            .tools
            .iter()
            .find(|t| t.api_name == api_name)
            .cloned()
            .ok_or_else(|| anyhow!("unknown MCP tool '{api_name}'"))?;
        let server = self
            .servers
            .get_mut(binding.server_index)
            .ok_or_else(|| anyhow!("MCP server '{}' is not connected", binding.server_key))?;
        let result = server
            .rpc(
                "tools/call",
                json!({ "name": binding.original_name, "arguments": args }),
            )
            .await
            .with_context(|| format!("MCP tool '{}' failed", binding.original_name))?;
        let content = mcp_result_text(&result);
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            bail!("MCP tool error: {content}");
        }
        Ok(content)
    }
}

fn policy_read_only(config: &Value) -> bool {
    config
        .get("policy")
        .and_then(|policy| policy.get("read_only"))
        .and_then(Value::as_bool)
        == Some(true)
}

fn server_approval_prompt(path: &Path, root: &Path, key: &str, config: &Value) -> String {
    let cwd = config
        .get("cwd")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .map(|cwd| {
            if cwd.is_absolute() {
                cwd
            } else {
                root.join(cwd)
            }
        })
        .unwrap_or_else(|| root.to_path_buf());
    let details = if server_transport(config) == "stdio" {
        format!(
            "Command: {}\nArguments: {}\nWorking directory: {}\nEnvironment values are hidden.",
            config.get("command").unwrap_or(&Value::Null),
            config.get("args").unwrap_or(&Value::Null),
            cwd.display()
        )
    } else {
        // Never display URL credentials, query tokens or authorization headers.
        let endpoint = config
            .get("url")
            .and_then(Value::as_str)
            .and_then(|url| reqwest::Url::parse(url).ok())
            .map(|url| url.origin().ascii_serialization())
            .unwrap_or_else(|| "(invalid URL)".into());
        format!("Endpoint origin: {endpoint}\nURL credentials and headers are hidden.")
    };
    format!("Trust MCP server {key} for this session?\nConfiguration: {}\n{details}\nThis server may execute code or access data with your permissions.\nOnly policy.read_only: true bypasses this prompt; it is a declaration, not a sandbox.\nYes: allow startup for this configuration in this session. No: continue without this server.", path.display())
}

fn server_transport(config: &Value) -> &str {
    config
        .get("transport")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            if config.get("type").and_then(Value::as_str) == Some("remote")
                || (config.get("url").is_some() && config.get("command").is_none())
            {
                "http"
            } else {
                "stdio"
            }
        })
}

fn server_request_reply(message: &Value) -> Option<Value> {
    let method = message.get("method")?.as_str()?;
    let id = message.get("id").filter(|id| !id.is_null())?;
    Some(if method == "ping" {
        json!({ "jsonrpc": "2.0", "id": id, "result": {} })
    } else {
        json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "Method not supported" } })
    })
}

fn rpc_result(response: &Value) -> Result<Value> {
    if response.get("result").is_some() == response.get("error").is_some() {
        bail!("invalid MCP response: expected result or error");
    }
    if let Some(error) = response.get("error") {
        let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown JSON-RPC error");
        bail!("MCP JSON-RPC error {code}: {message}");
    }
    Ok(response.get("result").cloned().unwrap_or(Value::Null))
}

fn unique_api_name(server_key: &str, original_name: &str, used: &mut HashSet<String>) -> String {
    let raw = format!("mcp_{server_key}_{original_name}");
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let base: String = sanitized.chars().take(64).collect();
    if used.insert(base.clone()) {
        return base;
    }
    for suffix_num in 2u32.. {
        let suffix = format!("_{suffix_num}");
        let prefix: String = base.chars().take(64 - suffix.len()).collect();
        let candidate = format!("{prefix}{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("unique MCP tool name sequence exhausted")
}

fn mcp_result_text(result: &Value) -> String {
    let mut chunks = Vec::new();
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        chunks.push(text.to_string());
                    }
                }
                Some("image") => {
                    let mime = block
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    chunks.push(format!(
                        "[image result omitted from text-only tool channel; MIME: {mime}]"
                    ));
                }
                _ => chunks.push(block.to_string()),
            }
        }
    }
    if let Some(structured) = result.get("structuredContent") {
        chunks.push(format!("Structured content: {structured}"));
    }
    if chunks.is_empty() {
        chunks.push(result.to_string());
    }
    let text = crate::process_output::terminal_safe_text(&chunks.join("\n"));
    if text.chars().count() > MAX_TOOL_RESULT_CHARS {
        format!(
            "{}\n[output truncated]",
            text.chars().take(MAX_TOOL_RESULT_CHARS).collect::<String>()
        )
    } else {
        text
    }
}
