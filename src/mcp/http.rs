//! MCP Streamable HTTP transport: POST requests with JSON or SSE responses.

use super::{rpc_result, MAX_MESSAGE_BYTES, MCP_REQUEST_TIMEOUT};
use anyhow::{anyhow, bail, Context, Result};
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use reqwest::{Client, Response, StatusCode, Url};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::timeout;

const HTTP_PROTOCOL_VERSION: &str = "2025-06-18";

// Only a 404 rejecting the original RPC POST is safe to retry. A failure while
// replying to a server request on an already accepted SSE stream is not.
#[derive(Debug)]
struct SessionExpired;

impl std::fmt::Display for SessionExpired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HTTP MCP session expired (404)")
    }
}

impl std::error::Error for SessionExpired {}

pub(super) struct HttpServer {
    client: Client,
    url: Url,
    headers: HeaderMap,
    session_id: Option<HeaderValue>,
    protocol_version: Option<String>,
    next_id: u64,
}

impl HttpServer {
    pub(super) async fn connect(config: &Value) -> Result<Self> {
        let url = config
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("HTTP MCP server has no url"))?;
        let url = Url::parse(url).map_err(|_| anyhow!("HTTP MCP server has an invalid url"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            bail!("HTTP MCP url must use http/https without embedded credentials or a fragment");
        }
        let mut headers = HeaderMap::new();
        if let Some(values) = config.get("headers") {
            let values = values
                .as_object()
                .ok_or_else(|| anyhow!("MCP headers must be an object"))?;
            for (name, value) in values {
                let name = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| anyhow!("invalid MCP header name"))?;
                if matches!(name.as_str(), "mcp-session-id" | "mcp-protocol-version") {
                    bail!("MCP session and protocol headers are managed by the client");
                }
                let value = value
                    .as_str()
                    .ok_or_else(|| anyhow!("MCP header values must be strings"))?;
                let mut value = HeaderValue::from_str(value)
                    .map_err(|_| anyhow!("invalid MCP header value"))?;
                value.set_sensitive(true);
                headers.insert(name, value);
            }
        }
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(MCP_REQUEST_TIMEOUT)
            .build()?;
        let mut server = Self {
            client,
            url,
            headers,
            session_id: None,
            protocol_version: None,
            next_id: 0,
        };
        let initialized = timeout(MCP_REQUEST_TIMEOUT, server.initialize())
            .await
            .map_err(|_| anyhow!("HTTP MCP initialization timed out"))
            .and_then(|result| result);
        if let Err(error) = initialized {
            let _ = server.close().await;
            return Err(error);
        }
        Ok(server)
    }

    fn request_headers(&self) -> HeaderMap {
        let mut headers = self.headers.clone();
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some(session_id) = &self.session_id {
            headers.insert("mcp-session-id", session_id.clone());
        }
        if let Some(version) = &self.protocol_version {
            // Only supported, static protocol version strings are stored here.
            headers.insert(
                "mcp-protocol-version",
                HeaderValue::from_str(version).unwrap(),
            );
        }
        headers
    }

    async fn post(&self, message: &Value) -> Result<Response> {
        let response = self
            .client
            .post(self.url.clone())
            .headers(self.request_headers())
            .json(message)
            .send()
            .await
            .map_err(reqwest::Error::without_url)?;
        if response.status().is_redirection() {
            bail!("HTTP MCP endpoint redirected; configure its final URL instead");
        }
        // Do not echo URLs, response bodies or authentication headers in errors.
        Ok(response
            .error_for_status()
            .map_err(reqwest::Error::without_url)?)
    }

    async fn initialize(&mut self) -> Result<()> {
        self.session_id = None;
        self.protocol_version = None;
        let result = self
            .rpc_once(
                "initialize",
                json!({
                    "protocolVersion": HTTP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": "rust-agent", "version": env!("CARGO_PKG_VERSION") }
                }),
            )
            .await?;
        let version = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("HTTP MCP initialize response has no protocolVersion"))?;
        if !matches!(version, "2025-03-26" | "2025-06-18" | "2025-11-25") {
            bail!("HTTP MCP server negotiated an unsupported protocol version");
        }
        self.protocol_version = Some(version.to_string());
        self.post(
            &json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} }),
        )
        .await?;
        Ok(())
    }

    pub(super) async fn rpc(&mut self, method: &str, params: Value) -> Result<Value> {
        timeout(MCP_REQUEST_TIMEOUT, async {
            match self.rpc_once(method, params.clone()).await {
                Err(error) if error.is::<SessionExpired>() => {
                    // A 404 with a session ID means the session expired, not that the tool ran.
                    self.initialize()
                        .await
                        .context("could not renew HTTP MCP session")?;
                    self.rpc_once(method, params).await
                }
                result => result,
            }
        })
        .await
        .map_err(|_| anyhow!("MCP request '{method}' timed out"))?
    }

    async fn rpc_once(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = json!(self.next_id);
        let response = match self
            .post(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .await
        {
            Err(error)
                if self.session_id.is_some()
                    && error
                        .downcast_ref::<reqwest::Error>()
                        .is_some_and(|e| e.status() == Some(StatusCode::NOT_FOUND)) =>
            {
                return Err(SessionExpired.into());
            }
            result => result?,
        };
        if method == "initialize" {
            if let Some(session_id) = response.headers().get("mcp-session-id") {
                if session_id.as_bytes().is_empty()
                    || !session_id
                        .as_bytes()
                        .iter()
                        .all(|b| (0x21..=0x7e).contains(b))
                {
                    bail!("HTTP MCP server returned an invalid session ID");
                }
                let mut session_id = session_id.clone();
                session_id.set_sensitive(true);
                self.session_id = Some(session_id);
            }
        }
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match content_type.as_str() {
            "application/json" => {
                let message = read_json(response).await?;
                self.process_message(message, &id)
                    .await?
                    .ok_or_else(|| anyhow!("HTTP MCP response does not match the request ID"))
            }
            "text/event-stream" => self.read_sse(response, &id).await,
            _ => bail!(
                "HTTP MCP request '{method}' returned no JSON/SSE response (status {})",
                response.status()
            ),
        }
    }

    async fn process_message(&self, message: Value, id: &Value) -> Result<Option<Value>> {
        if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            bail!("invalid HTTP MCP JSON-RPC message");
        }
        if message.get("method").is_some() {
            if let Some(reply) = super::server_request_reply(&message) {
                self.post(&reply).await?;
            }
            return Ok(None);
        }
        if message.get("id") != Some(id) {
            return Ok(None);
        }
        if message.get("result").is_some() == message.get("error").is_some() {
            bail!("invalid HTTP MCP response: expected result or error");
        }
        rpc_result(&message).map(Some)
    }

    async fn read_sse(&self, response: Response, id: &Value) -> Result<Value> {
        let mut stream = response.bytes_stream();
        let mut decoder = SseDecoder::default();
        loop {
            let chunk = stream.next().await;
            let done = chunk.is_none();
            match chunk {
                Some(chunk) => decoder.feed(&chunk.map_err(reqwest::Error::without_url)?)?,
                None if decoder.buffer.last() == Some(&b'\r') => decoder.feed(b"\n")?,
                None => {}
            }
            while let Some(message) = decoder.next_message()? {
                if let Some(result) = self.process_message(message, id).await? {
                    return Ok(result);
                }
            }
            if done {
                bail!("HTTP MCP SSE stream closed before the matching response");
            }
        }
    }

    pub(super) async fn close(&mut self) -> Result<()> {
        if self.session_id.is_some() {
            let response = self
                .client
                .delete(self.url.clone())
                .headers(self.request_headers())
                .timeout(Duration::from_secs(3))
                .send()
                .await
                .map_err(reqwest::Error::without_url)?;
            self.session_id = None;
            if response.status() != StatusCode::METHOD_NOT_ALLOWED {
                response
                    .error_for_status()
                    .map_err(reqwest::Error::without_url)?;
            }
        }
        Ok(())
    }
}

async fn read_json(mut response: Response) -> Result<Value> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(reqwest::Error::without_url)?
    {
        if bytes.len() + chunk.len() > MAX_MESSAGE_BYTES {
            bail!("HTTP MCP JSON response is too large");
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).context("HTTP MCP server returned invalid JSON")
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
    data: Vec<String>,
    data_bytes: usize,
    started: bool,
}

impl SseDecoder {
    fn feed(&mut self, chunk: &[u8]) -> Result<()> {
        if self.buffer.len() + self.data_bytes + chunk.len() > MAX_MESSAGE_BYTES {
            bail!("HTTP MCP SSE message is too large");
        }
        self.buffer.extend_from_slice(chunk);
        Ok(())
    }

    fn next_message(&mut self) -> Result<Option<Value>> {
        while let Some(pos) = self.buffer.iter().position(|b| matches!(b, b'\n' | b'\r')) {
            let ending = if self.buffer[pos] == b'\r' {
                if pos + 1 == self.buffer.len() {
                    return Ok(None);
                }
                if self.buffer[pos + 1] == b'\n' {
                    2
                } else {
                    1
                }
            } else {
                1
            };
            let raw: Vec<u8> = self.buffer.drain(..pos + ending).collect();
            let mut line =
                std::str::from_utf8(&raw[..pos]).context("HTTP MCP SSE contains invalid UTF-8")?;
            if !self.started {
                line = line.trim_start_matches('\u{feff}');
                self.started = true;
            }
            if line.is_empty() {
                if !self.data.is_empty() {
                    let data = self.data.join("\n");
                    self.data.clear();
                    self.data_bytes = 0;
                    if data.is_empty() {
                        continue;
                    }
                    return serde_json::from_str(&data)
                        .context("HTTP MCP SSE contains invalid JSON")
                        .map(Some);
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                let data = data.strip_prefix(' ').unwrap_or(data);
                self.data_bytes += data.len() + 1;
                self.data.push(data.to_string());
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests;
