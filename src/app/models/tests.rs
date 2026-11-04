use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn server(status: u16, body: &str) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/custom/v1/", listener.local_addr().unwrap());
    let body = body.to_owned();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while !request.windows(4).any(|part| part == b"\r\n\r\n") {
            let mut buf = [0; 1024];
            let count = socket.read(&mut buf).await.unwrap();
            assert_ne!(count, 0);
            request.extend_from_slice(&buf[..count]);
        }
        let response = format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8(request).unwrap()
    });
    (base_url, task)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test]
async fn discovery_uses_configured_path_and_auth_and_normalizes_ids() {
    let (url, task) = server(
        200,
        r#"{"object":"list","data":[{"id":"z-model"},{"id":"a-model"},{"id":"z-model"},{"id":""},{"id":"  "},{"id":12},{"name":"missing-id"},{"id":"bad\nname"}]}"#,
    ).await;
    let models = fetch_models(&client(), &url, "test-key", MODELS_TIMEOUT)
        .await
        .unwrap();
    assert_eq!(models, ["a-model", "z-model"]);
    let request = task.await.unwrap().to_lowercase();
    assert!(request.starts_with("get /custom/v1/models http/1.1\r\n"));
    assert!(request.contains("\r\nauthorization: bearer test-key\r\n"));
}

#[tokio::test]
async fn empty_list_and_local_server_without_key_are_supported() {
    let (url, task) = server(200, r#"{"data":[]}"#).await;
    let models = fetch_models(&client(), url.trim_end_matches('/'), "", MODELS_TIMEOUT)
        .await
        .unwrap();
    assert!(models.is_empty());
    assert!(!task
        .await
        .unwrap()
        .to_lowercase()
        .contains("authorization:"));
}

#[tokio::test]
async fn unsupported_unauthorized_and_failed_endpoints_report_status() {
    for status in [401, 404, 500] {
        let (url, task) = server(status, "provider error").await;
        let error = fetch_models(&client(), &url, "", MODELS_TIMEOUT)
            .await
            .unwrap_err();
        assert!(error.contains(&format!("HTTP {status}")), "{error}");
        task.await.unwrap();
    }
}

#[tokio::test]
async fn invalid_json_or_missing_data_does_not_look_like_an_empty_list() {
    for body in ["not json", "{}", r#"{"data":null}"#] {
        let (url, task) = server(200, body).await;
        let error = fetch_models(&client(), &url, "", MODELS_TIMEOUT)
            .await
            .unwrap_err();
        assert_eq!(error, "Invalid model list response");
        task.await.unwrap();
    }
}

#[tokio::test]
async fn timeout_includes_waiting_for_response_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
            let mut buf = [0; 1024];
            let count = socket.read(&mut buf).await.unwrap();
            assert_ne!(count, 0);
            request.extend_from_slice(&buf[..count]);
        }
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let error = fetch_models(&client(), &url, "", Duration::from_millis(100))
        .await
        .unwrap_err();
    task.abort();
    assert_eq!(error, "Model list request timed out");
}

#[tokio::test]
async fn connection_failure_returns_a_manual_entry_friendly_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    drop(listener);
    let error = fetch_models(&client(), &url, "", MODELS_TIMEOUT)
        .await
        .unwrap_err();
    assert_eq!(error, "Cannot load model list");
}

#[test]
fn dropping_menu_cancels_its_pending_request() {
    let (sender, mut receiver) = tokio::sync::oneshot::channel::<()>();
    let menu = ModelMenu {
        items: Vec::new(),
        selected: 0,
        message: None,
        request: Some(Box::pin(async move {
            let _sender = sender;
            std::future::pending().await
        })),
    };
    drop(menu);
    assert_eq!(
        receiver.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Closed)
    );
}
