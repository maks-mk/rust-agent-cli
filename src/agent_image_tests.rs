use super::*;
use image::{DynamicImage, ImageFormat};
use std::io::Cursor;

fn image_file(cfg: &Config, name: &str, format: ImageFormat) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    DynamicImage::new_rgb8(3, 2)
        .write_to(&mut output, format)
        .unwrap();
    let bytes = output.into_inner();
    std::fs::write(cfg.root.join(name), &bytes).unwrap();
    bytes
}

fn visual_message() -> Value {
    json!({"role":"user", "content":[
        {"type":"text", "text":"Images returned by read_images (tool_call_id: image); sample.png: 3×2"},
        {"type":"image_url", "image_url":{"url":format!("data:image/png;base64,{}", "A".repeat(30_000))}}
    ]})
}

#[tokio::test]
async fn reads_supported_formats_by_signature_and_converts_when_needed() {
    let api = MockApi::start(vec![]).await;
    for (format, mime) in [
        (ImageFormat::Png, "image/png"),
        (ImageFormat::Jpeg, "image/jpeg"),
        (ImageFormat::WebP, "image/webp"),
        (ImageFormat::Gif, "image/png"),
        (ImageFormat::Bmp, "image/png"),
        (ImageFormat::Tiff, "image/png"),
    ] {
        let original = image_file(&api.cfg, "image.wrong-extension", format);
        let ((status, summary, content), parts) =
            read_images(&json!({"paths":"image.wrong-extension"}), &api.cfg).await;
        assert_eq!(status, ToolStatus::Ok, "{format:?}: {content}");
        assert_eq!(summary, "Read 1 images");
        assert!(content.contains("3×2"));
        assert!(!content.contains("base64"));
        assert_eq!(parts.len(), 2);
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with(&format!("data:{mime};base64,")));
        if matches!(
            format,
            ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP
        ) {
            assert_eq!(
                url,
                format!("data:{mime};base64,{}", base64_encode(&original))
            );
        } else {
            assert!(url.starts_with("data:image/png;base64,iVBORw0KGgo"));
        }
    }
}

#[tokio::test]
async fn image_arguments_paths_and_partial_failures_are_checked() {
    let api = MockApi::start(vec![]).await;
    image_file(&api.cfg, "good.png", ImageFormat::Png);
    for args in [
        json!({}),
        json!({"paths":[]}),
        json!({"paths":["good.png", 1]}),
        json!({"paths":" "}),
        json!({"paths":["a","b","c","d","e"]}),
        json!({"paths":"good.png", "max_images":0}),
        json!({"paths":"good.png", "max_images":5}),
        json!({"paths":"good.png", "max_images":"1"}),
        json!({"paths":"../outside.png"}),
        json!({"paths":"https://example.com/image.png"}),
        json!({"paths":"note.txt"}),
        json!({"paths":"missing.png"}),
        json!({"paths":"."}),
    ] {
        let ((status, _, _), parts) = read_images(&args, &api.cfg).await;
        assert_eq!(status, ToolStatus::Failed, "{args}");
        assert!(parts.is_empty());
    }
    let ((status, _, content), parts) = read_images(
        &json!({"paths":["good.png", "note.txt", "missing.png"]}),
        &api.cfg,
    )
    .await;
    assert_eq!(status, ToolStatus::Failed);
    assert_eq!(parts.len(), 2);
    assert!(content.contains("good.png"));
    assert!(content.contains("Error reading \"note.txt\""));
    let ((status, _, content), parts) = read_images(
        &json!({"paths":["good.png", "missing.png"], "max_images":1}),
        &api.cfg,
    )
    .await;
    assert_eq!(status, ToolStatus::Ok);
    assert_eq!(parts.len(), 2);
    assert!(content.contains("Skipped 1 paths"));
}

#[tokio::test]
async fn image_file_and_dimension_limits_are_enforced() {
    let api = MockApi::start(vec![]).await;
    let file = std::fs::File::create(api.cfg.root.join("oversized.png")).unwrap();
    file.set_len(10 * 1024 * 1024 + 1).unwrap();
    drop(file);
    let ((status, _, content), parts) =
        read_images(&json!({"paths":"oversized.png"}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Failed);
    assert!(content.contains("byte limit"));
    assert!(parts.is_empty());
    DynamicImage::new_rgb8(8193, 1)
        .save(api.cfg.root.join("wide.png"))
        .unwrap();
    let ((status, _, _), parts) = read_images(&json!({"paths":"wide.png"}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Failed);
    assert!(parts.is_empty());
    // A valid PNG header followed by a truncated payload must not be forwarded.
    let bytes = image_file(&api.cfg, "broken.png", ImageFormat::Png);
    std::fs::write(api.cfg.root.join("broken.png"), &bytes[..bytes.len() / 2]).unwrap();
    let ((status, _, _), parts) = read_images(&json!({"paths":"broken.png"}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Failed);
    assert!(parts.is_empty());
}

#[tokio::test]
async fn image_batch_bytes_and_pixel_count_are_limited() {
    let api = MockApi::start(vec![]).await;
    // Trailing bytes keep this a decodable PNG while exercising the input budget.
    image_file(&api.cfg, "padded.png", ImageFormat::Png);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(api.cfg.root.join("padded.png"))
        .unwrap();
    file.set_len(7 * 1024 * 1024).unwrap();
    drop(file);
    let ((status, _, content), parts) = read_images(
        &json!({"paths":["padded.png", "padded.png", "padded.png"]}),
        &api.cfg,
    )
    .await;
    assert_eq!(status, ToolStatus::Failed);
    assert_eq!(parts.len(), 4);
    assert!(content.contains("batch byte limit"));
    DynamicImage::new_luma8(4001, 4000)
        .save(api.cfg.root.join("pixels.png"))
        .unwrap();
    let ((status, _, content), parts) = read_images(&json!({"paths":"pixels.png"}), &api.cfg).await;
    assert_eq!(status, ToolStatus::Failed);
    assert!(content.contains("pixel limit"));
    assert!(parts.is_empty());
}

#[tokio::test]
async fn image_tool_sends_visual_parts_after_all_parallel_tool_replies() {
    let calls = json!([
        {"index":0,"id":"image-a","function":{"name":"read_images","arguments":"{\"paths\":\"sample.png\"}"}},
        {"index":1,"id":"text","function":{"name":"read_file","arguments":"{\"path\":\"note.txt\"}"}},
        {"index":2,"id":"image-b","function":{"name":"read_images","arguments":"{\"paths\":\"sample.png\"}"}}
    ]);
    let mut api = MockApi::start(vec![
        sse(vec![delta(json!({"tool_calls":calls}))]),
        sse(vec![delta(json!({"content":"Saw the images"}))]),
    ])
    .await;
    image_file(&api.cfg, "sample.png", ImageFormat::Png);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let history = run_turn(
        &reqwest::Client::new(),
        &api.cfg,
        vec![json!({"role":"user", "content":"Inspect the images"})],
        &tx,
        &Arc::new(AtomicBool::new(false)),
        &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
    )
    .await
    .unwrap();
    let requests = api.requests().await;
    assert!(requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["function"]["name"] == "read_images"));
    let messages = requests[1]["messages"].as_array().unwrap();
    for (index, id) in ["image-a", "text", "image-b"].into_iter().enumerate() {
        assert_eq!(messages[index + 3]["role"], "tool");
        assert_eq!(messages[index + 3]["tool_call_id"], id);
        assert!(!messages[index + 3]["content"]
            .as_str()
            .unwrap()
            .contains("base64"));
    }
    for (index, id) in ["image-a", "image-b"].into_iter().enumerate() {
        let message = &messages[index + 6];
        assert!(is_image_message(message));
        assert!(!is_user_request(message));
        assert!(message["content"][0]["text"].as_str().unwrap().contains(id));
        assert!(message["content"][2]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }
    assert_eq!(history.last().unwrap()["content"], "Saw the images");
    while let Ok(event) = rx.try_recv() {
        let text = match event {
            AgentEvent::ToolEnd { summary, .. } => summary,
            AgentEvent::ToolOutput { output, .. } => output,
            AgentEvent::Text(text) | AgentEvent::Notice(text) => text,
            _ => continue,
        };
        assert!(!text.contains("base64"));
    }
}

#[tokio::test]
async fn compression_omits_image_payload_and_retains_complete_latest_group() {
    for retain_images in [false, true] {
        let mut api =
            MockApi::start(vec![sse(vec![delta(json!({"content":"Short summary"}))])]).await;
        let user = json!({"role":"user", "content":"Inspect sample.png"});
        let group = vec![
            json!({"role":"assistant", "content":"", "tool_calls":[{"id":"image", "type":"function", "function":{"name":"read_images", "arguments":"{\"paths\":\"sample.png\"}"}}]}),
            json!({"role":"tool", "tool_call_id":"image", "content":"Read sample.png"}),
            visual_message(),
        ];
        let mut history = vec![
            json!({"role":"assistant", "content":"Old context ".repeat(1000)}),
            user.clone(),
        ];
        history.extend(group.clone());
        if !retain_images {
            history.push(json!({"role":"assistant", "content":"Image inspected"}));
        }
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let compressed = compress_context(&reqwest::Client::new(), &api.cfg, &history, &tx)
            .await
            .unwrap()
            .unwrap();
        assert!(compressed.contains(&user));
        if retain_images {
            assert_eq!(&compressed[compressed.len() - 3..], group.as_slice());
        } else {
            assert!(!compressed.iter().any(is_image_message));
        }
        let requests = api.requests().await;
        let body = requests[0].to_string();
        assert!(!body.contains("base64"));
        assert!(!body.contains(&"A".repeat(100)));
        if !retain_images {
            assert!(body.contains("sample.png"));
            assert!(body.contains("Image data omitted"));
        }
    }
}

#[test]
fn image_token_estimate_does_not_scale_with_base64_length() {
    let small = visual_message();
    let mut large = small.clone();
    large["content"][1]["image_url"]["url"] =
        json!(format!("data:image/png;base64,{}", "A".repeat(300_000)));
    assert_eq!(estimate_tokens(&small), estimate_tokens(&large));
    assert!(!text_only_message(&large).to_string().contains("base64"));
    assert!(!is_user_request(&large));
    assert!(is_user_request(
        &json!({"role":"user", "content":"Inspect image"})
    ));
}

#[tokio::test]
async fn cancelling_after_image_read_preserves_visuals_and_tool_pairs() {
    let calls = json!([
        {"index":0,"id":"image","function":{"name":"read_images","arguments":"{\"paths\":\"sample.png\"}"}},
        {"index":1,"id":"write","function":{"name":"write_file","arguments":"{\"path\":\"never.txt\",\"content\":\"never\"}"}}
    ]);
    let api = MockApi::start(vec![sse(vec![delta(json!({"tool_calls":calls}))])]).await;
    image_file(&api.cfg, "sample.png", ImageFormat::Png);
    let cfg = api.cfg.clone();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        run_turn(
            &reqwest::Client::new(),
            &cfg,
            vec![json!({"role":"user", "content":"Inspect then write"})],
            &tx,
            &Arc::new(AtomicBool::new(false)),
            &Arc::new(tokio::sync::Mutex::new(McpManager::default())),
        )
        .await
    });
    let mut history = vec![json!({"role":"user", "content":"Inspect then write"})];
    let approval = loop {
        match timeout(Duration::from_secs(10), rx.recv())
            .await
            .unwrap()
            .unwrap()
        {
            AgentEvent::ToolApproval { reply, .. } => break reply,
            event => apply_history_event(&mut history, event),
        }
    };
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    drop(approval);
    while let Ok(event) = rx.try_recv() {
        apply_history_event(&mut history, event);
    }
    assert_eq!(history[2]["tool_call_id"], "image");
    assert_eq!(history[3]["tool_call_id"], "write");
    assert!(is_image_message(&history[4]));
    assert!(history[3]["content"]
        .as_str()
        .unwrap()
        .contains("Interrupted"));
    assert!(!api.cfg.root.join("never.txt").exists());
}
