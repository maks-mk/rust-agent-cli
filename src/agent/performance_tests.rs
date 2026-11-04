use super::*;

fn config(mode: ApiMode) -> Config {
    Config {
        api_mode: mode,
        base_url: "https://example.invalid/v1/".into(),
        api_key: String::new(),
        model: "model".into(),
        reasoning_effort: None,
        reasoning_efforts: Vec::new(),
        config_dir: PathBuf::from("."),
        root: PathBuf::from("."),
        max_steps: DEFAULT_MAX_STEPS,
        context_tokens: DEFAULT_CONTEXT_TOKENS,
        auto_approve: false,
    }
}

fn body(cfg: &Config, messages: &[Value], tools: Option<&Value>) -> Value {
    let refs: Vec<_> = messages.iter().collect();
    let (_, request) = completion_request(cfg, &refs, tools);
    serde_json::to_value(request).unwrap()
}

#[test]
fn token_estimates_match_serialized_bytes_including_escapes_and_unicode() {
    let mut values = vec![
        Value::Null,
        json!([true, false, 1, -123, 1.25, 1e100, [], {}]),
        json!({"role":"user", "content":"Код 🙂\n\r\t\"\\\u{0000}\u{001f}"}),
        json!({"role":"assistant", "content":"Visible", "tool_calls":[{"function":{"arguments":"{\"a\":1}"}}],
            "responses_output_items":[{"type":"reasoning", "encrypted_content":"a".repeat(100_000)}]}),
        json!({"role":"user", "content":[{"type":"text","text":"first\nline"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},
            {"type":"text","text":"second 🙂"}]}),
    ];
    for n in 0..64 {
        values.push(json!({"role":"assistant","content":"x".repeat(n)}));
    }
    for value in values {
        let expected = if value.get("responses_output_items").is_some() {
            let mut visible = value.clone();
            visible
                .as_object_mut()
                .unwrap()
                .remove("responses_output_items");
            visible.to_string().len().div_ceil(3)
        } else if is_image_message(&value) {
            text_only_message(&value).to_string().len().div_ceil(3) + 4096
        } else {
            value.to_string().len().div_ceil(3)
        };
        assert_eq!(estimate_tokens(&value), expected);
    }
}

#[test]
fn chat_request_keeps_exact_json_fields_and_omits_absent_options() {
    let mut cfg = config(ApiMode::Chat);
    let messages = vec![
        json!({"role":"system","content":"instructions"}),
        json!({"role":"user","content":"text \"🙂\n"}),
        json!({"role":"assistant","content":"","reasoning_content":"reasoning", "custom":{"keep":true}}),
    ];
    assert_eq!(
        body(&cfg, &messages, None),
        json!({
            "model":"model", "messages":messages, "stream":true,
            "stream_options":{"include_usage":true}
        })
    );
    let tools =
        json!([{"type":"function", "function":{"name":"example","parameters":{"type":"object"}}}]);
    cfg.reasoning_effort = Some("high".into());
    assert_eq!(
        body(&cfg, &messages, Some(&tools)),
        json!({
            "model":"model", "messages":messages, "stream":true,
            "stream_options":{"include_usage":true}, "tools":tools, "reasoning_effort":"high"
        })
    );
}

#[test]
fn responses_request_keeps_raw_continuation_and_non_string_tool_results() {
    let cfg = config(ApiMode::Responses);
    let output = json!([
        {"type":"reasoning","id":"r1","encrypted_content":"secret opaque bytes","summary":[]},
        {"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]},
        {"type":"function_call","id":"fc1","call_id":"call1","name":"example","arguments":"{}","custom":true}
    ]);
    let messages = vec![
        json!({"role":"assistant","content":"hello", "responses_output_items":output}),
        json!({"role":"tool","tool_call_id":"call1","content":{"result":42}}),
    ];
    let request = body(&cfg, &messages, None);
    let mut expected = output.as_array().unwrap().clone();
    expected
        .push(json!({"type":"function_call_output","call_id":"call1","output":"{\"result\":42}"}));
    assert_eq!(
        request,
        json!({"model":"model","input":expected,"stream":true,"store":false})
    );
}

#[test]
fn taking_assistant_payload_preserves_tool_calls_and_reasoning() {
    let output = vec![json!({"type":"message","content":[{"type":"output_text","text":"answer"}]})];
    let mut completion = Completion {
        text: "answer".into(),
        reasoning: "reasoning".into(),
        calls: vec![ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            args: "{}".into(),
        }],
        responses_output_items: Some(output.clone()),
        usage: None,
    };
    let message = completion.take_assistant_message();
    assert_eq!(
        message,
        json!({"role":"assistant","content":"answer","reasoning_content":"reasoning",
        "responses_output_items":output, "tool_calls":[{"type":"function","id":"c1","function":{"name":"read_file","arguments":"{}"}}]})
    );
    assert_eq!(completion.calls[0].id, "c1");
    assert_eq!(completion.calls[0].args, "{}");
}
