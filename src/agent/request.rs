//! Borrow history and schemas until reqwest serializes the request body.
use super::{responses_input, responses_tools, ApiMode, Config};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct Reasoning<'a> {
    effort: &'a str,
}

#[derive(Serialize)]
#[serde(untagged)]
enum RequestBody<'a> {
    Chat {
        model: &'a str,
        messages: &'a [&'a Value],
        stream: bool,
        stream_options: StreamOptions,
        #[serde(skip_serializing_if = "Option::is_none")]
        tools: Option<&'a Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<&'a str>,
    },
    Responses {
        model: &'a str,
        input: Vec<Cow<'a, Value>>,
        stream: bool,
        store: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        tools: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<Reasoning<'a>>,
    },
}

pub(super) fn completion_request<'a>(
    cfg: &'a Config,
    messages: &'a [&'a Value],
    tools: Option<&'a Value>,
) -> (String, impl Serialize + 'a) {
    match cfg.api_mode {
        ApiMode::Chat => (
            format!("{}/chat/completions", cfg.base_url.trim_end_matches('/')),
            RequestBody::Chat {
                model: &cfg.model,
                messages,
                stream: true,
                stream_options: StreamOptions {
                    include_usage: true,
                },
                tools,
                reasoning_effort: cfg.reasoning_effort.as_deref(),
            },
        ),
        ApiMode::Responses => (
            format!("{}/responses", cfg.base_url.trim_end_matches('/')),
            RequestBody::Responses {
                model: &cfg.model,
                input: responses_input(messages.iter().copied()),
                stream: true,
                store: false,
                tools: tools.map(responses_tools),
                reasoning: cfg
                    .reasoning_effort
                    .as_deref()
                    .map(|effort| Reasoning { effort }),
            },
        ),
    }
}
