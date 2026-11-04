//! Model discovery and the selector opened by `/model`.

use super::{App, Entry};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::Value;
use std::time::Duration;

const MODELS_TIMEOUT: Duration = Duration::from_secs(10);

pub struct ModelMenu {
    pub items: Vec<String>,
    /// The final item (`items.len()`) always opens manual entry.
    pub selected: usize,
    pub message: Option<String>,
    request: Option<BoxFuture<'static, Result<Vec<String>, String>>>,
}

impl ModelMenu {
    pub fn loading(&self) -> bool {
        self.request.is_some()
    }
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<Value>,
}

fn request_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "Model list request timed out".into()
    } else if let Some(status) = error.status() {
        format!("Model list unavailable (HTTP {})", status.as_u16())
    } else if error.is_decode() {
        "Invalid model list response".into()
    } else {
        "Cannot load model list".into()
    }
}

async fn fetch_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    timeout: Duration,
) -> Result<Vec<String>, String> {
    let mut request = client
        .get(format!("{}/models", base_url.trim_end_matches('/')))
        .timeout(timeout);
    if !api_key.is_empty() {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(request_error)?;
    let list: ModelList = response.json().await.map_err(request_error)?;
    let mut items: Vec<String> = list
        .data
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .filter(|id| !id.trim().is_empty() && !id.chars().any(char::is_control))
        .map(str::to_owned)
        .collect();
    items.sort();
    items.dedup();
    Ok(items)
}

impl App {
    pub fn model_menu_visible(&self) -> bool {
        self.pending.is_none() && self.model_menu.is_some()
    }

    pub fn models_loading(&self) -> bool {
        self.model_menu.as_ref().is_some_and(ModelMenu::loading)
    }

    pub(super) fn open_model_menu(&mut self) {
        let client = self.client.clone();
        let base_url = self.cfg.base_url.clone();
        let api_key = self.cfg.api_key.clone();
        self.model_menu = Some(ModelMenu {
            items: Vec::new(),
            selected: 0,
            message: Some("Loading models... You can enter a name manually now.".into()),
            request: Some(Box::pin(async move {
                fetch_models(&client, &base_url, &api_key, MODELS_TIMEOUT).await
            })),
        });
        self.session_menu = None;
        self.slash = None;
    }

    /// Polled alongside terminal events. The future stays in the menu when
    /// another event wins; closing the menu drops it and cancels the request.
    pub async fn poll_models(&mut self) {
        let Some(menu) = &mut self.model_menu else {
            return std::future::pending().await;
        };
        let Some(request) = &mut menu.request else {
            return std::future::pending().await;
        };
        let result = request.await;
        menu.request = None;
        match result {
            Ok(items) => {
                menu.selected = items
                    .iter()
                    .position(|id| id == &self.cfg.model)
                    .unwrap_or(0);
                menu.message = items
                    .is_empty()
                    .then(|| "No models returned. Enter a name manually.".into());
                menu.items = items;
            }
            Err(error) => {
                menu.message = Some(format!("{error}. Enter a name manually."));
            }
        }
    }

    pub(super) fn model_move(&mut self, delta: isize) {
        if let Some(menu) = &mut self.model_menu {
            let len = menu.items.len() + 1;
            menu.selected = (menu.selected as isize + delta).rem_euclid(len as isize) as usize;
        }
    }

    pub(super) fn model_manual_entry(&mut self) {
        self.model_menu = None;
        self.input.set("/model ");
        self.hist_pos = None;
        self.sync_slash();
    }

    pub(super) fn model_accept(&mut self) {
        let Some(menu) = self.model_menu.take() else {
            return;
        };
        if let Some(model) = menu.items.get(menu.selected) {
            self.set_model(model);
        } else {
            self.model_manual_entry();
        }
    }

    pub(super) fn set_model(&mut self, model: &str) {
        self.model_menu = None;
        self.cfg.model = model.to_string();
        self.entries
            .push(Entry::Notice(format!("Model set to {model}")));
        self.autosave();
    }
}

#[cfg(test)]
mod tests;
