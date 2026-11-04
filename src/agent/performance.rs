//! Opt-in, local-only performance measurements. See docs/performance.md.
use super::*;
use std::hint::black_box;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

#[cfg(all(feature = "profile-allocations", not(feature = "profile-heap")))]
#[global_allocator]
static ALLOCATOR: &stats_alloc::StatsAlloc<std::alloc::System> = &stats_alloc::INSTRUMENTED_SYSTEM;

#[cfg(feature = "profile-heap")]
#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

struct Scenario {
    name: &'static str,
    history: Vec<Value>,
    responses: Arc<Vec<String>>,
    api_mode: ApiMode,
    chunk_size: usize,
    expected: String,
}

fn chat(text: &str) -> String {
    format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content":text},"finish_reason":"stop"}]})
    )
}

fn scenarios() -> Vec<Scenario> {
    let history = || vec![json!({"role":"user", "content":"Explain this code."})];
    let tool = |index| {
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":format!("call_{index}"),"function":{"name":"read_file","arguments":"{\"path\":\"note.txt\"}"}}]},"finish_reason":"tool_calls"}]})
        )
    };
    let mut small = String::new();
    for _ in 0..2048 {
        small.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"delta":{"content":"я🙂"}}]})
        ));
    }
    small.push_str("data: [DONE]\n\n");
    let large = "Большой ответ 🙂\n".repeat(12_000);
    let completed = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"type":"response.completed","response":{"output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":large}]}]}})
    );
    vec![
        Scenario { name: "plain", history: history(), responses: Arc::new(vec![chat("Ready.")]), api_mode: ApiMode::Chat, chunk_size: 4096, expected: "Ready.".into() },
        Scenario { name: "history_30", history: (0..30).map(|i| json!({"role":if i % 2 == 0 {"user"} else {"assistant"},"content":"Code and context Я🙂\n".repeat(2048)})).collect(), responses: Arc::new(vec![chat("Ready.")]), api_mode: ApiMode::Chat, chunk_size: 4096, expected: "Ready.".into() },
        Scenario { name: "tools_3", history: history(), responses: Arc::new(vec![tool(0), tool(1), tool(2), chat("Ready.")]), api_mode: ApiMode::Chat, chunk_size: 4096, expected: "Ready.".into() },
        Scenario { name: "stream_2048", history: history(), responses: Arc::new(vec![small]), api_mode: ApiMode::Chat, chunk_size: 17, expected: "я🙂".repeat(2048) },
        Scenario { name: "large_json", history: history(), responses: Arc::new(vec![completed]), api_mode: ApiMode::Responses, chunk_size: 4096, expected: large },
    ]
}

struct Harness {
    cfg: Config,
    client: reqwest::Client,
    mcp: Arc<tokio::sync::Mutex<McpManager>>,
    auto: Arc<AtomicBool>,
    _root: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("note.txt"), "test file\n".repeat(2000)).unwrap();
        Self {
            cfg: Config {
                api_mode: ApiMode::Chat,
                base_url: String::new(),
                api_key: String::new(),
                model: "mock".into(),
                reasoning_effort: Some("high".into()),
                reasoning_efforts: vec![],
                config_dir: root.path().into(),
                root: root.path().into(),
                max_steps: 25,
                context_tokens: 0,
                auto_approve: true,
            },
            client: reqwest::Client::builder().no_proxy().build().unwrap(),
            mcp: Arc::new(tokio::sync::Mutex::new(McpManager::default())),
            auto: Arc::new(AtomicBool::new(true)),
            _root: root,
        }
    }

    async fn run(&mut self, scenario: &Scenario) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        self.cfg.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        self.cfg.api_mode = scenario.api_mode;
        let responses = Arc::clone(&scenario.responses);
        let chunk_size = scenario.chunk_size;
        let server = tokio::spawn(async move {
            for response in responses.iter() {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let header_end = loop {
                    let mut bytes = [0; 8192];
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&bytes[..n]);
                    if let Some(pos) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let length: usize = std::str::from_utf8(&request[..header_end])
                    .unwrap()
                    .lines()
                    .filter_map(|l| l.split_once(':'))
                    .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                    .unwrap()
                    .1
                    .trim()
                    .parse()
                    .unwrap();
                while request.len() < header_end + length {
                    let mut bytes = [0; 8192];
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&bytes[..n]);
                }
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", response.len()).as_bytes()).await.unwrap();
                for chunk in response.as_bytes().chunks(chunk_size) {
                    socket.write_all(chunk).await.unwrap();
                    tokio::task::yield_now().await;
                }
                socket.shutdown().await.unwrap();
            }
        });
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let consumer = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                black_box(event);
            }
        });
        let result = run_turn(
            &self.client,
            &self.cfg,
            scenario.history.clone(),
            &tx,
            &self.auto,
            &self.mcp,
        )
        .await
        .unwrap();
        assert_eq!(result.last().unwrap()["content"], scenario.expected);
        assert_eq!(
            result.iter().filter(|m| m["role"] == "tool").count(),
            scenario.responses.len() - 1
        );
        drop(result);
        drop(tx);
        server.await.unwrap();
        consumer.await.unwrap();
        tokio::task::yield_now().await;
    }
}

#[test]
#[ignore = "local Criterion benchmark; run explicitly in release mode"]
fn timing() {
    if cfg!(any(
        feature = "profile-allocations",
        feature = "profile-heap"
    )) {
        panic!("time without allocator instrumentation");
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut harness = Harness::new();
    let mut criterion = criterion::Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .without_plots();
    for scenario in scenarios() {
        criterion.bench_function(scenario.name, |b| {
            b.iter(|| runtime.block_on(harness.run(black_box(&scenario))))
        });
    }
    let scenario = scenarios().remove(0);
    criterion.bench_function("requests_100", |b| {
        b.iter(|| {
            runtime.block_on(async {
                for _ in 0..100 {
                    harness.run(black_box(&scenario)).await;
                }
            });
        })
    });
    criterion.final_summary();
}

#[test]
#[ignore = "allocation/heap profile; select a profile feature and run explicitly"]
fn profile() {
    if !cfg!(any(
        feature = "profile-allocations",
        feature = "profile-heap"
    )) {
        panic!("select profile-allocations or profile-heap");
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut harness = Harness::new();
    let mut all = scenarios();
    let mut repeated = scenarios().remove(0);
    repeated.name = "requests_100";
    all.push(repeated);
    for scenario in &all {
        runtime.block_on(harness.run(scenario));
        let repeats = if scenario.name == "requests_100" {
            100
        } else {
            1
        };
        #[cfg(feature = "profile-heap")]
        let profiler = dhat::Profiler::builder().testing().build();
        #[cfg(all(feature = "profile-allocations", not(feature = "profile-heap")))]
        let region = stats_alloc::Region::new(ALLOCATOR);
        runtime.block_on(async {
            for _ in 0..repeats {
                harness.run(scenario).await;
            }
        });
        #[cfg(feature = "profile-heap")]
        {
            let stats = dhat::HeapStats::get();
            println!(
                "PROFILE {} total_bytes={} peak_bytes={} retained_bytes={}",
                scenario.name, stats.total_bytes, stats.max_bytes, stats.curr_bytes
            );
            drop(profiler);
        }
        #[cfg(all(feature = "profile-allocations", not(feature = "profile-heap")))]
        {
            let stats = region.change();
            println!("PROFILE {} allocations={} reallocations={} allocated_bytes={} retained_delta_bytes={}", scenario.name, stats.allocations, stats.reallocations, stats.bytes_allocated, stats.bytes_allocated as i128 - stats.bytes_deallocated as i128);
        }
    }
}
