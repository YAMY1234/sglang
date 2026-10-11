// Zero-GPU regression: real P/D HTTP stubs, production PD router and load monitor.
// Run: cargo test --test pd_load_lifecycle_test -- --nocapture
use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicIsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{CacheAwareConfig, CacheAwarePolicy, PowerOfTwoPolicy},
    protocols::generate::GenerateRequest,
    routers::{http::pd_router::PDRouter, RouterTrait},
};
use tokio::{
    sync::watch,
    task::JoinHandle,
    time::{sleep, timeout},
};
use tracing_subscriber::fmt::MakeWriter;

#[allow(dead_code)]
#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);
impl Write for Logs {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl<'a> MakeWriter<'a> for Logs {
    type Writer = Logs;
    fn make_writer(&'a self) -> Logs {
        self.clone()
    }
}
#[allow(dead_code)]
impl Logs {
    fn events(&self) -> Vec<Value> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .filter_map(|line| {
                let v: Value = serde_json::from_str(line).ok()?;
                serde_json::from_str(v["fields"]["score_trace"].as_str()?).ok()
            })
            .collect()
    }
    fn request(&self, id: &str) -> Vec<Value> {
        self.events()
            .into_iter()
            .filter(|v| v["x_request_id"] == id)
            .collect()
    }
}

#[derive(Clone)]
struct Stub {
    name: String,
    role: &'static str,
    gate: watch::Receiver<bool>,
    receipts: Arc<Mutex<Vec<Value>>>,
    completed: Arc<Mutex<HashSet<u64>>>,
    polls: Arc<Mutex<Vec<(Instant, isize)>>>,
    score: Arc<AtomicIsize>,
    rank_shape: Arc<AtomicBool>,
}

async fn wait_gate(mut gate: watch::Receiver<bool>) {
    while !*gate.borrow_and_update() {
        gate.changed().await.unwrap();
    }
}

async fn generate(State(s): State<Stub>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let id = headers
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let room = body["bootstrap_room"].as_u64().unwrap();
    let text = body["text"].as_str().unwrap_or("").to_string();
    s.receipts
        .lock()
        .unwrap()
        .push(json!({"id": id, "role": s.role, "name": s.name,
        "room": room, "port": body["bootstrap_port"], "host": body["bootstrap_host"]}));
    if (s.role == "prefill" && text == "P_HTTP_ERROR")
        || (s.role == "decode" && text.starts_with("D_HTTP_ERROR"))
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "stub failure"}})),
        )
            .into_response();
    }
    if s.role == "prefill" {
        if text == "D_HTTP_ERROR_P_HEADERS" {
            wait_gate(s.gate.clone()).await;
        }
        // Headers precede P completion: checking only send().await would release too early.
        let stream = futures_util::stream::once(async move {
            wait_gate(s.gate).await;
            s.completed.lock().unwrap().insert(room);
            Ok::<_, io::Error>(Bytes::from_static(
                b"{\"meta_info\":{\"input_token_logprobs\":[[0.1,1,\"p\"]]}}",
            ))
        });
        return Response::new(Body::from_stream(stream));
    }
    // Realistic bootstrap pairing: decode cannot produce a first token until its
    // room's selected P completed. HTTP headers themselves are returned early.
    let st = s.clone();
    if body["stream"] != true {
        while !st.completed.lock().unwrap().contains(&room) {
            sleep(Duration::from_millis(2)).await;
        }
        wait_gate(st.gate).await;
        return Json(
            json!({"text": "stub-token", "meta_info": {"output_token_logprobs": [[0.2,2,"d"]]}}),
        )
        .into_response();
    }
    let stream = futures_util::stream::unfold((s, text, 0u8), move |(s, text, stage)| async move {
        match stage {
            0 => {
                if text != "D_EARLY" {
                    while !s.completed.lock().unwrap().contains(&room) {
                        sleep(Duration::from_millis(2)).await;
                    }
                }
                Some((
                    Ok(Bytes::from_static(b"data: {\"text\":\"stub-token\"}\n\n")),
                    (s, text, 1),
                ))
            }
            1 => {
                wait_gate(s.gate.clone()).await;
                if text == "D_STREAM_ERROR" {
                    Some((Err(io::Error::other("truncated stub stream")), (s, text, 2)))
                } else if text == "D_EOF" {
                    None
                } else {
                    Some((Ok(Bytes::from_static(b"data: [DONE]\n\n")), (s, text, 2)))
                }
            }
            _ => None,
        }
    });
    Response::new(Body::from_stream(stream))
}

async fn loads(State(s): State<Stub>) -> Json<Value> {
    let score = s.score.load(Ordering::SeqCst);
    s.polls.lock().unwrap().push((Instant::now(), score));
    if score == -1 {
        Json(json!({"aggregate": {"malformed": true}}))
    } else if score == -2 {
        Json(json!({"loads": []}))
    } else if s.rank_shape.load(Ordering::SeqCst) {
        let mut payload: Value =
            serde_json::from_str(include_str!("../fixtures/loads/frozen_7a841e_core.json"))
                .unwrap();
        payload["loads"][0]["num_total_tokens"] = json!(score / 2);
        payload["loads"][1]["num_total_tokens"] = json!(score - score / 2);
        Json(payload)
    } else {
        Json(json!({"aggregate": {"total_tokens": score}}))
    }
}

#[allow(dead_code)]
struct Rig {
    router: Arc<PDRouter>,
    workers: Vec<Arc<dyn Worker>>,
    p_gate: watch::Sender<bool>,
    d_gate: watch::Sender<bool>,
    receipts: Arc<Mutex<Vec<Value>>>,
    polls: Vec<Arc<Mutex<Vec<(Instant, isize)>>>>,
    scores: Vec<Arc<AtomicIsize>>,
    rank_shapes: Vec<Arc<AtomicBool>>,
    servers: Vec<JoinHandle<()>>,
}
impl Drop for Rig {
    fn drop(&mut self) {
        for server in &self.servers {
            server.abort();
        }
    }
}
impl Rig {
    async fn new(trace: bool, p_count: usize, d_count: usize, abs_threshold: usize) -> Self {
        let mut config = RouterConfig::default();
        config.disable_retries = true;
        let mut config_json = serde_json::to_value(config).unwrap();
        // Unknown fields are ignored by the frozen config, so the same stubs
        // can exercise the original router in the compatibility test.
        config_json["score_trace"] = json!(trace);
        let config: RouterConfig = serde_json::from_value(config_json).unwrap();
        let context = Arc::new(AppContext::from_config(config, 10).await.unwrap());
        let registry = context.worker_registry.clone();
        let policies = context.policy_registry.clone();
        policies.set_prefill_policy(Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: abs_threshold,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 60,
            max_tree_size: 2usize.pow(26),
        })));
        policies.set_decode_policy(Arc::new(PowerOfTwoPolicy::new()));
        let receipts = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(Mutex::new(HashSet::new()));
        let (p_gate, p_rx) = watch::channel(true);
        let (d_gate, d_rx) = watch::channel(true);
        let mut workers = Vec::new();
        let mut servers = Vec::new();
        let mut polls = Vec::new();
        let mut scores = Vec::new();
        let mut rank_shapes = Vec::new();
        for idx in 0..p_count + d_count {
            let is_p = idx < p_count;
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let poll = Arc::new(Mutex::new(Vec::new()));
            let score = Arc::new(AtomicIsize::new(if is_p {
                0
            } else {
                (idx * 100) as isize
            }));
            let rank_shape = Arc::new(AtomicBool::new(false));
            let stub = Stub {
                name: url.clone(),
                role: if is_p { "prefill" } else { "decode" },
                gate: if is_p { p_rx.clone() } else { d_rx.clone() },
                receipts: receipts.clone(),
                completed: completed.clone(),
                polls: poll.clone(),
                score: score.clone(),
                rank_shape: rank_shape.clone(),
            };
            let app = Router::new()
                .route("/generate", post(generate))
                .route("/v1/loads", get(loads))
                .with_state(stub);
            servers.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            }));
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(if is_p {
                        WorkerType::Prefill {
                            bootstrap_port: Some(19000 + idx as u16),
                        }
                    } else {
                        WorkerType::Decode
                    })
                    .build(),
            );
            registry.register(worker.clone());
            workers.push(worker);
            polls.push(poll);
            scores.push(score);
            rank_shapes.push(rank_shape);
        }
        policies.init_pd_cache_aware_policies(
            &registry.get_prefill_workers(),
            &registry.get_decode_workers(),
        );
        let router = Arc::new(PDRouter::new(&context).await.unwrap());
        Self {
            router,
            workers,
            p_gate,
            d_gate,
            receipts,
            polls,
            scores,
            rank_shapes,
            servers,
        }
    }
    fn spawn(&self, id: &str, text: &str, stream: bool) -> JoinHandle<Response> {
        let router = self.router.clone();
        let id = id.to_string();
        let text = text.to_string();
        tokio::spawn(async move {
            let body: GenerateRequest =
                serde_json::from_value(json!({"text": text, "stream": stream})).unwrap();
            let mut headers = HeaderMap::new();
            headers.insert("x-request-id", id.parse().unwrap());
            headers.insert("x-smg-routing-key", "stub-session".parse().unwrap());
            router.route_generate(Some(&headers), &body, None).await
        })
    }
    fn local_loads(&self) -> Vec<usize> {
        self.workers.iter().map(|w| w.load()).collect()
    }
    async fn zero(&self) {
        until(|| {
            self.local_loads().iter().all(|&n| n == 0)
                && self
                    .workers
                    .iter()
                    .all(|w| w.worker_routing_key_load().value() == 0)
        })
        .await;
        assert!(self
            .workers
            .iter()
            .all(|w| w.worker_routing_key_load().value() == 0));
    }
    async fn consume(handle: JoinHandle<Response>) -> (StatusCode, Bytes) {
        let response = timeout(Duration::from_secs(5), handle)
            .await
            .unwrap()
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, bytes)
    }
    fn assert_pairing(&self) {
        let mut grouped = HashMap::<String, Vec<Value>>::new();
        for r in self.receipts.lock().unwrap().iter() {
            grouped
                .entry(r["id"].as_str().unwrap().into())
                .or_default()
                .push(r.clone());
        }
        let mut rooms = HashSet::new();
        for (id, rr) in grouped {
            if rr.len() != 2 {
                continue;
            } // pre-dispatch cancellation need not reach both stubs
            let p = rr.iter().find(|r| r["role"] == "prefill").unwrap();
            let d = rr.iter().find(|r| r["role"] == "decode").unwrap();
            assert_eq!(p["room"], d["room"], "room mismatch {id}");
            assert_eq!(p["port"], d["port"], "bootstrap port mismatch {id}");
            assert_eq!(p["host"], d["host"]);
            let w = self
                .workers
                .iter()
                .find(|w| w.url() == p["name"].as_str().unwrap())
                .unwrap();
            assert_eq!(
                p["port"].as_u64().unwrap(),
                w.bootstrap_port().unwrap() as u64
            );
            assert!(
                rooms.insert(p["room"].as_u64().unwrap()),
                "room reused for {id}"
            );
        }
    }
}

async fn until(check: impl Fn() -> bool) {
    timeout(Duration::from_secs(5), async {
        while !check() {
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("condition timed out");
}
#[allow(dead_code)]
fn selection(logs: &Logs, id: &str, role: &str) -> Value {
    logs.request(id)
        .into_iter()
        .find(|v| v["phase"] == "selection" && v["role"] == role)
        .unwrap()["data"]
        .clone()
}
#[allow(dead_code)]
fn release(logs: &Logs, id: &str, role: &str) -> Value {
    logs.request(id)
        .into_iter()
        .find(|v| v["phase"] == "release" && v["role"] == role)
        .unwrap()
}
