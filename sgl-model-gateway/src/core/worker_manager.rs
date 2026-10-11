//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    observability::score_trace::LoadSample,
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        Self::get_all_worker_loads_with_samples(worker_registry, client)
            .await
            .0
    }

    async fn get_all_worker_loads_with_samples(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> (WorkerLoadsResult, HashMap<String, LoadSample>) {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    let sample = LoadSample::new(load);
                    (
                        WorkerLoadInfo {
                            worker: url,
                            worker_type,
                            load,
                        },
                        sample,
                    )
                }
            })
            .collect();

        let results = future::join_all(futures).await;
        let samples = results
            .iter()
            .map(|(info, sample)| (info.worker.clone(), *sample))
            .collect();
        let loads: Vec<_> = results.into_iter().map(|(info, _)| info).collect();
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        (
            WorkerLoadsResult {
                loads,
                total_workers,
                successful,
                failed,
            },
            samples,
        )
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let load_url = format!("{}/v1/loads?include=core", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => Self::parse_load_payload(&json).unwrap_or(-1),
                _ => -1,
            },
            _ => -1,
        }
    }

    /// Accept the aggregate API and the frozen model's per-DP-rank core API.
    /// The monitor polls an entire worker endpoint, so rank loads are summed.
    /// Reject partial/malformed/negative/overflowing reports instead of silently
    /// undercounting them. A valid aggregate takes precedence if both exist.
    fn parse_load_payload(json: &Value) -> Option<isize> {
        let nonnegative = |value: &Value| value.as_u64().and_then(|n| isize::try_from(n).ok());
        if let Some(total) = json
            .pointer("/aggregate/total_tokens")
            .and_then(nonnegative)
        {
            return Some(total);
        }
        let ranks = json.get("loads")?.as_array()?;
        if ranks.is_empty() {
            return None;
        }
        ranks.iter().try_fold(0isize, |total, rank| {
            total.checked_add(nonnegative(rank.get("num_total_tokens")?)?)
        })
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let (result, samples) =
                WorkerManager::get_all_worker_loads_with_samples(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_load_samples(&samples);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod load_payload_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frozen_core_fixture_reproduces_aggregate_only_mismatch() {
        let payload: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/loads/frozen_7a841e_core.json"
        ))
        .unwrap();
        assert!(
            payload.pointer("/aggregate/total_tokens").is_none(),
            "frozen router would publish -1"
        );
        assert_eq!(WorkerManager::parse_load_payload(&payload), Some(60));
    }

    #[test]
    fn aggregate_and_rank_shapes_accept_nonnegative_totals() {
        let parse = WorkerManager::parse_load_payload;
        assert_eq!(
            parse(&json!({"aggregate": {"total_tokens": 120}})),
            Some(120)
        );
        assert_eq!(parse(&json!({"aggregate": {"total_tokens": 0}})), Some(0));
        assert_eq!(
            parse(
                &json!({"loads": [{"dp_rank": 0, "num_total_tokens": 17}, {"dp_rank": 1, "num_total_tokens": 43}]})
            ),
            Some(60)
        );
        assert_eq!(parse(&json!({"loads": [{"num_total_tokens": 0}]})), Some(0));
        assert_eq!(
            parse(&json!({"aggregate": {"total_tokens": 99}, "loads": [{"num_total_tokens": 60}]})),
            Some(99)
        );
        assert_eq!(
            parse(
                &json!({"aggregate": {"total_tokens": "bad"}, "loads": [{"num_total_tokens": 60}]})
            ),
            Some(60)
        );
    }

    #[test]
    fn invalid_payloads_never_publish_partial_or_overflowed_totals() {
        let parse = WorkerManager::parse_load_payload;
        for bad in [
            json!(null),
            json!({}),
            json!({"loads": []}),
            json!({"loads": {}}),
            json!({"loads": [{"num_total_tokens": 3}, {}]}),
            json!({"loads": [{"num_total_tokens": 3}, {"num_total_tokens": -1}]}),
            json!({"loads": [{"num_total_tokens": "4"}]}),
            json!({"loads": [{"num_total_tokens": 1.5}]}),
            json!({"loads": [{"num_total_tokens": true}]}),
            json!({"aggregate": {"total_tokens": -1}}),
            json!({"aggregate": {"total_tokens": 1.5}}),
            json!({"aggregate": {"total_tokens": "4"}}),
            json!({"aggregate": {"total_tokens": u64::MAX}}),
            json!({"loads": [{"num_total_tokens": isize::MAX}, {"num_total_tokens": 1}]}),
        ] {
            assert_eq!(parse(&bad), None, "unexpected valid score: {bad}");
        }
    }
}
