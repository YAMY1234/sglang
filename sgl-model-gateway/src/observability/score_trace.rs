//! Opt-in, prompt-free request score trace. No process-global tracing switch.
use std::{
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::core::{Worker, WorkerLoadGuard};

pub fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// One load observation. Its clock starts when this worker's HTTP poll completes,
/// rather than when the slowest worker in the monitoring batch finishes.
#[derive(Debug, Clone, Copy)]
pub struct LoadSample {
    pub value: isize,
    pub sampled_at_unix_ms: u64,
    pub sampled_at: Instant,
}

impl LoadSample {
    pub fn new(value: isize) -> Self {
        Self {
            value,
            sampled_at_unix_ms: unix_ms(),
            sampled_at: Instant::now(),
        }
    }
}

#[derive(Debug)]
pub struct RequestScoreTrace {
    x_request_id: String,
    attempt_id: String,
    attempt: u32,
    started: Instant,
    bootstrap: Mutex<Value>,
}

impl RequestScoreTrace {
    pub fn request_id(headers: Option<&http::HeaderMap>) -> String {
        headers
            .and_then(|h| h.get("x-request-id").or_else(|| h.get("x_request_id")))
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
    }

    pub fn new(id: String, attempt: u32) -> Arc<Self> {
        Arc::new(Self {
            x_request_id: id,
            attempt_id: uuid::Uuid::new_v4().to_string(),
            attempt,
            started: Instant::now(),
            bootstrap: Mutex::new(Value::Null),
        })
    }

    pub fn set_bootstrap(&self, room: Value) {
        *self.bootstrap.lock() = room;
    }

    pub fn emit(&self, phase: &str, role: &str, data: Value) {
        let event = json!({"x_request_id": self.x_request_id, "attempt_id": self.attempt_id,
            "attempt": self.attempt, "phase": phase, "role": role,
            "at_unix_ms": unix_ms(), "elapsed_ms": self.started.elapsed().as_millis(),
            "bootstrap_room": *self.bootstrap.lock(), "data": data});
        tracing::info!(target: "smg::score_trace", score_trace = %event, "router_score_trace");
    }
}

#[derive(Debug, Clone)]
pub struct SelectionTrace {
    pub request: Arc<RequestScoreTrace>,
    pub role: &'static str,
}

impl SelectionTrace {
    pub fn emit(&self, data: Value) {
        self.request.emit("selection", self.role, data);
    }
}

/// Wrap the existing guard, adding timestamps only when tracing is enabled.
/// The reservation itself remains RAII-managed on cancellation and all returns.
pub struct PDLoadGuard {
    guard: Option<WorkerLoadGuard>,
    worker: Arc<dyn Worker>,
    trace: Option<Arc<RequestScoreTrace>>,
    role: &'static str,
    reason: &'static str,
}

impl PDLoadGuard {
    pub fn new(
        worker: Arc<dyn Worker>,
        headers: Option<&http::HeaderMap>,
        trace: Option<Arc<RequestScoreTrace>>,
        role: &'static str,
    ) -> Self {
        let guard = WorkerLoadGuard::new(worker.clone(), headers);
        if let Some(t) = &trace {
            t.emit(
                "reserve",
                role,
                json!({"worker": worker.url(), "load": worker.load()}),
            );
        }
        Self {
            guard: Some(guard),
            worker,
            trace,
            role,
            reason: "handler_cancelled",
        }
    }

    pub fn release(mut self, reason: &'static str) {
        self.reason = reason;
    }
    pub fn set_reason(&mut self, reason: &'static str) {
        self.reason = reason;
    }
}

impl Drop for PDLoadGuard {
    fn drop(&mut self) {
        drop(self.guard.take());
        if let Some(t) = &self.trace {
            t.emit("release", self.role,
            json!({"worker": self.worker.url(), "load": self.worker.load(), "reason": self.reason}));
        }
    }
}
