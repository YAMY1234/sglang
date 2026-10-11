//! Zero-GPU regression with real P/D HTTP stubs and production router/monitor.
//! Run: cargo test --test pd_load_lifecycle_test -- --nocapture
include!("common/pd_load_stub.rs");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cpu_stub_lifecycle_trace_refresh_and_defaults() {
    let logs = Logs::default();
    tracing_subscriber::fmt()
        .json()
        .with_writer(logs.clone())
        .with_env_filter("smg::score_trace=info")
        .try_init()
        .unwrap();
    let rig = Rig::new(true, 2, 2, 64).await;
    let secret = "PROMPT_CONTENT_MUST_NOT_APPEAR_IN_SCORE_TRACE";
    Rig::consume(rig.spawn("warm", secret, true)).await;
    rig.zero().await;
    let hot = selection(&logs, "warm", "prefill")["selected"]
        .as_str()
        .unwrap()
        .to_string();
    let hot_idx = rig.workers.iter().position(|w| w.url() == hot).unwrap();
    rig.p_gate.send_replace(false);
    let mut pending = Vec::new();
    // Prefix affinity sends 65 queued requests to the same P. With the original
    // late guard, every one reads load zero and request #66 stays on that P.
    for i in 0..65 {
        let id = format!("queue-{i}");
        pending.push(rig.spawn(&id, secret, true));
        until(|| {
            logs.request(&id)
                .iter()
                .filter(|v| v["phase"] == "reserve")
                .count()
                == 2
        })
        .await;
    }
    assert_eq!(rig.workers[hot_idx].load(), 65);
    assert_eq!(rig.workers.iter().map(|w| w.load()).sum::<usize>(), 130);
    let balanced = rig.spawn("balance-66", secret, true);
    until(|| {
        logs.request("balance-66")
            .iter()
            .any(|v| v["phase"] == "reserve")
    })
    .await;
    let s = selection(&logs, "balance-66", "prefill");
    assert_eq!(s["branch"], "balance");
    assert_eq!(s["max_load"], 65);
    assert_eq!(s["min_load"], 0);
    assert_ne!(s["selected"], hot);
    assert!(s["match_rate"].is_null());
    assert!(
        pending.iter().all(|h| !h.is_finished()),
        "P body headers must not release reservations"
    );
    println!("PASS P queue includes 65 waiting requests; default >64 and >1.5 thresholds trigger balance on #66");
    rig.p_gate.send_replace(true);
    for h in pending {
        assert_eq!(Rig::consume(h).await.0, StatusCode::OK);
    }
    Rig::consume(balanced).await;
    rig.zero().await;

    // Fast P / slow D: only D remains reserved. The client body is retained.
    rig.d_gate.send_replace(false);
    let response = rig.spawn("slow-decode", secret, true).await.unwrap();
    assert!(rig
        .router
        .worker_registry
        .get_prefill_workers()
        .iter()
        .all(|w| w.load() == 0));
    assert_eq!(
        rig.router
            .worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| w.load())
            .sum::<usize>(),
        1
    );
    assert_eq!(
        release(&logs, "slow-decode", "prefill")["data"]["reason"],
        "prefill_complete"
    );
    rig.d_gate.send_replace(true);
    rig.zero().await;
    // Guard must release on upstream DONE even before the consumer drops/reads body.
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(body.ends_with(b"data: [DONE]\n\n"));
    println!("PASS P releases before slow D; D releases on DONE while client retains body");

    // Slow P / fast D headers and stream: P work still owns its reservation.
    rig.p_gate.send_replace(false);
    let handle = rig.spawn("slow-prefill", "D_EARLY", true);
    until(|| {
        rig.receipts
            .lock()
            .unwrap()
            .iter()
            .filter(|v| v["id"] == "slow-prefill")
            .count()
            == 2
    })
    .await;
    assert_eq!(rig.local_loads().iter().sum::<usize>(), 2);
    assert!(!handle.is_finished());
    rig.p_gate.send_replace(true);
    Rig::consume(handle).await;
    rig.zero().await;
    println!("PASS slow P / fast D counts both reservations until P body completes");

    // Aborting the dispatch future is handler cancellation before response headers.
    rig.p_gate.send_replace(false);
    let handle = rig.spawn("cancel-dispatch", secret, true);
    until(|| rig.local_loads().iter().sum::<usize>() == 2).await;
    handle.abort();
    let _ = handle.await;
    rig.zero().await;
    rig.p_gate.send_replace(true);
    // Dropping an actual response body cancels the upstream D stream task.
    rig.d_gate.send_replace(false);
    let response = rig.spawn("cancel-body", secret, true).await.unwrap();
    drop(response);
    rig.zero().await;
    rig.d_gate.send_replace(true);
    assert_eq!(
        release(&logs, "cancel-body", "decode")["data"]["reason"],
        "client_cancelled"
    );
    println!("PASS dispatch cancellation and response-body cancellation release without leaks");

    rig.p_gate.send_replace(false);
    let p_outcomes: Vec<_> = rig
        .router
        .worker_registry
        .get_prefill_workers()
        .iter()
        .map(|w| w.circuit_breaker().total_successes())
        .collect();
    let handle = rig.spawn("early-d-error", "D_HTTP_ERROR", true);
    until(|| {
        logs.request("early-d-error")
            .iter()
            .any(|v| v["phase"] == "release" && v["role"] == "decode")
    })
    .await;
    assert!(rig
        .router
        .worker_registry
        .get_decode_workers()
        .iter()
        .all(|w| w.load() == 0));
    assert_eq!(
        Rig::consume(handle).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    rig.zero().await;
    assert_eq!(
        release(&logs, "early-d-error", "prefill")["data"]["reason"],
        "paired_decode_error"
    );
    let handle = rig.spawn(
        "early-d-error-before-p-headers",
        "D_HTTP_ERROR_P_HEADERS",
        true,
    );
    assert_eq!(
        Rig::consume(handle).await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    rig.zero().await;
    assert_eq!(
        release(&logs, "early-d-error-before-p-headers", "prefill")["data"]["reason"],
        "paired_decode_error"
    );
    assert_eq!(
        p_outcomes,
        rig.router
            .worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| w.circuit_breaker().total_successes())
            .collect::<Vec<_>>()
    );
    rig.p_gate.send_replace(true);
    println!("PASS early D HTTP error releases D and cancels P before headers or during body (no error-response hang or synthetic SSE re-reservation)");

    for (id, text) in [("p-error", "P_HTTP_ERROR"), ("d-error", "D_HTTP_ERROR")] {
        assert_eq!(
            Rig::consume(rig.spawn(id, text, true)).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        rig.zero().await;
    }
    rig.d_gate.send_replace(false);
    let response = rig
        .spawn("stream-error", "D_STREAM_ERROR", true)
        .await
        .unwrap();
    let mut body = response.into_body().into_data_stream();
    assert!(body.next().await.unwrap().is_ok());
    rig.d_gate.send_replace(true);
    assert!(body.next().await.unwrap().is_err());
    drop(body);
    rig.zero().await;
    assert_eq!(
        release(&logs, "stream-error", "decode")["data"]["reason"],
        "decode_stream_error"
    );
    Rig::consume(rig.spawn("eof", "D_EOF", true)).await;
    rig.zero().await;
    assert_eq!(
        release(&logs, "eof", "decode")["data"]["reason"],
        "decode_eof"
    );
    println!("PASS P/D HTTP errors, D body error and clean EOF release all reservations");

    let (status, bytes) = Rig::consume(rig.spawn("nonstream", secret, false)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&bytes).unwrap()["text"],
        "stub-token"
    );
    rig.zero().await;
    for wave in 0..20 {
        let handles: Vec<_> = (0..25)
            .map(|i| rig.spawn(&format!("long-{wave}-{i}"), secret, true))
            .collect();
        for h in handles {
            assert_eq!(Rig::consume(h).await.0, StatusCode::OK);
        }
        rig.zero().await;
    }
    rig.assert_pairing();
    println!("PASS nonstream and 500-request soak: worker and routing-key counts zero; bootstrap room/host/selected P port paired");

    // All but one endpoint return the exact frozen model loads[] shape; the
    // remaining D retains the aggregate API, exercising mixed worker versions.
    for shape in rig.rank_shapes.iter().take(rig.rank_shapes.len() - 1) {
        shape.store(true, Ordering::SeqCst);
    }
    // The production AppContext constructor must use refresh=1 despite startup=30.
    let config = RouterConfig::builder()
        .regular_mode(vec!["http://unused".into()])
        .worker_startup_check_interval_secs(30)
        .load_refresh_interval_secs(1)
        .build()
        .unwrap();
    let context = AppContext::from_config(config, 10).await.unwrap();
    for w in &rig.workers {
        context.worker_registry.register(w.clone());
    }
    context
        .policy_registry
        .set_decode_policy(rig.router.policy_registry.get_decode_policy());
    let monitor = context.load_monitor.as_ref().unwrap();
    let samples = monitor.subscribe();
    monitor.start().await;
    until(|| samples.borrow().len() == rig.workers.len()).await;
    Rig::consume(rig.spawn("fresh-score", secret, true)).await;
    rig.zero().await;
    sleep(Duration::from_millis(550)).await;
    Rig::consume(rig.spawn("aged-score", secret, true)).await;
    rig.zero().await;
    let fresh = selection(&logs, "fresh-score", "decode");
    let aged = selection(&logs, "aged-score", "decode");
    assert_eq!(fresh["branch"], "token_score");
    for c in fresh["candidates"].as_array().unwrap() {
        assert!(c["score"].as_i64().unwrap() >= 0);
        assert!(c["sampled_at_unix_ms"].as_u64().unwrap() > 0);
    }
    for c in aged["candidates"].as_array().unwrap() {
        assert!(c["age_ms"].as_u64().unwrap() >= 500);
    }
    sleep(Duration::from_millis(2650)).await;
    monitor.stop().await;
    for poll in &rig.polls {
        let samples = poll.lock().unwrap();
        assert_eq!(samples.len(), 4, "1s monitor should poll at t=0,1,2,3");
        for (_, score) in samples.iter() {
            assert!(*score >= 0);
        }
        for pair in samples.windows(2) {
            let d = pair[1].0.duration_since(pair[0].0).as_secs_f64();
            assert!((0.85..1.15).contains(&d), "poll gap {d}");
        }
    }
    println!("PASS production monitor refresh=1s/startup=30s: every endpoint four GET /v1/loads?include=core at t=0,1,2,3; mixed frozen loads[]/aggregate payloads nonnegative; score age retained");

    // Preserve raw -1 evidence, but compare local reservations for BOTH candidates.
    let policy = rig.router.policy_registry.get_decode_policy();
    let decodes = rig.router.worker_registry.get_decode_workers();
    policy.update_loads(&HashMap::from([
        (decodes[0].url().to_string(), -1),
        (decodes[1].url().to_string(), 10),
    ]));
    Rig::consume(rig.spawn("negative-score", secret, true)).await;
    rig.zero().await;
    let neg = selection(&logs, "negative-score", "decode");
    assert_eq!(neg["branch"], "request_count_fallback");
    for c in neg["candidates"].as_array().unwrap() {
        assert_eq!(c["score_source"], "local_reservations");
        assert_eq!(c["score"], c["local_load"]);
        assert_eq!(c["fallback_reason"], "invalid_token_snapshot");
    }
    assert!(neg["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["token_score"] == -1));
    policy.update_loads(&HashMap::new());
    Rig::consume(rig.spawn("fallback-score", secret, true)).await;
    rig.zero().await;
    assert_eq!(
        selection(&logs, "fallback-score", "decode")["branch"],
        "request_count_fallback"
    );
    println!("PASS trace distinguishes cached -1, token-score age and missing-score request-count fallback; invalid snapshots use live reservations");

    let events = logs.events();
    assert!(!String::from_utf8(logs.0.lock().unwrap().clone())
        .unwrap()
        .contains(secret));
    let mut accounting = HashMap::<(String, String), (usize, usize)>::new();
    for v in &events {
        let key = (
            v["attempt_id"].as_str().unwrap().to_string(),
            v["role"].as_str().unwrap().to_string(),
        );
        if v["phase"] == "reserve" {
            accounting.entry(key).or_default().0 += 1;
            assert!(!v["bootstrap_room"].is_null());
        } else if v["phase"] == "release" {
            accounting.entry(key).or_default().1 += 1;
        }
    }
    assert!(
        accounting.values().all(|&a| a == (1, 1)),
        "exactly one reservation/release per role per attempt"
    );
    let p_release = release(&logs, "slow-decode", "prefill");
    let d_release = release(&logs, "slow-decode", "decode");
    assert!(p_release["elapsed_ms"].as_u64().unwrap() <= d_release["elapsed_ms"].as_u64().unwrap());
    println!("PASS score trace prompt privacy, request/attempt IDs, bootstrap and exactly-once reserve/release timestamps ({} role lifecycles)", accounting.len());

    let off = Rig::new(false, 1, 1, 64).await;
    Rig::consume(off.spawn("trace-off", secret, true)).await;
    off.zero().await;
    assert!(logs.request("trace-off").is_empty());
    let defaults = RouterConfig::default();
    assert_eq!(defaults.load_refresh_interval_secs, 30);
    assert!(!defaults.score_trace);
    let config = RouterConfig::builder()
        .regular_mode(vec!["http://unused".into()])
        .worker_startup_check_interval_secs(1)
        .build()
        .unwrap();
    assert_eq!(config.load_refresh_interval_secs, 30);
    let ctx = AppContext::from_config(config, 10).await.unwrap();
    for w in &off.workers {
        ctx.worker_registry.register(w.clone());
    }
    ctx.policy_registry
        .set_decode_policy(off.router.policy_registry.get_decode_policy());
    let default_monitor = ctx.load_monitor.as_ref().unwrap();
    default_monitor.start().await;
    until(|| off.polls.iter().all(|p| !p.lock().unwrap().is_empty())).await;
    sleep(Duration::from_millis(1150)).await;
    default_monitor.stop().await;
    assert!(
        off.polls.iter().all(|p| p.lock().unwrap().len() == 1),
        "default 30s must not follow startup=1s"
    );
    let mut old_json = serde_json::to_value(defaults).unwrap();
    old_json.as_object_mut().unwrap().remove("score_trace");
    old_json
        .as_object_mut()
        .unwrap()
        .remove("load_refresh_interval_secs");
    let restored: RouterConfig = serde_json::from_value(old_json).unwrap();
    assert_eq!(restored.load_refresh_interval_secs, 30);
    assert!(!restored.score_trace);
    assert!(RouterConfig::builder()
        .regular_mode(vec!["http://unused".into()])
        .load_refresh_interval_secs(0)
        .build()
        .is_err());
    println!("PASS default trace off/refresh30, startup independence, old JSON configuration compatibility and zero-interval rejection");
}
