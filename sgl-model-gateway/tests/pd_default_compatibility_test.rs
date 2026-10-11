//! Identical zero-GPU test runs against the frozen and patched gateway.
//! Baseline: RGUARD_EXPECT_LATE_GUARD=1 cargo test --test pd_default_compatibility_test -- --nocapture
//! Patched: cargo test --test pd_default_compatibility_test -- --nocapture
include!("common/pd_load_stub.rs");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn default_behavior_compatibility_except_guard_lifetime() {
    let baseline = std::env::var_os("RGUARD_EXPECT_LATE_GUARD").is_some();
    let rig = Rig::new(false, 1, 1, 64).await;
    let text = "default compatibility prompt";
    let (status, body) = Rig::consume(rig.spawn("default-stream", text, true)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body.as_ref(),
        b"data: {\"text\":\"stub-token\"}\n\ndata: [DONE]\n\n"
    );
    rig.zero().await;
    let (status, body) = Rig::consume(rig.spawn("default-nonstream", text, false)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["text"],
        "stub-token"
    );
    rig.zero().await;

    rig.p_gate.send_replace(false);
    rig.d_gate.send_replace(false);
    let pending = rig.spawn("default-lifetime", text, true);
    until(|| {
        rig.receipts
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r["id"] == "default-lifetime")
            .count()
            == 2
    })
    .await;
    assert_eq!(
        rig.local_loads(),
        if baseline { vec![0, 0] } else { vec![1, 1] }
    );
    rig.p_gate.send_replace(true);
    let response = timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        rig.local_loads(),
        if baseline { vec![1, 1] } else { vec![0, 1] }
    );
    rig.d_gate.send_replace(true);
    assert!(axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap()
        .ends_with(b"data: [DONE]\n\n"));
    rig.zero().await;
    for (id, text) in [
        ("default-p-error", "P_HTTP_ERROR"),
        ("default-d-error", "D_HTTP_ERROR"),
    ] {
        assert_eq!(
            Rig::consume(rig.spawn(id, text, true)).await.0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        rig.zero().await;
    }
    rig.d_gate.send_replace(false);
    let response = rig.spawn("default-cancel", text, true).await.unwrap();
    drop(response);
    rig.zero().await;
    rig.d_gate.send_replace(true);
    for i in 0..50 {
        Rig::consume(rig.spawn(&format!("default-soak-{i}"), text, true)).await;
    }
    rig.zero().await;
    rig.assert_pairing();

    let config = RouterConfig::default();
    assert_eq!(config.worker_startup_check_interval_secs, 30);
    let context = AppContext::from_config(config, 10).await.unwrap();
    for w in &rig.workers {
        context.worker_registry.register(w.clone());
    }
    context
        .policy_registry
        .set_decode_policy(rig.router.policy_registry.get_decode_policy());
    let monitor = context.load_monitor.as_ref().unwrap();
    monitor.start().await;
    until(|| rig.polls.iter().all(|p| !p.lock().unwrap().is_empty())).await;
    sleep(Duration::from_millis(30_100)).await;
    monitor.stop().await;
    for polls in &rig.polls {
        let pp = polls.lock().unwrap();
        assert_eq!(pp.len(), 2);
        let gap = pp[1].0.duration_since(pp[0].0).as_secs_f64();
        assert!((29.8..30.2).contains(&gap));
        assert!(pp.iter().all(|(_, n)| *n >= 0));
    }
    println!("PASS {}: identical stream/nonstream payloads, P/D error statuses, cancellation, bootstrap pairing, 50-request zero counts and default 30s load polls; only guard lifetime differs", if baseline { "frozen455e7a" } else { "patched" });
}
