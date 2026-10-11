//! Production worker manager/monitor consuming both real HTTP payload shapes.
include!("common/pd_load_stub.rs");
use smg::core::{LoadMonitor, WorkerManager};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mixed_aggregate_and_frozen_rank_payloads_drive_decode_selection() {
    let rig = Rig::new(false, 1, 2, 64).await;
    rig.rank_shapes[1].store(true, Ordering::SeqCst);
    rig.scores[1].store(60, Ordering::SeqCst);
    rig.scores[2].store(17, Ordering::SeqCst);
    let result =
        WorkerManager::get_all_worker_loads(&rig.router.worker_registry, &rig.router.client).await;
    let actual: HashMap<_, _> = result
        .loads
        .into_iter()
        .map(|v| (v.worker, v.load))
        .collect();
    assert_eq!(actual[rig.workers[0].url()], 0);
    assert_eq!(actual[rig.workers[1].url()], 60);
    assert_eq!(actual[rig.workers[2].url()], 17);
    assert_eq!(result.successful, 3);
    assert_eq!(result.failed, 0);
    println!("PASS real HTTP GET: frozen model loads[] sums to 60, legacy aggregate gives 17, P zero accepted; all scores nonnegative");

    let monitor = LoadMonitor::new(
        rig.router.worker_registry.clone(),
        rig.router.policy_registry.clone(),
        rig.router.client.clone(),
        1,
    );
    let cached = monitor.subscribe();
    monitor.start().await;
    until(|| cached.borrow().get(rig.workers[1].url()) == Some(&60)).await;
    for i in 0..20 {
        let id = format!("rank-heavy-{i}");
        Rig::consume(rig.spawn(&id, "http payload selection", true)).await;
        let receipts = rig.receipts.lock().unwrap();
        let d = receipts
            .iter()
            .find(|r| r["id"] == id && r["role"] == "decode")
            .unwrap();
        assert_eq!(
            d["name"],
            rig.workers[2].url(),
            "60-token ranks must lose to 17-token aggregate; old parser chose -1 incorrectly"
        );
    }
    rig.zero().await;
    rig.scores[1].store(3, Ordering::SeqCst);
    rig.scores[2].store(70, Ordering::SeqCst);
    until(|| {
        cached.borrow().get(rig.workers[1].url()) == Some(&3)
            && cached.borrow().get(rig.workers[2].url()) == Some(&70)
    })
    .await;
    for i in 0..20 {
        let id = format!("rank-light-{i}");
        Rig::consume(rig.spawn(&id, "http payload selection", true)).await;
        let receipts = rig.receipts.lock().unwrap();
        let d = receipts
            .iter()
            .find(|r| r["id"] == id && r["role"] == "decode")
            .unwrap();
        assert_eq!(
            d["name"],
            rig.workers[1].url(),
            "updated 3-token ranks must win against 70-token aggregate"
        );
    }
    rig.zero().await;
    monitor.stop().await;
    rig.assert_pairing();
    println!("PASS production monitor and power_of_two: 20/20 choose aggregate17 over ranks60; after refresh 20/20 choose ranks3 over aggregate70");

    rig.scores[1].store(-1, Ordering::SeqCst);
    let invalid =
        WorkerManager::get_all_worker_loads(&rig.router.worker_registry, &rig.router.client).await;
    assert_eq!(invalid.failed, 1);
    assert_eq!(invalid.successful, 2);
    assert_eq!(
        invalid
            .loads
            .iter()
            .find(|v| v.worker == rig.workers[1].url())
            .unwrap()
            .load,
        -1
    );
    println!("PASS malformed HTTP payload retains invalid sentinel -1 rather than partial/nonnegative fabrication");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn empty_and_missing_http_snapshots_use_live_reservations_and_recover() {
    let rig = Rig::new(true, 1, 2, 64).await;
    rig.scores[1].store(-2, Ordering::SeqCst); // real GPU shape: loads: []
    rig.scores[2].store(-1, Ordering::SeqCst); // missing field
    let monitor = LoadMonitor::new(
        rig.router.worker_registry.clone(),
        rig.router.policy_registry.clone(),
        rig.router.client.clone(),
        1,
    );
    let cached = monitor.subscribe();
    monitor.start().await;
    until(|| {
        cached.borrow().get(rig.workers[1].url()) == Some(&-1)
            && cached.borrow().get(rig.workers[2].url()) == Some(&-1)
    })
    .await;
    // Hold D0 reservations beyond a refresh: the choice must read the current
    // guard count, not a count captured in the HTTP monitoring batch.
    let guards: Vec<_> = (0..9)
        .map(|_| smg::core::WorkerLoadGuard::new(rig.workers[1].clone(), None))
        .collect();
    for i in 0..20 {
        let id = format!("empty-fallback-{i}");
        Rig::consume(rig.spawn(&id, "empty snapshot", true)).await;
        assert_eq!(
            rig.receipts
                .lock()
                .unwrap()
                .iter()
                .find(|r| r["id"] == id && r["role"] == "decode")
                .unwrap()["name"],
            rig.workers[2].url()
        );
    }
    drop(guards);
    rig.zero().await;
    // Once both reports are valid again, resume token scoring, including zero.
    rig.scores[1].store(0, Ordering::SeqCst);
    rig.scores[2].store(100, Ordering::SeqCst);
    until(|| {
        cached.borrow().get(rig.workers[1].url()) == Some(&0)
            && cached.borrow().get(rig.workers[2].url()) == Some(&100)
    })
    .await;
    Rig::consume(rig.spawn("token-recovery", "recovery", true)).await;
    assert_eq!(
        rig.receipts
            .lock()
            .unwrap()
            .iter()
            .find(|r| r["id"] == "token-recovery" && r["role"] == "decode")
            .unwrap()["name"],
        rig.workers[1].url()
    );
    rig.zero().await;
    monitor.stop().await;
    rig.assert_pairing();
    println!("PASS actual HTTP loads:[]/missing field -> 20/20 choose free D using live reservations; valid zero/100 refresh resumes token scoring");
}
