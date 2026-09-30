"""#151: execute each missing component producer and its real release consumer."""

import ast
import dataclasses
import logging
import time
import types
import unittest
from enum import Enum

import torch
from test_pp_commit_integration import ROOT, Cluster, load, method

component_module = load(
    "sglang.srt.mem_cache.unified_cache.component_type",
    "mem_cache/unified_cache/component_type.py",
)
actions = load(
    "sglang.srt.mem_cache.unified_cache.cache_action",
    "mem_cache/unified_cache/cache_action.py",
)
CT = component_module.ComponentType


class Pools(str, Enum):
    MAMBA = "mamba"
    SWA = "swa"

    def __str__(self):
        return self.value


class Phase(Enum):
    BACKUP_HOST = 1
    LOAD_BACK = 2
    PREFETCH = 3


# Extract the checked-in dataclass, not a substitute release envelope.
tree = ast.parse((ROOT / "mem_cache/hicache_storage.py").read_text())
nodes = [
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef)
    and n.name in ("PoolName", "PoolHitPolicy", "PoolTransfer")
]
ns = {"dataclass": dataclasses.dataclass, "Enum": Enum, "__name__": __name__}
exec(  # noqa: S102 -- checked-in dataclass extraction, no external code
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    *nodes,
                ],
                type_ignores=[],
            )
        ),
        "hicache_storage.py",
        "exec",
    ),
    ns,
)
Transfer = ns["PoolTransfer"]


def production(file, name):
    f = method("mem_cache/" + file, name)
    f.__globals__.update(
        ComponentType=CT,
        CacheTransferPhase=Phase,
        PoolName=Pools,
        PoolTransfer=Transfer,
        FreeComponentHostSlot=actions.FreeComponentHostSlot,
        FreeComponentDeviceSlot=actions.FreeComponentDeviceSlot,
        MambaEvictExcessPathStates=type("MambaEvictExcessPathStates", (), {}),
        logger=logging.getLogger(__name__),
    )
    return f


MAMBA = "unified_cache/components/mamba_component.py"
SWA = "unified_cache/components/swa_component.py"


def consumer(cluster, rank, pool):
    c = cluster.caches[rank]
    cc = c.cache_controller
    cc.mem_pool_host.entry_map[str(pool)] = types.SimpleNamespace(
        host_pool=types.SimpleNamespace(get_size_per_token=lambda: 8)
    )
    pending = []

    def append(extra_pools, commit_key=None):
        for transfer in extra_pools:
            cc._append_host_mem_release_pages(
                cc.host_mem_release_queue,
                transfer.host_indices,
                1,
                commit_key=commit_key,
                pool=str(transfer.name),
            )
        pending.append(commit_key)

    cc.append_host_mem_release = append
    c.token_to_kv_pool_allocator = object()
    comp = types.SimpleNamespace(cache=c)
    f = production(MAMBA if pool == Pools.MAMBA else SWA, "apply_component_action")
    return lambda action: f(comp, action), pending


class ProducerTest(unittest.TestCase):
    def commit_actions(self, per_rank, pool, origin):
        cluster = Cluster()
        identities = []
        for rank, acts in enumerate(per_rank):
            apply, _ = consumer(cluster, rank, pool)
            for action in acts:
                apply(action)
            cc = cluster.caches[rank].cache_controller
            identities.append(tuple(cc.pp_commit_bridge.issued_releases))
            while not cc.host_mem_release_queue.empty():
                cc.pp_commit_bridge.stage_release(
                    cc.host_mem_release_queue.get_nowait()
                )
            self.assertEqual(cc.mem_pool_host.freed, [])
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(identities[0], identities[2])
        cluster.tick()
        cluster.tick()
        for c in cluster.caches:
            b = c._pp_commit
            self.assertTrue(c.cache_controller.mem_pool_host.freed)
            self.assertEqual(
                b.committed_by_kind["release:" + str(pool)], len(per_rank[0])
            )
            self.assertEqual(b.releases_by_origin[origin], len(per_rank[0]))
            self.assertEqual(b.issued_releases, {})
        return cluster

    def mamba_actions(self, rank, reason="existing", parent="req-1"):
        ct = CT.MAMBA
        node = types.SimpleNamespace(
            component_data={
                ct: types.SimpleNamespace(
                    host_value=torch.tensor([999]) if reason == "existing" else None
                )
            }
        )
        comp = types.SimpleNamespace(
            component_type=ct,
            tree_core=types.SimpleNamespace(node_by_id=lambda _: node),
        )
        transfer = Transfer(
            name=Pools.MAMBA,
            host_indices=torch.tensor([10 * rank]),
            keys=["content-key"],
            pp_commit_parent=parent,
        )
        result = types.SimpleNamespace(
            inserted_host_node=None if reason == "no_target" else rank + 70,
            mamba_exist=False,
        )
        acts = []
        production(MAMBA, "commit_hicache_transfer")(
            comp,
            node,
            Phase.PREFETCH,
            [transfer],
            cache_actions=acts,
            insert_result=result,
            pool_storage_result=types.SimpleNamespace(
                extra_pool_hit_pages={Pools.MAMBA: 0 if reason == "not_loaded" else 1}
            ),
        )
        self.assertTrue(result.mamba_exist)
        return acts

    def test_r8_mamba_unused_prefetch_slot_has_shared_parent_and_common_free(self):
        for reason in ("existing", "not_loaded", "no_target"):
            with self.subTest(reason=reason):
                self.commit_actions(
                    [self.mamba_actions(r, reason) for r in range(3)],
                    Pools.MAMBA,
                    "mamba_prefetch_unused",
                )

    def swa_actions(self, rank, reason):
        root = object()
        anchor = root if reason in ("missing", "existing", "prefix") else object()
        host = torch.tensor([100 * rank + i for i in range(4)])
        target = types.SimpleNamespace(
            component_data={
                CT.SWA: types.SimpleNamespace(host_value=torch.tensor([9]))
            },
            key=[0] * 4,
            parent=anchor,
        )
        if reason == "prefix":
            target = anchor  # Entire loaded buffer lies before the anchor boundary.
        tree = types.SimpleNamespace(
            root_node=root, page_size=1, node_by_id=lambda _: target
        )
        comp = types.SimpleNamespace(
            component_type=CT.SWA,
            tree_core=tree,
            full_window_pages=5 if reason == "short" else 4,
        )
        comp._release_swa_host = types.MethodType(
            production(SWA, "_release_swa_host"), comp
        )
        acts = []
        production(SWA, "_commit_prefetch")(
            comp,
            anchor,
            [
                Transfer(
                    name=Pools.SWA,
                    host_indices=host,
                    keys=["k0", "k1", "k2", "k3"],
                    pp_commit_parent="req-swa",
                )
            ],
            cache_actions=acts,
            insert_result=types.SimpleNamespace(inserted_host_node=1, total_len=4),
            pool_storage_result=types.SimpleNamespace(
                extra_pool_hit_pages={Pools.SWA: 0 if reason == "missing" else 4}
            ),
        )
        return acts

    def test_r9a_short_swa_window(self):
        self.commit_actions(
            [self.swa_actions(r, "short") for r in range(3)],
            Pools.SWA,
            "swa_prefetch_unused",
        )

    def test_r9b_swa_missing_pool_hit(self):
        self.commit_actions(
            [self.swa_actions(r, "missing") for r in range(3)],
            Pools.SWA,
            "swa_prefetch_unused",
        )

    def test_r9c_swa_existing_overlap(self):
        self.commit_actions(
            [self.swa_actions(r, "existing") for r in range(3)],
            Pools.SWA,
            "swa_prefetch_unused",
        )

    def test_r9d_swa_prefix_outside_anchor(self):
        self.commit_actions(
            [self.swa_actions(r, "prefix") for r in range(3)],
            Pools.SWA,
            "swa_prefetch_unused",
        )

    def test_r10_rehydrate_unattached_slot_has_request_and_content_identity(self):
        per_rank = []
        for rank in range(3):
            acts = []
            c = types.SimpleNamespace(
                _pp_commit=object(),
                _prefetch_outcome_stats={},
                ongoing_rehydrate={"rid": object()},
                tree_core=types.SimpleNamespace(node_by_id=lambda _: None),
                _apply_cache_actions=acts.extend,
                dec_host_lock_ref=lambda *args: None,
            )
            op = types.SimpleNamespace(
                request_id="rid",
                node_id=rank + 500,
                slot=torch.tensor([rank + 300]),
                transfer=Transfer(name=Pools.MAMBA, keys=["same-rehydrate-key"]),
                lock_params=None,
                ok=0,
                believed=0,
                children=0,
                start_time=time.monotonic(),
            )
            production("unified_radix_cache.py", "_finish_mamba_rehydrate")(c, op)
            self.assertEqual(c.ongoing_rehydrate, {})
            self.assertEqual(c._prefetch_outcome_stats["mamba_rehydrate_fail"], 1)
            per_rank.append(acts)
        self.commit_actions(per_rank, Pools.MAMBA, "mamba_rehydrate_unused")

    def test_missing_parent_still_rejected_and_flag_off_still_frees(self):
        acts = self.mamba_actions(0, parent=None)
        self.assertIsNone(acts[0].commit_key)
        cluster = Cluster()
        apply, _ = consumer(cluster, 0, Pools.MAMBA)
        with self.assertRaisesRegex(RuntimeError, "lacks logical parent"):
            apply(acts[0])
        disabled = Cluster(enabled=False)
        apply, _ = consumer(disabled, 0, Pools.MAMBA)
        apply(acts[0])
        item = disabled.caches[0].cache_controller.host_mem_release_queue.get_nowait()
        self.assertTrue(torch.equal(item, torch.tensor([0])))

    def test_distinct_logical_ranges_do_not_alias(self):
        bridge = Cluster().caches[0]._pp_commit
        a = bridge.note_release(
            ("req", "swa_prefetch_unused", ("k",), 0, 4), Pools.SWA, torch.arange(4), 1
        )
        b = bridge.note_release(
            ("req", "swa_prefetch_unused", ("k",), 4, 8), Pools.SWA, torch.arange(4), 1
        )
        self.assertNotEqual(a.key, b.key)
        self.assertEqual(a.generation, b.generation)


if __name__ == "__main__":
    unittest.main()
