"""Opt-in batching of independent P31 deep-prefix factor commits.

The caller owns the N-1/recurrent split. Only its deep emitters may use this
collector: shallow layers still publish before their recurrent token reads.
"""
import copy
import os


class DeepPrefixBatch:
    def __init__(self, pool, metadata, *, split_layer_limit=31):
        if not pool.batch_prefill or pool.cfg.init_method != 'k31':
            raise ValueError('deep prefix batch requires the native k31 batched pool')
        for flag in ('SGLANG_GDN_PSIDE_COMPOSITE', 'TWINSTAR_PD_EMITTER_GRAPH'):
            if os.environ.get(flag) == '1':
                raise ValueError('deep prefix batch has not admitted '+flag)
        if getattr(pool, '_exact_tail_transaction', None) is not None:
            raise ValueError('deep prefix batch cannot overlap an exact-tail transaction')
        if pool.prefill_factor_graph is not None:
            raise ValueError('deep prefix batch has not admitted a factor graph')
        self.pool = pool
        self.layers = tuple(lid for lid in pool.layer_ids if lid >= split_layer_limit)
        if not self.layers:
            raise ValueError('no deep layers in the factor pool')
        self.metadata = copy.copy(metadata)
        self.plan = copy.copy(metadata.factored_extend)
        if self.plan.pending or self.plan.batch_collector is not None or self.plan.checkpoint_group is not None:
            raise ValueError('deep prefix batch requires an uncollected prefix plan')
        self.plan.pending = []
        self.plan.preserve_layer_sink = True
        self.plan.next_layer = pool.layer_index(self.layers[0])
        self.plan.last_layer = pool.layer_index(self.layers[-1])
        self.metadata.factored_extend = self.plan
        self.final_src, self.final_dst = metadata.track_ssm_final_src, metadata.track_ssm_final_dst
        # An all-layer snapshot would overwrite shallow count-eight prefixes
        # with their already-appended count-nine live states.
        self.metadata.track_ssm_final_src = self.metadata.track_ssm_final_dst = None
        self.index = 0
        self.complete = False

    def before(self, layer_id):
        if self.complete or self.index >= len(self.layers) or layer_id != self.layers[self.index]:
            raise RuntimeError('deep prefix layers must arrive once in pool order')
        if self.plan.next_layer != self.pool.layer_index(layer_id):
            raise RuntimeError('deep prefix native plan did not advance')
        return self.metadata

    def after(self, layer_id, observer=None):
        if layer_id != self.layers[self.index] or self.plan.next_layer != self.pool.layer_index(layer_id)+1:
            raise RuntimeError('deep prefix native commit was not called')
        self.index += 1
        if self.index != len(self.layers):
            return
        self.pool.pside_join()
        if self.plan.pending:
            raise RuntimeError('deep prefix factors did not commit before publication')
        required = self.final_src is not None and self.final_src.numel() > 0
        if required:
            for lid in self.layers:
                self.pool.copy_slots_layer(lid, self.final_src, self.final_dst)
        self.complete = True
        if observer is not None:
            for _ in self.layers:
                observer(required)

    def finish(self):
        if not self.complete:
            raise RuntimeError('incomplete deep prefix batch; handoff prohibited')

    def abort(self):
        # Do not publish an incomplete prefix or let a later reader flush it.
        deferred = getattr(self.pool, '_pside_deferred_commit', None)
        if deferred is not None and deferred[1] is self.plan:
            self.pool._pside_deferred_commit = None
        self.plan.pending.clear()
