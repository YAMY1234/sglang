"""Accepted-input replay for r8/W8 verify transactions (stage2 #587).

Default-off owner; GPU admission is required before enabling in performance.
The original recurrence computes verify outputs and replays accepted inputs;
no candidate factor checkpoint is allocated. Persistent state stays unchanged
until commit. Conv, PLE, and KV retain their separate transaction owners.
"""
import json
import os
from pathlib import Path

import torch

from .gdn_factored_spec import FactoredGDNVerifyState


class FactoredGDNReplayState(FactoredGDNVerifyState):
    replay_inputs = True

    def __init__(self, pool, max_batch_size, draft_tokens, *, qkv_width,
                 input_dtype=torch.bfloat16, batched_commit=None, verify_window_fused=None,
                 snapshot_kernel=None, graph_commit=None):
        super().__init__(pool, max_batch_size, draft_tokens,
                         direct_checkpoints=False, _checkpoint_storage=False)
        if qkv_width < 1 or input_dtype != torch.bfloat16:
            raise ValueError('factor replay currently requires packed BF16 GDN inputs')
        layers, _, heads, _ = pool.a.shape
        shape = (layers, max_batch_size, draft_tokens)
        self.inputs = {
            'mixed': torch.zeros(*shape, qkv_width, dtype=input_dtype, device=pool.a.device),
            'a': torch.zeros(*shape, heads, dtype=input_dtype, device=pool.a.device),
            'b': torch.zeros(*shape, heads, dtype=input_dtype, device=pool.a.device),
        }
        self.layer_arguments = [None] * layers
        self.replay_indices = torch.full_like(self.work_indices, -1)
        self.batched_commit = (os.environ.get('SGLANG_GDN_VERIFY_REPLAY_BATCHED', '0') == '1'
                               if batched_commit is None else batched_commit)
        self.batched_constants = None
        self.verify_window_fused = (os.environ.get("SGLANG_GDN_VERIFY_WINDOW_FUSED", "0") == "1"
                                   if verify_window_fused is None else verify_window_fused)
        # Copy policy is independent of candidate-checkpoint storage. Replay
        # still owns no per-candidate factor checkpoints.
        self.snapshot_kernel = (os.environ.get('SGLANG_GDN_VERIFY_SNAPSHOT_KERNEL', '0') == '1'
                                if snapshot_kernel is None else snapshot_kernel)
        self.graph_commit = (os.environ.get('SGLANG_GDN_VERIFY_REPLAY_GRAPH', '0') == '1'
                             if graph_commit is None else graph_commit)
        if self.graph_commit and not self.batched_commit:
            raise ValueError('replay graph requires batched replay')
        self.commit_graphs = {}
        self.defer_cut = os.environ.get('SGLANG_GDN_VERIFY_DEFER_CUT', '0') == '1'
        self.record_fused = os.environ.get('SGLANG_GDN_VERIFY_RECORD_FUSED', '0') == '1'
        self.raw_append = os.environ.get('SGLANG_GDN_VERIFY_APPEND_RAW', '0') == '1'
        if self.raw_append and (not self.defer_cut or not self.verify_window_fused):
            raise ValueError('raw append is confined to deferred verification')
        self.commit_fused = os.environ.get('SGLANG_GDN_VERIFY_COMMIT_FUSED', '0') == '1'
        if self.commit_fused and (not self.defer_cut or not self.graph_commit):
            raise ValueError('fused commit requires deferred cut with graph replay')
        if self.record_fused and (not self.defer_cut or not self.verify_window_fused or
                os.environ.get('SGLANG_GDN_VERIFY_APPEND_RESIDENT','0')=='1'):
            raise ValueError('fused input record requires the deferred append window')
        # Explicit numerical-only diagnostic. CPU synchronization must never
        # be enabled in an event, profiler, formal or trace performance window.
        self.cadence_audit = os.environ.get('SGLANG_GDN_VERIFY_CADENCE_AUDIT')
        if self.defer_cut and (not self.batched_commit or not self.verify_window_fused or pool.U.shape[-2] != 32):
            raise ValueError('deferred cut requires batched replay, fused append and padded capacity 32')

    def bytes(self):
        return (super().bytes() + self.replay_indices.numel() * self.replay_indices.element_size()
                + sum(t.numel() * t.element_size() for t in self.inputs.values())
                + sum(t.numel() * t.element_size() for t in (self.batched_constants or {}).values())
                + sum(t.numel() * t.element_size() for entry in self.commit_graphs.values()
                      for t in entry['inputs']))

    def forward_layer(self, layer, mixed_qkv, a, b):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode

        tokens = self.draft_tokens
        batch = mixed_qkv.shape[0] // tokens
        if mixed_qkv.shape[0] != batch * tokens or not 0 < batch <= self.capacity:
            raise ValueError('factor replay requires fixed four-input verify rows')
        li = self.pool.layer_index(layer.layer_id)
        mixed = mixed_qkv.reshape(batch, tokens, -1)
        gates_a = a.reshape(batch, tokens, -1)
        gates_b = b.reshape(batch, tokens, -1)
        args = dict(A_log=layer.A_log, dt_bias=layer.dt_bias,
                    scale=layer.head_k_dim ** -.5, vbar=self.pool.vbar[li],
                    num_q_heads=layer.num_q_heads, num_v_heads=layer.num_v_heads,
                    head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
                    r=self.pool.cfg.r, rfull=self.pool.cfg.rfull,
                    truncate=True, **self.pool.cfg.kernel_kwargs())
        self.record_inputs(li, mixed, gates_a, gates_b, args)
        if self.verify_window_fused:
            from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_verify_window
            recording = None
            if self.record_fused:
                recording = {name: tensor[li, :batch] for name, tensor in self.inputs.items()}
                recording['written'] = self.written[li, :batch]
            output = factored_verify_window(mixed, gates_a, gates_b,
                fa=self.working['a'][li], fu=self.working['U'][li],
                fw=self.working['W'][li], fcount=self.working['count'][li],
                stale=self.stale, indices=self.work_indices[:batch], arguments=args, recording=recording)
            return output.reshape(1, batch * tokens, layer.num_v_heads, layer.head_v_dim)
        output = mixed_qkv.new_empty(batch, tokens, layer.num_v_heads, layer.head_v_dim)
        for step in range(tokens):
            out = factored_packed_decode(mixed[:, step], gates_a[:, step], gates_b[:, step],
                fa=self.working['a'][li], fu=self.working['U'][li],
                fw=self.working['W'][li], fcount=self.working['count'][li],
                stale=self.stale, ssm_state_indices=self.work_indices[:batch], **args)
            output[:, step].copy_(out[:, 0])
        return output.reshape(1, batch * tokens, layer.num_v_heads, layer.head_v_dim)

    def record_inputs(self, li, mixed, a, b, arguments):
        """Copy before model scratch can be reused; graph addresses stay fixed."""
        batch = mixed.shape[0]
        if not 0 <= li < len(self.layer_arguments) or not 0 < batch <= self.capacity:
            raise ValueError('invalid replay layer or batch')
        for name, source in (('mixed', mixed), ('a', a), ('b', b)):
            target = self.inputs[name][li, :batch]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError('replay must preserve raw input shape and precision')
            if not self.record_fused:
                target.copy_(source)
        previous = self.layer_arguments[li]
        if previous is not None:
            for name, value in arguments.items():
                old = previous[name]
                if isinstance(value, torch.Tensor):
                    if old.data_ptr() != value.data_ptr() or old.shape != value.shape:
                        raise ValueError('replay layer constants moved after graph setup')
                elif old != value:
                    raise ValueError('replay recurrence parameters changed')
        self.layer_arguments[li] = arguments
        if not self.record_fused:
            self.written[li, :batch].fill_(True)

    def _restore_entry(self, slots):
        if slots.is_cuda:
            from sglang.srt.layers.attention.linear.kernels.gdn_verify_io import snapshot_factors
            snapshot_factors(self.pool, self.working, slots)
        else:
            for name in self.names:
                self.working[name][:, :slots.numel()].copy_(getattr(self.pool, name).index_select(1, slots))

    def _publish_layer(self, li, slots, valid):
        """One working version; negative rows must not publish to padding slot 0."""
        steps = torch.where(valid, 0, -1).long()
        for name in self.names:
            self._scatter(getattr(self.pool, name)[li:li+1],
                          self.working[name][li:li+1].unsqueeze(2), slots, steps)

    def _publish_metadata(self, slots, valid):
        if self.meta_fused:
            from sglang.srt.layers.attention.linear.kernels.gdn_verify_meta import publish_metadata
            publish_metadata(self, slots, valid)
            return
        steps = torch.where(valid, 0, -1).long()
        for target, value in ((self.pool.stale, 1), (self.pool.dense_of, -1),
                              (self.pool.dense_required, 0), (self.pool.prefix_valid, 0)):
            if target is not None:
                self._scatter(target.view(1, -1, 1), self.constants[value], slots, steps)

    def _prepare_batched(self):
        if self.batched_constants is not None:
            return
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import DEFAULT_KERNEL, TRUNC_METHOD
        first = self.layer_arguments[0]
        if not (first['r'] == 8 and first['rfull'] == 16 and first.get('truncate') and
                first.get('post_order') and first.get('async_stream') is None and
                (first.get('kernel') or DEFAULT_KERNEL) == 'split' and TRUNC_METHOD == 'mgs'):
            raise ValueError('batched replay requires unchanged split post-order r8/W8 MGS')
        for li, args in enumerate(self.layer_arguments):
            for name, value in first.items():
                if isinstance(value, torch.Tensor):
                    if args[name].shape != value.shape or args[name].dtype != value.dtype:
                        raise ValueError('heterogeneous replay layer constants')
                elif args[name] != value:
                    raise ValueError('heterogeneous replay recurrence parameters')
            if args['vbar'].data_ptr() != self.pool.vbar[li].data_ptr():
                raise ValueError('replay vbar must match the owned layer pool')
        self.batched_constants = {name: torch.stack([a[name] for a in self.layer_arguments])
                                  for name in ('A_log', 'dt_bias')}

    def _publish_layers(self, slots, valid):
        from sglang.srt.layers.attention.linear.kernels.gdn_verify_io import publish_factors
        publish_factors(self.pool, self.working, slots, valid)

    def _commit_batched(self, slots, steps, track_slots, track_steps):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_replay_layers
        n = steps.numel()
        # A candidate cannot precede the previous candidate's W8 cut. Layers
        # are independent; batching them changes launch grouping, not math.
        for step in range(self.draft_tokens):
            self.replay_indices[:n].copy_(torch.where(steps >= step, self.row_ids[:n], -1))
            factored_packed_replay_layers(self.inputs['mixed'][:, :n, step],
                self.inputs['a'][:, :n, step], self.inputs['b'][:, :n, step],
                **self.batched_constants, vbar=self.pool.vbar, working=self.working,
                stale=self.stale, indices=self.replay_indices[:n], arguments=self.layer_arguments[0],
                deferred_cut=self.defer_cut)
            if track_slots is not None:
                self._publish_layers(track_slots, track_steps == step)
        if self.defer_cut:
            from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_expiry_truncate_layers
            args = self.layer_arguments[0]
            if track_slots is not None:
                # Each tracked prefix was copied at its accepted position.
                # Cut that prefix independently before publishing the final
                # accepted state, preserving aliases and publication order.
                tracked = torch.where(track_steps >= 0, track_slots, -1)
                factored_expiry_truncate_layers(self.pool.U, self.pool.W, self.pool.count,
                    tracked, args['r'], args['rfull'], deferred_cut=True,
                    trunc_warps=args.get('trunc_warps'), trunc_iters=args.get('trunc_iters'))
            factored_expiry_truncate_layers(self.working['U'], self.working['W'], self.working['count'],
                self.row_ids[:n], args['r'], args['rfull'], deferred_cut=True,
                trunc_warps=args.get('trunc_warps'), trunc_iters=args.get('trunc_iters'))
        self._publish_layers(slots, steps >= 0)

    def _commit_graph_body(self, slots, steps, track_slots, track_steps):
        """Original ordered recurrence/cuts and publication in fixed graph nodes."""
        if self.commit_fused and track_slots is None:
            from sglang.srt.layers.attention.linear.kernels.gdn_commit_window import factored_commit_window
            factored_commit_window(self.pool,self.working,self.inputs,self.batched_constants,
                self.stale,slots,self.row_ids[:slots.numel()],steps,self.layer_arguments[0])
        else:
            self._restore_entry(slots)
            self._commit_batched(slots, steps, track_slots, track_steps)
        if track_slots is not None:
            self._publish_metadata(track_slots, track_steps >= 0)
        self._publish_metadata(slots, steps >= 0)

    def _run_commit_graph(self, slots, steps, track_slots, track_steps):
        if not slots.is_cuda:
            # CPU interpretation checks the real body, not CUDA capture.
            self._commit_graph_body(slots, steps, track_slots, track_steps)
            return
        key = (slots.numel(), track_slots is not None)
        entry = self.commit_graphs.get(key)
        if entry is None:
            inputs = tuple(torch.full_like(slots, -1) for _ in range(4 if key[1] else 2))
            args = inputs if key[1] else (*inputs, None, None)
            stream = torch.cuda.Stream(device=slots.device)
            current = torch.cuda.current_stream(slots.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._commit_graph_body(*args)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                self._commit_graph_body(*args)
            current.wait_stream(stream)
            entry = dict(graph=graph, inputs=inputs)
            self.commit_graphs[key] = entry
        actual = (slots, steps, track_slots, track_steps) if key[1] else (slots, steps)
        for dst, src in zip(entry['inputs'], actual):
            dst.copy_(src)
        entry['graph'].replay()

    def commit(self, ticket, last_consumed_indices, *, track_slots=None, track_steps=None,
               _decode=None):
        """Replay exactly 1 + accepted drafts, including each original W8 cut."""
        if _decode is None:
            from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode
            _decode = factored_packed_decode
        steps = last_consumed_indices.long()
        self._validate(ticket, steps)
        if any(args is None for args in self.layer_arguments):
            raise RuntimeError('replay constants missing for a GDN layer')
        if track_slots is not None:
            if track_steps is None or track_slots.shape != steps.shape or track_steps.shape != steps.shape:
                raise ValueError('incomplete factor tracking coordinates')
            torch._assert_async(torch.all((track_steps < 0) | ((track_slots >= 0) &
                (track_slots < self.pool.a.shape[1]) & (track_steps <= steps))),
                'tracking beyond accepted prefix')
        elif track_steps is not None:
            raise ValueError('tracking steps without slots')
        if self.batched_commit:
            self._prepare_batched()
        audit = None
        if self.cadence_audit:
            audit = (self.pool.count[:, ticket.slots].detach().cpu(), steps.detach().cpu() + 1)
        if self.graph_commit:
            self._run_commit_graph(ticket.slots, steps, track_slots, track_steps)
            if audit is not None:
                self._record_cadence(ticket, audit)
            self.invalidate_slots(ticket.slots)
            ticket.closed = True
            return
        # Restore every layer before publishing anything, including track rows.
        self._restore_entry(ticket.slots)
        n = steps.numel()
        if self.batched_commit:
            self._commit_batched(ticket.slots, steps, track_slots, track_steps)
        else:
            for li, arguments in enumerate(self.layer_arguments):
                for step in range(self.draft_tokens):
                    self.replay_indices[:n].copy_(torch.where(steps >= step, self.row_ids[:n], -1))
                    _decode(self.inputs['mixed'][li, :n, step],
                            self.inputs['a'][li, :n, step], self.inputs['b'][li, :n, step],
                            fa=self.working['a'][li], fu=self.working['U'][li],
                            fw=self.working['W'][li], fcount=self.working['count'][li],
                            stale=self.stale, ssm_state_indices=self.replay_indices[:n], **arguments)
                    if track_slots is not None:
                        self._publish_layer(li, track_slots, track_steps == step)
                self._publish_layer(li, ticket.slots, steps >= 0)
        if track_slots is not None:
            self._publish_metadata(track_slots, track_steps >= 0)
        self._publish_metadata(ticket.slots, steps >= 0)
        if audit is not None:
            self._record_cadence(ticket, audit)
        self.invalidate_slots(ticket.slots)
        ticket.closed = True

    def _record_cadence(self, ticket, audit):
        before, accepted = audit
        after = self.pool.count[:, ticket.slots].detach().cpu()
        expected = 8 + (before - 8 + accepted[None, :, None]) % 8
        if not torch.equal(after, expected):
            raise RuntimeError('accepted-token W8 cadence differs from committed counts')
        if not torch.all(before == before[:1, :, :1]) or not torch.all(after == after[:1, :, :1]):
            raise RuntimeError('W8 cadence differs across layers or heads')
        cuts = (before[0, :, 0] - 8 + accepted) // 8
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = Path(self.cadence_audit) / f'rank{rank}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(dict(epoch=ticket.epoch, accepted=accepted.tolist(),
                before=before[0, :, 0].tolist(), after=after[0, :, 0].tolist(), cuts=cuts.tolist(),
                deferred=self.defer_cut, numerical_only=True)) + '\n')
