from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    LogitsProcessorOutput,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker


class EAGLEDraftExtendCudaGraphRunner:
    def __init__(self, eagle_worker: EAGLEWorker):
        # Parse args
        self.eagle_worker = eagle_worker
        if not hasattr(eagle_worker, "model_runner"):
            # V2: EagleDraftWorker
            self.model_runner = model_runner = eagle_worker.draft_runner
        else:
            self.model_runner = model_runner = eagle_worker.model_runner

        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.padded_static_len = -1
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # Attention backend
        self.num_tokens_per_bs = self.speculative_num_steps + 1
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.eagle_worker.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        self.seq_len_fill_value = (
            self.eagle_worker.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.ones((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros(
                (3, self.max_num_token), dtype=torch.int64
            )

            if self.eagle_worker.speculative_algorithm.is_eagle3():
                self.hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        (
                            self.model_runner.model_config.hf_config.target_hidden_size
                            * 3
                            if hasattr(
                                self.model_runner.model_config.hf_config,
                                "target_hidden_size",
                            )
                            else self.model_runner.model_config.hidden_size * 3
                        ),
                    ),
                    dtype=self.model_runner.dtype,
                )
            else:
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            self.seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            self.accept_length = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                self.global_num_tokens_gpu = None
                self.global_num_tokens_for_logprob_gpu = None

            if hasattr(
                self.model_runner.model_config.hf_config, "draft_vocab_size"
            ):  # llama_eagle
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(
                self.model_runner.model_config.hf_config, "hot_vocab_size"
            ):  # llama_eagle3
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size

            self.next_token_logits_buffer = torch.zeros(
                (self.max_bs, vocab_size),
                dtype=torch.float,
            )

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.seq_lens.numel()

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        return is_bs_supported

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        seq_lens_cpu = self.seq_lens_cpu[:bs]
        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        accept_length = self.accept_length[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        mrope_positions = self.mrope_positions[:, :num_tokens]
        hidden_states = self.hidden_states[:num_tokens]
        next_token_logits_buffer = self.next_token_logits_buffer[:bs]

        if self.require_mlp_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs] * self.dp_size,
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs],
                    dtype=torch.int32,
                    device=self.input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            accept_length=accept_length,
        )
        spec_info.positions = None

        self.deepep_adapter.capture(is_extend_in_batch=True)

        # Forward batch
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            attn_backend=self.eagle_worker.draft_extend_attn_backend,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            padded_static_len=self.padded_static_len,
        )

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=spec_info,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            # Backup two fields, which will be modified in-place in `draft_forward`.
            output_cache_loc_backup = forward_batch.out_cache_loc
            hidden_states_backup = forward_batch.spec_info.hidden_states

            ret = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            probs = torch.softmax(ret.next_token_logits, dim=-1)
            ret.topk_p, ret.topk_index = fast_topk(probs, self.topk, dim=-1)

            forward_batch.out_cache_loc = output_cache_loc_backup
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

        with torch.cuda.graph(
            graph, pool=get_global_graph_memory_pool(), stream=stream
        ):
            out = run_once()

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def replay(self, forward_batch: ForwardBatch):
        assert forward_batch.out_cache_loc is not None
        self.deepep_adapter.replay()

        # batch_size and num_seqs can be different in case there are finished examples
        # in the batch, which will not be counted as num_seqs
        raw_bs = forward_batch.batch_size
        num_tokens = forward_batch.input_ids.shape[0]
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = (
                max_num_tokens // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max_num_tokens
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        if bs * self.num_tokens_per_bs != num_tokens:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.positions.zero_()
            self.accept_length.fill_(1)
            self.extend_seq_lens.fill_(1)

        # Common inputs
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.extend_seq_lens is not None:
            self.extend_seq_lens[:raw_bs].copy_(forward_batch.extend_seq_lens)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.positions[:num_tokens].copy_(forward_batch.positions)
        if (
            forward_batch.spec_info.hidden_states.shape[1]
            == self.hidden_states.shape[1]
        ):
            self.hidden_states[:num_tokens].copy_(forward_batch.spec_info.hidden_states)
        if forward_batch.spec_info.accept_length is not None:
            self.accept_length[:raw_bs].copy_(forward_batch.spec_info.accept_length)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)

        # TODO(ch-wan): support num_token_non_padded
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs)

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)

        if forward_batch.extend_seq_lens_cpu is not None:
            self.extend_seq_lens_cpu[:raw_bs] = forward_batch.extend_seq_lens_cpu

        if bs != raw_bs:
            forward_batch.spec_info.positions = self.positions[:num_tokens]
            forward_batch.spec_info.accept_length = self.accept_length[:bs]

        self.eagle_worker.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum
            + (bs - raw_bs) * self.seq_len_fill_value,
            encoder_lens=None,
            forward_mode=ForwardMode.DRAFT_EXTEND,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # ===== DEBUG ASSERT 8: 检查 CUDA graph replay 前的状态 (draft_extend) =====
        import os
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        # 初始化计数器（使用类属性，在所有实例间共享）
        if not hasattr(self.__class__, '_replay_call_count'):
            self.__class__._replay_call_count = 0
            self.__class__._last_states = {}  # 保存最近的状态，用于崩溃分析
        
        self.__class__._replay_call_count += 1
        current_count = self.__class__._replay_call_count
        
        # 先执行所有检查
        check_passed = True
        error_msg = None
        try:
            # 检查 bs 是否在合法范围
            assert bs in self.graphs, f"bs {bs} not in captured graphs: {list(self.graphs.keys())}"
            assert raw_bs <= bs, f"raw_bs ({raw_bs}) should not exceed bs ({bs})"
            
            # 检查关键张量
            assert self.input_ids is not None, "input_ids is None"
            assert self.positions is not None, "positions is None"
            assert self.req_pool_indices is not None, "req_pool_indices is None"
            assert self.seq_lens is not None, "seq_lens is None"
            
            # 检查张量形状和设备
            if hasattr(self.input_ids, 'shape') and self.input_ids.numel() > 0:
                assert self.input_ids.device.type == 'cuda', f"input_ids on wrong device: {self.input_ids.device}"
            if hasattr(self.positions, 'shape') and self.positions.numel() > 0:
                assert self.positions.device.type == 'cuda', f"positions on wrong device: {self.positions.device}"
            
            # 检查 spec_info
            if forward_batch.spec_info is not None:
                spec_info = forward_batch.spec_info
                if hasattr(spec_info, 'verified_id') and spec_info.verified_id is not None:
                    assert spec_info.verified_id.numel() == raw_bs, \
                        f"verified_id size mismatch: {spec_info.verified_id.numel()} != {raw_bs}"
                if hasattr(spec_info, 'hidden_states') and spec_info.hidden_states is not None:
                    assert spec_info.hidden_states.shape[0] == raw_bs, \
                        f"hidden_states size mismatch: {spec_info.hidden_states.shape[0]} != {raw_bs}"
        except Exception as e:
            check_passed = False
            error_msg = str(e)
        
        # 收集当前状态信息（轻量级）
        current_state = {
            'count': current_count,
            'bs': bs,
            'raw_bs': raw_bs,
            'check_passed': check_passed,
            'error_msg': error_msg,
            'input_ids_ptr': self.input_ids.data_ptr() if hasattr(self.input_ids, 'data_ptr') else None,
            'positions_ptr': self.positions.data_ptr() if hasattr(self.positions, 'data_ptr') else None,
            'seq_lens_sum': forward_batch.seq_lens_sum,
        }
        
        # 保存最近10次调用的状态（环形缓冲）
        state_key = f"bs_{bs}"
        if state_key not in self.__class__._last_states:
            self.__class__._last_states[state_key] = []
        self.__class__._last_states[state_key].append(current_state)
        if len(self.__class__._last_states[state_key]) > 10:
            self.__class__._last_states[state_key].pop(0)
        
        # 如果检查失败，立即记录详细日志
        if not check_passed:
            debug_dir = "/sgl-workspace/files/mtp_debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"eagle_draft_extend_error_count{current_count}.txt")
            
            with open(debug_file, "w") as f:
                import time
                f.write(f"=== EAGLE DRAFT EXTEND CUDA GRAPH REPLAY ERROR ===\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call count: {current_count}\n")
                f.write(f"Error: {error_msg}\n\n")
                f.write(f"Current state:\n")
                f.write(f"  bs: {bs}, raw_bs: {raw_bs}\n")
                f.write(f"  Available graph bs: {list(self.graphs.keys())}\n")
                f.write(f"  forward_batch.seq_lens_sum: {forward_batch.seq_lens_sum}\n")
                f.write(f"\nDetailed tensor info:\n")
                if hasattr(self.input_ids, 'shape'):
                    f.write(f"  input_ids: shape={self.input_ids.shape}, device={self.input_ids.device}, ptr={self.input_ids.data_ptr()}\n")
                if hasattr(self.positions, 'shape'):
                    f.write(f"  positions: shape={self.positions.shape}, device={self.positions.device}, ptr={self.positions.data_ptr()}\n")
                if hasattr(self.seq_lens, 'shape'):
                    f.write(f"  seq_lens: shape={self.seq_lens.shape}, values={self.seq_lens.tolist() if self.seq_lens.numel() < 20 else 'too large'}\n")
                
                # 记录最近10次调用的历史
                f.write(f"\nLast 10 calls for bs={bs}:\n")
                for i, state in enumerate(self.__class__._last_states.get(state_key, [])):
                    f.write(f"  [{i}] count={state['count']}, bs={state['bs']}, raw_bs={state['raw_bs']}, "
                           f"ptr_changed={state['input_ids_ptr'] != current_state['input_ids_ptr']}\n")
            
            logger.error(f"EAGLE draft extend check FAILED at call #{current_count}! Details: {debug_file}")
            raise RuntimeError(error_msg)

        # Replay (with error capture)
        try:
            self.graphs[bs].replay()
        except Exception as e:
            # Replay 崩溃了，记录状态
            debug_dir = "/sgl-workspace/files/mtp_debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"eagle_draft_extend_replay_crash_count{current_count}.txt")
            
            with open(debug_file, "w") as f:
                import time
                f.write(f"=== EAGLE DRAFT EXTEND CUDA GRAPH REPLAY CRASH ===\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call count: {current_count}\n")
                f.write(f"Exception: {type(e).__name__}: {str(e)}\n\n")
                f.write(f"Current state at crash:\n")
                f.write(f"  bs: {bs}, raw_bs: {raw_bs}\n")
                f.write(f"  seq_lens_sum: {forward_batch.seq_lens_sum}\n")
                f.write(f"  input_ids_ptr: 0x{current_state['input_ids_ptr']:x}\n")
                f.write(f"  positions_ptr: 0x{current_state['positions_ptr']:x}\n")
                
                # 记录最近10次调用历史
                f.write(f"\nLast 10 successful calls for bs={bs}:\n")
                for i, state in enumerate(self.__class__._last_states.get(state_key, [])):
                    f.write(f"  [{i}] count={state['count']}, raw_bs={state['raw_bs']}, "
                           f"seq_lens_sum={state['seq_lens_sum']}, "
                           f"ptr_changed={state['input_ids_ptr'] != current_state['input_ids_ptr']}\n")
                
                # GPU 内存信息
                try:
                    f.write(f"\nGPU Memory:\n")
                    f.write(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")
                    f.write(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\n")
                except:
                    pass
            
            logger.error(f"EAGLE draft extend CUDA graph replay CRASHED at call #{current_count}! Details: {debug_file}")
            raise  # 重新抛出原始异常
        out = self.output_buffers[bs]
        if bs != raw_bs:
            forward_batch.spec_info.accept_length = self.accept_length[:raw_bs]
            out_copy = out
            out = LogitsProcessorOutput(
                next_token_logits=out.next_token_logits[:raw_bs],
                hidden_states=out.hidden_states[:raw_bs],
            )
            out.topk_p = out_copy.topk_p[:raw_bs]
            out.topk_index = out_copy.topk_index[:raw_bs]
        return out
