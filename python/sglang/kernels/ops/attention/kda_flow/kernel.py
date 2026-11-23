# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM's Kimi K3 specialized decode (cutedsl/kda/flow.py, KdaFlow) for
# SGLang's KDA verify contract: up to 5 verify rows, our conv state layout (tap-major
# [slot, 3, 3D] read, per-token windows written to the DSpARK intermediate buffer for
# the commit's rollback), records and pending counts indexed by the mamba slot, and
# either a precomputed raw forget gate or the fused F_B projection.
"""Persistent KDA verify with the deferred RecoverSSM fold (KdaFlow): one state pass
per item, two independent item streams per SM, the per-item state as asynchronous bulk
copies through a 3-page ring.

One CTA per SM (256 threads) walks a contiguous, head-major range of (request, head)
items (at most two heads, at most MAX_ITEMS items) in two groups of 4 warps; group g
runs the CTA's items g, g + 2, ... start to finish.

Per item: **prep** (thread c = channel c, shared memory only) runs the convolution +
SiLU, q / k normalization, decays and the fold's per-key operands, and writes the
token windows the commit rolls the conv state back to; **scan** (thread (b, l): rows
8 b + [0, 8), keys 4 (l + 8 j) + [0, 4)) folds the pending commit into the item's FP32
state page, S = S_0 * D_n + sum_t (u_t * kn_t) * P_t (u_t the previous verify's
corrections), writes the committed state back, and accumulates the 2q dot products
S (G_t k_t), S (G_t q_t); **solve** turns them into this verify's corrections (the
records the next fold reads) and outputs (gated RMSNorm, BF16).

Precision: the stock Triton RecoverSSM verify's formulas (BF16 conv outputs, raw
gates and recurrence outputs, FP32 state); only FP32 reduction order differs. Not bit
for bit with SGLang's recurrent verify + replay fold.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_words,
    mma_into,
    scatter_reduce,
    wait,
    zeros,
)

K = 128
GROUPS = 2
GROUP_THREADS = 128
THREADS = GROUPS * GROUP_THREADS
ROWS = 8  # state rows per scan thread: 8 (c // 8) + [0, 8)
LANES = 8  # scan threads per row
CHUNKS = 4  # 4-key chunks per thread and row: keys 4 (c % 8 + 8 j)
SLOTS = 3  # the CTA's state page ring
MAX_ITEMS = 16  # items per CTA (tables, raw gates)
PAGE_BYTES = K * K * 4
COPY_BYTES = 16384  # per bulk copy of a state page quarter (one warp's 32 rows)
# Rubin's oversized shared memory mode (327 KiB per CTA; budget 320 KiB).
OVERSIZED_BUDGET = 320 * 1024
PACKED = True  # FP32 pair ops as packed f32x2 instructions (scalar: same bits)
FOLD_FENCE = 1  # a warp sync after every FOLD_FENCE scan chunks (bounds load hoisting)
# Item table columns.
SLOT, HEAD, REQUEST, SROW, PENDING_N, RECORD = range(6)
META = 8
# Per-head channel constants: conv weights [3, 4], biases [3], exp(A_log), dt.
CONSTS = 17
A_EXP, DT_ROW = 15, 16

# Operand positions (``KdaFlow.__call__(*args)``).
(X, CONV_W, CONV, GATE_IN, BETA, A_LOG, DT, SLOTS_IN, STATE, OUT, OGATE, NORM_W, FB,
 ICONV, IROWS, CORR, KG, PENDING, LB, SCALE, EPS) = range(21)  # fmt: skip
NUM_ARGS = 21


def sums_count(tokens):
    """Block sums per prep: 2q norms, q^2 couplings, q recorded-key norms, q previous
    outputs; one 32-value round up to q = 4, two at q = 5."""
    return 32 if tokens * tokens + 4 * tokens <= 32 else 64


def padded_vectors(tokens):
    """The 2q scan vectors rounded up to the 4-wide swizzle (q = 5: 12)."""
    return -(-2 * tokens // 4) * 4


def shared_bytes(tokens, conv_size):
    pages = SLOTS * PAGE_BYTES
    heads = 2 * CONSTS * K * 4
    gates = MAX_ITEMS * tokens * K * 2
    tables = 2 * MAX_ITEMS * 8 * 4
    vectors = padded_vectors(tokens) + 2 * tokens  # x (padded), v, w
    block = vectors * K * 4 + tokens * K * 4 + K * 4 + tokens * tokens * 4  # + u, D, c
    staging = tokens * K * 2 + 2 * 9 * K * conv_size + 3 * tokens * K * 2 + tokens * K * 2
    staging += tokens * 2 * K * 2 + tokens * K * 4 + tokens * 2 * K * 2  # records in / out
    group = block + staging + sums_count(tokens) * 16
    return pages + heads + gates + tables + GROUPS * group + 64 * 8 + 4096


def smem_budget(major, minor, optin):
    return max(optin, OVERSIZED_BUDGET) if (major, minor) == (10, 7) else optin


def max_items_per_cta(requests):
    """A CTA's head-major item range spans at most two heads, at most MAX_ITEMS items."""
    return min(requests + 1, MAX_ITEMS)


# ------------------------------------------------------------------ inline PTX
@asm(Float32, pure=True)
def triton_exp(value):
    code = "{ .reg .f32 t; mul.f32 t,$1,0f3FB8AA3B; ex2.approx.ftz.f32 $0,t; }"
    return code, "=f,f", (value,)


@asm(Float32, pure=True)
def div_full(numerator, denominator):
    """The quotient by a positive denominator as div.full.f32 computes it."""
    code = "{ .reg .f32 a,b,r; .reg .pred big; mov.f32 a,$1; mov.f32 b,$2; "
    code += "setp.gt.f32 big,b,0f7E800000; @big mul.f32 a,a,0f3E800000; "
    code += "@big mul.f32 b,b,0f3E800000; rcp.approx.ftz.f32 r,b; mul.f32 $0,a,r; }"
    return code, "=f,f,f", (numerator, denominator)


@asm(Float32, pure=True)
def reciprocal(value):
    return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def sqrt_approx(value):
    return "sqrt.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(Float32, pure=True)
def rsqrt_ftz(value):
    return "rsqrt.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@cute.jit
def triton_sigmoid(value):
    return div_full(Float32(1), Float32(1) + triton_exp(-value))


def load_shared_f32x4(pointer):
    return load_words(pointer, count=4, shared=True, mutable=True,
                      result_type=Float32, word_type="f32")  # fmt: skip


def load_shared_f32x2(pointer):
    return load_words(pointer, count=2, shared=True, mutable=True,
                      result_type=Float32, word_type="f32")  # fmt: skip


def load_shared_u32x2(pointer):
    return load_words(pointer, count=2, shared=True, mutable=True, word_type="u32")


@asm()
def sync_threads(barrier_id, count):
    return "bar.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm()
def fence_async_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def wait_bulk_reads():
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def store_shared_f32x4(pointer, values):
    return "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f,~{memory}", (
        pointer,
        *values,
    )


@asm((Uint32,) * 4)
def load_global_u32x4(pointer):
    return "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", (pointer,)


@asm()
def store_shared_u32x4(pointer, values):
    return "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", (
        pointer,
        *values,
    )


@asm()
def store_global_f32x4(pointer, values):
    """16 B register -> global store, streaming (evict-first) cache policy."""
    return "st.global.cs.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f,~{memory}", (
        pointer,
        *values,
    )


@asm()
def store_bulk(dst, src, size):
    """Shared -> global bulk copy (in the open bulk group)."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def bulk_commit():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


class KdaFlow:
    """One KDA layer's verify over ``requests`` requests of ``tokens`` rows at
    ``heads`` heads; ``ctas`` persistent CTAs; ``fused_fb``: the gate input is F_A
    [R, 128] and the kernel applies F_B, else it is the raw gate [R, heads * 128]."""

    def __init__(self, heads, tokens, requests, ctas, fused_fb, smem_bytes, prefetch=False):
        assert 2 <= tokens <= 5
        self.heads, self.tokens, self.requests = heads, tokens, requests
        self.ctas, self.fused_fb, self.smem_bytes = ctas, fused_fb, smem_bytes
        # Pages issued before the PDL wait: the state was written by this layer's
        # previous step, never by the kernels just before it. The fused F_B rows
        # stage in the last page slot, so that mode prefetches one page fewer.
        self.early_pages = (SLOTS - 1 if fused_fb else SLOTS) if prefetch else 0
        self.packed, self.fold_fence = PACKED, FOLD_FENCE

    @cute.jit
    def __call__(self, *args):
        # Explicit size: the inferred one left the last mbarriers 16 B past the CTA's
        # window, where a co-resident CTA (PDL successor, side stream) overwrote them.
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1,
               smem=self.smem_bytes)  # fmt: skip

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def block_sums(self, values, sums, p, count: cutlass.Constexpr):
        """Per-value totals over the 128 group threads (thread p: channel p):
        sums[value, warp]; the caller syncs."""
        lane, warp = p % 32, p // 32
        total, selected = scatter_reduce(values, lane, count)
        if lane % (32 // count) == 0:
            sums[selected, warp] = total

    @staticmethod
    def quad_sum(r):
        return (r[0] + r[2]) + (r[1] + r[3])

    @cute.jit
    def silu(self, value):
        """The convolution's SiLU, x / (1 + exp(-x)), stored BF16."""
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def decay(self, a, raw_g, dt, lower_bound):
        """``a`` = exp(A_log)."""
        gate = lower_bound * triton_sigmoid(a * (raw_g + dt))
        return triton_exp(gate)

    @cute.jit
    def group_sync(self, g):
        cute.arch.sync_warp()
        sync_threads(1 + g, GROUP_THREADS)

    @cute.jit
    def item_info(self, args, item):
        """(head, request, state slot, scratch row, pending n, record slot) of an item
        (items are head-major); the record slot is the state slot."""
        requests: cutlass.Constexpr = self.requests
        head, request = item // requests, item % requests
        slot = Int32(args[SLOTS_IN][request])
        srow = Int32(args[IROWS][request])
        pending = Int32(0)
        if slot > 0:
            pending = Int32(args[PENDING][slot])
            if pending > self.tokens:
                pending = Int32(self.tokens)
            if pending < 0:
                pending = Int32(0)
        return head, request, slot, srow, pending, slot

    @cute.jit
    def load_quarter(self, args, pages, s, table, j, state_full, w):
        """Bulk-copy rows 32 w + [0, 32) of table item j's state page into page slot
        ``s``: one of the slot barrier's four arrivals (an empty item's carries no
        bytes)."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = state_full + s
        if slot > 0:
            rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, COPY_BYTES)
            source = args[STATE][Int64(slot), None, None, None]
            copy_g2s(at(pages, s, w * rows * K), at(source, head, w * rows, 0),
                     COPY_BYTES, barrier)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def load_page(self, args, pages, s, table, j, state_full):
        for w in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
            self.load_quarter(args, pages, s, table, j, state_full, w)

    @cute.jit
    def stage_item(self, args, table, j, buffers, barrier, parity, c):
        """Bulk-copy table item j's inputs (group thread c; lane 0 of warps 0-3 issue
        one share each): q / k / v and output gate rows, the 3 conv taps of each part
        (buffer ``parity``) and, with a pending commit, its records."""
        x, conv_state, gate = args[X], args[CONV], args[OGATE]
        convs_all, xs, gs, kgs, uss = buffers
        convs = convs_all[parity, None, None]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        slot, head, pending = table[j, SLOT], table[j, HEAD], table[j, PENDING_N]
        if slot > 0:
            rows: cutlass.Constexpr = 4 * T * K * 2 + 9 * K * size
            records: cutlass.Constexpr = T * 2 * K * 2 + T * K * 4
            total = Int32(rows)
            if pending > 0:
                total = total + records
            token = table[j, REQUEST] * T
            if c == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, total)
                for t in cutlass.range_constexpr(T):
                    copy_g2s(at(gs, t, 0), at(gate, token + t, head * K), K * 2, barrier)
            for part in cutlass.range_constexpr(3):
                if c == 32 * (part + 1):
                    for t in cutlass.range_constexpr(T):
                        copy_g2s(at(xs, t, part, 0),
                                 at(x, token + t, part * dim + head * K), K * 2,
                                 barrier)  # fmt: skip
            if c == 32:
                page = conv_state[Int64(slot), None, None]
                for part in cutlass.range_constexpr(3):
                    for tap in cutlass.range_constexpr(3):
                        copy_g2s(at(convs, part, tap * K),
                                 at(page, tap, part * dim + head * K), K * size,
                                 barrier)  # fmt: skip
            if pending > 0:
                rec = table[j, RECORD]
                if c == 64:
                    copy_g2s(at(kgs, 0, 0), at(args[KG], rec, head, 0, 0),
                             T * 2 * K * 2, barrier)  # fmt: skip
                if c == 96:
                    copy_g2s(at(uss, 0, 0), at(args[CORR], rec, head, 0, 0), T * K * 4,
                             barrier)  # fmt: skip
        else:
            if c == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state = args[CONV]
        gate_in, beta, f_b = args[GATE_IN], args[BETA], args[FB]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = self.requests
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first
        conv_type: cutlass.Constexpr = conv_state.element_type

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (SLOTS, K * K), align=1024)
        hcs = alloc(smem, Float32, (2, CONSTS, K))  # per head slot, channel
        raw = alloc(smem, BFloat16, (MAX_ITEMS, T, K), align=16)  # raw gates
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        betas = alloc(smem, Float32, (MAX_ITEMS, 8))  # sigmoid(beta) per token
        # Per group: the 2q scan vectors (padded to the swizzle), v (then the
        # outputs), the fold weights w_t.
        NVP: cutlass.Constexpr = padded_vectors(T)
        BLOCK: cutlass.Constexpr = (NVP + 2 * T) * K
        qkvd = alloc(smem, Float32, (GROUPS, NVP + 2 * T, K), align=128)
        xv = cute.make_tensor(
            at(qkvd, 0, 0, 0), cute.make_layout((GROUPS, NVP * K), stride=(BLOCK, 1))
        )
        v_s = cute.make_tensor(
            at(qkvd, 0, NVP, 0),
            cute.make_layout((GROUPS, T, K), stride=(BLOCK, K, 1)),
        )
        fw_s = cute.make_tensor(
            at(qkvd, 0, NVP + T, 0),
            cute.make_layout((GROUPS, T, K), stride=(BLOCK, K, 1)),
        )
        fu_s = alloc(smem, Float32, (GROUPS, T, K))  # the old corrections u_t
        fd_s = alloc(smem, Float32, (GROUPS, K))  # D_n
        ce_s = alloc(smem, Float32, (GROUPS, T * T))  # the scan's token couplings
        g_s = alloc(smem, BFloat16, (GROUPS, T, K))
        convs = alloc(smem, conv_type, (GROUPS, 2, 3, 3 * K), align=128)
        xs = alloc(smem, BFloat16, (GROUPS, T, 3, K), align=128)
        gs = alloc(smem, BFloat16, (GROUPS, T, K), align=128)
        kgs = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)
        uss = alloc(smem, Float32, (GROUPS, T, K), align=128)
        kgo = alloc(smem, BFloat16, (GROUPS, T, 2 * K), align=128)  # new key/gate
        sums = alloc(smem, Float32, (GROUPS, sums_count(T), 4))
        bars = smem.allocate(cutlass.Uint64, SLOTS + GROUPS + 2)
        state_full, stage_full = bars, bars + SLOTS
        fb_full = bars + SLOTS + GROUPS
        # Prologue scratch (fused F_B): the F_B rows of the CTA's heads in the last
        # page slot (its page loads after the gate projection), the F_A rows of its
        # items over the scan blocks (unused until the first prep).
        fbs = cute.make_tensor(
            cute.recast_ptr(at(pages, SLOTS - 1, 0), dtype=BFloat16),
            cute.make_layout((2, K, K), stride=(K * K, K, 1)),
        )
        fas = cute.make_tensor(
            cute.recast_ptr(at(qkvd, 0, 0, 0), dtype=BFloat16),
            cute.make_layout((MAX_ITEMS, T, K), stride=(T * K, K, 1)),
        )

        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):  # one arrival per warp quarter
                cute.arch.mbarrier_init(state_full + s, PAGE_BYTES // COPY_BYTES)
            for g in cutlass.range_constexpr(GROUPS):
                cute.arch.mbarrier_init(stage_full + g, 1)
            cute.arch.mbarrier_init(fb_full, 1)
            cute.arch.mbarrier_init_fence()
        if tid < count:
            head, request, slot, srow, pending, rec = self.item_info(args, first + tid)
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, SROW] = srow
            table[tid, PENDING_N] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        h0 = first // requests
        h1 = (first + count - 1) // requests
        heads = h1 - h0 + 1
        # Before the PDL wait: the conv weight and dt rows of the CTA's heads (FP32,
        # bulk-copied straight into the head table) and the F_B rows.
        hslot = tid // K
        lane, hw = tid % 32, tid // 32
        if hw < heads:
            rows = hcs[hw, None, None]
            if lane < 12:
                copy_g2s(at(rows, lane, 0),
                         at(args[CONV_W], lane // 4, lane % 4, (h0 + hw) * K), K * 4,
                         fb_full)  # fmt: skip
            if lane == 12:
                copy_g2s(at(rows, DT_ROW, 0), at(args[DT], (h0 + hw) * K), K * 4, fb_full)
        if tid == 0:
            fb_bytes = heads * 13 * K * 4
            if cutlass.const_expr(self.fused_fb):
                fb_bytes = fb_bytes + heads * K * K * 2
            cute.arch.mbarrier_arrive_and_expect_tx(fb_full, fb_bytes)
            if cutlass.const_expr(self.fused_fb):
                for h in cutlass.range_constexpr(2):
                    if h < heads:
                        copy_g2s(at(fbs, h, 0, 0), at(f_b, (h0 + h) * K, 0), K * K * 2,
                                 fb_full)  # fmt: skip
        cute.arch.sync_warp()
        EARLY: cutlass.Constexpr = self.early_pages
        if cutlass.const_expr(EARLY > 0):
            if tid == 0:
                for s in cutlass.range_constexpr(EARLY):
                    if s < count:
                        self.load_page(args, pages, s, table, s, state_full)
            cute.arch.sync_warp()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # The gate rows (F_A, or the raw gates themselves; 16 B per thread, all in
        # flight at once) and the betas: synchronous loads before the page burst.
        CHUNK: cutlass.Constexpr = K * 2 // 16
        ROUNDS: cutlass.Constexpr = MAX_ITEMS * T * CHUNK // THREADS
        fav = cute.make_rmem_tensor((ROUNDS, 4), Uint32)
        for k in cutlass.range_constexpr(ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * CHUNK)
            if item < count:
                t, part = (idx // CHUNK) % T, idx % CHUNK
                row = table[item, REQUEST] * T + t
                column = part * 8
                if cutlass.const_expr(not self.fused_fb):
                    column = table[item, HEAD] * K + part * 8
                v = load_global_u32x4(at(gate_in, row, column))
                for e in cutlass.range_constexpr(4):
                    fav[k, e] = v[e]
        if tid < count * T:
            item, t = tid // T, tid % T
            row = table[item, REQUEST] * T + t
            betas[item, t] = triton_sigmoid(Float32(beta[row, table[item, HEAD]]))
        if hslot < heads:
            hcs[hslot, A_EXP, tid % K] = triton_exp(Float32(args[A_LOG][h0 + hslot]))
        for k in cutlass.range_constexpr(ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * CHUNK)
            if item < count:
                t, part = (idx // CHUNK) % T, idx % CHUNK
                if cutlass.const_expr(self.fused_fb):
                    store_shared_u32x4(at(fas, item, t, part * 8),
                                       [fav[k, e] for e in range(4)])  # fmt: skip
                else:
                    store_shared_u32x4(at(raw, item, t, part * 8),
                                       [fav[k, e] for e in range(4)])  # fmt: skip
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        buffers = (
            convs[g, None, None, None], xs[g, None, None, None], gs[g, None, None],
            kgs[g, None, None], uss[g, None, None],
        )  # fmt: skip
        if g < count:  # the group's first item's inputs
            self.stage_item(args, table, g, buffers, stage_full + g, 0, c)
        cute.arch.sync_warp()
        wait(fb_full, 0, scope="cta")
        # The state stream starts once the small copies the prologue waits on have
        # landed; its first pages fill during the gate projection.
        if tid == 0:
            for s in cutlass.range_constexpr(EARLY, SLOTS - 1):
                if s < count:
                    self.load_page(args, pages, s, table, s, state_full)
        cute.arch.sync_warp()
        cute.arch.barrier()  # gate rows
        if cutlass.const_expr(self.fused_fb):
            self.gate_projection(tid, count, h0, table, fbs, fas, raw)
        cute.arch.sync_warp()
        cute.arch.barrier()  # raw gates; the last page slot and the blocks are free
        if cutlass.const_expr(EARLY < SLOTS):
            if tid == 0:
                if SLOTS - 1 < count:
                    self.load_page(args, pages, SLOTS - 1, table, SLOTS - 1, state_full)
        cute.arch.sync_warp()
        ops = (xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s)
        self.group_items(args, g, c, count, h0, pages, table, betas, raw, hcs,
                         buffers, ops, kgo[g, None, None], sums[g, None, None],
                         state_full, stage_full + g)  # fmt: skip

    @cute.jit
    def gate_projection(self, tid, count, h0, table, fbs, fas, raw):
        """raw[j, t, f] = BF16(F_B[head_j] @ F_A[j, t]) for the CTA's items (warp w:
        features 32 (w % 4) + [0, 32), items w // 4, w // 4 + 2, ...). Results are
        held until every warp has read F_A (which overlays the scan blocks)."""
        T: cutlass.Constexpr = self.tokens
        lane, warp = tid % 32, tid // 32
        group_id, quarter = lane >> 2, lane & 3
        wq, half = warp % 4, warp // 4
        PER: cutlass.Constexpr = MAX_ITEMS // 2
        acc = cute.make_rmem_tensor((PER, 2, 4), Float32)
        wr = cute.make_rmem_tensor((2, 8, 4), Uint32)
        cached = Int32(-1)
        for i in cutlass.range_constexpr(PER):
            j = half + 2 * i
            if j < count:
                head = table[j, HEAD]
                if head != cached:
                    fb = fbs[head - h0, None, None]
                    for m in cutlass.range_constexpr(2):
                        for chunk in cutlass.range_constexpr(8):
                            feature = 32 * wq + 16 * m + group_id
                            col = 16 * chunk + 4 * quarter
                            low = load_shared_u32x2(at(fb, feature, col))
                            high = load_shared_u32x2(at(fb, feature + 8, col))
                            wr[m, chunk, 0], wr[m, chunk, 2] = low
                            wr[m, chunk, 1], wr[m, chunk, 3] = high
                    cached = head
                cute.arch.sync_warp()  # converged for mma.sync
                for m in cutlass.range_constexpr(2):
                    part = zeros((1, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        wa = tuple(wr[m, chunk, e] for e in range(4))
                        ab = load_shared_u32x2(
                            at(fas, j, group_id % T, 16 * chunk + 4 * quarter)
                        )
                        mma_into(part[0, None], wa, ab, bf16=True)
                    for e in cutlass.range_constexpr(4):
                        acc[i, m, e] = part[0, e]
        cute.arch.sync_warp()
        cute.arch.barrier()  # every warp has read F_A
        for i in cutlass.range_constexpr(PER):
            j = half + 2 * i
            if j < count:
                for m in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        t = 2 * quarter + e % 2
                        feature = 32 * wq + 16 * m + group_id + 8 * (e // 2)
                        if t < T:
                            raw[j, t, feature] = BFloat16(acc[i, m, e])

    # ------------------------------------------------------------- group loop
    @cute.jit
    def group_items(self, args, g, c, count, h0, pages, table, betas, raw, hcs,
                    buffers, ops, kgo, sums, state_full, stage_full):  # fmt: skip
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        state, output = args[STATE], args[OUT]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        lane, warp = c % 32, c // 32
        lane_in, base = c % LANES, (c // LANES) * ROWS  # rows base + [0, 8)
        own = base + lane_in
        nw = args[NORM_W][c]
        mine = (count - g + 1) // 2  # this group's items: g, g + 2, ...
        # The group's previous item (outputs before the gated norm, output gates,
        # valid / token / head): its norm rides on the next prep's block reduction.
        po = cute.make_rmem_tensor(T, Float32)
        pg = cute.make_rmem_tensor(T, Float32)
        pm = cute.make_rmem_tensor(3, Int32)
        for t in cutlass.range_constexpr(T):
            po[t] = Float32(0)
        pm[0] = Int32(0)
        for jj in cutlass.range(mine, unroll=1):
            j = g + GROUPS * jj
            slot, head = table[j, SLOT], table[j, HEAD]
            pending, rec = table[j, PENDING_N], table[j, RECORD]
            token = table[j, REQUEST] * T
            wait(stage_full, jj % 2, scope="cta")
            if c == 0:  # the previous item's bulk stores have read their buffers
                wait_bulk_reads()
            cute.arch.sync_warp()
            item_buffers = (buffers[0][jj % 2, None, None], *buffers[1:])
            self.prep_item(args, g, c, j, slot, pending, raw, hcs[head - h0, None, None],
                           item_buffers, ops, sums, kgo, po, table)  # fmt: skip
            if pm[0] != 0:  # the previous item's outputs
                if slot > 0:
                    self.finish(args, c, sums, NV + T * T + T, po, pg, pm, nw)
                else:
                    self.finish_alone(args, g, c, sums, po, pg, pm, nw)
                pm[0] = Int32(0)
            for t in cutlass.range_constexpr(T):  # this item's output gates
                pg[t] = Float32(g_s[g, t, c])
            self.group_sync(g)  # the block is complete; the staging buffers are read
            if c == 0:
                if slot > 0:  # the new key / gate records
                    store_bulk(at(args[KG], rec, head, 0, 0), at(kgo, 0, 0),
                               T * 2 * K * 2)  # fmt: skip
                    bulk_commit()  # read by the next item's start
            if jj + 1 < mine:  # the group's next item's inputs
                self.stage_item(args, table, j + GROUPS, buffers, stage_full,
                                (jj + 1) % 2, c)  # fmt: skip
            cute.arch.sync_warp()
            s = j % SLOTS
            wait(state_full + s, (j // SLOTS) % 2, scope="cta")
            if slot > 0:
                acc = cute.make_rmem_tensor((ROWS, NV), Float32)
                destination = state[Int64(slot), head, None, None]
                self.scan(acc, g, s, pages, pending, lane_in, base, xv, fw_s, fu_s,
                          fd_s, destination, pending > 0)  # fmt: skip
                # The checkpoint left from registers: the warp's quarter of the slot
                # (its own rows, read) is free for the next page at once.
                fence_async_shared()
                cute.arch.sync_warp()
                if lane == 0:
                    if j + SLOTS < count:
                        self.load_quarter(args, pages, s, table, j + SLOTS,
                                          state_full, warp)  # fmt: skip
                cute.arch.sync_warp()
                values = self.rows_reduce(acc, lane_in)
                self.solve(values, g, betas, j, own, v_s, ce_s,
                           args[CORR][rec, head, None, None], po)  # fmt: skip
                pm[0], pm[1], pm[2] = Int32(1), token, head
            else:
                if c == 0:
                    if j + SLOTS < count:
                        self.load_page(args, pages, s, table, j + SLOTS, state_full)
                cute.arch.sync_warp()
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
        if pm[0] != 0:  # the group's last item
            self.finish_alone(args, g, c, sums, po, pg, pm, nw)

    # ------------------------------------------------------------------- scan
    def pfma(self, a, b, c):
        """Lanewise a * b + c of FP32 pairs (packed, or two FFMAs: the same bits)."""
        if self.packed:
            return fma2(a, b, c)
        return (fma(a[0], b[0], c[0]), fma(a[1], b[1], c[1]))

    @cute.jit
    def scan(self, acc, b, s, pages, pending, lane_in, base, xv, fw_s, fu_s, fd_s,
             destination, stored):
        """One pass over this thread's state elements (rows base + [0, 8), keys
        4 (lane_in + 8 j) + [0, 4)): the pending commit is folded in,
        S = fma(S_0, D_n, fc), fc = sum_t fma(u_t, w_t, fc) over t = q - 1 .. 0
        (w_t = kn_t P_t; tokens t >= n are zeros), the committed element goes back
        to the checkpoint, and acc[r, m] += S[r, key] x_m[key] over the 2q scan
        vectors (x = [G_t k_t; G_t q_t])."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        NVP: cutlass.Constexpr = padded_vectors(T)
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        us = cute.make_rmem_tensor((ROWS, T), Float32)
        for t in cutlass.range_constexpr(T):
            for h in cutlass.range_constexpr(2):
                u4 = load_shared_f32x4(at(fu_s, b, t, base + 4 * h))
                for e in cutlass.range_constexpr(4):
                    us[4 * h + e, t] = u4[e]
        for rr in cutlass.range_constexpr(ROWS):
            for m in cutlass.range_constexpr(NV):
                acc[rr, m] = Float32(0)
        for ch in cutlass.range_constexpr(CHUNKS):
            key = 4 * (lane_in + LANES * ch)
            d4 = load_shared_f32x4(at(fd_s, b, key))
            w4 = [load_shared_f32x4(at(fw_s, b, t, key)) for t in range(T)]
            xs = cute.make_rmem_tensor((4, NV), Float32)
            for e in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(NVP // 4):
                    # swizzled: [chunk][e][h][lane][4] (lanes read 128 B in a row)
                    pos = (((ch * 4 + e) * (NVP // 4) + h) * LANES + lane_in) * 4
                    if cutlass.const_expr(4 * h + 4 <= NV):
                        x4 = load_shared_f32x4(at(xv, b, pos))
                        for i in cutlass.range_constexpr(4):
                            xs[e, 4 * h + i] = x4[i]
                    else:  # q odd: the last slot holds 2 vectors (+ 2 pad)
                        x2 = load_shared_f32x2(at(xv, b, pos))
                        for i in cutlass.range_constexpr(NV - 4 * h):
                            xs[e, 4 * h + i] = x2[i]
            for rr in cutlass.range_constexpr(ROWS):
                address = at(pages, s, (base + rr) * K + key)
                sv = load_shared_f32x4(address)
                new = [Float32(0) for _ in range(4)]
                for p in cutlass.range_constexpr(2):
                    fc = (start, start)
                    for i in cutlass.range_constexpr(T):
                        t = T - 1 - i
                        fc = self.pfma((us[rr, t], us[rr, t]),
                                       (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # fmt: skip
                    pair = self.pfma((sv[2 * p], sv[2 * p + 1]),
                                     (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                    new[2 * p], new[2 * p + 1] = pair
                for e in cutlass.range_constexpr(4):
                    for p in cutlass.range_constexpr(NV // 2):
                        pair = (acc[rr, 2 * p], acc[rr, 2 * p + 1])
                        pair = self.pfma((new[e], new[e]),
                                         (xs[e, 2 * p], xs[e, 2 * p + 1]), pair)  # fmt: skip
                        acc[rr, 2 * p], acc[rr, 2 * p + 1] = pair
                if stored:
                    store_global_f32x4(at(destination, base + rr, key), new)
            if cutlass.const_expr(ch % self.fold_fence == self.fold_fence - 1):
                cute.arch.sync_warp()  # bounds ptxas's load hoisting (spills)

    @cute.jit
    def rows_reduce(self, acc, lane_in):
        """The 8 lanes' partials of rows base + [0, 8) -> lane l holds the 2q full
        sums of row base + l (transposed butterfly: lanes ^4, ^2, ^1)."""
        NV: cutlass.Constexpr = 2 * self.tokens
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(ROWS)]
        for stage in cutlass.range_constexpr(3):
            offset: cutlass.Constexpr = 4 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            upper = (lane_in & offset) != 0
            merged = []
            for i in cutlass.range_constexpr(half):
                row = []
                for m in cutlass.range_constexpr(NV):
                    send = rows[i][m] if upper else rows[i + half][m]
                    keep = rows[i + half][m] if upper else rows[i][m]
                    row.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                merged.append(row)
            rows = merged
        return rows[0]

    @cute.jit
    def solve(self, values, b, betas, j, own, v_s, ce_s, u_dst, po):
        """Row ``own``'s tokens from its scan sums A_t = S_0 (G_t k_t),
        B_t = S_0 (G_t q_t): dot_t = A_t + sum_s<t c_ts u_s, u_t = (v_t - dot_t) beta_t,
        out_t = B_t + sum_s<=t e_ts u_s (BF16-rounded, into ``po``); the corrections
        leave from registers into ``u_dst``."""
        T: cutlass.Constexpr = self.tokens
        index = couplings(T)
        u = [Float32(0) for _ in range(T)]
        vs = [v_s[b, t, own] for t in range(T)]
        for t in cutlass.range_constexpr(T):
            dot = values[t]
            for s in cutlass.range_constexpr(t):
                dot = fma(ce_s[b, index[("c", t, s)]], u[s], dot)
            u[t] = (vs[t] - dot) * betas[j, t]
            u_dst[t, own] = u[t]
        for t in cutlass.range_constexpr(T):
            o = values[T + t]
            for s in cutlass.range_constexpr(t + 1):
                o = fma(ce_s[b, index[("e", t, s)]], u[s], o)
            po[t] = Float32(BFloat16(o))

    @cute.jit
    def finish(self, args, c, sums, at_sum, po, pg, pm, nw):
        """Gated RMSNorm of the group's previous item (thread c: channel c of every
        token) from its squares' block sums sums[at_sum + t] (synced by the caller)."""
        eps, output = args[EPS], args[OUT]
        T: cutlass.Constexpr = self.tokens
        token, head = pm[1], pm[2]
        for t in cutlass.range_constexpr(T):
            sumsq = self.quad_sum(sums[at_sum + t, None])
            variance = sumsq * (1.0 / 128) + eps
            rstd = reciprocal(sqrt_approx(variance))
            sigmoid_gate = triton_sigmoid(pg[t])
            result = po[t] * rstd * nw * sigmoid_gate
            output[token + t, head * K + c] = BFloat16(result)

    @cute.jit
    def finish_alone(self, args, g, c, sums, po, pg, pm, nw):
        """``finish`` with its own block reduction (no prep to ride on)."""
        T: cutlass.Constexpr = self.tokens
        count: cutlass.Constexpr = 2 if T <= 2 else (4 if T <= 4 else 8)
        terms = [Float32(0) for _ in range(count)]
        for t in cutlass.range_constexpr(T):
            terms[t] = po[t] * po[t]
        self.block_sums(terms, sums, c, count)
        self.group_sync(g)
        self.finish(args, c, sums, 0, po, pg, pm, nw)
        self.group_sync(g)  # sums free again

    # ------------------------------------------------------------------- prep
    @cute.jit
    def prep_item(self, args, g, c, j, slot, pending, raw, hc, buffers, ops, sums, kgo,
                  po, table):  # fmt: skip
        """Item j's scan operands into the group's block (thread c = channel / key c):
        the 2q scan vectors G_t k_t, G_t q_t, the token couplings c_ts = k_s R_ts k_t,
        e_ts = k_s R_ts q_t, v, the fold weights w_t = kn_t P_t and D_n; its new key /
        raw-gate records into ``kgo``; the per-token conv windows to the scratch row.
        One block reduction for the q / k norms, the couplings, the recorded-key norms
        and the previous item's output squares."""
        lower_bound, scale = args[LB], args[SCALE]
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        convs, xs, gs, kgs, uss = buffers
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        blk = g
        conv_type: cutlass.Constexpr = convs.element_type
        consts = hc[None, c]
        index = couplings(T)
        if slot > 0:
            # The pending commit's records (staged before this verify writes the new
            # ones); tokens t >= n fold as zeros (+0, or -0 corrections when nothing
            # is pending: the fold's identity).
            masked = Float32(0)
            if pending == 0:
                masked = Float32(-0.0)
            old_keys = zeros(T, Float32)
            old_gates = zeros(T, Float32)
            for t in cutlass.range_constexpr(T):
                fu_s[blk, t, c] = masked
                if t < pending:
                    old_keys[t] = Float32(kgs[t, c])
                    old_gates[t] = Float32(kgs[t, K + c])
                    fu_s[blk, t, c] = uss[t, c]
            a = consts[A_EXP]
            dt = consts[DT_ROW]
            srow, head = table[j, SROW], table[j, HEAD]
            iconv = args[ICONV]

            # Convolution + SiLU of q, k, v (thread c: channel c of each part); the
            # window after token t, (seq[t + 1], seq[t + 2], seq[t + 3]) with
            # seq = [taps, x_0 .. x_q-1], goes to the scratch row for the rollback.
            values = cute.make_rmem_tensor((3, T), Float32)
            for part in cutlass.range_constexpr(3):
                taps = cute.make_rmem_tensor(3, conv_type)
                for t in cutlass.range_constexpr(3):
                    taps[t] = convs[part, t * K + c]
                inputs = [xs[t, part, c] for t in range(T)]
                for t in cutlass.range_constexpr(T):
                    accum = Float32(0)
                    for i in cutlass.range_constexpr(4):
                        tap = t + i
                        term = taps[tap] if tap < 3 else inputs[tap - 3]
                        accum = fma(Float32(term), consts[4 * part + i], accum)
                    values[part, t] = self.silu(accum)
                seq = [taps[0], taps[1], taps[2]] + [conv_type(v) for v in inputs]
                channel = part * dim + head * K + c
                for t in cutlass.range_constexpr(T):
                    for w in cutlass.range_constexpr(3):
                        iconv[srow, t, w, channel] = seq[t + 1 + w]
            decays = zeros(T, Float32)
            for t in cutlass.range_constexpr(T):
                v_s[blk, t, c] = values[2, t]
                g_s[blk, t, c] = gs[t, c]
                kgo[t, c] = BFloat16(values[1, t])
                raw_g = raw[j, t, c]
                kgo[t, K + c] = raw_g
                decays[t] = self.decay(a, Float32(raw_g), dt, lower_bound)
            # The decay products of this key: G_t (tokens 0 .. t), R_ts (s + 1 .. t).
            cumulative = [decays[0]]
            for t in cutlass.range_constexpr(1, T):
                cumulative.append(mul_rn(cumulative[t - 1], decays[t]))
            ratio = {}
            for t in cutlass.range_constexpr(1, T):
                ratio[(t, t - 1)] = decays[t]
                for s in cutlass.range_constexpr(t - 1):
                    ratio[(t, s)] = mul_rn(ratio[(t - 1, s)], decays[t])

            # The pending commit's recorded-key decays.
            old_decays = zeros(T, Float32)
            if pending > 0:
                for t in cutlass.range_constexpr(T):
                    old_decays[t] = self.decay(a, old_gates[t], dt, lower_bound)

            # One block reduction: q / k squares, the couplings (raw q / k), the
            # recorded-key squares, the previous item's output squares.
            SUMS_T: cutlass.Constexpr = sums_count(T)
            terms = [Float32(0) for _ in range(SUMS_T)]
            for t in cutlass.range_constexpr(T):
                terms[2 * t] = values[0, t] * values[0, t]
                terms[2 * t + 1] = values[1, t] * values[1, t]
                terms[NV + T * T + t] = old_keys[t] * old_keys[t]
                terms[NV + T * T + T + t] = po[t] * po[t]
            for i in cutlass.range_constexpr(len(index)):
                key = list(index)[i]
                kind, t, s = key
                weighted = values[1, s]
                if cutlass.const_expr(s < t):
                    weighted = mul_rn(weighted, ratio[(t, s)])
                other = values[1, t] if kind == "c" else values[0, t]
                terms[NV + i] = mul_rn(weighted, other)
            if cutlass.const_expr(SUMS_T == 32):
                self.block_sums(terms, sums, c, 32)
            else:  # two 32-value rounds (q = 5: 45 sums)
                high = cute.make_tensor(
                    at(sums, 32, 0), cute.make_layout((32, 4), stride=(4, 1))
                )
                self.block_sums(terms[:32], sums, c, 32)
                self.block_sums(terms[32:], high, c, 32)
            fence_async_shared()  # the records leave by bulk stores
            self.group_sync(g)  # the partial sums
            fq = [Float32(0) for _ in range(T)]
            fk = [Float32(0) for _ in range(T)]
            for t in cutlass.range_constexpr(T):
                qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                fq[t] = reciprocal(sqrt_approx(qn))
                fk[t] = reciprocal(sqrt_approx(kn))
            # The scan vectors (swizzled: [chunk][e][h][lane][4], key c =
            # 4 (lane + 8 chunk) + e; padded to a multiple of 4).
            NVP: cutlass.Constexpr = padded_vectors(T)
            vec = [Float32(0) for _ in range(NVP)]
            for t in cutlass.range_constexpr(T):
                q_t = mul_rn(mul_rn(values[0, t], fq[t]), scale)
                k_t = mul_rn(values[1, t], fk[t])
                vec[t] = mul_rn(cumulative[t], k_t)
                vec[T + t] = mul_rn(cumulative[t], q_t)
            chunk, lane_in, e = c // (4 * LANES), (c // 4) % LANES, c % 4
            for h in cutlass.range_constexpr(NVP // 4):
                pos = (((chunk * 4 + e) * (NVP // 4) + h) * LANES + lane_in) * 4
                store_shared_f32x4(at(xv, blk, pos), [vec[4 * h + i] for i in range(4)])
            # The couplings, normalized (thread i: coupling i).
            norm = Float32(0)
            for i in cutlass.range_constexpr(len(index)):
                key = list(index)[i]
                kind, t, s = key
                norm_i = mul_rn(fk[s], fk[t])
                if cutlass.const_expr(kind == "e"):
                    norm_i = mul_rn(fk[s], mul_rn(fq[t], scale))
                norm = Float32(cutlass.select_(c == i, norm_i, norm))
            if c < len(index):
                ce_s[blk, c] = mul_rn(self.quad_sum(sums[NV + c, None]), norm)
            # The commit's reverse token loop over this key: w_t = kn_t P_t and D_n;
            # tokens t >= n are zeros.
            final_decay = Float32(1)
            for i in cutlass.range_constexpr(T):
                t = T - 1 - i
                w_t = Float32(0)
                if t < pending:
                    total = self.quad_sum(sums[NV + T * T + t, None]) + 1.0e-6
                    kn_t = mul_rn(rsqrt_ftz(total), old_keys[t])
                    w_t = mul_rn(kn_t, final_decay)
                    final_decay = mul_rn(final_decay, old_decays[t])
                fw_s[blk, t, c] = w_t
            fd_s[blk, c] = final_decay


def couplings(tokens):
    """Index of each token coupling: ("c", t, s) for s < t (key-key), ("e", t, s)
    for s <= t (key-query)."""
    index = {}
    for t in range(tokens):
        for s in range(t):
            index[("c", t, s)] = len(index)
        for s in range(t + 1):
            index[("e", t, s)] = len(index)
    return index
