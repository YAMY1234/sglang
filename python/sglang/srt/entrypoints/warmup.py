from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.triton_load_watch import PRECOMPILE_RID_PREFIX

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


# (phase, prompt lengths, max_new_tokens). A prompt "k+n" resends prompt k plus n
# new tokens: a prefix hit with an n-token extend ("k+0" extends 1 token). Triton
# specializes integer arguments as == 1, divisible by 16, or other, and the mHC
# prefill path changes at 4,096 rows; the lengths hit each class for extend and
# prefix sizes, a prompt above the chunk size adds chunks with a prefix, and the
# batch phases cover prefills that arrive while others decode, at batch 2-4.
_PRECOMPILE_PLAN = (
    ("single", (1, 13, 16, 211, 4096, 4817, 20011), 8),
    ("single", ("5+0", "5+7", "5+160", "6+4321", "6+0"), 8),
    ("batch", (300, 5000), 16),
    ("batch", (64, 2000, 9000), 16),
    ("batch", (17, 32, 1100, 4100), 16),
    ("batch", ("2+5", "3+40", "4+4099"), 16),
)
# One batched request per size and token-count class: pre-tokenized batches reach
# the scheduler as one message, so each is one n-request extend, then decode at
# batch n. Kernels keyed on the batch (bs_upper / BS_P2 = next_power_of_2(bs),
# N = bs * verify width) see every size, with a total divisible by 16 and not.
_PRECOMPILE_GROUP_SIZES = range(1, 17)
# Enough for an eager verify step at every size, and the sampling-accept graph's
# capture (second step) at the decode-graph sizes.
_PRECOMPILE_GROUP_NEW_TOKENS = 4

# Decodes that a prompt prefill joins (a mixed batch, --enable-mixed-chunk only):
# 1, then 2 decode rows. 256 tokens outlast both 8K chunks of the prompt by >10x.
_PRECOMPILE_MIXED = ((1, 12000), (2, 9500))
_PRECOMPILE_MIXED_DECODE_TOKENS = 256


def _group_lengths(size, divisible_by_16):
    lengths = [64 + 37 * i for i in range(size)]
    total = sum(lengths)
    if divisible_by_16:
        lengths[-1] += -total % 16
    elif total % 16 == 0:
        lengths[-1] += 1
    return lengths


def _precompile_prompts(lengths, sources, rng):
    prompts = []
    for length in lengths:
        if isinstance(length, str):
            base, extra = (int(x) for x in length.split("+"))
            prompt = sources[base] + rng.integers(1000, 60000, size=extra).tolist()
        else:
            prompt = rng.integers(1000, 60000, size=length).tolist()
            sources.append(prompt)
        prompts.append(prompt)
    return prompts


async def dsv41_precompile(
    disaggregation_mode: str, tokenizer_manager, *, mixed_chunk: bool = True
):
    """Compile every serving-time kernel variant before the HTTP server listens.

    Warm-up and graph capture only run short decode-shaped batches, so prefill
    variants (sizes, prefix hits, chunks), mixed batches and the scheduler's
    batch helpers compiled inside the first requests that reached them (1-8 s
    stalls). These internal requests take the real serving path at each shape
    class, alternating greedy and server-default sampling; the cache is flushed
    afterwards so no prefix, KV or metric state survives.
    """
    import asyncio
    import time

    if disaggregation_mode != "null":
        logger.warning("Prefill precompile is for aggregated serving; skipped.")
        return
    rng = np.random.default_rng(0)
    sources = []
    count = 0

    def make(prompt, max_new_tokens, stream=False):
        nonlocal count
        count += 1
        params = {"max_new_tokens": max_new_tokens, "ignore_eos": True}
        if count % 2 == 0:
            params["temperature"] = 0.0
        return GenerateReqInput(
            input_ids=prompt,
            sampling_params=params,
            rid=f"{PRECOMPILE_RID_PREFIX}{count}",
            stream=stream,
        )

    def make_group(prompts, greedy):
        nonlocal count
        count += len(prompts)
        params = {"max_new_tokens": _PRECOMPILE_GROUP_NEW_TOKENS, "ignore_eos": True}
        if greedy:
            params["temperature"] = 0.0
        return GenerateReqInput(
            input_ids=prompts,
            sampling_params=params,
            rid=[f"{PRECOMPILE_RID_PREFIX}{count}-{i}" for i in range(len(prompts))],
        )

    async def drain(gen):
        async for _ in gen:
            pass

    async def run(prompt, max_new_tokens):
        await drain(
            tokenizer_manager.generate_request(make(prompt, max_new_tokens), None)
        )

    start = time.perf_counter()
    for phase, lengths, max_new_tokens in _PRECOMPILE_PLAN:
        prompts = _precompile_prompts(lengths, sources, rng)
        if phase == "single":
            for prompt in prompts:
                await run(prompt, max_new_tokens)
        else:
            await asyncio.gather(*(run(p, max_new_tokens) for p in prompts))

    for size in _PRECOMPILE_GROUP_SIZES:
        for divisible in (True, False):
            lengths = _group_lengths(size, divisible)
            prompts = _precompile_prompts(lengths, sources, rng)
            group = make_group(prompts, greedy=divisible)
            await drain(tokenizer_manager.generate_request(group, None))

    for num_decodes, prefill_len in _PRECOMPILE_MIXED if mixed_chunk else ():
        streams = []
        for prompt in _precompile_prompts((500,) * num_decodes, sources, rng):
            gen = tokenizer_manager.generate_request(
                make(prompt, _PRECOMPILE_MIXED_DECODE_TOKENS, True), None
            )
            await gen.__anext__()  # decoding from here on
            streams.append(asyncio.ensure_future(drain(gen)))
        await run(_precompile_prompts((prefill_len,), sources, rng)[0], 8)
        await asyncio.gather(*streams)

    flushed = False
    for _ in range(50):
        if (await tokenizer_manager.flush_cache()).success:
            flushed = True
            break
        await asyncio.sleep(0.1)
    logger.info(
        "Prefill precompile: %d requests in %.1f s; cache flushed: %s.",
        count,
        time.perf_counter() - start,
        flushed,
    )
    if not flushed:
        raise RuntimeError("Prefill precompile could not flush the cache.")


_warmup_registry["dsv41_precompile"] = dsv41_precompile
