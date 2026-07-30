# Kimi-K3 Experts-Only NVFP4 Accuracy Fixes

## Result

Kimi-K3 experts-only NVFP4 has been accuracy-verified on 8x B300 with TP8,
CUDA Graph, and concurrency 4:

- fused-front enabled;
- K3 AR fusion enabled;
- `trtllm_mla` used for both attention and decode attention;
- GSM8K-200: 193/200, score = 0.965;
- all 200 requests completed with no retries or CUDA errors.

This branch contains only the code changes required by the default correct
execution path.

## Baseline capabilities

This fix stack uses `7972c0909196cdc72f32205797f36d3ec7856ac6` as its base.
The baseline already provides:

1. SiTU activation support for K3 ModelOpt NVFP4 MoE;
2. SiTU cubin support in the FlashInfer TRT-LLM FP4 MoE runner.

Both changes are also required when applying this stack to an older K3 branch.

## Commits in this branch

### 1. Correct SiTU NVFP4 scaling

Commit: `5124dfe95ad9b4529cf10f19705900e6d5c655d3`

This required correctness fix has two parts:

- pass the K3 SiTU parameters `alpha=4` and `beta=25` to the TRT-LLM SiTU
  kernel;
- keep only the GEMM2 input requantization factor in SiTU's `g1_scale_c`
  instead of folding the W13 up-half dequantization scale into the SiTU
  intermediate scale.

SiTU is nonlinear in both its gate and up inputs. Moving the up-half scale
after `tanh` changes the computation and caused parts of the old output to be
approximately 8192 times too small.

### 2. Preserve FP32 fused-front router semantics

Commit: `32b8273c300a83b7e0261a9bd5327d3b0ba00e3f`

This is a required correctness fix.

K3's `MoEGate` contract is a BF16 x BF16 GEMM with FP32 router logits, followed
by sigmoid, correction bias, and top-k. The plain-TP fused-front path previously
placed the router projection in a BF16-output merged GEMM. That changed the
top-16 expert set for a fraction of tokens at each layer.

After the fix, the merged projection produces FP32 output. After splitting it:

- router logits remain FP32;
- shared gate/up and routed latent tensors return to their original BF16 dtype;
- downstream non-router kernel contracts remain unchanged.

### 3. Use stride-aware precomputed routing

Commit: `cddef56b1e311583efc2a679a5ad24b661c49143`

This is a required correctness fix.

The router view split from multi-token fused-front output has a row stride
larger than 896. The FlashInfer TRT-LLM from-logits FFI accepts only a data
pointer and shape, with no row-stride parameter. It therefore interpreted this
view as dense `[tokens, 896]` storage and read incorrect addresses beginning
with the second token.

After the fix, K3 SiTU NVFP4 uses SGLang's FP32 radix router to precompute top-k
IDs and weights before calling the FlashInfer routed-MoE entry point. This
preserves FP32 routing and does not pass a strided logits view through an
interface that cannot represent its layout.

### 4. Document the accuracy fixes

This commit records the delivery scope, root cause, and validation evidence. It
does not change runtime behavior.

## Required runtime fixes

On top of the stated baseline, accuracy requires these three code fixes:

1. `5124dfe95a`: correct the SiTU kernel parameters and scale composition;
2. `32b8273c30`: restore the fused-front FP32 router contract;
3. `cddef56b1e`: restore the multi-token router layout contract.

These fixes do not require disabling fused-front, CUDA Graph, or AR fusion, and
do not require switching to `trtllm_mha`.

## Root-cause ownership

The failure was not caused by checkpoint corruption or by a remaining SiTU
cubin arithmetic defect.

- The checkpoint BF16 tensors, NVFP4 cast, and independent dequantization
  comparisons all passed.
- After the SiTU scaling fix, real full-layer and TP8 kernel-vs-emulation
  cosine similarity reached 0.999414 and 0.999412, respectively, with 100%
  routing agreement.
- The remaining defects were precision and layout contract regressions between
  the SGLang K3 fused-front path and the ModelOpt/FlashInfer TRT-LLM routing
  interface.

## Validation scope

The complete accuracy validation was job `210747`, which scored 0.965 on
GSM8K-200. It validated the default internal fix path with fused-front enabled.

The completed validation targeted correctness. The performance impact of the
FP32 merged output has not yet been measured independently. Before upstreaming,
run a performance A/B and add router dtype/layout regression coverage.
