// Modified by NVIDIA: selects 12 KV CTAs for the opt-in verify launcher; fmhaRunner.cuh beside it is a verbatim copy.
#pragma once
#define SGLANG_DEP16_VERIFY_KV_CTAS 12
#include "../../../../include/flashinfer/trtllm/fmha/fmhaKernels.cuh"
