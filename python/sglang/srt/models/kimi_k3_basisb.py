# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kimi-K3.1 load-time MXFP8 for the shared experts and the KDA projections.

Under ``SGLANG_K31_BASISB_MXFP8`` those linears load the BF16 checkpoint and
cast it to MXFP8 after loading, with ModelOpt MXFP8QTensor numerics, so the
served weights equal the TRT-LLM mixed checkpoint byte for byte.
"""

import logging
from functools import lru_cache
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    mxfp8_quantize_modelopt,
    resolve_mxfp8_dense_gemm_backend,
)
from sglang.srt.utils.common import log_info_on_rank0

logger = logging.getLogger(__name__)


class _BasisBMxfp8LinearMethod(Fp8LinearMethod):
    def _mxfp8_weight_cast(self, weight: torch.Tensor):
        return mxfp8_quantize_modelopt(weight)

    def apply_into(self, layer, x, output, bias=None):
        # SP o_proj writes into K3's symmetric buffer; FlashInfer's MXFP8 GEMM
        # has no out= here, so this costs one [rows, hidden] copy.
        output.copy_(self.apply(layer, x, bias=bias))
        return output


class _BasisBMxfp8Config(Fp8Config):
    def __init__(self) -> None:
        super().__init__(is_checkpoint_fp8_serialized=False, use_mxfp8=True)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        assert isinstance(layer, LinearBase), prefix
        return _BasisBMxfp8LinearMethod(self)


def k31_basisb_linear_config(
    quant_config: Optional[QuantizationConfig],
) -> Optional[QuantizationConfig]:
    """The config for a linear in the basis-B set: online MXFP8 when the switch is on."""
    if not envs.SGLANG_K31_BASISB_MXFP8.get():
        return quant_config
    _log_engaged()
    return _BasisBMxfp8Config()


@lru_cache(maxsize=1)
def _log_engaged() -> None:
    log_info_on_rank0(
        logger,
        "K3.1 basis B ENGAGED: shared experts and KDA q/k/v/g/b/f_a/o load BF16 and "
        "cast to MXFP8 (ModelOpt MXFP8QTensor numerics), dense backend "
        f"{resolve_mxfp8_dense_gemm_backend().name}",
    )


def k31_basisb_enabled() -> bool:
    return envs.SGLANG_K31_BASISB_MXFP8.get()
