"""SGLANG_DSV41_TAIL_STAGE: a shared DeepSeek-V4.1 tail server for encoder-only PD.

Each encoder-only prefill (stage 0) runs the early layers over a late-fed request's
prefix [0, e) and pushes the layer-20 c1 KV of that prefix and the late layers' input
of the last window [e - 128, e) into a slot of one tail server (stage 1). The tail
server batches those windows through the late layers and sends their SWA rows and
DSpark captures to decode as PP rank 1 of every prefill's bootstrap table; decode
admits the request with one fused step.
"""
