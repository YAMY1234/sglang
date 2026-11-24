# Decode warm-up

`d_warmup.py` runs once after the servers are up and before the benchmark client starts. It uses only the Python standard library. It sends waves of synthetic PD requests through the frontend, so that every decode DP rank decodes at every running-batch size the measured window uses before the client exists. The prompts are random token ids from a fixed seed, so they cannot hit the client's prefix cache.

```bash
SRT_FRONTEND_HOST=<FRONTEND_HOST> SRT_FRONTEND_PORT=<FRONTEND_PORT> MODEL=moonshotai/Kimi-K3.1-Pro \
GSM_LOGS=<LOG_DIR> python3 d_warmup.py --dp 4
```

- `<LOG_DIR>` must hold the decode workers' logs as `*_decode_w*.out`. The script exits 3 unless every decode DP rank printed at least one `Decode batch` line there.
- It writes `<LOG_DIR>/D-WARMUP.json` with per-request timings.
- Defaults: input lengths 1k, 8k, 32k (plus 64k and 128k), 128 output tokens, 1 to 20 running requests per rank (`--isls`, `--osl`, `--ks`).
