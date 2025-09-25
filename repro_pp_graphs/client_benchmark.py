import argparse, time, json, random, pathlib, requests

def load_prompts(path):
    p = pathlib.Path(path)
    if p.exists():
        xs = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if xs: return xs
    return ["Explain the impact of CUDA graphs on LLM decoding throughput in 3 bullet points."]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--prompts", default="prompts.txt")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/generate"   # sglang 默认 HTTP 生成端点
    prompts = load_prompts(args.prompts)
    payload = {
        "text": [random.choice(prompts) for _ in range(args.batch)],
        "sampling_params": {"max_new_tokens": args.max_new_tokens, "temperature": 0.0}
    }

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=600)
    t1 = time.time(); r.raise_for_status()
    out = r.json()

    # 若返回里没有逐 token 时间戳，就用总时长/生成 token 粗估 TPOT
    gen_tokens = 0
    if isinstance(out, list):
        # 批量请求返回列表
        for o in out:
            meta_info = o.get("meta_info", {})
            gen_tokens += meta_info.get("completion_tokens", args.max_new_tokens)
    else:
        # 单个请求
        meta_info = out.get("meta_info", {})
        gen_tokens = meta_info.get("completion_tokens", args.max_new_tokens) * args.batch
    
    tpot = (t1 - t0) / max(gen_tokens, 1)

    print(json.dumps({
        "total_time_s": round(t1 - t0, 3),
        "gen_tokens": gen_tokens,
        "approx_tpot_s": round(tpot, 6)
    }, indent=2))

if __name__ == "__main__":
    main() 