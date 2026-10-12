"""lm-eval model `sglang-duet`: lm-eval's own `sglang-generate` API backend plus what a fair paired exam of a DUET release
against its unchanged base needs (same for both arms):

* chat-template kwargs, so thinking runs at its highest setting where the model has one
  (model_args `chat_template_kwargs=enable_thinking:true;reasoning_effort:xhigh`);
* the thinking segment is removed before grading (`think_end=</think>`); with `think_required=1` a response that never
  closes its thinking (cut off by the context) is graded as an empty answer and counted, never silently dropped;
* generation budget = min(max_gen_toks, context - prompt) per request (context = server_context, default max_length) (the server would otherwise reject the request or
  clamp it silently); a prompt that lm-eval left-truncated (longer than max_length - max_gen_toks) is refused loudly;
* deterministic paired sampling: `sampling_seed` derived from (pair_seed, prompt tokens, k-th occurrence of the same
  prompt in this process), so both arms draw the same random stream for the same question and repetition (the texts stay
  identical only until the first logit difference: batched kernels are not batch-invariant, so neither a rerun nor the two
  arms reproduce a reply token for token); a request that lm-eval retries (tenacity calls `_create_payload` again with the same message
  object) keeps its occurrence and seed (SGLang's seeded sampler, honoured per request by this fork's sampling_batch_info; the server
  must run the pytorch sampling backend when top-k/top-p are in use -- flashinfer rejects seeds there);
* a sidecar jsonl per process: finish reason, token counts, seed, the raw text of every response (resumed_from: the request id of
  an interrupted run whose reply the balancer replayed, lb.py --resume).

model_args example:
  base_url=http://127.0.0.1:31001/generate,model=/work/models/X,num_concurrent=48,max_length=262144,timeout=86400,
  chat_template_kwargs=enable_thinking:true;reasoning_effort:xhigh,think_required=1,pair_seed=20260930,sidecar=/out/sidecar.jsonl
"""
import hashlib
import json
import os
import threading

from lm_eval.api.registry import register_model
from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI

from .common import ids_key, sha1


def parse_kv(spec):
    """'enable_thinking:true;reasoning_effort:xhigh' -> {'enable_thinking': True, 'reasoning_effort': 'xhigh'}
    (model_args are split on ',' and '=' by lm-eval, so nested settings use ':' and ';')."""
    out = {}
    for item in filter(None, str(spec or "").split(";")):
        key, _, value = item.partition(":")
        low = value.strip().lower()
        out[key.strip()] = True if low == "true" else False if low == "false" else value.strip()
    return out


def derive_seed(pair_seed, key, occurrence):
    digest = hashlib.blake2b(f"{pair_seed}:{key}:{occurrence}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


@register_model("sglang-duet")
class SGLangDuetAPI(SGLANGGENERATEAPI):
    def __init__(self, base_url=None, chat_template_kwargs="", think_end="</think>", think_required=0, pair_seed=None,
                 sidecar=None, context_margin=8, server_context=None, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        if self.tokenizer is None or not self.tokenized_requests:
            raise ValueError("sglang-duet needs the HF tokenizer and tokenized requests (the defaults)")
        if self._batch_size != 1:
            raise ValueError("sglang-duet sends one request per call so each gets its own seed and budget (batch_size=1)")
        self.chat_template_kwargs = parse_kv(chat_template_kwargs)
        self.think_end = think_end or None
        self.think_required = str(think_required).lower() in ("1", "true", "yes")
        self.pair_seed = None if pair_seed in (None, "", "none", "None") else int(pair_seed)
        self.context_margin = int(context_margin)
        # the server's context; max_length is lm-eval's, which also bounds the prompt by max_length - max_gen_toks (default: the same)
        self.server_context = None if server_context in (None, "", "none", "None") else int(server_context)
        self.sidecar = open(sidecar, "a", encoding="utf-8") if sidecar else None
        self._lock = threading.Lock()
        self._occurrence = {}
        self._attempts = {}            # id(message object) -> [message object, occurrence, attempts]: retries keep their seed
        self._rid_owner = {}           # rid -> id(message object), released when the response arrives
        self._prompt_sha1 = {}
        self._meta = {}

    # ------------------------------------------------------------------ prompt side
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt, **self.chat_template_kwargs)

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=False, truncation=False, **kwargs):
        encoded = super().tok_encode(string, left_truncate_len=left_truncate_len, add_special_tokens=add_special_tokens,
                                     truncation=truncation, **kwargs)
        strings = [string] if isinstance(string, str) else list(string)
        rows = [encoded] if isinstance(string, str) else encoded
        with self._lock:
            for text, ids in zip(strings, rows):
                self._prompt_sha1[ids_key(ids)] = sha1(text)
        return encoded

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        request = super()._create_payload(messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kwargs)
        if not generate or "input_ids" not in request:
            return request
        rows = request["input_ids"]
        rows = rows if isinstance(rows[0], list) else [rows]
        if len(rows) != 1:
            raise ValueError("sglang-duet expects one prompt per request")
        ids = rows[0]
        params = request["sampling_params"]
        context = self.server_context or self.max_length + 1          # TemplateAPI keeps max_length - 1
        budget = context - len(ids) - self.context_margin
        if budget < 1:
            raise ValueError(f"a prompt of {len(ids)} tokens leaves no room to generate in a {context}-token context")
        params["max_new_tokens"] = min(int(params["max_new_tokens"]), budget)
        key = ids_key(ids)
        with self._lock:
            slot = self._attempts.get(id(messages))
            if slot is not None and slot[0] is messages:          # a retry of the same lm-eval request (the slot holds a reference)
                slot[2] += 1
                occurrence, attempt = slot[1], slot[2]
            else:
                occurrence, attempt = self._occurrence.get(key, 0), 0
                self._occurrence[key] = occurrence + 1
                self._attempts[id(messages)] = [messages, occurrence, 0]
            prompt_sha1 = self._prompt_sha1.get(key)
        if prompt_sha1 is None:
            raise ValueError(f"a prompt of {len(ids)} tokens was left-truncated by lm-eval (max_length - max_gen_toks); "
                             "raise max_length or lower the task's max_gen_toks instead of grading a truncated prompt")
        if self.pair_seed is not None:
            params["sampling_seed"] = derive_seed(self.pair_seed, key, occurrence)
        rid = f"{os.getpid()}-{key}-{occurrence}" + (f"-retry{attempt}" if attempt else "")   # unique: repetitions share the servers
        request["input_ids"] = ids
        request["rid"] = rid
        with self._lock:
            self._rid_owner[rid] = id(messages)
        self._meta[rid] = {"prompt_sha1": prompt_sha1, "prompt_tokens": len(ids), "occurrence": occurrence, "retries": attempt,
                           "seed": params.get("sampling_seed"), "max_new_tokens": params["max_new_tokens"]}
        return request

    # ------------------------------------------------------------------ response side
    def _strip_thinking(self, text):
        if self.think_end and self.think_end in text:
            return text.split(self.think_end, 1)[1], False       # the thinking segment ends at the FIRST tag
        if self.think_required:
            return "", True
        return text, False

    def parse_generations(self, outputs, **kwargs):
        outs = outputs if isinstance(outputs, list) else [outputs]
        answers = []
        for out in outs:
            text = out["text"]
            info = out["meta_info"]
            finish = (info.get("finish_reason") or {}).get("type")
            answer, no_think_end = self._strip_thinking(text)
            rid = info.get("id")
            with self._lock:
                owner = self._rid_owner.pop(rid, None)
                if owner is not None:
                    self._attempts.pop(owner, None)
            record = dict(self._meta.pop(rid, {}), rid=rid, finish=finish, completion_tokens=info.get("completion_tokens"),
                          length_truncated=finish == "length", no_think_end=no_think_end, answer_sha1=sha1(answer),
                          text_raw=text, resumed_from=info.get("duet_resumed"))
            if self.sidecar is not None:
                with self._lock:
                    self.sidecar.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self.sidecar.flush()
            answers.append(answer)
        return answers
