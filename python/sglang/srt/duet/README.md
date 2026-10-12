# DUET serving: families, releases, arm check, environment

Overview (installation, the releases, serving, parity, exams): `benchmark/duet_eval/README.md`.

## Families

| family (`spec.json` `model`) | base architecture | served by | start with |
|---|---|---|---|
| `kimi-linear` | `KimiLinearForCausalLM` | `python/sglang/srt/models/kimi_linear_duet/` | `python -m sglang.launch_server --model-path <base> --duet-release <release> ...` or canonical `benchmark/duet_eval/launch_duet.py` |
| `lightning` | `NemotronHForCausalLM` | `python/sglang/srt/models/lightning_duet/` | `python -m sglang.launch_server --model-path <base> --duet-release <release dir or Hub id> --duet-numerics reference ...` or `launch_duet.py` |
| `glm5-next` | `Glm5NextForConditionalGeneration` | `python/sglang/srt/models/glm_duet/` | `python -m sglang.launch_server --model-path <base> --tp-size 4 --duet-release <release dir or Hub id> --duet-numerics reference ...` or `launch_duet.py` (TP 4; twinstar docs/79) |
| `flash-next` | `Qwen4ExpForConditionalGeneration` | `python/sglang/srt/models/flash_next_duet/` | native `sglang.launch_server --duet-release <release> ...` or canonical `launch_duet.py` |

`launch_duet.py` (`benchmark/duet_eval/`) starts either arm of any family with identical server settings except the DUET
enablement: `--arm duet` serves the release, `--arm stock` the unchanged base; `--dry-run` prints the complete command and
environment.  A plain `sglang.launch_server` needs what the family's loader requires, or it stops at load: Lightning
`--disable-radix-cache --chunked-prefill-size -1 --disable-overlap-schedule` and `SGLANG_MAMBA_SSM_DTYPE=float32`; GLM
`--disable-radix-cache` and `SGLANG_MAMBA_SSM_DTYPE=float32`; both `--duet-numerics reference` (the exam settings also pin
`SGLANG_MAMBA_CONV_DTYPE=float32`).

A requested release is never answered by the base model.  At model registration (`adapters.register_from_environment`,
run when `SGLANG_DUET_DIR` / `--duet-release` is set) the release's `spec.json` names the family; the family's package
replaces the base architecture, and the server stops (`UnservedRelease`) when the package is not importable, when the
base model's architecture is not the family's, or when the release does
not resolve.

## How a release is resolved

* A directory is used as is.
* `<owner>/<repo>[@revision]` is resolved to a commit (`HfApi().model_info`), and exactly `spec.json`, `manifest.json`
  and `duet_components.safetensors` of that commit are downloaded as regular files (a symlink is refused) into
  `<hf cache>/duet-releases/<owner>--<repo>/<commit>`.  `launch_duet.py --hf-cache` (default `$HF_HOME`, else
  `~/.cache/huggingface`) hands the server the directory. A plain `sglang.launch_server --duet-release <owner>/<repo>[@<revision>]`
  downloads it once in the launching process, before the arguments are exported to the workers
  (`adapters.localize_release`, cache root `$HF_HOME`), so the model loaders read a directory.
* The Hub repository's name and the manifest's `name` are independent (the released `Latent-Voyager/<model>-SEED` repositories
  hold manifests named `<model>-DUET-v3.5`): the identity of a set is its component sha256, and nothing compares names.
* Without `@revision` the default branch's current head is used (it can change between launches).  Resolution always
  queries the Hub (`model_info`), so offline (`HF_HUB_OFFLINE=1`) a Hub id fails even when its copy exists: pass the
  directory `<hf cache>/duet-releases/<owner>--<repo>/<commit>` instead (the exam job does this when `HF_OFFLINE=1`).  A
  path that does not exist is taken for a Hub id.
* When the Hub reports the LFS sha256 of `duet_components.safetensors` (`model_info(files_metadata=True)`), the downloaded file
  must have it (`release.fetch_release`, `hub_lfs_sha256`); `verify_release` then compares it with the manifest.  An optional
  pin of one's own: `launch_duet.py --duet-release-sha256` / `fetch_duet.py --release-sha256` (`release.check_pin`).
* The base checkpoint may be a Hub id as well (`launch_duet.py --model <owner>/<repo>[@revision]`, `fetch_duet.py --base`):
  `release.fetch_base` refuses a repository other than the manifest's `base.hf_id` and a revision that does not resolve to
  `base.revision` (the default when none is given), fetches `config.json` first and compares its sha256 with
  `base.config_sha256`, and only then downloads the snapshot into `<hf cache>/hub` (the standard layout); `check_base` runs on
  the snapshot as on any directory.  Private and gated repositories need a login (`hf auth login`, stored in `$HF_HOME/token`, or
  `HF_TOKEN`); `huggingface_hub` reads it, nothing here handles or records a token, and an access error names that cause.

Every loader verifies the release before serving it: `spec.json` is valid and equals `manifest.provenance.spec`, the
component file's size and sha256 match the manifest, and the tensor set and shapes match the family's contract (Kimi,
Lightning and GLM derive it from the spec and the base config; the canonical Flash-Next loader checks its release
contract without an external config-view pre-step).

Base identity: a manifest may name its training base as

    manifest["base"] = {"hf_id": str, "revision": str | None, "config_sha256": str}

where `config_sha256` is the sha256 of the base's `config.json` bytes.  `release.check_base` (called by
`launch_duet.py` before launching and by each family's load path) refuses a base whose `config.json` differs, a
`hf_id` that differs from `manifest["base_model"]` (when both are given), a Hub snapshot directory of another 40-hex
`revision`, and a malformed `manifest["base"]`.  A manifest without `base`, or a base that is not a local directory with a
`config.json`, is served with a warning that the base is not identity-checked.

## What the arm check verifies (`launch_duet.check_arm`, on this launch's `server.log`)

* Canonical Kimi and Flash-Next: combine the healthy native server's `/get_server_info` adapter,
  in-tree status, verified release path and actual trim/r/W values with model load/depth markers.
  Kimi additionally reports the loaded component SHA256. Flash-Next does not emit a component hash:
  the launcher verifies the release, the native loader checks it again, and the runtime release path
  must match; this is distinct from a server-emitted SHA256. No legacy `--twinstar-root` route remains.
* GLM and Lightning: the model load marker and `DUET served cut` k/trim/r/W/SHA256 must match
  the verified release and requested ablations.
* Stock arm: no DUET marker of any family.
* `--no-verify` skips the launcher's own verification (a warning names it); the check then compares with the unverified
  `spec.json` and the manifest's declared sha256, and the server's loader still hashes the component file.
* `--extra` accepts only `--no-prefill-layer-trim`, `--decode-ssm-r N`, `--decode-ssm-w N` and applies them to the
  duet arm; anything else is refused.

Runtime guards: a request the scheduler would retract is aborted on a DUET server (`retraction.py`; its re-prefill
would code the generated tokens); extend batches the Kimi DUET prefill does not take and Lightning requests that get
depth-k logits are counted and logged.  Memory: the state cut (`dense_state_policy.py`) keeps one warm basis
((H, N, r) fp32) per (layer, slot) until SGLang reuses the slot; `state_factor.project_state_batched` stores each as its own
tensor (before fork 028001518c a view kept the whole cut chunk alive per finished request).

## Environment

`launch_duet.py` starts the server with the caller's environment minus every `SGLANG_DUET_*`, `TWINSTAR_*`,
`SGLANG_FLASHNEXT_*`, `SGLANG_QWEN4_*` variable and `SGLANG_EXTERNAL_MODEL_PACKAGE` (printed as `ENV DROPPED`, recorded
in `launch.json`; Flash-Next also drops `NVFP4_OVERLAY_MAP` and `DUET_BASE_SRT_ROOT`), then sets what the family needs:

| variable | set by | value |
|---|---|---|
| `SGLANG_DUET_HOOK_PROFILE` | passed through from the caller | 1 = wall time of the state-cut hook every 500 decode steps (host timing only) |
| `NVIDIA_TF32_OVERRIDE` | launch_duet, every arm | 0 |
| `CUDA_VISIBLE_DEVICES`, `PYTHONPATH` | launch_duet | `--gpus` (Flash-Next: the first `--tp` of them); `PYTHONPATH` = this fork's `python/`, then the caller's |
| `SGLANG_MAMBA_CONV_DTYPE`, `SGLANG_MAMBA_SSM_DTYPE` | launch_duet, all families and both arms | float32 |
| `SGLANG_DUET_DIR`, `SGLANG_EXTERNAL_MODEL_PACKAGE` | Lightning's existing native recipe | release directory / `sglang.srt.models.lightning_duet` |
| `SGLANG_DUET_DIR` | native `--duet-release` localization for Kimi, Flash-Next and GLM | verified release path; canonical adapters registered in tree |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | launch_duet, GLM (both arms) | 0 (DeepGEMM kernels compile on first use; no numeric effect) |
| `SGLANG_DUET_STATE_OVERSAMPLE` | diagnostic plain-server caller; launch_duet drops it | warm-state sketch controls; not a performance qualification |

Without `launch_duet.py` the server code also reads (a CLI flag beats its variable): `SGLANG_DUET_DIR`
(`--duet-release`), `SGLANG_DUET_PREFILL_LAYER_TRIM`, `SGLANG_DUET_PREFILL_SAVING_POLICY`, `SGLANG_DUET_DECODE_SSM_R`,
`SGLANG_DUET_DECODE_SSM_W`, `SGLANG_DUET_EMITTER_PRECISION`, `SGLANG_DUET_PREFIX_STATE`, `SGLANG_DUET_CODE_PRECISION`,
`SGLANG_DUET_NUMERICS`, `SGLANG_DUET_ALLOW_UNVALIDATED` (the flag or the variable enables it), `SGLANG_DUET_STATE_OVERSAMPLE`,
`SGLANG_DUET_HOOK_PROFILE`, and, legacy, `SGLANG_DUET_ENABLED` (Lightning only) and the `TWINSTAR_LIGHTNING_DUET_DIR` /
`TWINSTAR_KIMI_DUET_DIR` aliases (read by the loaders, not by registration: they need `SGLANG_EXTERNAL_MODEL_PACKAGE`).
What `--duet-numerics` fills and what it does not enforce is stated in `numerics.py`.

## Canonical launcher qualification

Kimi and both Flash-Next releases use the same in-tree adapters as the measured native
performance entries. The benchmark launcher redirect is qualified by CPU tests and CLI dry runs;
it has not been measured separately on GPU. Its launcher defaults are not the sealed performance
recipes. Legacy `kimi_duet` and external `twinstar_sgl` sources may remain as historical reference
code, but `launch_duet.py` does not offer a second service-stack route.
