"""DUET accuracy exams on this SGLang fork.

    launch_duet.py                one launcher for both arms of an exam (base model unchanged / base + DUET release), any family
    fetch_duet.py                 a released set and its base from the Hugging Face Hub, verified (README.md "Evaluate a released
                                  DUET set from the Hub")
    lmeval/                       the paired official exams: lm-eval 0.4.13 against SGLang servers (lmeval/README.md: protocol,
                                  reproduction)
    flash_next/                   the Flash-Next DUET model package (twinstar_sgl) the exams serve and the reference side of its
                                  parity check (guard_reference.py) (flash_next/README.md)
    forced_logits.py, forced_nll.py   teacher-forced raw-logit scoring of a served model (validation only)
    validate_fn_parity_job.sh     Flash-Next served-vs-reference parity of a base / release pair
    validate_fn_port_job.sh       the in-fork Flash-Next package against the original tree (bitwise, else statistical)
    validate_hooked_parity_job.sh Kimi-Linear / Lightning served-vs-reference parity of a base / release pair
    parity_verdict.py             the verdict of those parity jobs: served DUET - served base against reference DUET - base
    validate_glm_parity_job.sh    GLM-5.3-Flash: the served side of its parity (the reference side is in the twinstar repository)
    reply_survey.py               reply lengths under the exam prompts, decode throughput, sanity replies of a served base
    long_prompt_probe.py          can a served arm take an N-token prompt, and how long does it take (long-context RULER sizing)
    longctx_probe_job.sh          the long-context RULER sizing job: document builds for the RULER cache + long_prompt_probe per arm
    bench_state_cut.py            GPU micro-benchmark of the DUET state cut (Kimi / Lightning geometries)
    archive/                      one-off regrades and diagnostics that recorded runs used (archive/README.md)

The earlier `/generate` evaluator (legacy/) was superseded by lmeval/ and removed on 2026-10-03 (git history keeps it).
"""
