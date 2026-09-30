"""Model-independent DUET/SEED serving layer (docs/162 §3.5).

spec           -- DuetSpec field set and validation (twinstar/duet/spec.py at origin/minma/0913 dd9c7bdbd)
options        -- CLI / SGLANG_DUET_* / spec precedence for the serving switches (docs/162 §3.2)
latent_codec   -- the residual code: fake-quantised forward, NVFP4 / gap8 packing (twinstar/duet/latent.py, latentfmt.py)
state_factor   -- explicit sink + warm-started rank-r truncation of a recurrent state (twinstar/duet/state.py)

F5 skeleton (lead #006-3): the code below is MOVED from the Kimi line (twinstar_sgl/kimi_duet_math.py,
duet_options.py) and the Lightning line (models/lightning_duet/{latent,state,options,components}.py) without
changing any arithmetic; those modules now import from here.  Model adapters keep everything model-specific.
"""
