# Project memory

This file is the persistent record for this project. It lives in the git repo (not in a user profile / session-local location) specifically because this lab machine has already lost context once to a restart/wipe — anything that matters needs to be here, not just in a Claude session.

## Dissertation goal

User is doing an MSc dissertation at Trinity College Dublin (TCD), supervised by **Rachel McDonnell**. Topic: using AI to generate diverse clothing textures for crowds in games/simulations.

Plan: use [Paint3D](https://github.com/OpenTexture/Paint3D) (git submodule at `external/paint3d`) to generate 2K UV textures on base mesh clothing templates. Core contribution is **diversity** across a crowd, not just per-texture quality — any method should be judged on "does this produce meaningfully diverse outputs across a crowd," not "does this look good for one character."

## The 3 sampling methods (reconstructed/reconfirmed 2026-07-02, still being refined)

The original 3 methods were designed in a prior chat session lost to a lab-computer restart. Reconstructed in conversation on 2026-07-02 as:

1. **Prompt-space (attribute-grammar) sampling** — Paint3D's text-conditioned pipeline (`pipeline_paint3d_stage1.py` / `pipeline_paint3d_stage2.py`). Sample diverse text prompts from a combinatorial grammar over garment attributes (color, material, pattern, wear/condition, style).

2. **Quasi-Monte Carlo (QMC) sampling over Paint3D's existing latent space** — instead of independent random noise draws (the Stable Diffusion initial noise latent `z_T`), sample via a low-discrepancy sequence (Sobol/Halton, or Latin Hypercube Sampling) for more even, provably-diverse coverage per sample budget. Explicitly confirmed: **not** a hand-built CLIP subspace — applied directly to Paint3D's own latent space.

3. **CLIP-embedding-distance-constrained/guided sampling around "garment"** — check or actively steer generation so a candidate embedding stays within some distance threshold of `CLIP("garment")` (or a small set of category words), keeping outputs garment-like without discrete prompt grammar.

As of 2026-07-02 the user was explicit that the exact technical formalization of methods 2 and 3 (which low-discrepancy sequence; filter vs. active guidance for the CLIP-distance mechanism) is **not settled** — confirm current thinking before assuming specifics in future work.

A reading list (foundations + one paper per method + crowd-diversity grounding via McDonnell's own "Clone Attack! Perception of Crowd Variety", SIGGRAPH 2008) was compiled 2026-07-02 in conversation — worth regenerating/checking for if not otherwise saved in the repo.

## Lab environment

Works from a shared TCD lab computer (Windows, PowerShell) that has previously been restarted/wiped, losing session state that wasn't committed to this repo. GPU: NVIDIA RTX A4000 (as of 2026-07-02, host driver 581.15, CUDA 13.0 capable, confirmed working through WSL2 passthrough).

**Environment setup is documented and scripted in `.claude/setup/`** — see `install_paint3d_env.sh` and `README.md` there for the full WSL2 conda environment (Paint3D's own stack: Python 3.8.5, PyTorch 1.12.1+cu113, kaolin 0.13.0, pytorch3d 0.7.2) plus every gotcha hit while building it (conda ToS acceptance, missing `libgomp1`, an execstack ELF incompatibility, kaolin's exact wheel URL, a pytorch3d dependency-resolution trap). As of 2026-07-02 this was built and fully verified (`torch.cuda.is_available() == True`, all three packages import cleanly) but Paint3D's actual pipeline scripts haven't been run end-to-end yet.

**Going forward: don't rely on Claude's own session/user-profile memory for anything load-bearing on this project** — it does not survive a machine wipe. Anything worth remembering across sessions belongs in this repo (this file, `.claude/setup/`, or code/config itself).
