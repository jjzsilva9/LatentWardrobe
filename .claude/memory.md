# Project memory

This file is the persistent record for this project. It lives in the git repo (not in a user profile / session-local location) specifically because this lab machine has already lost context once to a restart/wipe — anything that matters needs to be here, not just in a Claude session.

## Dissertation goal

User is doing an MSc dissertation at Trinity College Dublin (TCD), supervised by **Rachel McDonnell**. Topic: using AI to generate diverse clothing textures for crowds in games/simulations.

Plan: use [Paint3D](https://github.com/OpenTexture/Paint3D) (git submodule at `external/paint3d`) to generate 2K UV textures on base mesh clothing templates. Core contribution is **diversity** across a crowd, not just per-texture quality — any method should be judged on "does this produce meaningfully diverse outputs across a crowd," not "does this look good for one character."

## Methodology: two-stage pipeline (redesigned 2026-07-02)

**Full detail, open questions, and reading list now live in `.claude/dissertation-plan.md` — read that file for the complete picture, this is just a summary.** A verified 22-paper literature review on diffusion output diversity is at `.claude/lit-review-diffusion-diversity.md` (compiled 2026-07-02); the plan's Methodology predates it and a rethink of the Stage-1 methods against its taxonomy is **pending** — do not treat the 3 methods below as final.

The original framing ("3 independent methods for ensuring variety across the whole garment space") was redesigned 2026-07-02 into a two-stage pipeline, because "sample the whole garment space" was never actually the right goal for a *crowd* — crowds need scenario-bound coherence (a medieval crowd doesn't need spacesuit coverage), not maximal unbounded diversity:

- **Stage 1 — theme-bounded candidate generation.** User supplies an open-ended scenario prompt (e.g. "aliens on the moon"). One of the 3 original methods (below) samples Paint3D's latent space *scoped to that theme*, producing a coherent candidate pool. Goal: bounded, coherent variance, not maximal diversity.
- **Stage 2 — output-space diversity-maximizing selection (NEW).** From that pool, select the final crowd-sized subset via k-DPP or greedy farthest-point sampling over rendered-view embeddings — this is what actually *guarantees* no two crowd members are near-duplicates, rather than hoping Stage 1 produced that property incidentally. Directly targets the "clone detection" concern from McDonnell's own Clone Attack! paper.

**Current state (2026-07-02, end of session): baseline running overnight.** Baseline implemented in `scripts/generate_crowd_baseline.py`: crowd prompt passed verbatim per member; outfit sampling (1 top + 1 bottom merged into one mesh with a split UV atlas — top left half, bottom right half — via pure-OBJ merge in the driver, OR 1 fullbody garment, default 25% prob); explicit per-member seeds (Paint3D's stock configs hardcode a FIXED seed; its `seed:-1` fallback is time-based at 1s resolution and collides across members); per-prompt manifest.json; `--resume`; `--dry-run`. Mesh library: MakeHuman CC0 packs staged in `assets/meshes/` (43 tops, 41 bottoms, 59 fullbody; no outerwear — nothing fit that category). Only the `.obj` per garment is kept (UVs verified present; Paint3D normalizes scale on load so units don't matter).

**Overnight run 2026-07-02 FAILED, relaunched 2026-07-03:** the first launch produced nothing — (1) driver passed relative paths to Paint3D subprocesses running with `cwd=external/paint3d` → instant FileNotFoundError per member (fixed: absolute-resolve outdir/mesh_dir at startup); (2) `nohup`-in-WSL launch method dies with the WSL2 VM teardown — detached runs must use `Start-Process wsl.exe` from Windows (see `scripts/run_overnight.sh` + post-mortem in `outputs/overnight_20260702/README.md`). Lesson recorded: never skip the 1-member end-to-end smoke test before batch runs. 4 prompts × 25 members → `outputs/overnight_20260702/`. Every future method must beat this baseline.

The former Stage 1 methods are **parked** pending baseline results and a methods rethink against the lit review's taxonomy: (1) prompt-grammar sampling (theme concatenated onto fixed grammar — LLM-per-theme grammar rejected as nondeterministic); (2) ~~QMC over the latent~~ **removed 2026-07-02** (no published precedent, not the focus — its slot belongs to the published joint-noise-design family in the lit review §3.1); (3) CLIP-distance-constrained sampling with theme-specific anchor (filter-vs-guidance undecided). Stage 2 (k-DPP vs. farthest-point output-space selection) still planned but must engage the *Don't Settle at the Mode* (ECCV 2026) claim that during-sampling repulsion beats post-hoc selection.

Evaluation axes: diversity + quality/plausibility + theme-coherence; pairwise LPIPS + CLIP distance decided, rest open — see plan doc.

## Working notes moved out of the dissertation plan (2026-07-03 rewrite)

The plan doc was cut down to a concise dissertation-facing document at the user's request (confirmed methods, one-line future methods, lean evaluation section). Details that used to live there:

- **Naming change:** the plan now calls the baseline "Method 1" and CLIP mesh sampling "Method 2" (formerly "Baseline 0" and "Method 1a"). QMC is fully deleted, not just parked.
- **Method 2 embedding-source history:** text descriptions were drafted (from filenames) then **rejected and deleted** (2026-07-03) — user's rationale: filenames describe original textures, which we strip; renders-only is final. A text/hybrid variant could theoretically return as an ablation but the user was emphatic — do not re-propose it.
- **Parked observations for the write-up:** (1) *Don't Settle at the Mode* (ECCV 2026) claims during-sampling repulsion beats overgenerate-then-select — must be engaged if k-DPP/farthest-point methods are pursued; (2) CFG guidance scale is a large diversity confound (Paint3D defaults 7/3) — fix or sweep it in comparisons; guidance-interval restriction (Kynkäänniemi et al., NeurIPS 2024) composes with everything; (3) Method 2 extension idea: sample bottom conditioned on chosen top (outfit-internal compatibility); (4) theme test set should eventually span 6-8 themes of varying specificity, not just the current 3; (5) shape-only matching example for the write-up: the "medievaldress" mesh ranks as just "a long dress" (qipao outranked it for the medieval theme) — correct behavior, good explanatory figure; tutu → 0.2% for office/medieval themes is the motivating example validated.
- **Operational details:** overnight 2026-07-03 batch = `scripts/run_overnight.sh` (3 themes × {baseline, clip τ=0.02} × N=30, interleaved theme-by-theme so partial nights yield complete pairs; ~20-24h, `--resume` spillover safe; launch via visible-window Start-Process only). Evaluation harness pre-downloaded LPIPS+CLIP weights; open_clip 2.0.2 has no `get_tokenizer` (use module-level `tokenize`). Garment CLIP embeddings cached at `assets/meshes/garment_clip_cache.pt`, invalidated by render mtimes; renders cached in `assets/meshes/renders/`.

## Lab environment

Works from a shared TCD lab computer (Windows, PowerShell) that has previously been restarted/wiped, losing session state that wasn't committed to this repo. GPU: NVIDIA RTX A4000 (as of 2026-07-02, host driver 581.15, CUDA 13.0 capable, confirmed working through WSL2 passthrough).

**Environment setup is documented and scripted in `.claude/setup/`** — see `install_paint3d_env.sh` and `README.md` there for the full WSL2 conda environment (Paint3D's own stack: Python 3.8.5, PyTorch 1.12.1+cu113, kaolin 0.13.0, pytorch3d 0.7.2) plus every gotcha hit while building it (conda ToS acceptance, missing `libgomp1`, an execstack ELF incompatibility, kaolin's exact wheel URL, a pytorch3d dependency-resolution trap, a `huggingface_hub`/`diffusers` version-pin skew, and a `libcuda.so` unversioned-symlink/`LD_LIBRARY_PATH` issue specific to running the actual pipeline scripts).

**As of 2026-07-02: fully verified end-to-end**, not just the import stack — ran both `pipeline_paint3d_stage1.py` and `pipeline_paint3d_stage2.py` against the Suzanne monkey demo mesh and confirmed a correctly textured, seamless result (checked the UV albedo map and a rendered turntable frame). Model checkpoints (SD1.5 + ControlNets, ~7-8GB) download automatically from HuggingFace Hub on first run, no manual step needed. Next real step is applying this working pipeline to actual garment meshes / one of the 3 sampling methods, not further environment debugging.

**Going forward: don't rely on Claude's own session/user-profile memory for anything load-bearing on this project** — it does not survive a machine wipe. Anything worth remembering across sessions belongs in this repo (this file, `.claude/setup/`, or code/config itself).
