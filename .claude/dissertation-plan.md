# Dissertation plan

AI-generated diverse clothing textures for crowds in games. MSc, TCD, supervised by Rachel McDonnell. This document is the concise plan used to structure and write the dissertation — working notes, session logs, and environment details live in `.claude/memory.md`, not here.

## Pipeline

Paint3D (CVPR 2024) textures untextured meshes from a text prompt via a two-stage process (depth-conditioned multi-view generation, then UV-space inpainting/refinement). We texture **outfits**: per crowd member, either a top + bottom (merged into one mesh sharing a split UV atlas) or a fullbody garment, drawn from a 143-garment CC0 MakeHuman mesh library (43 tops / 41 bottoms / 59 fullbody). The user supplies one crowd-level theme prompt (e.g. "medieval villagers"); every member is generated from it with an explicit per-member seed. All sampling choices are recorded in per-crowd manifests, making every crowd exactly reproducible.

Implementation: `scripts/generate_crowd_baseline.py` (generation), `scripts/render_garments.py` + `scripts/mesh_sampling.py` (Method 2), `scripts/evaluate_crowd.py` (metrics).

## Methods (confirmed)

**Method 1 — random meshes, shared prompt (baseline).** Garments sampled uniformly at random per member; the theme prompt passed verbatim to Paint3D; diversity arises only from seed stochasticity and mesh variation. Every other method must beat this.

**Method 2 — CLIP-sampled meshes, shared prompt.** Garments sampled with probability weighted by the cosine similarity between (a) CLIP *image* embeddings of each garment's untextured front/back renders and (b) the CLIP *text* embedding of the theme ("clothing worn by {prompt}"), via a softmax with temperature τ. τ→∞ recovers Method 1 exactly; small τ concentrates on the most theme-apt silhouettes — so τ is a controllable coherence↔mesh-diversity knob. Renders are deliberately the *only* garment representation: asset filenames/descriptions mostly describe original textures, which are stripped and repainted to the theme — shape is the only property a mesh contributes, and renders capture exactly that.

**Method 4 — Pink-noise initialization.** Orthogonal to Methods 1-3 (a noise-space change, not a mesh- or prompt-sampling change, so it composes with any of them): replace Paint3D's initial Gaussian latent noise with pink noise before stage 1/stage 2 sampling, following *It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models* (Harrington, Koepke, Karthik, Darrell, Efros — arXiv 2601.00090). Construction: FFT the initial noise, reweight each frequency component by `1/(1+f)^α` (f = radial frequency, α=0.2 per their ablation — quality degrades above α=0.3), inverse-FFT, renormalize to unit variance. No optimization, gradients, or batch coupling required — a one-time transform on the noise sampler. Their own ablation (no models tested at SD1.5, Paint3D's backbone, so magnitude here is unverified but the mechanism is architecture-agnostic) shows a real diversity gain from pink noise *alone*, before any of the paper's optimization: DINO-diversity +9.2% (SDXL-Turbo), +4.7% (SANA-Sprint), +23.7% (PixArt-α).

## Future methods (not committed — one line each)

- **CLIP-embedding-distance sampling of Paint3D's generative space** — steer or filter the diffusion sampling itself by embedding distance to a theme anchor, rather than only the mesh choice.
- **Determinantal point processes (k-DPP)** — overgenerate a candidate pool, select the final crowd as a fixed-size DPP over rendered-view embeddings to guarantee pairwise distinctness.
- **Greedy farthest-point sampling** — cheaper alternative to k-DPP for the same overgenerate-then-select role.
- **Pareto frontier analysis** — sweep control knobs (τ, guidance scale) and report coherence/diversity/quality as a frontier rather than single scores.
- **CLIP relevance gating** — filter or re-rank generated textures by CLIP similarity to the theme prompt as an output-side coherence gate.
- **Combinatoric attribute grammar** — sample garment-attribute strings (color/material/pattern/wear) from a fixed grammar and prepend them to the theme prompt for controlled prompt-level variation.
- **LLM-per-theme grammar** — an LLM mutates per-member prompt variants for the crowd theme rather than a fixed grammar; previously rejected as nondeterministic, reconsidered on the strength of *WANDER* ("Evolve to Inspire: Novelty Search for Diverse Image Generation," arXiv 2511.00686), which steers LLM prompt mutation along semantic "emitters" (style/composition/atmosphere) and scores novelty via CLIP embeddings of the resulting images — grounding the mutation in a measured objective rather than raw LLM freeform output, and making it as logged/reproducible as every other method here (record the seed + the chosen mutation per member). 2026-07-06 literature pass (see `.claude/lit-review-diffusion-diversity.md` §3.6) found WANDER's precursor, *Prompt Expansion* (arXiv:2312.16720, LLM expands one prompt into N diverse variants, no novelty-search loop) and a second data point, *DIAGen* (arXiv:2408.14584, GPT-4-diversified prompts → diffusion → measured recall gain) — but confirmed no published work combines LLM prompt diversification with crowd/character generation or garment texturing specifically, so this method would be a genuine (if modest) methodological combination, not a replication of prior work.
- **Particle Guidance** — joint (not independent) denoising of a member batch with a pairwise-repulsion term added to the score at each step; the strongest methodological fit for directly countering Clone Attack!'s clone-from-independent-sampling finding, but requires refactoring generation from per-member sequential calls to joint batch sampling.
- **CADS (condition-annealed sampling)** — anneal noise into the conditioning embedding on a schedule; cheapest per-step diversity knob, no batch coupling needed, drops into the existing per-member loop unchanged.

## Evaluation

All metrics computed on a fixed set of frames from each member's turntable render (never raw UV maps — mostly background). No human perceptual study (out of time-scope); stated as a limitation, with McDonnell et al.'s Clone Attack! (SIGGRAPH 2008) as the perceptual grounding the quantitative proxies stand in for.

Metric set follows *It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models* (their diversity/quality battery), minus HPSv2 (dropped — not worth the extra dependency for this project's scope). Runs in its own `eval` conda env (`.claude/setup/install_eval_env.sh`), kept separate from the `paint3d` generation env so upgrading eval-only packages (DINOv2 via torch.hub, DreamSim, open_clip) can never break generation.

- **Diversity:** patchwise DINOv2 distance (their Eq. 3: mean cosine distance per patch, averaged over patches, over all member pairs), DreamSim pairwise distance, pairwise LPIPS, low-resolution (32×32) pixel L2, and Vendi score + DPP score (determinant) both over a DINOv2 `[CLS]`-token similarity kernel (switched from our earlier CLIP-based Vendi kernel to match the paper). All pairwise stats report mean, minimum nearest-neighbour (clone detection), and std across member pairs.
- **Color diversity:** pairwise hue-histogram distance, background-masked (ties directly to Clone Attack!'s finding that color variation masks clones) — already implemented pre-dating this paper's own color-histogram metric.
- **Theme coherence:** CLIPScore-style similarity between the theme prompt and rendered views, per member and crowd mean + std.
- **Quality/plausibility:** not yet chosen — candidates: CLIP-IQA / LAION aesthetic score (cheap) or precision-recall against a real-garment reference set.
- Comparisons at matched compute budget where methods differ in per-sample cost.

## Current experiment

Overnight 2026-07-03: **Method 1 vs Method 2 (τ=0.02)**, 3 themes ("medieval villagers", "business casual office workers", "aliens on the moon"), N=30 per crowd → 180 members. Doubles as the candidate pool for future selection-based methods (k=10 from N=30). Evaluate per crowd with `scripts/evaluate_crowd.py`; first baseline-vs-method comparison lands the following morning.

## Background reading

Core (read first): Paint3D (Zeng et al., CVPR 2024) · Clone Attack! (McDonnell et al., SIGGRAPH 2008) · Latent Diffusion/SD (Rombach et al., CVPR 2022) · CLIP (Radford et al., ICML 2021) · ControlNet (Zhang et al., ICCV 2023) · Classifier-Free Guidance (Ho & Salimans, 2022).

Evaluation & selection: CLIPScore (Hessel et al., EMNLP 2021) · LPIPS (Zhang et al., CVPR 2018) · Vendi score (Friedman & Dieng, 2023) · k-DPPs (Kulesza & Taskar, ICML 2011) · Consistency-Diversity-Realism Pareto fronts (Astolfi et al., 2024).

Predecessor texturing systems (related work): TEXTure (Richardson et al., 2023) · Text2Tex (Chen et al., ICCV 2023).

Diffusion-model diversity methods (for the future-methods section and related work): see the dedicated verified review at `.claude/lit-review-diffusion-diversity.md` (30 papers, taxonomy, per-paper verification status). Two additions from the 2026-07-06 ranking pass, verified via WebSearch/WebFetch: *It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models* (Harrington et al., arXiv 2601.00090 — Method 4, and the source of the evaluation metric set above) and *WANDER: Evolve to Inspire: Novelty Search for Diverse Image Generation* (arXiv 2511.00686 — backs the LLM-per-theme grammar future method). A same-day follow-up pass (§3.6 of the review) surveyed LLM-driven prompt-rewriting specifically and confirmed the WANDER/LLM-per-theme-grammar angle is genuinely unexplored for crowds/clothing — no forcing of weak citations where the gap is real.
