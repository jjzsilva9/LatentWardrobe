# Dissertation plan

AI-generated diverse clothing textures for crowds in games. MSc, TCD, supervised by Rachel McDonnell. This document is the concise plan used to structure and write the dissertation — working notes, session logs, and environment details live in `.claude/memory.md`, not here.

## Pipeline

Paint3D (CVPR 2024) textures untextured meshes from a text prompt via a two-stage process (depth-conditioned multi-view generation, then UV-space inpainting/refinement). We texture **outfits**: per crowd member, either a top + bottom (merged into one mesh sharing a split UV atlas) or a fullbody garment, drawn from a 143-garment CC0 MakeHuman mesh library (43 tops / 41 bottoms / 59 fullbody). The user supplies one crowd-level theme prompt (e.g. "medieval villagers"); every member is generated from it with an explicit per-member seed. All sampling choices are recorded in per-crowd manifests, making every crowd exactly reproducible.

Implementation: `scripts/generate_crowd_baseline.py` (generation), `scripts/render_garments.py` + `scripts/mesh_sampling.py` (Method 2), `scripts/evaluate_crowd.py` (metrics).

## Methods (confirmed)

**Method 1 — random meshes, shared prompt (baseline).** Garments sampled uniformly at random per member; the theme prompt passed verbatim to Paint3D; diversity arises only from seed stochasticity and mesh variation. Every other method must beat this.

**Method 2 — CLIP-sampled meshes, shared prompt.** Garments sampled with probability weighted by the cosine similarity between (a) CLIP *image* embeddings of each garment's untextured front/back renders and (b) the CLIP *text* embedding of the theme ("clothing worn by {prompt}"), via a softmax with temperature τ. τ→∞ recovers Method 1 exactly; small τ concentrates on the most theme-apt silhouettes — so τ is a controllable coherence↔mesh-diversity knob. Renders are deliberately the *only* garment representation: asset filenames/descriptions mostly describe original textures, which are stripped and repainted to the theme — shape is the only property a mesh contributes, and renders capture exactly that.

## Future methods (not committed — one line each)

- **CLIP-embedding-distance sampling of Paint3D's generative space** — steer or filter the diffusion sampling itself by embedding distance to a theme anchor, rather than only the mesh choice.
- **Determinantal point processes (k-DPP)** — overgenerate a candidate pool, select the final crowd as a fixed-size DPP over rendered-view embeddings to guarantee pairwise distinctness.
- **Greedy farthest-point sampling** — cheaper alternative to k-DPP for the same overgenerate-then-select role.
- **Pareto frontier analysis** — sweep control knobs (τ, guidance scale) and report coherence/diversity/quality as a frontier rather than single scores.
- **CLIP relevance gating** — filter or re-rank generated textures by CLIP similarity to the theme prompt as an output-side coherence gate.
- **Combinatoric attribute grammar** — sample garment-attribute strings (color/material/pattern/wear) from a fixed grammar and prepend them to the theme prompt for controlled prompt-level variation.

## Evaluation

All metrics computed on a fixed set of frames from each member's turntable render (never raw UV maps — mostly background). No human perceptual study (out of time-scope); stated as a limitation, with McDonnell et al.'s Clone Attack! (SIGGRAPH 2008) as the perceptual grounding the quantitative proxies stand in for.

- **Diversity:** pairwise LPIPS and pairwise CLIP-image distance — reported as mean *and* minimum nearest-neighbour (clone detection), plus Vendi score (effective number of distinct members).
- **Color diversity:** pairwise hue-histogram distance, background-masked (ties directly to Clone Attack!'s finding that color variation masks clones).
- **Theme coherence:** CLIPScore-style similarity between the theme prompt and rendered views, per member and crowd mean.
- **Quality/plausibility:** not yet chosen — candidates: CLIP-IQA / LAION aesthetic score (cheap) or precision-recall against a real-garment reference set.
- Comparisons at matched compute budget where methods differ in per-sample cost.

## Current experiment

Overnight 2026-07-03: **Method 1 vs Method 2 (τ=0.02)**, 3 themes ("medieval villagers", "business casual office workers", "aliens on the moon"), N=30 per crowd → 180 members. Doubles as the candidate pool for future selection-based methods (k=10 from N=30). Evaluate per crowd with `scripts/evaluate_crowd.py`; first baseline-vs-method comparison lands the following morning.

## Background reading

Core (read first): Paint3D (Zeng et al., CVPR 2024) · Clone Attack! (McDonnell et al., SIGGRAPH 2008) · Latent Diffusion/SD (Rombach et al., CVPR 2022) · CLIP (Radford et al., ICML 2021) · ControlNet (Zhang et al., ICCV 2023) · Classifier-Free Guidance (Ho & Salimans, 2022).

Evaluation & selection: CLIPScore (Hessel et al., EMNLP 2021) · LPIPS (Zhang et al., CVPR 2018) · Vendi score (Friedman & Dieng, 2023) · k-DPPs (Kulesza & Taskar, ICML 2011) · Consistency-Diversity-Realism Pareto fronts (Astolfi et al., 2024).

Predecessor texturing systems (related work): TEXTure (Richardson et al., 2023) · Text2Tex (Chen et al., ICCV 2023).

Diffusion-model diversity methods (for the future-methods section and related work): see the dedicated verified review at `.claude/lit-review-diffusion-diversity.md` (22 papers, taxonomy, per-paper verification status).
