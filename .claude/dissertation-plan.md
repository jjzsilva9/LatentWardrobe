# Dissertation plan: AI-generated diverse crowd clothing textures

Working plan for the methodology chapter(s). This is the evolving source of truth for what the methods are and how they'll be evaluated — update it as decisions get made rather than letting them live only in chat. See also `.claude/memory.md` (general project/environment context) and `.claude/setup/` (environment reproduction).

Status legend: **DECIDED** / **OPEN QUESTION** / **IDEA (not committed)**.

**2026-07-02: major redesign.** The original framing ("3 independent methods for ensuring variety across the whole garment space") has been replaced by a two-stage pipeline: Stage 1 generates a *theme-bounded* candidate pool (any of the original 3 methods, now scoped to a user-supplied scenario prompt like "aliens on the moon" rather than the unconstrained space of all clothing), and Stage 2 guarantees individual distinctiveness between final crowd members via output-space diversity-maximizing selection. This resolves several problems flagged in the original critique — see Methodology below for what changed and why.

---

## Background

Reading list compiled 2026-07-02, updated 2026-07-02 (Stage 2 redesign).

> **2026-07-02, later:** a dedicated literature review on diffusion-model output diversity now lives at `.claude/lit-review-diffusion-diversity.md` — 22 papers, taxonomy (noise design / particle methods / condition perturbation / guidance repair / post-hoc selection), per-paper verification status. It supersedes the "Method 2 — QMC" subsection below (QMC-for-diffusion-diversity turns out to have no published precedent; the joint-noise-design family it belongs to does, with SD1.5-tested methods). **A methods rethink against that taxonomy is pending — the Methodology section below predates it.**

**Foundation — the pipeline itself**
- Paint3D: Paint Anything 3D with Lighting-Less Texture Diffusion Models — Zeng et al., CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zeng_Paint3D_Paint_Anything_3D_with_Lighting-Less_Texture_Diffusion_Models_CVPR_2024_paper.html
- TEXTure: Text-Guided Texturing of 3D Shapes — Richardson et al., SIGGRAPH 2023. https://arxiv.org/abs/2302.01721
- Text2Tex: Text-driven Texture Synthesis via Diffusion Models — Chen et al., ICCV 2023. https://arxiv.org/abs/2303.11396

**Core mechanics underlying the Stage 1 methods**
- High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion) — Rombach et al., CVPR 2022. https://arxiv.org/abs/2112.10752
- Learning Transferable Visual Models From Natural Language Supervision (CLIP) — Radford et al., ICML 2021. https://arxiv.org/abs/2103.00020
- Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet) — Zhang et al., ICCV 2023. https://arxiv.org/abs/2302.05543
- Classifier-Free Diffusion Guidance — Ho & Salimans, 2022. https://arxiv.org/abs/2207.12598 (directly relevant: this is the mechanism that trades fidelity against diversity, and is a live risk for Method 2 — see Methodology below)

**Method 2 — QMC / low-discrepancy sampling**
- Quasi-Monte Carlo method (overview) — https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method
- Discrepancy-based Inference for Intractable Generative Models using Quasi-Monte Carlo — https://arxiv.org/abs/2106.11561
- Quasi Monte Carlo methods enable extremely low-dimensional deep generative models (2026) — https://arxiv.org/html/2601.18676v1
- McKay, Beckman & Conover (1979), *A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code*, Technometrics — original Latin Hypercube Sampling paper (look up directly, no confirmed link)

**Method 3 — CLIP-embedding-distance-guided sampling**
- More Control for Free! Image Synthesis with Semantic Diffusion Guidance — Liu et al., WACV 2023. https://openaccess.thecvf.com/content/WACV2023/papers/Liu_More_Control_for_Free_Image_Synthesis_With_Semantic_Diffusion_Guidance_WACV_2023_paper.pdf
- GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models — Nichol et al., 2021 (recalled from memory, not search-confirmed — verify before citing)
- DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation — Kim et al., CVPR 2022 (recalled from memory, not search-confirmed — verify before citing)

**Stage 2 — output-space diverse selection (added 2026-07-02)**
- Determinantal Point Processes for Machine Learning — Kulesza & Taskar, 2012. https://arxiv.org/pdf/1207.6083 — the standard reference for diversity-maximizing subset selection; DPPs jointly model quality and diversity via a single kernel, which maps directly onto Stage 2's actual goal (distinct *and* plausible crowd members, not just distinct).
- k-DPPs: Fixed-Size Determinantal Point Processes — Kulesza & Taskar, ICML 2011. https://icml.cc/2011/papers/611_icmlpaper.pdf — the variant that fixes the output set size k. This is the one we actually want: plain DPPs model set size and content jointly, which biases content selection in an unwanted way when (like here) the crowd size k is known up front.
- Greedy farthest-point / k-center sampling — no single canonical paper; a simple, well-established diversity-selection baseline, cheaper than DPP sampling. Worth implementing as a baseline to compare k-DPP against before committing to the more complex option.

**Theme-coherence and the coherence/diversity/quality tradeoff (added 2026-07-02)**
- CLIPScore: A Reference-free Evaluation Metric for Image Captioning — Hessel et al., EMNLP 2021. https://arxiv.org/abs/2104.08718 — the natural instrument for the new theme-coherence metric: cosine similarity between the theme prompt's CLIP text embedding and a rendered view's CLIP image embedding.
- Consistency-Diversity-Realism Pareto Fronts of Conditional Image Generative Models (2024). https://arxiv.org/html/2406.10429v1 — directly relevant: studies exactly this three-way tradeoff (their "consistency" ≈ our theme-coherence, "realism" ≈ our plausibility/quality, "diversity" ≈ ours) as a Pareto frontier rather than a single scalar per method. Good methodological template for the results chapter — see Evaluation below.


**Crowd-diversity grounding — supervisor's own line of work**
- Clone Attack! Perception of Crowd Variety — McDonnell et al., SIGGRAPH 2008. https://www.scss.tcd.ie/rachel.mcdonnell/papers/Siggraph08.pdf — read this first; it's the founding justification for the whole dissertation question, and its finding that *color variation alone can mask clones* directly motivates the color-diversity metric under Evaluation below.
- Shape Shifters: Does Body Shape Change the Perception of Small-Scale Crowd Motions? (2024) — https://arxiv.org/abs/2412.16151 — check if same lab; adjacent perceptual-diversity-in-crowds work either way.

---

## Methodology

### Overview — two-stage pipeline (redesigned 2026-07-02)

Built on the Paint3D pipeline (verified working end-to-end 2026-07-02, see `.claude/setup/README.md`).

- **Stage 1 — theme-bounded candidate generation.** The user supplies an open-ended scenario/theme prompt (e.g. "aliens on the moon"). One of the three original methods samples Paint3D's latent space *scoped to that theme*, producing a pool of N candidate garment textures that are stylistically coherent with the scenario. The goal here is deliberately **bounded** variance — not maximal diversity across the whole space of possible clothing. A crowd needs to look like it belongs together in the scene, not like a random sample of all clothing that has ever existed.
- **Stage 2 — output-space diversity-maximizing selection.** From that theme-coherent pool, select the final crowd-sized subset of k textures using a diversity-maximizing selection algorithm (k-DPP or greedy farthest-point) over rendered-view embeddings. This is what actually *guarantees* no two selected crowd members are near-duplicates, rather than hoping Stage 1's sampling mechanism produced that property incidentally.

This resolves the earlier open question of whether the three original methods are independent experiments or composable axes: they're now three interchangeable **Stage 1 mechanisms**, each needing its own theme → conditioning translation step, all feeding into one shared Stage 2 selection.

### Baseline 0: verbatim crowd prompt + random mesh + random seed — **DECIDED 2026-07-02, implemented, not yet run**

The dissertation's baseline and current jumping-off point: the user's crowd-level prompt is passed verbatim to Paint3D for every member; each member gets a randomly selected garment mesh and a fresh random SD seed. Whatever diversity emerges comes purely from seed stochasticity and mesh variation — every proposed method must beat this to justify its existence.

Implemented in `scripts/generate_crowd_baseline.py` (per-member config generation with explicit seeds — necessary because Paint3D's stock configs hardcode a fixed seed, and its `seed: -1` fallback is `int(time.time())` with 1-second resolution which can collide across members; manifest.json for exact reproducibility; `--resume` for crash recovery).

**Meshes (DECIDED 2026-07-02):** MakeHuman community asset packs (CC0), organized by outfit slot in `assets/meshes/` (tops / bottoms / outerwear / fullbody — see its README for export conventions). **Outfit-level texturing (DECIDED):** each member samples 1 top + 1 bottom (or 1 fullbody), the garments are merged into ONE mesh with a shared repacked UV atlas (each garment's existing UV islands transformed into its own atlas region — reversible, so per-garment textures can be cropped back out for engine use), and textured in a single Paint3D run for outfit coherence. Driver update + UV-combine utility + mesh validator pending mesh arrival; not yet smoke-tested.

### Stage 1 methods (PARKED 2026-07-02 pending baseline results and methods rethink)

The lit review (`.claude/lit-review-diffusion-diversity.md`) reframed this space; the methods below predate it. QMC was **removed** entirely (no published precedent for diffusion batch diversity, not the dissertation's focus); the diversity-methods literature (repulsive noise coupling, particle guidance, CADS, guidance-interval — see lit review §3) is kept in the back pocket to draw on once the baseline is measured.

#### Method 1: Prompt-space (attribute-grammar) sampling — **DECIDED: keep the grammar fixed, concatenate the theme prompt**
Paint3D's text-conditioned pipeline (`pipeline_paint3d_stage1/2.py`), sampling prompts from a fixed combinatorial grammar over garment attributes (color, material, pattern, wear/condition, style). The theme prompt is appended directly onto each sampled attribute-string before it's passed to Paint3D — e.g. grammar output "red denim jacket, worn" + theme "aliens on the moon" → "aliens on the moon, red denim jacket, worn".

Considered and rejected 2026-07-02: using an LLM to regenerate/brainstorm a theme-scoped attribute vocabulary per theme. Decided against — adds a nondeterministic, harder-to-reproduce component for a dissertation, when the fixed generic grammar plus Stable Diffusion's own conditioning can plausibly reconcile theme + attributes without it.

Open concerns:
- Prompt diversity ≠ output diversity under SD's mode-seeking behavior (especially at guidance_scale 7) — must validate against actual rendered output, not input prompts.
- Grammar doesn't enforce plausibility; incoherent attribute combos may need filtering, which biases the effective sampled distribution.
- Simple concatenation may occasionally produce theme-incoherent combinations (e.g. "aliens on the moon, denim jacket"). Not treated as a design flaw to pre-solve — the theme-coherence metric (CLIPScore, see Evaluation) is exactly the tool that will empirically reveal whether plain concatenation is good enough, or whether closer integration is needed later.

#### Method 2: ~~QMC sampling over Paint3D's latent space~~ — **REMOVED 2026-07-02**
Removed: no published precedent exists for QMC/low-discrepancy initial-noise batches as a diffusion diversity method (checked during the lit review), and it is not the focus of this dissertation. Its conceptual slot — "make the batch of initial noises jointly diverse instead of i.i.d." — is exactly the *joint noise design* family in the lit review (§3.1: repulsive Gaussian coupling, antithetic pairing, noise optimization), which has published, SD1.5-tested members. If a noise-design method is wanted later, start from those, not QMC.

#### Method 3: CLIP-embedding-distance-constrained/guided sampling — **anchor becomes theme-specific, meaningfully stronger than before**
Check or actively steer generation so a candidate embedding stays within some distance threshold of a theme-derived CLIP anchor (e.g. `CLIP("aliens on the moon clothing")`), rather than the generic `CLIP("garment")` anchor from the original framing.

This directly addresses the original critique that the generic single-word anchor was too loose to do meaningful work (Paint3D's geometry conditioning likely already rules out non-garment content for free). A theme-specific anchor is a materially tighter, more useful constraint.

Open concerns (carried over, still unresolved):
- Not yet decided: passive filter (reject samples too far from anchor) vs. active gradient guidance during denoising — opposite risk profiles (filtering can only shrink diversity; guidance can reshape it but risks adversarial/incoherent textures and costs 2-4x more compute per sample).
- CLIP's embedding space is anisotropic and has a text/image modality gap — a fixed distance threshold needs empirical/percentile calibration against the theme-specific anchor, not an arbitrary cutoff.
- Still needs a specified sampling distribution *within* the constrained region, not just the boundary — otherwise this collapses into Method 1 or 2 with a different-shaped boundary.

### Stage 2: Output-space diversity-maximizing selection — **NEW, added 2026-07-02**

Overgenerate N candidates via whichever Stage 1 method, then select the final k via **k-DPP** or **greedy farthest-point sampling** over rendered-view embeddings (LPIPS feature space, CLIP space, or a combination — not yet decided which).

- Naturally incorporates a quality gate: filter or weight the candidate pool by the plausibility metric (see Evaluation) before/during selection, so incoherent candidates can't win a selection round even in principle.
- Tunable knobs: pool size N, target crowd size k, quality-vs-diversity weighting in the selection objective, and — the main lever for the whole pipeline — how tightly Stage 1 bounds around the theme before Stage 2 even runs.
- **OPEN QUESTION:** k-DPP vs. greedy farthest-point. k-DPP is more principled/probabilistic and has a direct fixed-size formulation (Kulesza & Taskar, ICML 2011); farthest-point is simpler, cheaper, and a strong common baseline. Worth implementing both and comparing before committing.

---

## Evaluation

No human perceptual study — out of scope for time. Relying on quantitative proxies instead; the perceptual-diversity-vs-latent-diversity gap this creates is a known limitation to state explicitly (see Clone Attack! in Background — perceived variety is the thing that actually matters, and these are proxies for it, not the thing itself). All metrics below are computed on the **final Stage-2-selected crowd**, not the raw Stage-1 candidate pool, unless otherwise noted.

**Diversity metrics — DECIDED**
- Pairwise LPIPS (low-level perceptual difference)
- Pairwise CLIP-image embedding distance (semantic/style difference)

**Theme-coherence metric — NEW axis, added 2026-07-02, OPEN QUESTION on exact instantiation**
Turns evaluation from a two-way (diversity vs. quality) into a three-way axis: diversity (between characters) / plausibility (is each texture well-formed) / **coherence** (does the crowd actually look like it belongs in the requested scenario). Natural instrument: CLIPScore-style cosine similarity between the theme prompt's CLIP text embedding and each rendered view's CLIP image embedding (Hessel et al., 2021). Natural experimental framing: sweep the Stage 1↔Stage 2 coherence-vs-diversity knob and report the resulting Pareto frontier across all three axes, following the template of the *Consistency-Diversity-Realism Pareto Fronts* paper (2024) — a much stronger result than a single number per method.

**Quality/plausibility metric — IDEA, not yet decided**
Needed to prevent "diversity" being gamed by incoherent/garbled output. Candidates:
- CLIP-IQA or a LAION aesthetic-predictor score — cheap, zero/no-reference, reuses the CLIP model already in the pipeline.
- Precision/Recall for generative models (Kynkäänniemi et al. 2019) against a real-garment reference set (e.g. DeepFashion crops) — Precision = plausibility, Recall = coverage of real-world diversity. More informative than FID, more robust at smaller N. Needs a reference dataset assembled.
- Skip FID — biased estimator, needs sample counts (hundreds-to-thousands) likely infeasible given per-mesh generation cost.

**Clone/duplicate-detection metric — IDEA, not yet decided**
Mean pairwise distance can hide a bad pair even when the average looks healthy — exactly the failure mode Clone Attack! studies. Note: Stage 2's selection algorithm should directly optimize against this, so this metric doubles as a check that Stage 2 is actually working, not just a general diversity report.
- Minimum nearest-neighbor distance across the generated set, reported alongside the mean.
- Vendi Score (Friedman & Dieng, 2023) as the headline aggregate diversity number instead of raw mean pairwise distance — an effective-rank measure over the pairwise similarity kernel, correctly penalizes sets that are only "diverse" because they split into tight clusters far apart.

**Color-diversity metric — IDEA, agreed worth doing**
Pairwise color-histogram distance (chi-square or EMD between hue histograms). Cheap, and ties directly to Clone Attack!'s own finding that color variation alone can mask clones — strong link back to the supervisor's established work for the discussion section.

**Methodological note — DECIDED**
Compute all metrics on rendered multi-view images (Paint3D already outputs `render_rgb.mp4` turntables), not on the raw UV albedo map. The raw UV layout is mostly background/empty space with the actual texture warped into a small region (confirmed visually during the Suzanne-monkey test run) — running metrics directly on it would dilute every score with irrelevant background pixels. Use a small fixed set of camera angles, averaged, since a crowd is seen from multiple angles.

**Not yet addressed**
- Fair-comparison budget: if Method 3's guidance variant costs more per sample than Methods 1/2, comparing at equal sample count (N) isn't fair — may need equal wall-clock/compute budget instead.
- Which reference dataset (if any) to use for Precision/Recall, if that's the direction chosen.
- Which embedding space Stage 2's k-DPP/farthest-point selection should operate over (LPIPS, CLIP, or a combination), and how to weight quality into the selection objective.
- Need a small set of test theme prompts (not just "aliens on the moon") spanning different levels of specificity/exoticism, to demonstrate the pipeline generalizes rather than resting on one anecdote.
