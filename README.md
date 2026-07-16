# LatentWardrobe

**Diversity-aware diffusion texturing for 3D crowds.** Given a single theme
prompt (e.g. *"medieval villagers"*), LatentWardrobe generates a crowd of
textured 3D garment meshes that stay on-theme while remaining perceptually
diverse — no two members should read as clones.

MSc dissertation project, Trinity College Dublin, supervised by Rachel McDonnell.

![Crowds by method](outputs/crowd_library_20260715_stage1eval/plots/end_to_end/contact_sheets/medieval_villagers.png)
*One row per method (seven methods), five members per row, all generated from
the same theme prompt. White/pink noise pairs (rows 2–3, 4–5, 6–7) share
meshes and diffusion seeds, so texture-level differences read straight down a
column.*

## Motivation

Texture-diffusion pipelines such as Paint3D (Zeng et al., CVPR 2024) can paint
a garment mesh from a text prompt, but conditional diffusion models
systematically under-deliver diversity when sampled repeatedly under the same
conditioning: a naively generated crowd collapses toward near-duplicate
outfits. Perceptual crowd research (McDonnell et al., *Clone Attack!*,
SIGGRAPH 2008) shows viewers reliably detect such appearance clones, and that
clothing colour/texture variation is precisely what masks them. This project
studies training-free interventions that restore appearance diversity without
sacrificing theme coherence.

## Method

Each crowd member is an outfit (top + bottom, or fullbody garment) drawn from
a 139-piece CC0 MakeHuman mesh library and textured by Paint3D
(Stable Diffusion 1.5 + depth ControlNet). Interventions are applied at three
independent points and compared factorially:

| Factor | Levels |
|---|---|
| Garment mesh selection | uniform random · CLIP-similarity to theme (softmax, τ) |
| Initial latent noise | white · pink (FFT reweight `1/(1+f)^α`, α = 0.2; Harrington et al., CVPR 2026) |
| Per-member prompt | plain theme · CLIP-sampled 6-axis clothing description · LLM-evolved description |

CLIP mesh selection additionally draws outfits *pair-coherently*: one slot
(top or bottom, random which) is sampled first against the theme, then the
other slot's weights are re-weighted by CLIP image–image similarity to the
chosen garment (softmax, τ_pair) — independent per-slot draws can pair a top
and bottom that each fit the theme but not each other. The uniform-random
baseline keeps truly independent draws.

The LLM prompt method is a WANDER-inspired novelty search (arXiv 2511.00686):
a local llama-3.1-8B evolves a population of structured clothing descriptions
(colour / formality / material / pattern / condition / accessories), accepting
mutations only when they increase k-NN novelty in CLIP text space. Each
member's evolved description also becomes the CLIP anchor for its own mesh
selection, coupling prompt content to silhouette choice. An earlier 12-axis
variant that scored novelty on the full wearer description converged to a
near-monochrome crowd — an instructive failure documented in
`scripts/llm_prompt_mutation.py`; the corrected 6-axis, clothing-novelty
search is the study's best-performing method:

![Failed 12-axis search vs corrected 6-axis search](outputs/factorial7_20260706/plots/wander_before_after.png)
*Same theme, same pipeline: the initial search optimised text the renderer
cannot see (top); restricting evolution to renderable clothing attributes
recovers colour and pattern diversity (bottom).*

## Evaluation

Evaluation separates three questions, each with its own generation batch,
metrics file, and chart folder:

1. **Mesh selection** (`plots/mesh/`, `shape_metrics.json`) — what does
   sampling alone change? Untextured, grey-shaded turntables of the
   end-to-end batch, scored with the perceptual battery minus colour and
   CLIPScore (meaningless without paint) plus pairwise silhouette 1−IoU on
   the camera-aligned foreground masks — a pure-geometry diversity number
   with no pretrained model in the loop.
2. **Texture generation** (`plots/texture/`, mesh-fixed batch) — what do
   prompt strategy and noise change when the meshes can't? Every texturing
   method replayed against one shared set of garment meshes and diffusion
   seeds, pinned per member index from the CLIP-mesh reference cell
   (`--fixed-members-from`).
3. **End-to-end** (`plots/end_to_end/`) — meshes and textures both vary; the
   as-deployed comparison.

The appearance battery follows Harrington et al. — patchwise DINOv2 distance,
DreamSim, LPIPS, low-resolution pixel L2, hue-histogram χ², and Vendi/DPP
scores over a DINOv2 kernel — alongside CLIPScore theme coherence, since
diversity is trivial if the crowd may drift off-theme. All metrics are
computed on foreground-masked turntable renders; each chart is emitted both
per-theme and averaged across themes (`per_prompt/` and `avg/` subfolders),
plus per-theme contact sheets. Current batches generate stage 1 only (~1/9
the per-member cost of full two-stage texturing); colour diversity on these
uses a painted-texel mask derived from Paint3D's exact placeholder colour,
flagged `color_source: "stage1_derived_mask"` in `metrics.json`.

![Vendi score by method](outputs/factorial7_20260706/plots/vendi_score_dinov2.png)

Headline results from the earlier `medieval villagers` factorial pilot
(N = 10 per cell, pre pool-fix, full two-stage texturing): theme coherence is
flat across all methods (0.19–0.21), so diversity gains were not paid for
with theme drift; a single random colour word triples colour diversity
(χ² 0.53 → 0.86); and the corrected 6-axis LLM search achieves the pilot's
highest effective-member count (Vendi 5.01 vs 4.73 for the random-mesh
baseline) while more than doubling the colour diversity of its 12-axis
predecessor. Cross-theme replication for that pilot is in
`outputs/multitheme_20260710/`.

The consolidated crowd-library study (5 themes × 7 methods, reduced mesh
pool, pair-coherent sampling) regenerates these comparisons across all three
evaluation stages; its charts land in
`outputs/crowd_library_20260715_stage1eval/plots/{mesh,texture,end_to_end}/`
and this section's numbers will be replaced from them.

## Repository layout

```
scripts/
  generate_crowd_baseline.py   crowd generation driver (all methods, manifests, resume)
  mesh_sampling.py             CLIP-anchored garment mesh selection (+ pair-coherent outfits)
  pink_noise.py                pink-noise initial-latent patch
  llm_prompt_mutation.py       WANDER-inspired LLM novelty search (Ollama)
  evaluate_crowd.py            appearance diversity + coherence battery
  evaluate_shape.py            shape-only battery (+ silhouette IoU)
  generate_shape_eval.py       untextured turntable renders for the shape eval
  plot_crowd_comparison.py     per-metric bar charts (per-theme + theme-mean)
  plot_crowd_contact_sheet.py  member-grid contact sheets
  run_crowd_library*.sh        active experiment: end-to-end + mesh-fixed generation,
                               3 eval passes, plots, supervisor, pipeline watcher
  run_factorial7*.sh           medieval-villagers factorial (historical)
  run_multitheme*.sh           cross-theme replication (historical)
assets/meshes/                 139 CC0 garment meshes (MakeHuman-derived)
assets/meshes_removed/         meshes excluded from the pool (broken UV/texturing)
external/paint3d/              Paint3D (git submodule)
outputs/<batch>/<cell>/<prompt>/          manifests, metrics.json, renders
outputs/<batch>/plots/{mesh,texture,end_to_end}/{avg,per_prompt}/  stage-grouped charts
```

Generation runs inside WSL2 (`paint3d` conda env); evaluation runs in a
separate `eval` env (`.claude/setup/`). Every crowd records a manifest of
seeds, garments, and prompts; LLM evolution — the one non-reproducible
component — persists its full population and mutation history alongside.

## Key references

- Zeng et al., *Paint3D: Paint Anything 3D with Lighting-less Texture Diffusion Models*, CVPR 2024
- McDonnell et al., *Clone Attack! Perception of Crowd Variety*, SIGGRAPH 2008
- Harrington et al., *It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models*, CVPR 2026
- *WANDER — Evolve to Inspire: Novelty Search for Diverse Image Generation*, arXiv 2511.00686
- Friedman & Dieng, *The Vendi Score: A Diversity Evaluation Metric for Machine Learning*, TMLR 2023
- Kulesza & Taskar, *k-DPPs: Fixed-Size Determinantal Point Processes*, ICML 2011

A fuller annotated review (31 papers, verification status per entry) is in
`.claude/lit-review-diffusion-diversity.md`.
