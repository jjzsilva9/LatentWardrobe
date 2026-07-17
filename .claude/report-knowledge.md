# Research-space knowledge for report writing — 2026-07-17

Companion to `.claude/project-state.md` (ops/state). This file is the
research understanding: how each piece sits in the literature, what the
findings mean, which claims are defensible at what strength, and the
narrative arcs worth building the report around. Verified references live in
`.claude/lit-review-diffusion-diversity.md` (per-entry verification status);
citations below use the IDs recorded there.

## 1. The problem, precisely stated

Conditional diffusion models systematically under-deliver diversity when
sampled repeatedly under fixed conditioning — mode collapse toward a few
high-probability outputs. For crowds this is not cosmetic: McDonnell et al.
(*Clone Attack!*, SIGGRAPH 2008) showed viewers reliably DETECT appearance
clones, and that clothing colour/texture variation is what masks them. So
the crowd setting turns "diversity" from a nice-to-have into the perceptual
requirement, and gives it an unusual grounding: the end metric is human
clone-detection, which most diffusion-diversity papers lack.

The project's niche: **training-free** interventions (no fine-tuning, no
gradient-based noise optimization), composable at three independent points
of an existing texture pipeline (Paint3D: SD1.5 + depth ControlNet), in a
domain (3D garment texturing) where the diversity literature has essentially
not been applied. The lit review's §3.6 confirmed the WANDER/LLM-prompt
angle is genuinely unexplored for crowds/clothing — the gap is real, not
manufactured.

Field framing to adopt: the literature measures success as a
**quality–diversity Pareto frontier**, not a single number (arXiv
2406.10429, 2510.03813). Our three-axis reporting (diversity / coherence /
quality) is the same idea with coherence added because theme drift is the
domain-specific failure mode.

## 2. Where each method sits in the literature

The lit review's taxonomy of diffusion-diversity interventions: (a) initial
noise design, (b) during-sampling particle interaction, (c) condition
perturbation, (d) guidance repair, (e) post-hoc selection. Our methods map:
pink noise → (a); axis sampling and WANDER → (c) (condition-side, via the
prompt); CLIP mesh selection → domain-specific retrieval, closest to (e)
but acting BEFORE generation. Nothing here is (b)/(d) — worth stating as
deliberate scope (training-free, no sampler surgery).

### Pink noise (→ Harrington et al., arXiv 2601.00090, CVPR 2026)

- Their claim we rely on: initial-noise FREQUENCY PROFILE matters for
  diversity; pink noise (boosted low frequencies) yields more diverse
  samples than white *with no optimization at all* — DINO-diversity +9.2%
  (SDXL-Turbo), +4.7% (SANA-Sprint), +23.7% (PixArt-α) in their ablation.
  α=0.2 is their operating point; quality degrades visibly above ~0.3.
- Mechanism intuition (safe to state as intuition): low-frequency noise
  components seed global appearance modes (overall colour, large regions);
  boosting them widens the spread of global-appearance basins the sampler
  falls into.
- **Honest caveat:** they did not test SD1.5 (Paint3D's backbone). The
  mechanism is architecture-agnostic; the magnitude was never promised.
- Our result fits the mechanism: pink helps **colour** diversity on PLAIN
  prompts (χ² 0.630 vs 0.596, texture stage) — i.e. when nothing else is
  injecting global-appearance variation — and washes out under
  description-driven prompting, which occupies the same "global appearance"
  channel more strongly. Zero measurable set-level (Vendi) cost either way:
  a free knob, but a small one. This "prompt variation dominates noise
  variation" reading is a genuine, defensible finding.

### WANDER-inspired novelty search (→ arXiv 2511.00686)

- Source idea: evolve prompts with LLM mutations, accepting only mutations
  that increase k-NN novelty in an embedding space — quality-diversity
  search transplanted to prompt space.
- Our adaptation (state all three, they're contributions): (1) structured
  mutation over six *clothing* axes (colour/formality/material/pattern/
  condition/accessories) rather than free text; (2) the crowd is the
  population — one evolution per (theme, crowd), members are the final
  population; (3) novelty scored in CLIP text space over the *clothing*
  description only.
- The 12-axis failure is the best methodological story in the project: the
  first version scored novelty on the full wearer description; the search
  optimized attributes the RENDERER CANNOT SEE (age, build, backstory) and
  converged to a near-monochrome crowd. Constraint: novelty must be scored
  on renderable attributes. This is a transferable lesson, not just a bug
  fix — frame it as such.
- Finding inversion worth featuring: WANDER is NOT the diversity winner
  (Vendi bottom of table) but IS the quality/coherence winner (best HPSv2
  0.142 e2e, best CLIPScore 0.216). Interpretation: LLM evolution produces
  *plausible, well-formed* descriptions — coherent phrasings SD renders
  cleanly — rather than maximally spread ones. Novelty search in text space
  optimizes distance between descriptions, but SD's rendering contracts
  text-space distances non-uniformly.

### CLIP-axis sampling (our own method; also WANDER's ablation)

- Framing that makes it look as smart as it is: axis sampling is the
  LLM-free ablation of WANDER's axes — same six-axis description grammar,
  but each axis sampled independently ∝ softmax(CLIP sim to theme / τ_axis)
  instead of evolved. That it BEATS the LLM on texture diversity (Vendi
  3.46 vs 3.24; colour 0.745 vs 0.624, texture stage) is the paper-worthy
  result: **independent per-axis sampling with CLIP relevance weighting is
  a stronger diversifier than sequential novelty search**, at a fraction of
  the cost (no LLM in the loop). The evolved search's advantage shows up in
  quality, not diversity.
- Both description methods share one design decision worth a paragraph:
  the member's own description becomes BOTH the diffusion prompt AND that
  member's mesh-selection CLIP anchor — coupling silhouette choice to
  intended appearance (a jacket-y description retrieves jacket-y meshes).

### CLIP-anchored mesh selection (+ pair-coherent outfits)

- Retrieval, not generation: garments embedded via CLIP image encoder over
  neutral preview renders; sampling ∝ softmax(sim(theme)/τ), τ=0.02. τ is
  the explicit coherence↔diversity knob (τ→∞ uniform, τ→0 argmax).
- Pair-coherence (our design): independent per-slot sampling can pick a top
  and bottom that each fit the theme but not each other; the fix draws one
  slot first (random which), then re-weights the other slot by CLIP
  image–image similarity to the chosen garment (τ_pair=0.05). Motivated by
  observed incoherent outfits; qualitative improvement visible in contact
  sheets.
- The measured tradeoff (mesh stage): CLIP selection *costs* silhouette
  diversity vs random (shape Vendi 2.35–2.58 vs 2.80; 1−IoU 0.54–0.58 vs
  0.60). This is the coherence↔diversity tradeoff made concrete — random
  selection is the diversity ceiling and the coherence floor. Frame as: the
  cost side is measurable; the benefit side (theme fit of meshes) currently
  rests on qualitative evidence because no adequate instrument exists (§4).

## 3. Evaluation: what each metric IS and why it's there

Axis design follows Harrington et al., whose human studies found SET-LEVEL
metrics align best with human diversity preference — the justification for
leading with Vendi/DPP rather than pairwise averages. A crowd is a set; the
perceptual object in McDonnell's work is the ensemble.

- **Vendi score** (Friedman & Dieng, TMLR 2023): exp(Shannon entropy of the
  eigenvalues) of the normalized member-similarity kernel — "effective
  number of distinct members," bounded by N. **At N=5 the ceiling is 5**;
  observed values ~3.1–3.5 mean differences compress — always report this
  ceiling when interpreting.
- **DPP score** (after Kulesza & Taskar, k-DPPs, ICML 2011): n-th-rooted
  determinant of the same kernel — geometric-mean eigenvalue, the "volume"
  reading; collapses toward 0 if ANY two members are near-clones (a
  clone-detector, which is exactly the McDonnell failure mode). Vendi and
  DPP are two summaries of one kernel — report both, but they are not
  independent evidence.
- **DreamSim** (Fu et al., NeurIPS 2023): ViT-ensemble similarity
  fine-tuned on human two-alternative forced-choice judgments — the
  human-perception-aligned pairwise measure; the bridge between machine
  metrics and McDonnell's human-detection framing.
- **Colour χ²**: hue-histogram chi-square over PAINTED UV texels (exact
  painted-texel mask, not a render heuristic) — the transparent low-level
  metric, directly linkable to McDonnell's finding that colour variation
  masks clones.
- **CLIPScore** (Hessel et al. 2021 style): prompt↔render cosine — theme
  coherence, the axis diversity must not be bought against. Flat across
  methods end-to-end (0.206–0.216) = "no method pays with drift."
- **HPSv2** (Wu et al. 2023): CLIP fine-tuned on human preference pairs —
  the quality axis, matching Harrington's coherence/quality reporting.
- **Silhouette 1−IoU** (ours): pairwise foreground-mask IoU at fixed camera
  angles — the only metric with NO pretrained model; immune to "the
  backbone is texture-biased" objections at the mesh stage.
- **Dropped** (computed initially, cut with reasons — one honest paragraph
  in the report): DINOv2 patchwise (Harrington's Eq. 3; local aligned
  distance — redundant with the set-level story for our claims), LPIPS
  (superseded by DreamSim, its successor on human agreement), pixel-L2
  (superseded by colour χ² + IoU for interpretability).

**The three-stage attribution design** is itself a methodological
contribution: mesh-only (untextured renders — what does selection alone
change), texture-only (mesh-fixed replay: garments+seeds pinned per member
index from a reference cell, so cross-method differences can ONLY come from
prompt strategy × noise), end-to-end (as deployed). The built-in validity
check: under pinning, the random-mesh and CLIP-mesh plain cells become the
same condition and produce near-identical scores — evidence the isolation
actually isolates. The seed-matched white/pink twin design does the same
for the noise factor (twins share garments and diffusion seeds; contact
sheet columns are directly comparable).

## 4. Instrument limits to disclose (state them BEFORE results)

- **CLIP on untextured renders is a weak instrument** — CLIP's signal is
  texture/colour dominated; grey silhouettes barely separate (0.190–0.193
  flat). Consequence: mesh-stage theme-fit is currently NOT machine-
  measurable here; it reads qualitatively in contact sheets.
- **CLIP circularity at the mesh stage**: selection samples BY CLIP
  similarity, so CLIP-based coherence evaluation of selection is partially
  self-fulfilling (mitigated but not removed by the pathway difference:
  sampling anchors on textured garment previews, eval sees grey outfit
  renders). An open problem — honest candidates: garment↔theme CLIP as a
  labeled manipulation check, an LLM judge over garment descriptions, or a
  small human study (the gold standard, and the natural future-work bridge
  back to McDonnell's paradigm).
- **Stage-1 scoring proxy**: crowds are scored on stage-1 turntables (the
  prompt/noise differences are already expressed at stage 1; ~1/9 the
  cost). Trades away final UV inpainting polish. Within-batch comparisons
  are unaffected; absolute values are not comparable to stage-2-scored
  runs.
- **N=5 members, 5 themes**: Vendi ceiling; cross-theme std is the error
  bar (n=5 themes) and whiskers overlap for most set-level differences —
  the texture-stage axes separation is the most robust quantitative gap.
  Avoid significance language; "consistent across themes" is the honest
  strength claim.
- **No human study** — DreamSim/HPSv2 are human-*aligned* proxies, not
  human data. Given the McDonnell grounding, name the perceptual study as
  the obvious next step.
- Single backbone (SD1.5 via Paint3D), one garment library (139 CC0
  meshes) — external validity limits.

## 5. Claim strength map (write to these, not past them)

| Claim | Strength |
|---|---|
| Axis sampling is the strongest texturing diversifier (Vendi + colour + DreamSim agree, texture stage) | Strong — largest gap, consistent across themes and metrics |
| WANDER buys quality/coherence at a small diversity cost | Strong on direction; HPSv2 gap is clear, Vendi cost small |
| CLIP mesh selection costs silhouette diversity vs random | Strong — two independent instruments (DINOv2-set + model-free IoU) agree |
| Pink noise adds colour diversity on plain prompts, washes out under descriptions | Moderate — visible, consistent, but small and colour-only |
| No method pays for diversity with theme drift | Moderate — CLIPScore flat, but CLIPScore is a blunt instrument |
| Mesh selection improves theme fit | Qualitative only — no adequate instrument (say so explicitly) |

## 6. Narrative arcs that make the report cohere

1. **Attribution**: three intervention points → three evaluation stages →
   each method's effect localized to the stage it acts on. The pipeline
   diagram, the stage design, and the results sections all mirror this.
2. **The diversity–coherence–quality triangle**: each method family
   occupies a corner (axes → diversity; WANDER → quality/coherence; τ and
   selection → the coherence↔diversity edge; pink noise → a free small
   push). No winner — a palette; recommend by use-case (e.g. background
   crowds: axes + pink; hero-adjacent crowds: WANDER).
3. **Failure-driven design**: 12-axis novelty → monochrome collapse →
   renderable-attributes constraint; incoherent outfits → pair-coherent
   sampling; broken meshes discovered THROUGH the pipeline's own renders →
   library curation. Each failure produced a design principle — this arc
   makes the methodology read as research, not plumbing.
4. **Training-free practicality**: every intervention composes with a
   stock pipeline at sampling time; total added cost is one CLIP pass +
   (optionally) a local 8B LLM. Contrast with optimization/fine-tuning
   families in the lit review.

## 7. Terminology discipline (consistency across the report)

- "mesh-fixed" (never "texture-fixed") for the pinned replay batch.
- "cells" for factorial conditions; "members" for crowd individuals;
  "themes" for prompts; "crowd" = one (cell, theme) set of N members.
- "set-level" vs "pairwise" metrics — keep the distinction sharp; it
  carries the Harrington justification.
- Pink noise: "1/(1+f)^α spectral reweighting, α=0.2" — avoid "equal
  energy per octave" (only true for α=1).
- WANDER method name in tables: "WANDER (6-axis)"; the failed variant:
  "12-axis full-description novelty" — always paired with its failure mode.
