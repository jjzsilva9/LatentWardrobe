## Slide 1 — Title page

> Hi, I'm Joe, and this is LatentWardrobe: diversity-aware diffusion
> texturing for 3D crowds. How to generate a crowd of characters
> from a prompt that can match a theme, while remaining perceptually diverse.

---

## Slide 2 — Overview

---

## Slide 3 — Background + Related Work

Three research areas meet in this project; the slide is built as three columns
with real citations, converging on the gap statement. Sources:
`.claude/dissertation-plan.md` (Background reading) + `.claude/lit-review-diffusion-diversity.md`
(31-paper verified review).

ON THE SLIDE — three columns + a gap line
1. **Generating 3D clothing**
   - Full text-to-3D garments: WordRobe (Srivastava et al. 2024); GaussianDreamer
     (Yi et al. 2023) / GarmentDreamer (3DGS garments)
     *(!! WordRobe + GaussianDreamer are NOT in the verified lit review yet —
     verify authors/venues before the slide goes in front of markers)*
   - Texturing fixed meshes (our line): TEXTure (Richardson et al. 2023),
     Text2Tex (Chen et al., ICCV 2023), Point-UV Diffusion (Yu et al., ICCV 2023),
     **Paint3D (Zeng et al., CVPR 2024)** — highlighted, it's our substrate
2. **Crowd perception — why diversity matters**
   - **Clone Attack! (McDonnell et al., SIGGRAPH 2008)** — viewers detect
     appearance clones in crowds; colour/texture variation is what masks them
   - (optional second: Eye-catching Crowds, McDonnell et al., SIGGRAPH 2009)
3. **Diversity in diffusion models**
   - Why it collapses: CFG (Ho & Salimans 2022); newer models measurably less
     diverse (Astolfi et al. 2024, Pareto fronts)
   - Noise-space: pink-noise / collapse recovery (*It's Never Too Late*, CVPR 2026)
   - Prompt-space: Prompt Expansion (Datta et al. 2023); WANDER novelty search
     (arXiv 2511.00686)
   - Selection: k-DPPs (Kulesza & Taskar, ICML 2011)
- Bottom line, verbatim: *"No published work combines all three: crowd-scale
  generation + garment texturing + diffusion diversity control."* (gap claim
  verified in the lit review, 2026-07-06)

SAY (~215 words)
> Three research areas meet in this project.
>
> First, generating 3D clothing. One line of work generates whole garments from
> text — WordRobe, GaussianDreamer. Impressive, but game crowds need predictable,
> game-ready assets. So the practical line textures *fixed* meshes instead:
> TEXTure, Text2Tex, and Paint3D, which paints a texture into a garment's UV map
> with depth-conditioned Stable Diffusion. Paint3D is our substrate: mesh in,
> prompt in, textured garment out.
>
> Second, crowd perception — why diversity is the thing to optimize. Clone
> Attack!, McDonnell and colleagues, SIGGRAPH 2008: viewers reliably spot
> appearance clones in crowds, and varying clothing colour and texture is
> precisely what masks them. That grounds this whole project perceptually —
> appearance variation is what buys crowd realism, and it's also what our
> quantitative metrics stand in for.
>
> Third, diversity in diffusion models. Sampled repeatedly, diffusion models
> under-deliver variety: classifier-free guidance explicitly trades diversity
> for fidelity, and newer models are measurably worse than older ones. The
> counter-measures are training-free interventions at different points: reshape
> the initial noise — pink noise; rewrite the prompt with an LLM — WANDER's
> novelty search; or over-generate and select with determinantal point processes.
>
> Nobody has published the combination of all three areas. That's the gap this
> project occupies.

---

## Slide 4 — Design

ON THE SLIDE
- Main visual: left-to-right pipeline diagram (build this properly — it's the
  slide markers will photograph):
  `theme prompt -> garment library (CLIP-anchored mesh sampling) ->
   per-member prompt (plain / +colour word / LLM-evolved) ->
   Paint3D 2-stage texturing (initial latent: white vs pink noise) ->
   turntable render -> metrics`
- Small factorial table: 3 factors x levels -> 7 cells (+2 Method 5b), N=10 each.
- Callout box: *"White/pink pairs are seed-matched: same garments, same seeds —
  the only difference is the initial latent."*

SAY (~205 words)
> The pipeline: a single theme prompt goes in — "medieval villagers" — and a
> textured crowd comes out. Three design decisions matter.
>
> First, mesh selection. Each member's garments are sampled from a mesh library,
> either uniformly at random, or weighted by CLIP similarity between the garment
> and the theme — so a villager is more likely to draw a tunic than a disco skirt.
> That's a diversity/coherence dial, and we measure both sides of it.
>
> Second, the per-member prompt. Three variants: the plain theme; the theme plus
> one random colour word — deliberately the dumbest possible interventionpoint;
> and a description evolved by a local LLM using a WANDER-style novelty search
> over structured clothing attributes.
>
> Third, the initial latent. Paint3D normally starts from white noise; we can
> swap in pink noise, which redistributes energy toward low frequencies and
> encourages globally different colour layouts rather than pixel-level jitter.
>
> Those three factors give a reduced factorial — seven cells, ten members each.
> And the pairs are controlled: every white-versus-pink pair is seed-matched,
> same garments, same seeds, so any difference is the latent and nothing else.
> Every crowd is scored on diversity *and* theme coherence, because diversity is
> trivial if you're allowed to drift off-theme.

---

## Slide 5 — Implementation / Implementation Issues

ON THE SLIDE
- Compact stack strip (one line of logos/names): Paint3D (SD 1.5 + depth
  ControlNet) · WSL2 + CUDA · llama-3.1-8B via Ollama, fully local · separate
  generation/evaluation conda envs · ~21 min per member -> ~27 h per batch.
- Three problem->fix pairs, two columns, one line each (below).
- Optional small photo/screenshot of the supervisor log — makes "unattended"
  concrete.

SAY (~195 words)
> Everything runs on one machine: Paint3D on Stable Diffusion 1.5 with a depth
> ControlNet inside WSL2, and the prompt evolution on a local llama-3.1-8B via
> Ollama — no cloud APIs, so everything is reproducible or logged. One member
> takes about twenty-one minutes of GPU time, so a full batch is twenty-seven
> unattended hours — and that's where the real engineering lived. Three examples.
>
> First: WSL's network-drive layer silently stalls under sustained writes — the
> GPU just sits idle, no error, no crash. This is a known, unresolved WSL issue.
> The fix: generate to local disk, mirror every member to the network share the
> moment it finishes, and run a supervisor with a watchdog that restarts the
> batch when progress stops. After that, a mid-batch machine crash cost one
> member instead of a night.
>
> Second: LLM evolution is not reproducible the way a seeded sampler is — the
> same seed does not give the same tokens. So every evolved population and every
> mutation, accepted or rejected, is persisted next to the outputs. Crowds are
> replayable and the search is auditable.
>
> Third: evaluation has to separate garment from background. A colour heuristic
> misclassified cream garments as background — so instead we recover the
> renderer's own per-pixel foreground mask, which it happens to bake into one of
> its diagnostic videos. Exact mask, no heuristic.

---

## Slide 6 — Demo of Artefact

Live generation is impossible in-slot (~21 min/member). The demo is pre-rendered
assets presented live. Have everything embedded AND a local fallback copy.

ON THE SLIDE (two builds)
1. Full-bleed: `plots/contact_sheet_frame000.png` — the money shot, whole slide.
2. Then 2-3 turntable videos, muted, looping:
   - `4_clip_white_color/medieval_villagers/member_03/stage2/tile_res_0/render_rgb.mp4` (blue paisley shirt)
   - `8_clip_white_llm6/medieval_villagers/member_00/stage2/tile_res_0/render_rgb.mp4` (plaid dress — Method 5b)
   - one brown member from the failed WANDER cells for contrast, e.g.
     `6_clip_white_llm/medieval_villagers/member_03/stage2/tile_res_0/render_rgb.mp4`

SAY (~125 words + ~15 s of letting a video loop)
> [contact sheet up] This is the artefact's output. Each row is one method, each
> column one crowd member — all from the same theme prompt.
>
> Row one is the baseline. Look at row four — the theme plus a single random
> colour word: greens, blues, gold, orange, and it still reads as one coherent
> medieval crowd. And the bottom two rows — prompts evolved by an LLM with
> novelty search over clothing attributes: plaids, stripes, pale blues, greens,
> and the strongest overall diversity in the study. Getting that method to work
> required diagnosing an instructive failure — hold that thought for the
> evaluation.
>
> [turntables] And these are full 3D assets, not images — textured meshes you
> can drop straight into an engine. [let one loop ~10s]

---

## Slide 7 — Evaluation                                    [1:45]

All 9 cells generated + evaluated (2026-07-10). Numbers final.

ON THE SLIDE
- TWO charts only: `plots/color_chi2_mean.png` + `plots/vendi_score_dinov2.png`
  (regenerated 2026-07-10 with all 9 methods). The other six exist for questions.
- One line naming the battery: DINOv2 patchwise, DreamSim, LPIPS, pixel-L2,
  colour χ², Vendi, DPP + CLIPScore coherence.
- Before/after strip: `plots/wander_before_after.png` — the failed 12-axis
  WANDER crowd (all brown) directly above the fixed 6-axis one (plaid, scarlet,
  stripes, pale blue, green). Same theme, same pipeline, fixed search.
  (Cells 6/7 are deliberately NOT in the main charts/contact sheet — they appear
  only in this strip, as the "before".)
- Three takeaway lines matching the three findings below.

KEY NUMBERS (all n=10)
| cell                          | Vendi | DreamSim | colour χ² | CLIP coherence |
|-------------------------------|-------|----------|-----------|----------------|
| 1 random mesh baseline        | 4.73  | 0.509    | 0.497     | 0.198          |
| 2 CLIP mesh                   | 3.56  | 0.417    | 0.526     | 0.208          |
| 4 + colour word               | 4.17  | 0.522    | 0.855     | 0.194          |
| 6 WANDER (12-axis, full nov.) | 4.54  | 0.413    | 0.296     | 0.198          |
| 8 WANDER 5b (6-axis, white)   | **5.01** | 0.525 | 0.715     | 0.196          |
| 9 WANDER 5b (6-axis, pink)    | 4.67  | 0.525    | 0.711     | 0.197          |

SAY (~230 words)
> Crowds are scored with a battery of diversity metrics — perceptual, feature-
> space, and colour — plus CLIP theme-coherence. Three findings.
>
> First, coherence is flat — 0.19 to 0.21 across every method. Nothing here
> paid for diversity with theme drift.
>
> Second, the cheapest intervention was a big win on colour: adding one random
> colour word raised colour diversity from about 0.50 to 0.86 — and you saw it
> directly in the contact sheet.
>
> Third — the failure, and what fixing it bought. Our first WANDER adaptation
> scored *below the plain baseline* on colour: every member brown. Diagnosis:
> we scored novelty on the whole person-description, so the search happily
> mutated age and posture — attributes a clothing texture can't even show —
> and colour was never once challenged in thirty generations. The fix stayed
> faithful to WANDER's own design: all ten of their emitters force substantive
> *visual* change, so we restricted evolution to six clothing-only attributes,
> forced large mutation steps, and scored novelty on exactly the text that
> drives rendering.
>
> The corrected method more than doubled colour diversity — 0.30 to 0.72 —
> and reached a Vendi score of 5.0 effective members, the highest in the
> entire study, beating even the random-mesh baseline. Same pipeline, same
> compute; the only change was making the search optimize what the renderer
> can actually see. And coherence didn't move.

---

## Slide 8 — Conclusions

** Draft — trim once slide 7 is final; total talk must land ≤ 9:00, target 8:00-8:30. **

ON THE SLIDE
- 3 contributions / 2 limitations / 2 future lines, one short line each:
  - C1: first controlled factorial of diversity interventions for diffusion-textured 3D crowds
  - C2: foreground-masked, multi-metric evaluation harness
  - C3: diagnosed failure mode of text-space novelty search + principled fix
  - L1: one theme, N=10 per cell; L2: no human perceptual study
  - F1: diversity-forced initialisation (Method 5c); F2: during-sampling methods (particle guidance / CADS)

SAY (~76 words)
> To conclude: this project contributes a controlled factorial comparison of
> diversity interventions for diffusion-textured crowds; a foreground-masked,
> multi-metric evaluation harness; and a diagnosed failure mode of text-space
> novelty search, with a fix grounded in the original paper. Limitations are
> scale — one theme, ten members per cell — and the absence of a human study.
> The clearest next steps are diversifying the search's initialisation and
> testing during-sampling methods. Thank you — happy to take questions.

---

## Q&A prep

- **Why not fine-tune for diversity?** Training-free methods compose with any
  checkpoint, need no data collection, and can't damage texture quality;
  orthogonal to fine-tuning, not competing with it.
- **How does pink noise work?** FFT reweighting of the initial latent by
  1/(1+f)^α with α=0.2 — boosts low-frequency energy so samples differ in global
  layout rather than pixel jitter; above α≈0.3 quality visibly degrades.
- **Why only textures, not geometry?** Texture is where diffusion collapse lives
  and what's automatable today; silhouette variety comes from the mesh library +
  sampling policy. Generative geometry is future work.
- **Why a local 8B LLM rather than GPT-4 class?** No API cost/rate limits inside
  a 30-generation × multi-cell loop, fully self-contained artefact; capability
  gap is acknowledged (WANDER used GPT-4o-mini) and is part of the discussion.
- **Isn't CLIP-text novelty a weak signal?** Yes — WANDER scores novelty on
  generated images; that would cost a full Paint3D run per candidate (~20 min)
  here. The substitution is documented as a downgrade, and its consequences are
  a finding, not an accident.
- **Is N=10 enough?** Each pairwise metric averages 45 pairs × 4 views; headline
  effects dwarf pairwise std, but formal significance over repeated crowds is
  future work — compute is the binding constraint (~3.5 h/cell).
- **How did Method 5b beat even the random-mesh baseline on Vendi (5.01 vs
  4.73)?** The baseline's diversity comes only from mesh variety + seed
  stochasticity; 5b adds deliberate texture-content variation on top of
  CLIP-selected meshes — so members differ in silhouette AND painted
  appearance, and Vendi (effective distinct members in DINOv2 space) rewards
  both. It didn't beat the colour-word cell on colour χ² though (0.72 vs
  0.86) — random colour words explore hue space more aggressively than an 8B
  LLM's "plausible villager" palette. Fair trade: 5b buys more overall
  diversity and semantic richness (materials, patterns, condition), not just hue.
- **Did pink noise matter in the end?** Within seed-matched pairs it nudged
  perceptual metrics up in the plain and colour cells, was ~neutral for the
  LLM cells, and slightly lowered Vendi in some pairs — honest summary:
  modest, inconsistent, and much smaller than the prompt-side effects.
- **Were both WANDER runs identical apart from the fix?** Same master seed,
  same RNG stream (deliberately not keyed on the new options), so the initial
  drafts and the mutate/crossover schedule are comparable; the LLM calls
  themselves are not bit-reproducible, which is why every population and
  mutation is logged.
