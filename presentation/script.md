## Slide 1 — Title page

> Hi, I'm Joe, and this is LatentWardrobe: diversity-aware diffusion
> texturing for 3D crowds. How to generate a crowd of characters
> from a prompt that can match a theme, while remaining perceptually diverse.

---

## Slide 2 — Overview

---

## Slide 3 — Background: crowds in production                    [~0:40]

ON THE SLIDE
- One full-slide photo of a modern crowd scene: a UE5 City Sample street crowd
  still (Epic's docs/press material — screenshot-fair in an academic talk,
  credit "Epic Games, City Sample (2022)") or a comparable film crowd frame.
- Over/under the photo, three short problem lines:
  - crowds are everywhere: streets, stadiums, battles — thousands of members
  - authoring a unique character per member does not scale (artist time, memory)
  - so production recombines a FINITE template library: mix-and-match parts,
    tint variation, impostors at distance
  - state of the art today: MetaHuman's wardrobe system (UE 5.6+) — clothing
    variation = Mutable-plugin mix-and-match of wardrobe assets + runtime
    material colour parameters on a parametric body. Even the newest tooling
    varies by recombining and re-tinting FIXED assets; the textures themselves
    are never new.
- Citations pinned small at the bottom (full refs below).

REFERENCES (slide 3)
- Epic Games. *City Sample Project — Unreal Engine 5 Demonstration.* Official
  UE5 documentation, 2022. (crowds = MetaHuman library + "mix and match crowd
  system" + impostor LoD)
  https://dev.epicgames.com/documentation/unreal-engine/city-sample-project-unreal-engine-demonstration
- El-Ali, M., Tong, J., et al. *Zootopia Crowd Pipeline.* ACM SIGGRAPH 2016
  Talks. DOI: 10.1145/2897839.2927467.
- Walt Disney Animation Studios. *Creating the Mudskipper Pile in Disney's
  "Moana 2": A Slippery Problem Space.* ACM SIGGRAPH 2025 Talks.
  DOI: 10.1145/3721239.3734086. (template-crowd pipelines are current practice)
- Epic Games. *Tailoring Your Own Wardrobe Items.* MetaHuman documentation
  (UE 5.6+ parametric body + Mutable-based outfit customization; variation =
  wardrobe recombination + runtime material colour parameters).
  https://dev.epicgames.com/documentation/en-us/metahuman/tailoring-your-own-wardrobe-items

SAY (~110 words)
> Crowds are everywhere in games and film — streets, stadiums, battlefields —
> and every member needs a model, textures, animation. Authoring a unique
> character per member simply doesn't scale. So this is how it's actually
> done: Unreal's City Sample — the Matrix Awakens demo — builds its street
> crowds by mixing and matching a finite library of MetaHuman characters,
> with impostors past a certain distance. Disney's film pipelines do the same
> at larger scale — Zootopia populated a city with millions of agents from
> character-variant libraries, and they're still presenting crowd-pipeline
> talks at SIGGRAPH this year. And the newest tooling hasn't changed the
> structure: MetaHuman's current wardrobe system varies clothing by mixing and
> matching wardrobe assets and swapping material colours at runtime — the
> textures themselves are always fixed, pre-authored assets. Variety, in
> production, means recombining and re-tinting a finite library.

---

## Slide 4 — Background: perception of crowd variety             [~0:35]

ON THE SLIDE
- One figure from Clone Attack! — Figure 1 (the crowd with maximum clones) or
  Figure 2b (the 9 HSV-recoloured outfits — perfect because it *shows* their
  colour-modulation mechanism). Open PDF:
  https://www.scss.tcd.ie/Rachel.McDonnell/papers/Siggraph08.pdf, credit
  "McDonnell et al., SIGGRAPH 2008".
- Finding lines — all experimentally measured, across the two papers:
  - appearance clones detected far faster than motion clones (mean 5.7 s vs
    18 s in baselines; 28 s in crowd scenes) → spend the variety budget on
    appearance (Clone Attack!, 2008)
  - colour modulation was TESTED, not assumed: recolouring the clone pair
    roughly doubled detection time, 5.7 s → 12.3 s, p < 0.0001 (2008)
  - WHAT A CHARACTER WEARS drives perceived identity: clone false positives
    were "mainly due to clothing" — two suit-wearing models confused "even
    though their faces and bodies were very different" (2008)
  - clothing-texture variation is among the MOST effective variety channels:
    "head accessories, top texture and face texture variation are all equally
    effective at creating variety, whereas facial geometry alterations are
    less so" (Eye-catching Crowds, 2009)
- TAKEAWAY line (the slide's punchline, verbatim): *"What a character wears
  dominates perceived individuality — so diversifying the garments
  themselves, in both silhouette and texture, directly targets what viewers
  actually judge."*
- Footnote: the template assumption is still current — 2025 crowd research
  builds from 14 avatar templates (CrowdSplat).
- HONESTY NOTE (not on slide, for Q&A): the 2009 geometry finding concerns
  *facial* geometry, not garment silhouette — the silhouette claim rests on
  the 2008 clothing-confusion finding (same garment type → confusion despite
  different bodies, which implies different garments → differentiation), an
  inference, not a measured silhouette experiment. Say "targets", not
  "is proven to maximize".

REFERENCES (slide 4)
- McDonnell, R., Larkin, M., Dobbyn, S., Collins, S., O'Sullivan, C.
  *Clone Attack! Perception of Crowd Variety.* ACM Transactions on Graphics
  27(3), SIGGRAPH 2008. DOI: 10.1145/1360612.1360625.
- McDonnell, R., Larkin, M., Hernández, B., Rudomin, I., O'Sullivan, C.
  *Eye-catching Crowds: Saliency based Selective Variation.* ACM Transactions
  on Graphics 28(3), SIGGRAPH 2009.
- Sun, X., Xu, Y., Dingliana, J., O'Sullivan, C. *CrowdSplat: Exploring
  Gaussian Splatting for Crowd Rendering.* arXiv:2501.17792, 2025.

SAY (~135 words)
> So if template crowds are unavoidable, the question becomes perceptual: how
> much variety does a viewer need, and where? McDonnell and colleagues
> measured this across two papers. Appearance clones are detected far faster
> than motion clones — seconds versus half a minute — so appearance is where
> the variety budget belongs. Colour masking was tested, not assumed:
> recolouring a cloned pair doubled detection time. And critically, *what a
> character wears* is what viewers judge identity by: when participants
> confused two different characters as clones, it was almost always the
> clothing — two models in suits, mistaken for each other despite completely
> different faces. The follow-up study confirmed it from the other side:
> varying top texture was among the most effective ways to create variety,
> more so than altering geometry of the face.
>
> That's the link that justifies our design: if clothing dominates perceived
> individuality, then diversifying the garments themselves — silhouette *and*
> texture — directly targets what viewers actually judge. That's exactly the
> pair of levers this project modulates.

---

## Slide 5 — Background: diffusion texturing, and its catch      [~0:45]

ON THE SLIDE
- Left: texture diffusion in one image — either Paint3D's teaser (mesh in →
  textured mesh out; credit "Zeng et al., CVPR 2024") or one of our own
  members' before/after (untextured mesh render vs stage-2 turntable frame —
  rights-safe and it's our substrate).
- Right: mode collapse in one image — 4-5 near-identical outputs from one
  prompt. Rights-safe option: crop 4 samey tan members from our plain-prompt
  cell (contact sheet row 2).
- Two lines:
  - depth-conditioned Stable Diffusion paints a garment's UV map from text
    (TEXTure → Text2Tex → Paint3D)
  - the catch: guidance trades diversity for fidelity; newer models are
    MEASURABLY less diverse (Pareto-front study)

REFERENCES (slide 5)
- Zeng, X., et al. *Paint3D: Paint Anything 3D with Lighting-less Texture
  Diffusion Models.* CVPR 2024. arXiv:2312.13913.
- Richardson, E., et al. *TEXTure: Text-Guided Texturing of 3D Shapes.*
  SIGGRAPH 2023. / Chen, D.Z., et al. *Text2Tex.* ICCV 2023.
- Ho, J., Salimans, T. *Classifier-Free Diffusion Guidance.* 2022.
  arXiv:2207.12598.
- Astolfi, P., et al. *Consistency-Diversity-Realism Pareto Fronts of
  Conditional Image Generative Models.* 2024. arXiv:2406.10429.

SAY (~110 words)
> That texture lever is exactly what generative models have transformed.
> Texture diffusion — TEXTure, Text2Tex, and in our case Paint3D — paints a
> brand-new texture into a garment's UV map from a text prompt, conditioned
> on depth so it respects the geometry. So instead of recolouring one shared
> texture, every crowd member could get a genuinely unique outfit,
> automatically.
>
> Except diffusion models have their own cloning problem: mode collapse.
> Sample the same prompt ten times and you get near-duplicates. It's
> structural — classifier-free guidance explicitly trades diversity for
> fidelity, and a large Meta study showed newer models are measurably *less*
> diverse than older ones. Solving that, for crowds, is this project.

---

## Slide 6 — Related work: generating the garments themselves    [~0:25]

ON THE SLIDE
- Two teaser images side by side, credited: WordRobe (project page:
  https://wordrobe24.github.io/WordRobe_Page/) and GarmentDreamer
  (arXiv 2405.12420).
- One contrast line: per-asset optimisation, unposed / simulation-input
  outputs — vs a games pipeline that wants fixed, rigged, UV-stable meshes.
- Gap line, verbatim: *"No published work combines crowd-scale generation +
  garment texturing + diffusion diversity control."*

REFERENCES (slide 6)
- Srivastava, A., Manu, P., Raj, A., Jampani, V., Sharma, A. *WordRobe:
  Text-Guided Generation of Textured 3D Garments.* ECCV 2024. arXiv:2403.17541.
- Li, B., Li, X., Jiang, Y., Xie, T., Gao, F., Wang, H., Yang, Y., Jiang, C.
  *GarmentDreamer: 3DGS Guided Garment Synthesis with Diverse Geometry and
  Texture Details.* arXiv:2405.12420, 2024.

SAY (~75 words)
> One related line to position against: generating the garment geometry
> itself from text — WordRobe, GarmentDreamer. Impressive results, but these
> are per-asset optimisation pipelines producing unposed meshes; a games
> pipeline wants fixed, rigged, UV-stable assets. So we keep a fixed garment
> library and make *texture* carry the variety. And the combination this
> leaves open is exactly our gap: nobody has published crowd-scale generation
> plus garment texturing plus diffusion diversity control together.

---

## Slide 7 — Design

DIAGRAM SPEC — two diagrams, don't merge them (pipeline is linear, evolution
is a cycle; mixing them wrecks both):

Diagram 1 (main visual): left-to-right pipeline strip, three coloured factor
boxes with options stacked inside, real example images under each stage:

```
 THEME               FACTOR 2 (prompt)          FACTOR 1 (mesh)           FACTOR 3 (noise)
"medieval      ┌──────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
 villagers" ──►│ per-member prompt    │──►│ garment library     │──►│ Paint3D              │──► turntable ──► metrics
               │  ○ plain theme       │ │ │ (143 meshes)        │   │ 2-stage texturing    │    render
               │  ○ + colour word     │ │ │  ○ uniform random   │   │  ○ white noise       │
               │  ● WANDER-evolved ═╗ │ │ │  ● CLIP-weighted ◄──┼───┼─ (evolved desc. is   │
               └────────────────────╫─┘ │ └─────────────────────┘   │  the CLIP anchor)    │
                                    ║   └── prompt text ───────────►│  ○ pink noise        │
                                    ╚═► [expands into Diagram 2]    └──────────────────────┘
 images: prompt snippet | garment-render strip + score bars | noise patches | turntable frame | chart thumb
```
Wiring that must be right: the WANDER-evolved description feeds BOTH the
Paint3D prompt AND that member's CLIP mesh anchor (the novel coupling); in
plain/colour variants the mesh anchor is the theme itself.

Diagram 2 (second build on the slide, or backup slide): the evolution cycle:

```
        ┌───────────── POOL: n clothing descriptions ─────────────┐
        │ "worn rustic brown wool patched clothing and apron" ... │◄─────────────┐
        └──────────────┬───────────────────────────────────────── ┘              │
                       │ pick random parent(s) + random clothing axis            │
                       ▼                                                         │
         ┌─────────────────────────┐  50/50  ┌──────────────────────┐            │
         │ MUTATE (llama-3.1-8B):  │   or    │ CROSSOVER: child     │            │
         │ "COMPLETELY CHANGE      │         │ inherits each field  │            │
         │  the {axis} value"      │         │ from either parent   │            │
         └────────────┬────────────┘         └──────────┬───────────┘            │
                      └──────────────┬──────────────────┘                        │
                                     ▼                                           │
               ┌────────────────────────────────────────┐   child beats the      │
               │ SCORE: CLIP text embed → novelty =     │   least-novel member?  │
               │ mean cos dist to k=3 nearest in pool   │── yes: REPLACE it ─────┘
               └────────────────────────────────────────┘── no: discard
        repeat 3n generations → final pool = the n members' prompts
        annotate a real logged mutation:  colour "brown" → "scarlet"  (accepted)
```

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

## Slide 8 — Implementation / Implementation Issues

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

## Slide 9 — Demo of Artefact

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

## Slide 10 — Evaluation                                   [1:45]

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

## Slide 11 — Conclusions

** Draft — trim once slide 7 is final; total talk must land ≤ 9:00, target 8:00-8:30. **

ON THE SLIDE
- 3 contributions / 2 limitations / 2 future lines, one short line each:
  - C1: first controlled factorial of diversity interventions for diffusion-textured 3D crowds
  - C2: foreground-masked, multi-metric evaluation harness
  - C3: diagnosed failure mode of text-space novelty search + principled fix
  - L1: one theme, N=10 per cell; L2: no human perceptual study
  - F1: diversity-forced initialisation (Method 5c); F2: during-sampling methods (particle guidance / CADS);
    F3: a perceptual study measuring the effect of garment MESH (silhouette) variation
    on clone detection — the missing experiment in the Clone Attack!/Eye-catching
    Crowds line, and the one our silhouette-diversity claim would want

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
