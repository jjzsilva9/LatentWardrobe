# Literature review: Output diversity in diffusion models — causes and control

Compiled 2026-07-02 for the LatentWardrobe dissertation (see `.claude/dissertation-plan.md`). No published survey paper dedicated to this topic exists as of this date (checked explicitly — general diffusion surveys like Yang et al.'s *Diffusion Models: A Comprehensive Survey*, ACM Computing Surveys, treat diversity only in passing). This review was therefore compiled from primary sources.

**Verification levels** (tagged per paper):
- **[V]** = verified — the paper's arXiv abstract page was fetched and read directly; mechanism and claims below come from the paper's own text.
- **[S]** = search-level only — known from search results/snippets; verify against the actual paper before citing in the dissertation.

Venues stated where the paper page or search results named one; otherwise arXiv-only. Note several key papers here are 2025–2026 and not yet widely replicated — flagged inline where it matters.

---

## 1. Scope and problem statement

Conditional diffusion models (including Stable Diffusion 1.5, the backbone of Paint3D) systematically under-deliver diversity when sampled repeatedly under the same or similar conditioning: outputs collapse toward a small number of modes even across different random seeds. For LatentWardrobe this is the central technical obstacle — a crowd generated from one theme prompt must not look like clones — so the literature on *why* collapse happens and *how* to counteract it is the direct scientific context for Stage 1 of our pipeline.

The field has no consolidated name; relevant work appears under "diversity collapse," "mode collapse in diffusion models," "non-i.i.d. sampling," "initial noise design," and "guidance dynamics." It is young (essentially 2023–2026) and fragmented — hence this review.

## 2. Why diversity collapses: classifier-free guidance and its dynamics

- **Classifier-Free Diffusion Guidance** — Ho & Salimans, 2022. arXiv:2207.12598. **[V-prior]** (established earlier in project). CFG is the standard mechanism trading diversity for fidelity/alignment: sampling with guidance scale > 1 sharpens the conditional distribution. Paint3D's own configs use guidance_scale 7 (stage 1 txt2img) and 3 (inpaint), so we inherit this tradeoff by default.

- **Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models** — Jin, Shi & Gu. ICLR 2026. arXiv:2509.22007. **[V]** Decomposes CFG's effect over the sampling trajectory into three stages: (1) *Direction Shift* — early guidance accelerates convergence toward the weighted mean, introducing initialization bias and norm growth; (2) *Mode Separation* — inherited bias suppresses weaker modes, reducing global diversity; (3) *Concentration* — late guidance amplifies within-mode contraction, killing fine-grained variability. Empirically: early strong guidance erodes global diversity; late strong guidance suppresses within-mode variation. Proposes time-varying guidance schedules, which "consistently improve both quality and diversity." Key implication for us: *when* guidance is applied matters as much as how strong it is.

- **Consistency-Diversity-Realism Pareto Fronts of Conditional Image Generative Models** — Astolfi et al. (Meta). 2024. arXiv:2406.10429. **[V]** Large empirical study measuring prompt-consistency, diversity, and realism as a Pareto frontier across model generations. Central finding: "realism and consistency can both be improved simultaneously; however there exists a clear tradeoff between realism/consistency and diversity" — and newer models (SDXL-era and later) are *less* diverse than earlier ones; the original Latent Diffusion model outperformed recent versions on their combined criteria. Two consequences for us: (a) our three-axis evaluation design (coherence/diversity/quality, already in the plan) matches the field's best current methodology; (b) Paint3D's SD1.5 backbone being an *older* model is, for diversity purposes, arguably an advantage.

## 3. Taxonomy of diversity-enhancing methods

### 3.1 Initial-noise design (before sampling starts)

The cheapest intervention point: change how the batch of initial latents z_T is drawn, leave the sampler untouched.

- **Couple to Control: Joint Initial Noise Design in Diffusion Models** — Jia, Shen & Wang. 2026. arXiv:2605.11311. **[V]** The unifying framework for this family. Observation: drawing a batch of noises i.i.d. is just one choice among many valid *joint* distributions whose marginals are each standard Gaussian — and marginal-Gaussianity is all the pretrained model needs. Design the *dependence structure* across the batch instead: their repulsive Gaussian coupling (negative cross-sample covariance) "improves gallery diversity on SD1.5, SDXL, and SD3 while largely preserving prompt alignment and image quality," at the same sampling cost as independent generation, matching or beating test-time noise-optimization baselines. Framework explicitly covers several existing methods as special cases (the abstract does not enumerate which). Directly tested on SD1.5 — our backbone. **Strongest single foundation for a noise-design Stage 1 method.**

- **Antithetic Noise in Diffusion Models** — Jia, Liu, Song, Yuan, Shen & Wang. 2025 (rev. 2026). arXiv:2506.06185. **[V]** Same group; the precursor. Pairs each noise with its negation (z, −z). Primary contribution is *uncertainty quantification* (up to 90% narrower confidence intervals), explained via an "affine antisymmetric" property of learned score functions; diversity improvement is a secondary explored application, not the headline claim (earlier search snippets overstate this). Training-free, model-agnostic, zero runtime overhead — worth including as a near-free baseline, but cite it accurately.

- **Diverse Diffusion: Enhancing Image Diversity in Text-to-Image Generation** — Zameshina, Teytaud & Najman. 2023. arXiv:2310.12583. **[V]** Early, simple entry: sequentially draw SD latent vectors, keeping only those sufficiently *distant* from the ones already accepted, until the batch is full. Evaluated with color diversity, LPIPS, and demographic representation. Methodologically close to our project twice over: it is (a) distance-constrained sampling in exactly our latent space, and (b) evaluated with two of our three chosen metric families. A must-cite.

- **Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior (DivIn)** — Li, Liu & Kawaguchi. ICML 2026 spotlight. arXiv:2606.02453. **[V]** Argues standard Gaussian initialization is itself a cause of collapse because it ignores the guidance potential landscape; samples initial noise from a reweighted ("guidance potential posterior") prior via Langevin dynamics, steering initializations away from collapse-prone basins. "Combining [it] with trajectory-based diversity methods significantly expands the diversity-quality Pareto frontier" — i.e., initialization-level and sampling-level methods compose.

- **It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models** — CVPR 2026. arXiv:2601.00090. **[V]** Test-time noise optimization objective that mitigates mode collapse while preserving fidelity; additionally analyzes the *frequency characteristics* of initial noise, showing "alternative noise initializations with different frequency profiles can improve both optimization and search" (pink-noise-style structured initializations, per the paper's experiments). This — not CADS — is the source for any "low-frequency structure in the noise matters for diversity" claim.

- **Diverse Text-to-Image Generation via Contrastive Noise Optimization** — Kim, Um & Ye. ICLR 2026. arXiv:2510.03813. **[V]** Optimizes a batch of initial noises with a contrastive loss in the Tweedie-denoised data space: instances pushed apart for diversity, anchored to a reference for fidelity. Claims a superior quality-diversity Pareto frontier, robust to hyperparameters. Costlier than coupling (requires optimization passes) but stronger control.

- Adjacent (quality-oriented) noise design, establishing that initial noise is a recognized general-purpose lever: **InitNO** (CVPR 2024, arXiv:2404.04650) **[S]** optimizes initial noise for prompt faithfulness via attention scores; **Golden Noise for Diffusion Models** (arXiv:2411.09502) **[S]** learns a noise-prompt mapping; **All Seeds Are Not Equal** (arXiv:2411.18810) **[S]** shows seed choice systematically affects compositional success. None target diversity directly; cite only as context.

- **Quasi-Monte Carlo / low-discrepancy noise: gap in the literature.** Our searches found *no published paper applying QMC/Sobol sequences to diffusion initial-noise batches for output diversity*. QMC tooling exists (`torch.quasirandom.SobolEngine`; BoTorch's `NormalQMCEngine` with inverse-CDF transform) and QMC-for-generative-models exists in other roles (e.g. discrepancy-based inference, arXiv:2106.11561 **[S]**), but the specific idea in our original Method 2 appears unpublished. Double-edged: it would be novel, but there is also *no evidence* input-space low-discrepancy survives the sampler's nonlinearity — whereas repulsive coupling (Couple to Control) has direct SD1.5 evidence. If pursued, frame QMC as one instance of joint noise design and benchmark against repulsive coupling and antithetic pairing.

### 3.2 During-sampling interaction between samples (particle methods)

Change the reverse-diffusion dynamics so the batch's samples repel each other while denoising.

- **Particle Guidance: non-I.I.D. Diverse Sampling with Diffusion Models** — Corso, Xu, De Bortoli, Barzilay & Jaakkola. 2023. arXiv:2310.13102. **[V]** The foundational paper: augments the reverse process with a joint-particle time-evolving potential whose gradient acts as a repulsive force between samples (pairwise-similarity kernels). Claims increased diversity without quality loss in conditional image generation; 13% median error reduction in molecular conformer generation. Code released (gcorso/particle-guidance).

- **Diverse Sampling in Diffusion Models with Marginal Preserving Particle Guidance (EDDY)** — Vinograd, Achituve & Fetaya. 2026. arXiv:2605.06553. **[V]** Fixes the known defect of naive repulsion (it distorts each sample's marginal distribution, i.e. can hurt per-sample quality): exploits Fokker-Planck symmetries to build divergence-free, anti-symmetric drift perturbations that alter joint trajectories while *provably preserving the evolving marginal*. Improves diversity with strong distributional fidelity; offers practical approximations for perceptual-embedding-based image use.

- **SPARKE: Scalable Prompt-Aware Diversity Guidance via RKE Score** — Jalali, Lei, Gohari & Farnia. 2025. arXiv:2506.10173. **[V]** Diversity guidance driven by a conditional Rényi kernel entropy measure, made prompt-aware (diversity is measured relative to semantically similar prompts) and reduced from O(n³) to O(n) — relevant if our candidate pools get large.

- **Don't Settle at the Mode! Mitigating Diversity Collapse in Pretrained Flow Models via Feature Self-Guidance** — Bhat, Parihar, Bhat & Venkatesh Babu. ECCV 2026. arXiv:2606.27371. **[V]** For flow-matching models: training-free batch-level *feature dispersal* inside the network plus manifold regularization. Notable for us mainly because it **explicitly benchmarks against post-hoc sample selection via external reward models and claims to beat it on both effectiveness and efficiency** — the most direct published challenge to an overgenerate-then-select design like our Stage 2. Our dissertation should cite this and answer it (see §5).

- **Repulsive Score Distillation** — arXiv:2406.16683. **[S]** Repulsion among an ensemble of particles in a variational/score-distillation setting; adjacent rather than central.

### 3.3 Condition-side perturbation

Leave noise and sampler alone; perturb the conditioning embedding.

- **CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling** — Sadat, Buhmann, Bradley, Hilliges & Weber (ETH/Disney). ICLR 2024. arXiv:2310.17347. **[V]** Adds scheduled, monotonically *decreasing* Gaussian noise to the conditioning vector during inference (high perturbation early → exact condition late). Directly targets diversity loss at high guidance scale: "increases generation diversity, especially at high guidance scales, with minimal loss of sample quality"; SOTA class-conditional ImageNet FIDs with a pretrained model. Compatible with any sampler. Natural fit for us since Paint3D runs at guidance 7.
  *(Correction note: an earlier working summary in this project misattributed a "low-frequency noise drives diversity" finding to CADS. CADS perturbs the condition embedding, not the initial noise; the frequency finding belongs to arXiv:2601.00090 above.)*

- **PromptMoG** — arXiv:2511.20251. **[S]** Samples prompt embeddings from a mixture-of-Gaussians around the encoded prompt to raise diversity in long-prompt generation. Same family as CADS.

### 3.4 Fixing the guidance mechanism itself

Diversity loss is partly an artifact of *how* CFG is applied; these change that, orthogonally to noise design.

- **Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models** — Kynkäänniemi, Aittala, Karras, Laine, Aila & Lehtinen. NeurIPS 2024. arXiv:2404.07724. **[V]** Guidance is "clearly harmful toward the beginning of the chain (high noise levels), largely unnecessary toward the end, and only beneficial in the middle." Restricting CFG to a middle noise-level interval improved ImageNet-512 FID 1.81→1.40 and validated on SDXL. Trivially applicable to Paint3D (one scheduler-level change) and composes with everything above. Authors recommend exposing the interval as a standard hyperparameter.

- **Guiding a Diffusion Model with a Bad Version of Itself (Autoguidance)** — Karras, Aittala, Kynkäänniemi, Lehtinen, Aila & Laine. NeurIPS 2024 (oral). arXiv:2406.02507. **[V]** Replaces the unconditional branch of CFG with a smaller/less-trained version of the same model, yielding "disentangled control over image quality without compromising the amount of variation" — record ImageNet FIDs (1.01 @ 64², 1.25 @ 512²). The cleanest evidence that CFG's quality-diversity entanglement is not fundamental. Caveat for us: needs a "bad" companion model; for SD1.5 an off-the-shelf weaker checkpoint would have to be chosen or made — heavier lift than guidance-interval.

- Time-varying guidance schedules — proposed within arXiv:2509.22007 **[V]** (§2 above) as the mitigation its theory implies.

### 3.5 Post-hoc selection (overgenerate, then select a diverse subset)

- Covered in the project plan already: **k-DPPs** (Kulesza & Taskar, ICML 2011) and greedy farthest-point selection over output-space embeddings; **Diverse Diffusion** (§3.1) is effectively a latent-space rejection variant of the same idea from 2023. The plan's Stage 2 lives here.
- Honest tension to address: *Don't Settle at the Mode* (§3.2) claims during-sampling guidance beats selection-based baselines in both effectiveness and efficiency. Their selection baseline used external reward models rather than perceptual-diversity selection over rendered views, and their setting (single images) differs from ours (textures judged on meshes after a two-stage UV pipeline, where per-candidate cost is high and the selection embedding is domain-specific) — but the dissertation must engage with this comparison rather than ignore it, ideally empirically (Stage 2 selection vs. a during-sampling repulsion method at equal compute).

## 4. Mechanistic findings worth building on

1. **Initial noise materially determines output diversity** — converging evidence from DivIn (initialization landscape), It's Never Too Late (frequency profile), Couple to Control / antithetic work (dependence structure), and the seed-effect literature (All Seeds Are Not Equal **[S]**). The "everything happens in the sampler, noise is interchangeable" intuition is dead.
2. **CFG's diversity cost is stage-dependent and partly avoidable** — early guidance harms global diversity, late guidance harms within-mode variation (arXiv:2509.22007); confining guidance to a middle interval (arXiv:2404.07724) or replacing the guiding model (arXiv:2406.02507) recovers much of the loss.
3. **Initialization-level and sampling-level interventions compose** — stated directly by DivIn; consistent with CADS + noise-design being orthogonal levers.
4. **The field measures success as a quality-diversity (or consistency-diversity-realism) Pareto frontier**, not a single number (arXiv:2406.10429, arXiv:2510.03813) — matching our planned evaluation design.

## 5. Implications for LatentWardrobe (our synthesis, not literature)

- **Original Method 2 (QMC over z_T) should be reframed, not deleted:** it is one point in the joint-noise-design family (§3.1), which now has a general framework (Couple to Control) and empirically validated members on our exact backbone. A defensible dissertation experiment: within Stage 1, compare i.i.d. baseline vs. antithetic pairing vs. repulsive Gaussian coupling vs. QMC/Sobol (novel, unpublished) at identical cost, measured on our three axes. QMC becomes a contribution *within* an established framing instead of an isolated bet.
- **A during-sampling method (Particle Guidance or EDDY) deserves a slot** — the field's evidence for these is strong, they're training-free, and *Don't Settle*'s challenge to selection-based designs makes an empirical noise-design vs. particle-repulsion vs. Stage-2-selection comparison genuinely interesting rather than decorative.
- **CADS is the natural third family representative** (condition-side), especially given Paint3D's high default guidance scale.
- **Guidance interval / guidance scale must be controlled for** in every comparison — per §2 it is a confound large enough to swamp method differences; consider adopting limited-interval guidance globally as it composes with all methods.
- **Stage 2 (k-DPP/farthest-point selection) stays**, but the dissertation should benchmark it against a during-sampling alternative and cite *Don't Settle* explicitly.
- Note the original three-method framing (prompt-grammar / QMC / CLIP-anchor) no longer matches the field's structure; the taxonomy above (noise design / particle interaction / condition perturbation / guidance repair / selection) is the literature-aligned way to organize the methodology chapter. Restructuring decision deferred to discussion — see plan doc.

### 3.6 LLM-driven prompt rewriting (before the prompt reaches the diffusion model at all)

Orthogonal to every method in §3.1-§3.5: leave the sampler/noise/guidance/conditioning-embedding untouched, and instead have an LLM rewrite or expand the *text* prompt itself into multiple diverse variants before each one goes through an otherwise-unmodified T2I pipeline. Compiled 2026-07-06 via WebSearch/WebFetch (fetched abstract/HTML pages directly, not from training memory), prompted by the user asking specifically for this angle after Method 4 (pink noise) was confirmed.

- **WANDER — Evolve to Inspire: Novelty Search for Diverse Image Generation.** arXiv:2511.00686. **[V]** Already the anchor for the plan's "LLM-per-theme grammar" future method (see dissertation-plan.md). LLM (GPT-4o-mini) mutates prompts along semantic "emitters" (style/composition/atmosphere); novelty scored via CLIP embeddings of resulting images against a FLUX-DEV reference. First/only paper found combining an evolutionary novelty-search loop with LLM prompt mutation for diffusion diversity — this framing appears genuinely novel, not an established subfield.

- **Prompt Expansion for Adaptive Text-to-Image Generation** — Datta, Ku, Ramachandran, Anderson. arXiv:2312.16720 (Dec 2023). **[V]** WANDER's clearest methodological precursor. A fine-tuned, prompt-tuned PaLM 2 (1B param) expands one user query into N=4 diverse text prompts, each independently sent to the T2I model, explicitly to fix "lack of diversity when sampling image sets." Diversity measured as variance of CoCa image embeddings across the resulting set. No evolutionary/novelty-search loop and no reference-model scoring (simpler than WANDER) — a good "minimal version" citation to establish the family before introducing WANDER's fuller machinery.

- **Using LLMs as prompt modifier to avoid biases in AI image generators** — Peinl. arXiv:2504.11104 (Apr 2025). **[V]** An LLM rewrites/reformulates prompts before SDXL / SD3.5 / Flux, explicitly to raise output diversity and counter demographic-bias collapse (a documented diffusion failure mode directly analogous to crowd-clone risk). Notes persistent failure on some attributes (e.g. disability representation) — an honest limitation worth citing rather than overclaiming LLM rewriting as a solved problem.

- **DIAGen: Semantically Diverse Image Augmentation with Generative Models for Few-Shot Learning** — Lingenberg et al. GCPR 2024 (arXiv:2408.14584, v2 2025). **[V]** GPT-4 (also tested LLaMA2-7B) generates varied class-specific prompts (viewpoint/environment/attribute) to steer a diffusion model for few-shot data augmentation; diversity quantified via improved recall (Kynkäänniemi et al. 2019) rather than CLIP novelty. Not clothing-specific, but a second concrete "LLM-diversified prompts → diffusion → measured diversity gain" data point beyond WANDER, useful for showing the mechanism generalizes.

- **Exploring Language Patterns of Prompts in Text-to-Image Generation and Their Impact on Visual Diversity** — De Rosa Palmini & Cetinic. arXiv:2504.14125 (Apr 2025). **[V]** Not a method paper — an empirical study of 6M+ CivitAI prompts showing lexical repetition in user prompts causes visual homogenization. Motivating evidence for *why* LLM-based prompt diversification should help at all; good to cite alongside the method papers rather than instead of them.

- Fashion/clothing-adjacent, weaker evidence: **Interactive Fashion Content Generation Using LLMs and Latent Diffusion Models** (Mantri & Sasikumar, arXiv:2306.05182, CVPR 2023 workshop) claims LLM-refined prompts give "culturally diversified fashion styles" but the abstract gives no mechanistic or quantitative diversity measure; **Masked Language Prompting for Generative Data Augmentation in Few-shot Fashion Style Recognition** (Hirakawa & Shimizu, arXiv:2504.19455) does bounded masked-word substitution on fashion captions, narrower than WANDER's open-ended mutation; **Character-based Outfit Generation with Vision-augmented Style Extraction via LLMs** (Forouzandehmehr et al., arXiv:2402.05941) conditions outfits on customer age/gender via LLM prompting — attribute personalization, not diversity-for-its-own-sake. All **[S]**-level (abstract only, thin on diversity evidence) — mention as adjacent-domain color, not load-bearing citations.

- **Explicit gap confirmed:** no published paper combines crowd/character generation + clothing/garment texturing + LLM-driven diffusion diversity. Searches for crowd-generation-adjacent LLM+diffusion work surfaced only spatial/steering-behavior papers (e.g. Gen-C, arXiv:2504.01924), and garment-texturing papers (FabricDiffusion, GarmentDreamer, GarmentPainter) have no LLM-diversity component. **This means the LLM-per-theme grammar future method, if implemented, would be a genuine methodological combination not yet published** — worth stating plainly in the dissertation rather than searching further for cover that doesn't exist.

- Confirmed **not** LLM-based prompt rewriting (checked and ruled out, useful only as related-work contrast): PromptMoG (arXiv:2511.20251, prompt-*embedding* MoG sampling, no LLM text rewriting), PAG/"Learning to Sample Effective and Diverse Prompts" (arXiv:2502.11477, GFlowNet-based), Minority-Focused T2I via Prompt Optimization (arXiv:2410.07838, CVPR 2025 Oral, continuous token optimization), TPSO (arXiv:2511.19811, token-embedding-space optimization), RainbowPlus (arXiv:2504.15047, LLM-driven evolutionary mutation but for adversarial *text* jailbreaks against LLMs, not image diversity — surface-similar to WANDER's evolutionary framing but different modality/objective entirely).

### 3.7 3D-native precedent for our LPIPS-diversity evaluation (checked 2026-07-07, one relevant hit)

User asked whether there's a 3D-generation-native (rather than 2D-image-diffusion) literature on diversity methods/metrics, given our whole metric suite (DINOv2, DreamSim, LPIPS, Vendi, DPP) is ported from 2D. Broad search via subagent (WebSearch/WebFetch) came back thin overall — SDS/NeRF-based diversity papers (DiverseDream arXiv:2312.02192, Dive3D arXiv:2506.13594) target a different failure mode (mode-seeking SDS loss) than Paint3D's single-pass depth-conditioned UV diffusion, and no paper addresses diversity for that specific mechanism. One directly useful hit kept:

- **Point-UV Diffusion: Texture Generation on 3D Meshes with Point-UV Diffusion** — Yu et al. ICCV 2023, arXiv:2308.10490. **[V]** Same problem setting as us: diverse texture generation on a *fixed* mesh (not shape generation). Evaluates diversity via **pairwise LPIPS across 5 samples per mesh**, explicitly justifying rejection of FID/KID because their training data is color-biased — same reasoning we've applied to our own metric choice. Independent precedent for LPIPS-based diversity evaluation in exactly this setting (diffusion-textured fixed 3D mesh templates); worth citing to justify the LPIPS half of our metric suite as domain-appropriate, not just imported wholesale from 2D image diversity work.

Rest of that research pass (3D-native metrics like COV/MMD/1-NNA/JSD from point-cloud literature, McDonnell's *Eye-catching Crowds* SIGGRAPH 2009) judged not worth folding in — either not applicable to a mesh-fixed/texture-only setting, or not adding enough beyond what's already covered.

## 6. Reference list (with verification status)

| # | Paper | Venue | arXiv | Status |
|---|-------|-------|-------|--------|
| 1 | Ho & Salimans — Classifier-Free Diffusion Guidance | NeurIPS workshop 2021/2022 | [2207.12598](https://arxiv.org/abs/2207.12598) | V (prior) |
| 2 | Jin, Shi, Gu — Stage-wise Dynamics of CFG | ICLR 2026 | [2509.22007](https://arxiv.org/abs/2509.22007) | V |
| 3 | Astolfi et al. — Consistency-Diversity-Realism Pareto Fronts | arXiv 2024 | [2406.10429](https://arxiv.org/abs/2406.10429) | V |
| 4 | Jia, Shen, Wang — Couple to Control | arXiv 2026 | [2605.11311](https://arxiv.org/abs/2605.11311) | V |
| 5 | Jia et al. — Antithetic Noise in Diffusion Models | arXiv 2025 | [2506.06185](https://arxiv.org/abs/2506.06185) | V |
| 6 | Zameshina, Teytaud, Najman — Diverse Diffusion | arXiv 2023 | [2310.12583](https://arxiv.org/abs/2310.12583) | V |
| 7 | Li, Liu, Kawaguchi — DivIn / Initialization is Half the Battle | ICML 2026 spotlight | [2606.02453](https://arxiv.org/abs/2606.02453) | V |
| 8 | It's Never Too Late — noise optimization, frequency profiles | CVPR 2026 | [2601.00090](https://arxiv.org/abs/2601.00090) | V |
| 9 | Kim, Um, Ye — Contrastive Noise Optimization | ICLR 2026 | [2510.03813](https://arxiv.org/abs/2510.03813) | V |
| 10 | Corso et al. — Particle Guidance | arXiv 2023 | [2310.13102](https://arxiv.org/abs/2310.13102) | V |
| 11 | Vinograd, Achituve, Fetaya — EDDY / Marginal-Preserving PG | arXiv 2026 | [2605.06553](https://arxiv.org/abs/2605.06553) | V |
| 12 | Jalali et al. — SPARKE | arXiv 2025 | [2506.10173](https://arxiv.org/abs/2506.10173) | V |
| 13 | Bhat et al. — Don't Settle at the Mode | ECCV 2026 | [2606.27371](https://arxiv.org/abs/2606.27371) | V |
| 14 | Sadat et al. — CADS | ICLR 2024 | [2310.17347](https://arxiv.org/abs/2310.17347) | V |
| 15 | Kynkäänniemi et al. — Guidance in a Limited Interval | NeurIPS 2024 | [2404.07724](https://arxiv.org/abs/2404.07724) | V |
| 16 | Karras et al. — Autoguidance | NeurIPS 2024 oral | [2406.02507](https://arxiv.org/abs/2406.02507) | V |
| 17 | InitNO — Initial Noise Optimization | CVPR 2024 | [2404.04650](https://arxiv.org/abs/2404.04650) | S |
| 18 | Golden Noise for Diffusion Models | arXiv 2024 | [2411.09502](https://arxiv.org/abs/2411.09502) | S |
| 19 | All Seeds Are Not Equal | arXiv 2024 | [2411.18810](https://arxiv.org/abs/2411.18810) | S |
| 20 | Repulsive Score Distillation | arXiv 2024 | [2406.16683](https://arxiv.org/abs/2406.16683) | S |
| 21 | PromptMoG | arXiv 2025 | [2511.20251](https://arxiv.org/abs/2511.20251) | S |
| 22 | Kulesza & Taskar — k-DPPs | ICML 2011 | [ICML PDF](https://icml.cc/2011/papers/611_icmlpaper.pdf) | V (prior) |
| 23 | WANDER — Evolve to Inspire | arXiv 2025 | [2511.00686](https://arxiv.org/abs/2511.00686) | V |
| 24 | Datta et al. — Prompt Expansion for Adaptive T2I Generation | arXiv 2023 | [2312.16720](https://arxiv.org/abs/2312.16720) | V |
| 25 | Peinl — LLMs as prompt modifier to avoid biases | arXiv 2025 | [2504.11104](https://arxiv.org/abs/2504.11104) | V |
| 26 | Lingenberg et al. — DIAGen | GCPR 2024 | [2408.14584](https://arxiv.org/abs/2408.14584) | V |
| 27 | De Rosa Palmini & Cetinic — Language Patterns of Prompts and Visual Diversity | arXiv 2025 | [2504.14125](https://arxiv.org/abs/2504.14125) | V |
| 28 | Mantri & Sasikumar — Interactive Fashion Content Generation | CVPR 2023 workshop | [2306.05182](https://arxiv.org/abs/2306.05182) | S |
| 29 | Hirakawa & Shimizu — Masked Language Prompting for Fashion Style Recognition | arXiv 2025 | [2504.19455](https://arxiv.org/abs/2504.19455) | S |
| 30 | Forouzandehmehr et al. — Character-based Outfit Generation via LLMs | IEEE BigData 2023 workshop | [2402.05941](https://arxiv.org/abs/2402.05941) | S |
| 31 | Yu et al. — Point-UV Diffusion | ICCV 2023 | [2308.10490](https://arxiv.org/abs/2308.10490) | V |

**Known limitations of this review:** compiled via web search + abstract-page reads, not full-paper reads; author lists for [V] entries come from arXiv abstract pages as summarized by an automated fetch and should be spot-checked when writing the dissertation's bibliography; 2025–2026 entries (2, 4, 5, 7, 8, 9, 11, 13) are recent and not yet widely replicated — treat single-paper claims as provisional. Searches were English-only, arXiv-centric.
