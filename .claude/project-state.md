# Project state & accumulated context — 2026-07-17

Handoff document for continuing work on another machine. Complements (does
not repeat) `README.md`, `.claude/dissertation-plan.md`,
`.claude/lit-review-diffusion-diversity.md`, `.claude/setup/README.md`.
Written at the completion of the crowd-library study's generation +
evaluation + tube_top surgical pass.

## Canonical data (what to trust)

- **`outputs/crowd_library_20260715_stage1eval/`** — the end-to-end batch.
  7 cells × 5 themes × 5 members, stage-1-only, reduced pool (139 meshes),
  pair-coherent sampling. Generation COMPLETE, eval 35/35 `metrics.json`,
  shape eval 20/20 `shape_metrics.json` (cells 1/2/4/8). All charts +
  contact sheets under its `plots/{mesh,texture,end_to_end}/{avg,per_prompt}`
  + `contact_sheets/`. This is the batch the README cites.
- **`outputs/crowd_library_20260715_meshfixed/`** — the mesh-fixed (texture
  isolation) batch, pinned per member index to the e2e batch's
  `2_clip_white_plain` manifests. Eval 33/33 (cell 2 = verbatim copies of the
  reference incl. its valid metrics; **cell 1 intentionally unscored /
  uncharted** — with meshes pinned it is a nondeterminism replicate of cell 2,
  and was removed from `run_crowd_library_mesh_fixed*.sh` for future runs).
- **FROZEN/UNUSABLE:** `crowd_library_20260713_stage1eval` (pre-pool-fix, 27
  rescueteam-contaminated members) and `crowd_library_20260713_meshfixed`
  (partial + later corrupted by a misdirected relaunch — see its
  `DO_NOT_RESUME.md`). Never resume either.
- **tube_top surgical pass (2026-07-17), complete:** `tube_top.obj` was
  geometrically broken (y_max 2.05, below skirt waistbands — renders as
  "just a skirt"). Removed from pool; replacement members drawn
  pair-coherently and twin-consistently: medieval_villagers/member_04 →
  `f_lusekofta + polkadotskirt`, superheroes/member_01 →
  `sweater_fisherman + shortjeans` (e2e cells 2/3 + all meshfixed
  propagations regenerated; everything rescored/replotted).

## Critical invariants (violating these silently corrupts data)

1. **The mesh pool is part of a batch's identity.**
   `generate_crowd_baseline.py` re-samples members and REWRITES
   `manifest.json` + `outfit.obj` on every run — `--resume` only skips
   Paint3D stages. Resuming any batch after a pool change re-draws every
   member. Escape hatch: `CROWD_LIBRARY_FIXED_FROM=self` (run_crowd_library.sh)
   pins each cell to its own manifest so only members with deleted dirs
   regenerate.
2. **rsync here never deletes and the SOURCE always wins.** The batch
   scripts' initial pull (remote→local) and the supervisor's final sync
   (local→remote) will resurrect deleted files and clobber patched ones if
   the two sides disagree. Any surgical edit (manifest patch, member
   deletion) must be applied to BOTH `outputs/...` on U: and the
   `/root/..._local` mirror before anything relaunches.
3. **Mesh-fixed launches need the FULL env set** —
   `CROWD_LIBRARY_OUT`, `CROWD_LIBRARY_LOCAL`, `CROWD_LIBRARY_MESHFIXED_OUT`,
   `CROWD_LIBRARY_MESHFIXED_LOCAL`, `CROWD_LIBRARY_BATCH_SCRIPT`,
   `CROWD_LIBRARY_REF_OUT`. Omitting the MESHFIXED pair once sent a run into
   the frozen 20260713 dirs (that's what corrupted them). The pipeline
   watcher's `launch_mf_supervisor` has the correct full set — copy from
   there.
4. **LLM evolution is not reproducible** — `llm_members.json` beside each
   llm-cell manifest is the only record; never delete it, never re-evolve
   mid-crowd (the driver guards this).

## Lab-machine ops (WSL) — condensed war stories

- **The quoting trap (bit us 3×):** `$vars` inside
  `wsl.exe ... bash -c '...'` command strings get EATEN at the Git-Bash
  boundary — loops nul out, `rm -rf "$X/path"` becomes `rm -rf /path`
  (silently deletes nothing). Safe channels: stdin heredoc
  (`wsl.exe ... bash -s <<'EOF'`), explicit literal paths, or looping on the
  Windows side. **Always `ls`-verify deletions before proceeding.**
- **WSL VM died 5× in 3 days** (`Wsl/Service/E_UNEXPECTED`). Mitigations:
  `C:\Users\silvajo\.wslconfig` (memory=24GB, swap=16GB) and the pipeline
  watcher serialized to one memory-heavy workload at a time. Recovery drill:
  `wsl --shutdown` from PowerShell, relaunch the watcher/supervisor
  one-liner — every phase re-detects completion from disk markers; relaunch
  is always safe. Leading crash suspect: drvfs under end-of-batch rsync
  storms (2 crashes at exactly batch completion). drvfs also wedges
  ("Transport endpoint is not connected") → `umount -l /mnt/u` + remount.
- Every WSL entry point needs the mount guard:
  `mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u`.
- Orchestration: `run_crowd_library_supervised.sh` (disk-count progress,
  health gates, 2h force-restart watchdog, generic via
  `CROWD_LIBRARY_BATCH_SCRIPT`) and `run_crowd_library_pipeline.sh` (chains
  gen→gen→3 sequential evals→plots; phases detected via
  `CROWD_LIBRARY_COMPLETE.txt` markers + log banners).
- Envs: `paint3d` (generation, py3.8/torch1.12), `eval` (scoring, py3.10).
  HPSv2's wheel is broken — `install_eval_env.sh` copies the missing BPE
  vocab from open_clip. Ollama (llama3.1:8b) serves the WANDER cells.

## Metric decisions (the reasoning chain)

- Axis design follows Harrington et al. (arXiv 2601.00090 — the source of
  BOTH the pink-noise method and the eval design; first author Anne
  Harrington, verified). Their human studies favor **set-level** diversity
  metrics → reported set: **Vendi + DPP** (same DINOv2-CLS kernel, two
  summaries), **DreamSim** (human-aligned pairwise), **color χ²**
  (UV-albedo-based, painted-texel mask; McDonnell link), **CLIPScore**
  (coherence), **HPSv2** (quality — restored after initially being skipped).
- **Cut outright** (computed nothing since 2026-07-17): DINOv2 patchwise,
  LPIPS, pixel-L2 — unreported metrics were pure compute.
- Stage-1 color diversity uses a mask derived from Paint3D's exact
  placeholder `texture_default_color` [0.8,0.1,0.8] → uint8 (204,25,204);
  verified zero near-miss pixels. Flagged `color_source:
  "stage1_derived_mask"` in metrics.json.
- Mesh stage adds **silhouette 1−IoU** (pure geometry, no pretrained model)
  and carries CLIPScore/HPSv2 on untextured renders **with stated caveats**:
  weak signal (CLIP is texture-dominated) and indirect circularity (sampling
  anchors on textured garment previews, eval sees grey outfit renders).
  Mesh-level coherence measurement remains an OPEN QUESTION (pinned).
- Chart labels are stage-aware: texture stage collapses cells 1+2 into
  "plain prompt" and drops mesh wording; mesh stage labels by selection
  anchor only. `plot_crowd_comparison.py` `--texture-labels` / `--shape`.

## Results story (theme means, N=5; full numbers in README)

Each method has a distinct signature:
- **CLIP-axis sampling** = strongest texturing diversifier (texture-stage
  Vendi 3.46 vs 3.22 plain; color χ² 0.745/0.758).
- **WANDER (6-axis)** = quality/coherence: best HPSv2 (0.142 e2e) and best
  CLIPScore, at a small Vendi cost.
- **CLIP mesh selection** measurably COSTS silhouette diversity vs random
  (shape Vendi 2.35–2.58 vs 2.80; IoU-dist 0.54–0.58 vs 0.60) — the
  coherence↔diversity tradeoff made visible; grey-render CLIPScore is too
  weak to resolve the coherence side (0.190–0.193 flat).
- **Pink noise** = free color nudge on plain prompts (0.630 vs 0.596),
  washes out under description-driven prompting.
- Sanity check that validates the mesh-fixed design: pinned cells 1/2
  produce near-identical scores.

## Report structure (agreed outline)

Methodology chapter = the system: pipeline overview (+ 3 intervention
points, stage-1-only decision), garment library & mesh selection (incl.
broken-mesh exclusions, pair-coherent pairing), pink noise, description-
driven prompting (axis sampling + WANDER as siblings; 12-axis failure story
inline). Evaluation chapter per template: **Experiments** = factorial
design table, three-stage attribution design, metric battery + caveats,
reproducibility; **Results** = per-stage findings (contact sheet first,
then charts), cross-stage synthesis carrying the thesis, Limitations block
(N=5 Vendi ceiling, stage-1 proxy, 5 themes). Infra war stories stay OUT of
the report (appendix at most).

## Working agreements (user preferences)

- **NEVER commit or push without being explicitly asked.** Task-scoped
  permission does not generalize.
- The replay batch is **"mesh-fixed"** — never "texture-fixed" (the mesh is
  fixed, not the texture).
- Terse status updates; lead with the answer; don't re-explain; don't
  narrate options that won't be pursued. Verify claims from disk/code
  before reporting ("is it done?" → check, don't infer from stale logs).

## Open items

- Mesh-level coherence metric: pinned open question (CLIP is circular-ish;
  candidates discussed: garment↔theme CLIP as manipulation check, LLM judge
  over category_phrase, human eval).
- README hero image renders on GitHub only after `outputs/.../plots/**` is
  pushed (gitignore exceptions already in place).
- Presentation PNGs: medieval_villagers' full pipeline stills (incl.
  untextured `shape_eval_views` renders) are unignored and push-ready.
- The 2026-07-17 script/eval changes are uncommitted, awaiting the user's
  own commit/push.
