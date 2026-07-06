"""Baseline crowd texture generation.

The simplest possible crowd-generation scheme, used as the dissertation's
baseline: the user supplies crowd-level text prompt(s); each crowd member gets
a randomly sampled OUTFIT and a fresh random seed, and the crowd prompt is
passed verbatim to Paint3D (stage 1 depth-conditioned generation + stage 2 UV
inpaint/refine).

Outfit sampling: with probability (1 - --fullbody-prob) a member wears
1 top + 1 bottom (merged into a single mesh, textured in ONE Paint3D run so
the outfit comes out coherent); otherwise a single fullbody garment (dress,
suit, ...). Merging offsets each garment's UV islands into its own half of a
shared atlas (top: u in [0, 0.5], bottom: u in [0.5, 1]) — reversible, so
per-garment textures can be cropped back out later.

All per-member choices (garments, seed, prompts) are recorded in a
manifest.json per prompt, so any crowd is exactly reproducible from
(--seed, prompt, --num-members, mesh library state).

Run INSIDE WSL in the paint3d conda env, from the repo root:

    conda activate paint3d
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    python scripts/generate_crowd_baseline.py \
        --prompt "superhero costume" --prompt "medieval villagers" \
        --num-members 20 --outdir outputs/overnight

Notes:
- Multiple --prompt flags run sequentially in one process (overnight use);
  results land in <outdir>/<prompt_slug>/member_XX/.
- Member -> (outfit, seed) assignment is derived from (--seed, prompt) only,
  so --resume re-runs are stable and adding prompts doesn't reshuffle others.
- Paint3D's stock configs hardcode a FIXED seed (same prompt+mesh => identical
  texture) and seed=-1 means int(time.time()) (1s resolution, collides across
  members). This driver writes an explicit per-member seed into per-member
  config files instead.
- --dry-run samples outfits and writes merged meshes + manifests but skips
  Paint3D entirely (fast validation of the mesh library / sampling logic).
- --resume skips any member stage whose final output already exists.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PAINT3D_DIR = REPO_ROOT / "external" / "paint3d"
STAGE1_TEMPLATE = PAINT3D_DIR / "controlnet/config/depth_based_inpaint_template.yaml"
STAGE2_TEMPLATE = PAINT3D_DIR / "controlnet/config/UV_based_inpaint_template.yaml"
RENDER_CONFIG = "paint3d/config/train_config_paint3d.py"  # relative to PAINT3D_DIR

STAGE1_FINAL = "res-0/albedo.png"
STAGE2_FINAL = "tile_res_0/albedo.png"

# Method 3 (--color-prompt): one random colour word per member, folded into
# the garment description alongside its zero-shot category.
COLORS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "teal", "maroon", "navy", "gold", "silver",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", action="append", required=True, dest="prompts",
                   help="crowd-level text prompt; repeat the flag for multiple sequential runs")
    p.add_argument("--num-members", type=int, default=4, help="crowd size per prompt")
    p.add_argument("--mesh-dir", type=Path, default=REPO_ROOT / "assets/meshes",
                   help="mesh library root containing tops/ bottoms/ fullbody/ subdirs")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="master seed; member assignments derive from (seed, prompt)")
    p.add_argument("--fullbody-prob", type=float, default=0.25,
                   help="probability a member wears a fullbody garment instead of top+bottom")
    p.add_argument("--mesh-sampling", choices=("uniform", "clip"), default="uniform",
                   help="garment selection: uniform (baseline) or clip (Method 1a: "
                        "theme-weighted via CLIP render embeddings)")
    p.add_argument("--tau", type=float, default=0.02,
                   help="softmax temperature for --mesh-sampling clip "
                        "(large tau -> uniform; small tau -> most theme-apt garments)")
    p.add_argument("--color-prompt", action="store_true",
                   help="Method 3: give each member a random colour word, inserted into "
                        "the garment description in the Paint3D prompt")
    p.add_argument("--pink-noise", action="store_true",
                   help="Method 4: replace Paint3D's initial latent noise with pink noise "
                        "(FFT 1/(1+f)^alpha reweighting -- see scripts/pink_noise.py). "
                        "Orthogonal to every mesh-/prompt-sampling method above.")
    p.add_argument("--pink-alpha", type=float, default=0.2,
                   help="pink-noise exponent (--pink-noise only); paper's chosen operating "
                        "point is 0.2, quality visibly degrades above ~0.3")
    p.add_argument("--llm-emitters", action="store_true",
                   help="Method 5: the whole crowd's wearer+clothing descriptions are evolved by "
                        "a local LLM (Ollama) as a WANDER-inspired population (arXiv 2511.00686; "
                        "see scripts/llm_prompt_mutation.py for the documented divergences from "
                        "the paper) over fixed emitter axes (age/build/posture/role/mood + "
                        "colour/formality/material/pattern/condition/accessories). Pool size = "
                        "--num-members; each generation mutates or crosses over a random pool "
                        "member and keeps it only if it beats the pool's current least-novel "
                        "member (CLIP-text k-NN novelty). Final descriptions become the per-member "
                        "Paint3D prompt AND (if --mesh-sampling clip) the per-member CLIP anchor "
                        "for garment selection. Supersedes --color-prompt (colour is one of its "
                        "emitter axes) -- mutually exclusive with it.")
    p.add_argument("--llm-generations", type=int, default=None,
                   help="evolution generations (--llm-emitters only); default 3x --num-members")
    p.add_argument("--llm-k-neighbors", type=int, default=3,
                   help="k for the k-NN novelty score within the pool (--llm-emitters only)")
    p.add_argument("--skip-stage2", action="store_true", help="run stage 1 only")
    p.add_argument("--dry-run", action="store_true",
                   help="sample outfits + write merged meshes + manifest, skip Paint3D")
    p.add_argument("--resume", action="store_true",
                   help="skip stages whose final output already exists")
    return p.parse_args()


# ---------------------------------------------------------------- mesh merge

def load_obj(path):
    """Parse a simple v/vt/f wavefront obj (MakeHuman exports: no normals, no mtl)."""
    v, vt, faces = [], [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            v.append(parts[1:4])
        elif parts[0] == "vt":
            vt.append([float(parts[1]), float(parts[2])])
        elif parts[0] == "f":
            corners = []
            for corner in parts[1:]:
                idx = corner.split("/")
                if len(idx) < 2 or idx[1] == "":
                    raise ValueError(f"{path}: face corner '{corner}' has no UV index — "
                                     f"mesh is unusable for UV-space texturing")
                corners.append((int(idx[0]) - 1, int(idx[1]) - 1))
            faces.append(corners)
    if not vt:
        raise ValueError(f"{path}: no UV coordinates (vt) found")
    return v, vt, faces


def write_outfit_obj(garment_paths, out_path):
    """Merge garments into one obj. Each garment's UVs are uniformly offset
    into its own vertical strip of the atlas (garment i of n: u in [i/n, (i+1)/n]).
    Faces are fan-triangulated. A single garment passes through unchanged
    (still rewritten, so downstream handling is uniform)."""
    n = len(garment_paths)
    lines = [f"# merged outfit: {', '.join(Path(g).name for g in garment_paths)}"]
    v_off = vt_off = 0
    for gi, gpath in enumerate(garment_paths):
        v, vt, faces = load_obj(gpath)
        lines.append(f"# --- {Path(gpath).name}")
        lines += [f"v {' '.join(c)}" for c in v]
        if n == 1:
            lines += [f"vt {u:.6f} {w:.6f}" for u, w in vt]
        else:
            lines += [f"vt {(u + gi) / n:.6f} {w:.6f}" for u, w in vt]
        for corners in faces:
            for a in range(1, len(corners) - 1):  # fan triangulation
                tri = (corners[0], corners[a], corners[a + 1])
                lines.append("f " + " ".join(f"{vi + v_off + 1}/{ti + vt_off + 1}" for vi, ti in tri))
        v_off += len(v)
        vt_off += len(vt)
    Path(out_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- sampling

def load_library(mesh_dir):
    lib = {slot: sorted((mesh_dir / slot).glob("*.obj")) for slot in ("tops", "bottoms", "fullbody")}
    if not (lib["tops"] and lib["bottoms"]) and not lib["fullbody"]:
        sys.exit(f"mesh library at {mesh_dir} needs tops/+bottoms/ or fullbody/ meshes")
    return lib


def sample_members(prompt, num, master_seed, lib, fullbody_prob, slot_weights=None, color_prompt=False):
    """slot_weights: optional {slot: (paths, weights)} from mesh_sampling.theme_weights
    (Method 1a); None -> uniform baseline sampling."""
    rng = random.Random(f"{master_seed}:{prompt}")

    def pick(slot):
        if slot_weights is not None and slot in slot_weights:
            paths, w = slot_weights[slot]
            return str(rng.choices(paths, weights=list(w), k=1)[0])
        return str(rng.choice(lib[slot]))

    members = []
    for i in range(num):
        can_tb = bool(lib["tops"] and lib["bottoms"])
        can_fb = bool(lib["fullbody"])
        use_fb = can_fb and (not can_tb or rng.random() < fullbody_prob)
        garments = [pick("fullbody")] if use_fb else [pick("tops"), pick("bottoms")]
        members.append({
            "member": f"member_{i:02d}",
            "outfit_type": "fullbody" if use_fb else "top_bottom",
            "garments": garments,
            "sd_seed": rng.randrange(1, 2**31 - 1),
            "color": rng.choice(COLORS) if color_prompt else None,
        })
    return members


def sample_members_with_llm(prompt, num, master_seed, lib, fullbody_prob, mesh_sampling_mode, tau,
                            generations, k_neighbors):
    """Method 5: evolve the whole crowd's wearer+clothing descriptions as one
    WANDER-inspired population (see llm_prompt_mutation.evolve_crowd), then
    sample garments per member from its own evolved clothing_desc as the
    CLIP anchor (Method 2 composition) instead of the crowd-wide theme."""
    import llm_prompt_mutation as lpm
    import mesh_sampling

    pool, full_descs, clothing_descs, theme_relevance, history = lpm.evolve_crowd(
        prompt, num, master_seed, generations, k_neighbors)

    rng = random.Random(f"{master_seed}:{prompt}:garments")
    members = []
    for i in range(num):
        if mesh_sampling_mode == "clip":
            slot_weights = mesh_sampling.theme_weights(clothing_descs[i], tau)
        else:
            slot_weights = None

        def pick(slot):
            if slot_weights is not None and slot in slot_weights:
                paths, w = slot_weights[slot]
                return str(rng.choices(paths, weights=list(w), k=1)[0])
            return str(rng.choice(lib[slot]))

        can_tb = bool(lib["tops"] and lib["bottoms"])
        can_fb = bool(lib["fullbody"])
        use_fb = can_fb and (not can_tb or rng.random() < fullbody_prob)
        garments = [pick("fullbody")] if use_fb else [pick("tops"), pick("bottoms")]

        members.append({
            "member": f"member_{i:02d}",
            "outfit_type": "fullbody" if use_fb else "top_bottom",
            "garments": garments,
            "sd_seed": rng.randrange(1, 2**31 - 1),
            "color": None,  # Method 5's colour emitter supersedes --color-prompt
            "llm": {
                "winner": pool[i],
                "full_desc": full_descs[i],
                "clothing_desc": clothing_descs[i],
                "theme_relevance": theme_relevance[i],  # reported, not enforced -- see module docstring
            },
        })
    return members, history


def garment_key(path):
    """slot/stem key matching mesh_sampling's cache keys, e.g. 'tops/tube_top'."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def category_phrase(member, categories):
    """Garment description clause for the Paint3D prompt: category (always,
    from zero-shot CLIP classification — see mesh_sampling.py) plus a random
    colour word when Method 3 (--color-prompt) assigned one."""
    color = member.get("color")
    prefix = f"{color} " if color else ""

    def cat(path):
        return categories.get(garment_key(path), "clothing")

    if member["outfit_type"] == "fullbody":
        return f"wearing a {prefix}{cat(member['garments'][0])}"
    top, bottom = (cat(g) for g in member["garments"])
    return f"wearing a {prefix}{top} and {prefix}{bottom}"


# ---------------------------------------------------------------- paint3d

def build_stage1_cfg(prompt, category_phrase, seed, dst, full_desc=None):
    cfg = OmegaConf.load(STAGE1_TEMPLATE)
    desc = f", {full_desc}" if full_desc else ""
    cfg.txt2img.prompt = f"turn around, {prompt}{desc}, {category_phrase}"
    cfg.txt2img.seed = seed
    # stage 1 itself copies prompt/negative_prompt/seed into its inpaint step;
    # negative prompt stays Paint3D's anti-lighting boilerplate on purpose.
    OmegaConf.save(cfg, dst)


def build_stage2_cfg(prompt, category_phrase, seed, dst, full_desc=None):
    cfg = OmegaConf.load(STAGE2_TEMPLATE)
    desc = f", {full_desc}" if full_desc else ""
    cfg.inpaint.prompt = f"UV map, {prompt}{desc}, {category_phrase}"
    cfg.inpaint.seed = seed
    # the img2img (tile refine) prompt is not reachable via stage 2's --prompt
    # flag and defaults to a "Sci-Fi digital painting" style — rewrite it.
    cfg.img2img.prompt = f"UV map, {prompt}{desc}, {category_phrase}, high quality, best quality"
    cfg.img2img.seed = seed
    OmegaConf.save(cfg, dst)


def run_stage(script, sd_config, mesh_path, outdir, texture_path=None):
    cmd = [sys.executable, script,
           "--sd_config", str(sd_config),
           "--render_config", RENDER_CONFIG,
           "--mesh_path", str(mesh_path),
           "--outdir", str(outdir)]
    if texture_path is not None:
        cmd += ["--texture_path", str(texture_path)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PAINT3D_DIR)
    secs = round(time.time() - t0, 1)
    if proc.returncode < 0:
        print(f"subprocess {Path(script).name} terminated by signal {-proc.returncode} after {secs}s",
              flush=True)
    else:
        print(f"subprocess {Path(script).name} exited rc={proc.returncode} after {secs}s",
              flush=True)
    return proc.returncode, secs


# ---------------------------------------------------------------- main

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def preflight():
    """Fail loudly at startup on anything that guarantees downstream failure."""
    for p in (PAINT3D_DIR, STAGE1_TEMPLATE, STAGE2_TEMPLATE):
        if not p.exists():
            sys.exit(f"PREFLIGHT FAILED: {p} does not exist")


def main():
    args = parse_args()
    # Paint3D stages run as subprocesses with cwd=PAINT3D_DIR, so every path
    # handed to them (configs, meshes, outdirs) must be absolute — relative
    # paths would resolve inside external/paint3d and fail or misplace output.
    args.outdir = args.outdir.resolve()
    args.mesh_dir = args.mesh_dir.resolve()
    if args.llm_emitters and args.color_prompt:
        sys.exit("--llm-emitters and --color-prompt are mutually exclusive: "
                  "colour is one of --llm-emitters' own emitter axes")
    preflight()
    if args.pink_noise:
        # Read by the opt-in hook in external/paint3d/pipeline_paint3d_stage{1,2}.py
        # -- Paint3D's stages run as separate subprocesses (see run_stage()), so
        # the pink-noise patch can't be installed from this process; it has to be
        # applied inside each child instead, which is what these env vars trigger.
        os.environ["PINK_NOISE_ALPHA"] = str(args.pink_alpha)
        os.environ["LATENTWARDROBE_SCRIPTS_DIR"] = str(Path(__file__).resolve().parent)
    lib = load_library(args.mesh_dir)
    print(f"mesh library: {len(lib['tops'])} tops, {len(lib['bottoms'])} bottoms, "
          f"{len(lib['fullbody'])} fullbody")
    print(f"plan: {len(args.prompts)} prompt(s) x {args.num_members} members"
          + (" [DRY RUN]" if args.dry_run else ""))

    import mesh_sampling  # pulls in torch/open_clip; cheap if the garment cache already exists
    categories = mesh_sampling.garment_categories(device="cpu")
    if args.llm_emitters:
        import llm_prompt_mutation

    total_failures = 0
    completed_any = False  # circuit breaker: if the very first member fails, the
    # problem is almost certainly systemic (env/config/code), not one bad garment
    # — abort the whole batch immediately instead of burning hours on 100
    # identical failures that nobody is watching.
    for prompt in args.prompts:
        pdir = args.outdir / slugify(prompt)
        pdir.mkdir(parents=True, exist_ok=True)
        if args.llm_emitters:
            generations = args.llm_generations if args.llm_generations is not None else 3 * args.num_members
            members, llm_history = sample_members_with_llm(
                prompt, args.num_members, args.seed, lib, args.fullbody_prob,
                args.mesh_sampling, args.tau, generations, args.llm_k_neighbors)
        else:
            llm_history = None
            slot_weights = None
            if args.mesh_sampling == "clip":
                slot_weights = mesh_sampling.theme_weights(prompt, args.tau, device="cpu")
            members = sample_members(prompt, args.num_members, args.seed, lib, args.fullbody_prob,
                                     slot_weights=slot_weights, color_prompt=args.color_prompt)
        for m in members:
            m["category_phrase"] = category_phrase(m, categories)
        manifest = {
            "crowd_prompt": prompt,
            "master_seed": args.seed,
            "fullbody_prob": args.fullbody_prob,
            "mesh_sampling": args.mesh_sampling,
            "tau": args.tau if args.mesh_sampling == "clip" else None,
            "color_prompt": args.color_prompt,
            "pink_noise": args.pink_noise,
            "pink_alpha": args.pink_alpha if args.pink_noise else None,
            "llm_emitters": args.llm_emitters,
            "llm_generations": generations if args.llm_emitters else None,
            "llm_k_neighbors": args.llm_k_neighbors if args.llm_emitters else None,
            "llm_model": llm_prompt_mutation.OLLAMA_MODEL if args.llm_emitters else None,
            "llm_evolution_history": llm_history,
            "stage1_prompt_template": "turn around, {prompt}, {full_desc}, {category_phrase}",
            "stage2_prompt_templates": {
                "inpaint": "UV map, {prompt}, {full_desc}, {category_phrase}",
                "img2img": "UV map, {prompt}, {full_desc}, {category_phrase}, high quality, best quality",
            },
            "members": members,
        }
        manifest_path = pdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        for m in members:
            mdir = pdir / m["member"]
            mdir.mkdir(exist_ok=True)
            print(f"\n=== [{prompt}] {m['member']}: "
                  f"{'+'.join(Path(g).stem for g in m['garments'])} seed={m['sd_seed']} ===",
                  flush=True)

            outfit_path = mdir / "outfit.obj"
            try:
                write_outfit_obj(m["garments"], outfit_path)
            except ValueError as e:
                m["outfit"] = {"status": f"MERGE FAILED: {e}"}
                total_failures += 1
                manifest_path.write_text(json.dumps(manifest, indent=2))
                if not completed_any:
                    sys.exit("ABORTING BATCH: the very first member failed (outfit merge) "
                             "— treating as systemic. Fix and rerun with --resume.")
                continue
            m["outfit"] = {"status": "ok", "path": str(outfit_path)}

            if args.dry_run:
                manifest_path.write_text(json.dumps(manifest, indent=2))
                continue

            s1_out, s2_out = mdir / "stage1", mdir / "stage2"
            s1_final, s2_final = s1_out / STAGE1_FINAL, s2_out / STAGE2_FINAL

            if args.resume and s1_final.exists():
                m["stage1"] = {"status": "skipped (resume)"}
            else:
                cfg = mdir / "stage1_cfg.yaml"
                build_stage1_cfg(prompt, m["category_phrase"], m["sd_seed"], cfg,
                                  full_desc=m.get("llm", {}).get("full_desc"))
                rc, secs = run_stage("pipeline_paint3d_stage1.py", cfg, outfit_path, s1_out)
                m["stage1"] = {"status": "ok" if rc == 0 else f"FAILED rc={rc}", "seconds": secs}
                if rc != 0:
                    total_failures += 1
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    if not completed_any:
                        sys.exit("ABORTING BATCH: the very first member failed at stage 1 "
                                 "— treating as systemic. Fix and rerun with --resume.")
                    continue
            m["stage1"]["final_texture"] = str(s1_final)

            if args.skip_stage2:
                m["stage2"] = {"status": "skipped (--skip-stage2)"}
            elif args.resume and s2_final.exists():
                m["stage2"] = {"status": "skipped (resume)", "final_texture": str(s2_final)}
            else:
                cfg = mdir / "stage2_cfg.yaml"
                build_stage2_cfg(prompt, m["category_phrase"], m["sd_seed"], cfg,
                                  full_desc=m.get("llm", {}).get("full_desc"))
                rc, secs = run_stage("pipeline_paint3d_stage2.py", cfg, outfit_path, s2_out,
                                     texture_path=s1_final)
                m["stage2"] = {"status": "ok" if rc == 0 else f"FAILED rc={rc}", "seconds": secs}
                if rc != 0:
                    total_failures += 1
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    if not completed_any:
                        sys.exit("ABORTING BATCH: the very first member failed at stage 2 "
                                 "— treating as systemic. Fix and rerun with --resume.")
                    continue
                m["stage2"]["final_texture"] = str(s2_final)

            completed_any = True
            manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\ndone. failures: {total_failures}")
    if total_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
