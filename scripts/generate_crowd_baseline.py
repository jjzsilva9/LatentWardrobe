"""Baseline crowd texture generation.

The simplest possible crowd-generation scheme, used as the dissertation's
baseline: the user supplies ONE crowd-level text prompt; each crowd member
gets a randomly selected mesh from --mesh-dir and a fresh random seed, and
the crowd prompt is passed verbatim to Paint3D (stage 1 depth-conditioned
generation + stage 2 UV inpaint/refine). All per-member choices (mesh, seed,
exact prompts) are recorded in <outdir>/manifest.json so any crowd is exactly
reproducible from (--prompt, --seed, --num-members, mesh set).

Run INSIDE WSL in the paint3d conda env, from the repo root:

    conda activate paint3d
    python scripts/generate_crowd_baseline.py \
        --prompt "aliens on the moon" --num-members 4 \
        --outdir outputs/crowd_baseline_aliens

Or from Windows PowerShell:

    wsl -u root -e bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate paint3d && cd /mnt/c/Users/silvajo/LatentWardrobe && python scripts/generate_crowd_baseline.py --prompt '...' --num-members 4 --outdir outputs/test"

Notes:
- Paint3D's stock configs hardcode a FIXED seed (same prompt+mesh => identical
  texture) and seed=-1 only means int(time.time()) (1s resolution — members
  generated in the same second would collide). This driver therefore writes an
  explicit per-member seed into generated config files.
- Prompt conventions follow Paint3D's demos: stage 1 prepends "turn around, "
  (drives the two-view front/back grid); stage 2 prepends "UV map, ". The crowd
  prompt itself is not otherwise modified.
- --resume skips any member whose outputs already exist (lab machine dies
  sometimes; also lets you run stage 1 and stage 2 as separate invocations).
"""

import argparse
import json
import random
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, help="crowd-level text prompt, used for every member")
    p.add_argument("--num-members", type=int, default=4)
    p.add_argument("--mesh-dir", type=Path, default=PAINT3D_DIR / "demo/objs",
                   help="directory searched recursively for *.obj meshes (default: Paint3D demo objs)")
    p.add_argument("--outdir", type=Path, required=True,
                   help="output directory (created); one member_XX/ subdir per crowd member")
    p.add_argument("--seed", type=int, default=0,
                   help="master seed; deterministically derives each member's mesh choice and SD seed")
    p.add_argument("--skip-stage2", action="store_true", help="run stage 1 only")
    p.add_argument("--resume", action="store_true",
                   help="skip stages whose final output already exists")
    return p.parse_args()


def build_stage1_cfg(prompt, seed, dst):
    cfg = OmegaConf.load(STAGE1_TEMPLATE)
    cfg.txt2img.prompt = f"turn around, {prompt}"
    cfg.txt2img.seed = seed
    # stage 1 copies prompt/negative_prompt/seed from txt2img into inpaint itself;
    # negative prompt is left as Paint3D's anti-lighting boilerplate on purpose.
    OmegaConf.save(cfg, dst)


def build_stage2_cfg(prompt, seed, dst):
    cfg = OmegaConf.load(STAGE2_TEMPLATE)
    cfg.inpaint.prompt = f"UV map, {prompt}"
    cfg.inpaint.seed = seed
    # img2img (tile refine) prompt is NOT overridden by stage 2's --prompt flag,
    # and the template's default injects a "Sci-Fi digital painting" style --
    # rewrite it to stay faithful to the crowd prompt.
    cfg.img2img.prompt = f"UV map, {prompt}, high quality, best quality"
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
    return proc.returncode, round(time.time() - t0, 1)


def main():
    args = parse_args()
    meshes = sorted(args.mesh_dir.rglob("*.obj"))
    if not meshes:
        sys.exit(f"no .obj meshes found under {args.mesh_dir}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "manifest.json"

    # Derive all per-member choices up front from the master seed, so the
    # member -> (mesh, sd_seed) mapping is identical across --resume runs.
    rng = random.Random(args.seed)
    members = []
    for i in range(args.num_members):
        members.append({
            "member": f"member_{i:02d}",
            "mesh": str(rng.choice(meshes)),
            "sd_seed": rng.randrange(1, 2**31 - 1),  # avoid -1 / 0 sentinels
        })

    manifest = {
        "crowd_prompt": args.prompt,
        "master_seed": args.seed,
        "mesh_dir": str(args.mesh_dir),
        "mesh_pool": [str(m) for m in meshes],
        "stage1_prompt_template": "turn around, {prompt}",
        "stage2_prompt_templates": {
            "inpaint": "UV map, {prompt}",
            "img2img": "UV map, {prompt}, high quality, best quality",
        },
        "members": members,
    }

    failures = 0
    for m in members:
        mdir = args.outdir / m["member"]
        mdir.mkdir(exist_ok=True)
        s1_out, s2_out = mdir / "stage1", mdir / "stage2"
        s1_final, s2_final = s1_out / STAGE1_FINAL, s2_out / STAGE2_FINAL

        print(f"\n=== {m['member']}: mesh={Path(m['mesh']).name} seed={m['sd_seed']} ===", flush=True)

        # --- stage 1
        if args.resume and s1_final.exists():
            m["stage1"] = {"status": "skipped (resume)"}
        else:
            cfg_path = mdir / "stage1_cfg.yaml"
            build_stage1_cfg(args.prompt, m["sd_seed"], cfg_path)
            rc, secs = run_stage("pipeline_paint3d_stage1.py", cfg_path, m["mesh"], s1_out)
            m["stage1"] = {"status": "ok" if rc == 0 else f"FAILED rc={rc}", "seconds": secs}
            if rc != 0:
                failures += 1
                manifest_path.write_text(json.dumps(manifest, indent=2))
                continue
        m["stage1"]["final_texture"] = str(s1_final)

        # --- stage 2
        if args.skip_stage2:
            m["stage2"] = {"status": "skipped (--skip-stage2)"}
        elif args.resume and s2_final.exists():
            m["stage2"] = {"status": "skipped (resume)", "final_texture": str(s2_final)}
        else:
            cfg_path = mdir / "stage2_cfg.yaml"
            build_stage2_cfg(args.prompt, m["sd_seed"], cfg_path)
            rc, secs = run_stage("pipeline_paint3d_stage2.py", cfg_path, m["mesh"], s2_out,
                                 texture_path=s1_final)
            m["stage2"] = {"status": "ok" if rc == 0 else f"FAILED rc={rc}", "seconds": secs}
            if rc != 0:
                failures += 1
            else:
                m["stage2"]["final_texture"] = str(s2_final)

        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {manifest_path}")
    if failures:
        sys.exit(f"{failures} stage run(s) failed -- see manifest")
    print("all members completed")


if __name__ == "__main__":
    main()
