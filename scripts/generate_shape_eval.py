"""Untextured turntable renders for silhouette/mesh-selection-only evaluation
(dissertation eval Part 1: does the mesh-selection method change what's worn,
independent of how it's textured?).

Reuses each member's already-sampled outfit.obj (written by
generate_crowd_baseline.py before any Paint3D texturing runs, so this needs
NO new mesh sampling and works against already-completed cells) and renders
it with external/paint3d/pipeline_paint3d_shapeonly.py: no texture, no
ControlNet/Stable Diffusion, just Paint3D's own differentiable renderer over
the same camera path used for the textured turntables. Cheap relative to real
generation (no diffusion), safe to run alongside a GPU-bound texturing batch.

Only 4 cells carry distinct mesh-selection behaviour and belong in this
comparison -- see .claude/memory.md for why: pink-noise variants (3, 5, 9)
are mesh-identical to their white-noise counterpart (pink noise only
perturbs the diffusion initial latent, never mesh sampling), so they add
nothing here. Method 5b (WANDER, cells 8/9) is mesh-*different* between its
own white/pink runs (Ollama evolution isn't reproducible), but that's
incidental to noise, not caused by it -- cell 8 stands in for the method by
convention (same convention already used everywhere else: compare the white
variant when a mesh-identical pink twin exists).

Usage (inside WSL, paint3d conda env):
    python scripts/generate_shape_eval.py \
        --library-dir outputs/crowd_library_20260713 [--resume]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAINT3D_DIR = REPO_ROOT / "external" / "paint3d"
RENDER_CONFIG = "paint3d/config/train_config_paint3d.py"  # relative to PAINT3D_DIR

SHAPE_CELLS = ["1_random_white_plain", "2_clip_white_plain", "4_clip_white_axes", "8_clip_white_llm6"]
SHAPE_EVAL_SUBDIR = "shape_eval"
SHAPE_EVAL_FINAL = "render_rgb.mp4"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library-dir", type=Path, required=True, help="e.g. outputs/crowd_library_20260713")
    p.add_argument("--cells", nargs="+", default=SHAPE_CELLS,
                   help="which cells to render (default: the 4 mesh-selection-distinct methods)")
    p.add_argument("--resume", action="store_true", help="skip members whose shape render already exists")
    return p.parse_args()


def render_member(outfit_path, outdir):
    cmd = [sys.executable, "pipeline_paint3d_shapeonly.py",
           "--render_config", RENDER_CONFIG,
           "--mesh_path", str(outfit_path),
           "--outdir", str(outdir)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PAINT3D_DIR)
    return proc.returncode, round(time.time() - t0, 1)


def main():
    args = parse_args()
    args.library_dir = args.library_dir.resolve()
    total, done, failed = 0, 0, 0
    for cell in args.cells:
        cell_dir = args.library_dir / cell
        if not cell_dir.is_dir():
            print(f"[shape-eval] cell={cell} has no dir yet, skipping")
            continue
        for prompt_dir in sorted(cell_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            for member_dir in sorted(prompt_dir.glob("member_*")):
                outfit = member_dir / "outfit.obj"
                if not outfit.exists():
                    continue
                total += 1
                outdir = member_dir / SHAPE_EVAL_SUBDIR
                final = outdir / SHAPE_EVAL_FINAL
                if args.resume and final.exists():
                    continue
                print(f"=== [{cell}] {prompt_dir.name}/{member_dir.name} ===", flush=True)
                rc, secs = render_member(outfit, outdir)
                if rc != 0:
                    failed += 1
                    print(f"FAILED rc={rc} after {secs}s", flush=True)
                else:
                    done += 1
                    print(f"ok after {secs}s", flush=True)
    print(f"\ndone. rendered {done}, failed {failed}, total members seen {total}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
