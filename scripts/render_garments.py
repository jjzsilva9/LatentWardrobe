"""Render untextured front/back preview images of every garment mesh.

Used by Method 1a (theme-aware mesh sampling): the renders are CLIP-image
embedded and compared against the theme prompt. Also handy as a visual
catalogue of the mesh library (assets/meshes/renders/).

Pure-CPU software renderer (painter's algorithm + cv2 polygon fill, Lambert
headlight shading, orthographic projection) — no OpenGL/display needed in
headless WSL, no GPU contention with generation runs. Quality is deliberately
modest; CLIP sees 224px inputs.

    python scripts/render_garments.py            # renders all garments
    python scripts/render_garments.py --only tops/tube_top
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = REPO_ROOT / "assets" / "meshes"
RENDER_DIR = MESH_DIR / "renders"
SIZE = 448          # supersampled; downscaled to OUT for mild antialiasing
OUT = 336

sys.path.insert(0, str(Path(__file__).parent))
from generate_crowd_baseline import load_obj  # reuse the obj parser


def render_views(obj_path):
    """Return (front, back) uint8 grayscale-shaded RGB images."""
    v, _, faces = load_obj(obj_path)
    verts = np.array(v, dtype=np.float64)
    # triangulate (fan), gather vertex indices only
    tris = []
    for corners in faces:
        vi = [c[0] for c in corners]
        for a in range(1, len(vi) - 1):
            tris.append((vi[0], vi[a], vi[a + 1]))
    tris = np.array(tris)

    # normalize into unit box, preserving aspect
    center = (verts.max(0) + verts.min(0)) / 2
    verts = verts - center
    scale = np.abs(verts).max()
    verts = verts / (scale + 1e-9)

    images = []
    for yaw in (0.0, np.pi):  # front, back (rotate about Y axis)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = verts @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
        # orthographic: x -> u, y -> v (flip), z -> depth toward camera
        uv = np.stack([rot[:, 0], -rot[:, 1]], axis=1)
        uv = ((uv * 0.45 + 0.5) * SIZE).astype(np.int32)
        depth = rot[:, 2]

        tri_uv = uv[tris]                       # (T, 3, 2)
        tri_depth = depth[tris].mean(axis=1)    # (T,)
        # face normal z for headlight shading
        p = rot[tris]                           # (T, 3, 3)
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        nz = np.abs(n[:, 2]) / (np.linalg.norm(n, axis=1) + 1e-12)
        shade = (0.25 + 0.65 * nz)              # ambient + diffuse

        img = np.full((SIZE, SIZE, 3), 255, np.uint8)
        order = np.argsort(tri_depth)           # far-to-near (painter)
        for t in order:
            g = int(shade[t] * 255)
            cv2.fillConvexPoly(img, tri_uv[t], (g, g, g), lineType=cv2.LINE_8)
        images.append(cv2.resize(img, (OUT, OUT), interpolation=cv2.INTER_AREA))
    return images


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="render just one garment, e.g. tops/tube_top")
    args = ap.parse_args()

    objs = []
    for slot in ("tops", "bottoms", "fullbody"):
        objs += sorted((MESH_DIR / slot).glob("*.obj"))
    if args.only:
        objs = [MESH_DIR / (args.only + ".obj")]

    done = failed = 0
    for obj in objs:
        slot = obj.parent.name
        outdir = RENDER_DIR / slot
        outdir.mkdir(parents=True, exist_ok=True)
        front_p = outdir / f"{obj.stem}_front.png"
        back_p = outdir / f"{obj.stem}_back.png"
        if front_p.exists() and back_p.exists() and \
                front_p.stat().st_mtime > obj.stat().st_mtime:
            done += 1
            continue
        try:
            front, back = render_views(obj)
        except Exception as e:
            print(f"RENDER FAILED {slot}/{obj.stem}: {e}")
            failed += 1
            continue
        cv2.imwrite(str(front_p), front)
        cv2.imwrite(str(back_p), back)
        done += 1
        print(f"rendered {slot}/{obj.stem}")
    print(f"\n{done} rendered/cached, {failed} failed -> {RENDER_DIR}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
