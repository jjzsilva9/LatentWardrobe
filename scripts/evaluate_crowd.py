"""Crowd evaluation harness.

Computes the dissertation's metrics for one generated crowd (one prompt dir
produced by generate_crowd_baseline.py):

  Diversity   - pairwise LPIPS (mean + min nearest-neighbour)
              - pairwise CLIP-image cosine distance (mean + min-NN)
              - Vendi score over the CLIP similarity kernel (effective number
                of distinct members; punishes clustered "fake" diversity)
              - pairwise hue-histogram chi-square distance (color diversity,
                background-masked)
  Coherence   - CLIPScore-style cosine similarity between the crowd prompt and
                each member's rendered views (per member + crowd mean)

All metrics are computed on a fixed set of frames from each member's stage-2
turntable render (render_rgb.mp4) — NOT on raw UV maps (see plan doc: UV
layouts are mostly background and would dilute every score). Frames at the
same index correspond to the same camera angle across members.

Usage (inside WSL, paint3d env):
    python scripts/evaluate_crowd.py --crowd-dir outputs/<batch>/<prompt_slug> \
        [--frames 0 15 30 45] [--device cuda]

Writes <crowd-dir>/metrics.json and prints a summary table.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RENDER_REL = "stage2/tile_res_0/render_rgb.mp4"
VIEWS_CACHE = "eval_views"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crowd-dir", type=Path, required=True, help="one prompt dir with manifest.json")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 15, 30, 45],
                   help="frame indices to sample from each turntable render (same angle across members)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="default: <crowd-dir>/metrics.json")
    return p.parse_args()


def extract_views(member_dir, frame_ids):
    """Grab fixed frames from the member's turntable video; cache as PNGs."""
    video = member_dir / RENDER_REL
    if not video.exists():
        return None
    cache = member_dir / VIEWS_CACHE
    cache.mkdir(exist_ok=True)
    paths = [cache / f"frame_{i:03d}.png" for i in frame_ids]
    if all(p.exists() for p in paths):
        return paths
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i, out in zip(frame_ids, paths):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, max(total - 1, 0)))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        cv2.imwrite(str(out), frame)
    cap.release()
    return paths


def load_rgb(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return img


def foreground_mask(rgb):
    """Renders sit on a near-white background — mask it out so background
    pixels don't dominate color histograms. Approximation: foreground =
    not-near-white."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    near_white = (hsv[..., 1] < 25) & (hsv[..., 2] > 230)
    return ~near_white


def hue_histogram(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = foreground_mask(rgb).astype(np.uint8)
    if mask.sum() < 100:  # nearly empty render — degenerate
        return None
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    return (hist / max(hist.sum(), 1e-8)).flatten()


def chi_square(h1, h2):
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-8)))


def pairwise_stats(mat_upper):
    vals = np.array(mat_upper)
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max())}


def vendi_score(sim_matrix):
    """Effective number of distinct members: exp(entropy of eigenvalues of
    the normalized similarity kernel). Friedman & Dieng 2023."""
    k = sim_matrix / sim_matrix.shape[0]
    eig = np.linalg.eigvalsh(k)
    eig = np.clip(eig, 1e-12, None)
    eig = eig / eig.sum()
    return float(np.exp(-(eig * np.log(eig)).sum()))


def main():
    args = parse_args()
    manifest = json.loads((args.crowd_dir / "manifest.json").read_text())
    prompt = manifest["crowd_prompt"]

    # --- collect views per completed member
    members, views, skipped = [], {}, []
    for m in manifest["members"]:
        mdir = args.crowd_dir / m["member"]
        paths = extract_views(mdir, args.frames)
        if paths is None:
            skipped.append(m["member"])
            continue
        members.append(m["member"])
        views[m["member"]] = paths
    if len(members) < 2:
        sys.exit(f"only {len(members)} member(s) with renders — need >= 2 for pairwise metrics "
                 f"(skipped: {skipped})")
    print(f"crowd '{prompt}': {len(members)} members evaluated, {len(skipped)} skipped {skipped}")

    device = torch.device(args.device)

    # --- LPIPS (view-matched pairwise, averaged over views)
    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    def to_lpips_tensor(path):
        img = load_rgb(path).astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear").to(device)

    lpips_imgs = {m: [to_lpips_tensor(p) for p in views[m]] for m in members}
    lpips_pairs = []
    with torch.no_grad():
        for a, b in itertools.combinations(members, 2):
            d = np.mean([float(lpips_net(x, y)) for x, y in zip(lpips_imgs[a], lpips_imgs[b])])
            lpips_pairs.append(d)
    del lpips_imgs, lpips_net
    torch.cuda.empty_cache()

    # --- CLIP embeddings (open_clip, ViT-B-32) for diversity + coherence
    import open_clip
    # NOTE: env pins open_clip 2.0.2 (Paint3D requirement) — old API:
    # module-level tokenize(), no get_tokenizer().
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()

    from PIL import Image
    with torch.no_grad():
        text_emb = clip_model.encode_text(open_clip.tokenize([prompt]).to(device)).float()
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        view_embs = {}
        for m in members:
            batch = torch.stack([preprocess(Image.open(p)) for p in views[m]]).to(device)
            e = clip_model.encode_image(batch).float()
            e /= e.norm(dim=-1, keepdim=True)
            view_embs[m] = e

    member_emb = torch.stack([view_embs[m].mean(dim=0) for m in members])
    member_emb /= member_emb.norm(dim=-1, keepdim=True)
    sim = (member_emb @ member_emb.T).cpu().numpy()
    clip_pairs = [1.0 - sim[i, j] for i, j in itertools.combinations(range(len(members)), 2)]
    coherence = {m: float((view_embs[m] @ text_emb.T).mean()) for m in members}

    # --- color diversity (hue histograms, background-masked, view-matched)
    hists = {}
    for m in members:
        hs = [hue_histogram(load_rgb(p)) for p in views[m]]
        hists[m] = [h for h in hs if h is not None]
    color_pairs = []
    for a, b in itertools.combinations(members, 2):
        ds = [chi_square(x, y) for x, y in zip(hists[a], hists[b])]
        if ds:
            color_pairs.append(float(np.mean(ds)))

    results = {
        "crowd_prompt": prompt,
        "num_members_evaluated": len(members),
        "skipped_members": skipped,
        "frames": args.frames,
        "diversity": {
            "lpips_pairwise": pairwise_stats(lpips_pairs),
            "clip_distance_pairwise": pairwise_stats(clip_pairs),
            "color_chi2_pairwise": pairwise_stats(color_pairs) if color_pairs else None,
            "vendi_score_clip": vendi_score(sim),
        },
        "coherence": {
            "clipscore_per_member": coherence,
            "clipscore_crowd_mean": float(np.mean(list(coherence.values()))),
        },
    }
    out = args.out or (args.crowd_dir / "metrics.json")
    out.write_text(json.dumps(results, indent=2))

    d, c = results["diversity"], results["coherence"]
    print(f"\n=== {prompt} (n={len(members)}) ===")
    print(f"LPIPS pairwise      mean {d['lpips_pairwise']['mean']:.4f}  min-NN {d['lpips_pairwise']['min']:.4f}")
    print(f"CLIP dist pairwise  mean {d['clip_distance_pairwise']['mean']:.4f}  min-NN {d['clip_distance_pairwise']['min']:.4f}")
    if d["color_chi2_pairwise"]:
        print(f"Color chi2 pairwise mean {d['color_chi2_pairwise']['mean']:.4f}")
    print(f"Vendi score (CLIP)  {d['vendi_score_clip']:.2f} effective members")
    print(f"Theme coherence     crowd mean {c['clipscore_crowd_mean']:.4f}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
