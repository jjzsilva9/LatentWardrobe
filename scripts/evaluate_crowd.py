"""Crowd evaluation harness.

Computes the dissertation's metrics for one generated crowd (one prompt dir
produced by generate_crowd_baseline.py). Metric set follows "It's Never Too
Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models"
(minus HPSv2, which we chose not to report):

  Diversity   - patchwise DINOv2 distance (Eq. 3 of that paper: mean cosine
                distance per patch, averaged over P patches, over all pairs)
              - DreamSim pairwise distance
              - pairwise LPIPS
              - pairwise low-res (32x32) pixel L2
              - pairwise hue-histogram chi-square distance (color diversity,
                background-masked)
              - Vendi score over a DINOv2 [CLS] cosine-similarity kernel
                (effective number of distinct members)
              - DPP score: determinant of that same DINOv2 [CLS] kernel
  Coherence   - CLIPScore-style cosine similarity between the crowd prompt and
                each member's rendered views (per member + crowd mean)

Every pairwise statistic reports mean, min (nearest-neighbour / clone
detection), and std across the contributing member pairs.

All metrics are computed on a fixed set of frames from each member's stage-2
turntable render (render_rgb.mp4) — NOT on raw UV maps (see plan doc: UV
layouts are mostly background and would dilute every score). Frames at the
same index correspond to the same camera angle across members, which is also
what makes DINOv2 patch-level correspondence (Eq. 3) meaningful: patch p is
the same grid cell, and roughly the same garment region, across members.

Usage (inside WSL, "eval" conda env -- see .claude/setup/install_eval_env.sh,
kept separate from the "paint3d" env so upgrading eval deps can't break
generation):
    conda activate eval
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
DINOV2_INPUT_SIZE = 224  # 16x16 = 256 patches at patch size 14

# DreamSim defaults its weight cache to "./models" (relative to cwd) and, as
# a side effect, calls torch.hub.set_dir() globally -- so torch.hub.load()
# for DINOv2 below lands in that same folder too. Left at the default
# location (repo root, gitignored) rather than redirected elsewhere, since
# it's just a downloaded-weights cache, not repo content.
MODEL_CACHE_DIR = "./models"


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


def low_res_l2(rgb):
    """32x32 pixel-space feature, following [58] (as cited in the collapse-
    recovery paper): a cheap global-layout diversity proxy independent of any
    learned feature extractor."""
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return small.flatten()


def pairwise_stats(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
            "std": float(vals.std())}


def vendi_score(sim_matrix):
    """Effective number of distinct members: exp(entropy of eigenvalues of
    the normalized similarity kernel). Friedman & Dieng 2023."""
    k = sim_matrix / sim_matrix.shape[0]
    eig = np.linalg.eigvalsh(k)
    eig = np.clip(eig, 1e-12, None)
    eig = eig / eig.sum()
    return float(np.exp(-(eig * np.log(eig)).sum()))


def dpp_score(sim_matrix):
    """DPP diversity statistic: determinant of the similarity kernel (Kulesza
    & Taskar 2011 via [31] in the collapse-recovery paper). Same DINOv2 [CLS]
    kernel as the Vendi score above -- det ~ 0 means near-duplicate members,
    det -> 1 means a maximally spread-out set."""
    sign, logdet = np.linalg.slogdet(sim_matrix)
    return float(sign * np.exp(logdet)) if sign > 0 else 0.0


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model.to(device).eval()


def dinov2_transform(rgb):
    """Resize + ImageNet-normalize a raw RGB uint8 array for DINOv2."""
    img = cv2.resize(rgb, (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def cosine_dist_matrix(a, b):
    """a: (N, D), b: (M, D), both L2-normalized -> (N, M) cosine distance."""
    return 1.0 - (a @ b.T)


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
    rgbs = {m: [load_rgb(p) for p in views[m]] for m in members}

    # --- LPIPS (view-matched pairwise, averaged over views)
    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    def to_lpips_tensor(rgb):
        img = rgb.astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear").to(device)

    lpips_imgs = {m: [to_lpips_tensor(r) for r in rgbs[m]] for m in members}
    lpips_pairs = []
    with torch.no_grad():
        for a, b in itertools.combinations(members, 2):
            d = np.mean([float(lpips_net(x, y)) for x, y in zip(lpips_imgs[a], lpips_imgs[b])])
            lpips_pairs.append(d)
    del lpips_imgs, lpips_net
    torch.cuda.empty_cache()

    # --- DreamSim (view-matched pairwise, averaged over views)
    from dreamsim import dreamsim
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=str(device), cache_dir=MODEL_CACHE_DIR)
    from PIL import Image

    dreamsim_imgs = {m: [dreamsim_preprocess(Image.fromarray(r)).to(device) for r in rgbs[m]] for m in members}
    dreamsim_pairs = []
    with torch.no_grad():
        for a, b in itertools.combinations(members, 2):
            d = np.mean([float(dreamsim_model(x, y)) for x, y in zip(dreamsim_imgs[a], dreamsim_imgs[b])])
            dreamsim_pairs.append(d)
    del dreamsim_imgs, dreamsim_model
    torch.cuda.empty_cache()

    # --- CLIP embeddings (open_clip, ViT-B-32) for coherence only
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    clip_tokenize = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        text_emb = clip_model.encode_text(clip_tokenize([prompt]).to(device)).float()
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        clip_view_embs = {}
        for m in members:
            batch = torch.stack([clip_preprocess(Image.fromarray(r)) for r in rgbs[m]]).to(device)
            e = clip_model.encode_image(batch).float()
            e /= e.norm(dim=-1, keepdim=True)
            clip_view_embs[m] = e
    coherence = {m: float((clip_view_embs[m] @ text_emb.T).mean()) for m in members}
    del clip_model, clip_view_embs
    torch.cuda.empty_cache()

    # --- DINOv2: patchwise diversity (Eq. 3), Vendi + DPP over the [CLS] kernel
    dinov2_model = load_dinov2(device)
    dinov2_cls = {}    # member -> (num_views, D) CLS embeddings
    dinov2_patch = {}  # member -> (num_views, P, D) patch-token embeddings
    with torch.no_grad():
        for m in members:
            batch = torch.stack([dinov2_transform(r) for r in rgbs[m]]).to(device)
            feats = dinov2_model.forward_features(batch)
            cls = feats["x_norm_clstoken"]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            patches = feats["x_norm_patchtokens"]
            patches = patches / patches.norm(dim=-1, keepdim=True)
            dinov2_cls[m] = cls
            dinov2_patch[m] = patches
    del dinov2_model
    torch.cuda.empty_cache()

    # patchwise distance (Eq. 3): per pair, per view, mean cosine distance
    # over P corresponding patches; averaged over views then over pairs.
    patchwise_pairs = []
    for a, b in itertools.combinations(members, 2):
        per_view = []
        for pa, pb in zip(dinov2_patch[a], dinov2_patch[b]):  # each (P, D)
            d = 1.0 - (pa * pb).sum(dim=-1)  # (P,) cosine distance per patch
            per_view.append(float(d.mean()))
        patchwise_pairs.append(float(np.mean(per_view)))

    # member-level [CLS] embedding (mean over views, re-normalized) for the
    # Vendi / DPP kernel, matching how the old CLIP diversity kernel was built.
    member_cls = torch.stack([dinov2_cls[m].mean(dim=0) for m in members])
    member_cls = member_cls / member_cls.norm(dim=-1, keepdim=True)
    dinov2_sim = (member_cls @ member_cls.T).cpu().numpy()

    # --- color diversity (hue histograms, background-masked, view-matched)
    hists = {}
    for m in members:
        hs = [hue_histogram(r) for r in rgbs[m]]
        hists[m] = [h for h in hs if h is not None]
    color_pairs = []
    for a, b in itertools.combinations(members, 2):
        ds = [chi_square(x, y) for x, y in zip(hists[a], hists[b])]
        if ds:
            color_pairs.append(float(np.mean(ds)))

    # --- low-res (32x32) pixel L2 (view-matched pairwise, averaged over views)
    lowres = {m: [low_res_l2(r) for r in rgbs[m]] for m in members}
    lowres_pairs = []
    for a, b in itertools.combinations(members, 2):
        ds = [float(np.linalg.norm(x - y)) for x, y in zip(lowres[a], lowres[b])]
        lowres_pairs.append(float(np.mean(ds)))

    results = {
        "crowd_prompt": prompt,
        "num_members_evaluated": len(members),
        "skipped_members": skipped,
        "frames": args.frames,
        "diversity": {
            "dinov2_patchwise_pairwise": pairwise_stats(patchwise_pairs),
            "dreamsim_pairwise": pairwise_stats(dreamsim_pairs),
            "lpips_pairwise": pairwise_stats(lpips_pairs),
            "pixel_l2_32_pairwise": pairwise_stats(lowres_pairs),
            "color_chi2_pairwise": pairwise_stats(color_pairs) if color_pairs else None,
            "vendi_score_dinov2": vendi_score(dinov2_sim),
            "dpp_score_dinov2": dpp_score(dinov2_sim),
        },
        "coherence": {
            "clipscore_per_member": coherence,
            "clipscore_crowd_mean": float(np.mean(list(coherence.values()))),
            "clipscore_crowd_std": float(np.std(list(coherence.values()))),
        },
    }
    out = args.out or (args.crowd_dir / "metrics.json")
    out.write_text(json.dumps(results, indent=2))

    d, c = results["diversity"], results["coherence"]
    print(f"\n=== {prompt} (n={len(members)}) ===")
    print(f"DINOv2 patchwise    mean {d['dinov2_patchwise_pairwise']['mean']:.4f}  "
          f"min-NN {d['dinov2_patchwise_pairwise']['min']:.4f}  std {d['dinov2_patchwise_pairwise']['std']:.4f}")
    print(f"DreamSim pairwise   mean {d['dreamsim_pairwise']['mean']:.4f}  "
          f"min-NN {d['dreamsim_pairwise']['min']:.4f}  std {d['dreamsim_pairwise']['std']:.4f}")
    print(f"LPIPS pairwise      mean {d['lpips_pairwise']['mean']:.4f}  "
          f"min-NN {d['lpips_pairwise']['min']:.4f}  std {d['lpips_pairwise']['std']:.4f}")
    print(f"Pixel-L2(32) pairwise mean {d['pixel_l2_32_pairwise']['mean']:.4f}  "
          f"min-NN {d['pixel_l2_32_pairwise']['min']:.4f}  std {d['pixel_l2_32_pairwise']['std']:.4f}")
    if d["color_chi2_pairwise"]:
        print(f"Color chi2 pairwise mean {d['color_chi2_pairwise']['mean']:.4f}  std {d['color_chi2_pairwise']['std']:.4f}")
    print(f"Vendi score (DINOv2) {d['vendi_score_dinov2']:.2f} effective members")
    print(f"DPP score (DINOv2)   {d['dpp_score_dinov2']:.4g}")
    print(f"Theme coherence     crowd mean {c['clipscore_crowd_mean']:.4f}  std {c['clipscore_crowd_std']:.4f}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
