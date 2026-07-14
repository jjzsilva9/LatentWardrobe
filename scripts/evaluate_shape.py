"""Shape/silhouette-only diversity evaluation (dissertation eval Part 1).

Companion to evaluate_crowd.py, scoped to what mesh SELECTION alone can
change -- run against the untextured turntables from generate_shape_eval.py
(member_XX/shape_eval/render_rgb.mp4 + render_all.mp4), not the final
textured renders. Same diversity batteries as evaluate_crowd.py (DINOv2
patchwise, DreamSim, LPIPS, pixel-L2, Vendi/DPP over a DINOv2 CLS kernel),
same background-masking approach (z-normals panel of render_all.mp4).

Deliberately DROPS two things evaluate_crowd.py reports, because they aren't
meaningful on a flat, untextured, grey-shaded mesh:
  - color diversity (chi-square over UV hue histograms) -- there is no paint
  - theme coherence (CLIPScore) -- CLIP's signal is dominated by texture/
    colour cues a silhouette-only render doesn't have

Usage (inside WSL, "eval" conda env):
    python scripts/evaluate_shape.py --crowd-dir outputs/<batch>/<cell>/<prompt> \
        [--frames 0 15 30 45] [--device cuda]

Writes <crowd-dir>/shape_metrics.json (kept separate from evaluate_crowd.py's
metrics.json, which scores the same dir's final textured output).
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

RENDER_REL = "shape_eval/render_rgb.mp4"
RENDER_ALL_REL = "shape_eval/render_all.mp4"
DINOV2_INPUT_SIZE = 224
PATCH_GRID = 16
BG_GRAY_THRESH = 250
BBOX_MARGIN_FRAC = 0.02
MODEL_CACHE_DIR = "./models"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crowd-dir", type=Path, required=True, help="one prompt dir with manifest.json")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 15, 30, 45])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch-coverage-tau", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None, help="default: <crowd-dir>/shape_metrics.json")
    return p.parse_args()


def extract_frames(video, frame_ids, cache_dir, prefix):
    if not video.exists():
        return None
    cache_dir.mkdir(exist_ok=True)
    paths = [cache_dir / f"{prefix}_{i:03d}.png" for i in frame_ids]
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
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def load_mask(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def extract_mask_frames(video, frame_ids, cache_dir, prefix):
    if not video.exists():
        return None
    cache_dir.mkdir(exist_ok=True)
    paths = [cache_dir / f"{prefix}_{i:03d}.png" for i in frame_ids]
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
        h, w, _ = frame.shape
        third = w // 3
        normals = frame[:, 2 * third:]
        gray = normals.mean(axis=2)
        fg = np.where(gray < BG_GRAY_THRESH, 255, 0).astype(np.uint8)
        cv2.imwrite(str(out), fg)
    cap.release()
    return paths


def bbox_from_mask(mask, margin_frac=BBOX_MARGIN_FRAC):
    if mask.sum() < 100:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = mask.shape
    my = int((y1 - y0) * margin_frac)
    mx = int((x1 - x0) * margin_frac)
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)


def crop_pad_square(rgb, bbox):
    if bbox is None:
        return rgb
    y0, y1, x0, x1 = bbox
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), 255, dtype=rgb.dtype)
    oy, ox = (side - h) // 2, (side - w) // 2
    square[oy:oy + h, ox:ox + w] = crop
    return square


def patch_coverage_grid(mask, grid=PATCH_GRID, size=DINOV2_INPUT_SIZE):
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    patch = size // grid
    return resized.reshape(grid, patch, grid, patch).mean(axis=(1, 3))


def low_res_l2(rgb):
    small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return small.flatten()


def pairwise_stats(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {"mean": float(vals.mean()), "min": float(vals.min()), "max": float(vals.max()),
            "std": float(vals.std())}


def vendi_score(sim_matrix):
    k = sim_matrix / sim_matrix.shape[0]
    eig = np.clip(np.linalg.eigvalsh(k), 1e-12, None)
    eig = eig / eig.sum()
    return float(np.exp(-(eig * np.log(eig)).sum()))


def dpp_score(sim_matrix):
    n = sim_matrix.shape[0]
    sign, logdet = np.linalg.slogdet(sim_matrix)
    return float(sign * np.exp(logdet / n)) if sign > 0 else 0.0


def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model.to(device).eval()


def dinov2_transform(rgb):
    img = cv2.resize(rgb, (DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def main():
    args = parse_args()
    manifest = json.loads((args.crowd_dir / "manifest.json").read_text())
    prompt = manifest["crowd_prompt"]

    members, views, masks, skipped = [], {}, {}, []
    for m in manifest["members"]:
        mdir = args.crowd_dir / m["member"]
        cache = mdir / "shape_eval_views"
        paths = extract_frames(mdir / RENDER_REL, args.frames, cache, "frame")
        if paths is None:
            skipped.append(m["member"])
            continue
        mask_paths = extract_mask_frames(mdir / RENDER_ALL_REL, args.frames, cache, "mask")
        members.append(m["member"])
        views[m["member"]] = paths
        masks[m["member"]] = mask_paths
    if len(members) < 2:
        sys.exit(f"only {len(members)} member(s) with shape renders — need >= 2 (skipped: {skipped})")
    print(f"crowd '{prompt}': {len(members)} members evaluated, {len(skipped)} skipped {skipped}")

    device = torch.device(args.device)
    rgbs = {m: [load_rgb(p) for p in views[m]] for m in members}
    fg_masks = {}
    for m in members:
        if masks[m] is None:
            fg_masks[m] = [np.ones(rgb.shape[:2], dtype=bool) for rgb in rgbs[m]]
        else:
            fg_masks[m] = [load_mask(p) for p in masks[m]]
    bboxes = {m: [bbox_from_mask(fg) for fg in fg_masks[m]] for m in members}
    cropped_rgbs = {m: [crop_pad_square(rgb, bbox) for rgb, bbox in zip(rgbs[m], bboxes[m])]
                    for m in members}
    patch_cov = {m: [patch_coverage_grid(fg) for fg in fg_masks[m]] for m in members}

    import lpips as lpips_lib
    lpips_net = lpips_lib.LPIPS(net="alex", verbose=False).to(device)

    def to_lpips_tensor(rgb):
        img = rgb.astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return torch.nn.functional.interpolate(t, size=(256, 256), mode="bilinear").to(device)

    lpips_imgs = {m: [to_lpips_tensor(r) for r in cropped_rgbs[m]] for m in members}
    lpips_pairs = []
    with torch.no_grad():
        for a, b in itertools.combinations(members, 2):
            d = np.mean([float(lpips_net(x, y)) for x, y in zip(lpips_imgs[a], lpips_imgs[b])])
            lpips_pairs.append(d)
    del lpips_imgs, lpips_net
    torch.cuda.empty_cache()

    from dreamsim import dreamsim
    dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device=str(device), cache_dir=MODEL_CACHE_DIR)
    from PIL import Image

    dreamsim_imgs = {m: [dreamsim_preprocess(Image.fromarray(r)).to(device) for r in cropped_rgbs[m]] for m in members}
    dreamsim_pairs = []
    with torch.no_grad():
        for a, b in itertools.combinations(members, 2):
            d = np.mean([float(dreamsim_model(x, y)) for x, y in zip(dreamsim_imgs[a], dreamsim_imgs[b])])
            dreamsim_pairs.append(d)
    del dreamsim_imgs, dreamsim_model
    torch.cuda.empty_cache()

    dinov2_model = load_dinov2(device)
    dinov2_patch, dinov2_cls_cropped = {}, {}
    with torch.no_grad():
        for m in members:
            batch_full = torch.stack([dinov2_transform(r) for r in rgbs[m]]).to(device)
            feats_full = dinov2_model.forward_features(batch_full)
            patches = feats_full["x_norm_patchtokens"]
            patches = patches / patches.norm(dim=-1, keepdim=True)
            dinov2_patch[m] = patches

            batch_crop = torch.stack([dinov2_transform(r) for r in cropped_rgbs[m]]).to(device)
            feats_crop = dinov2_model.forward_features(batch_crop)
            cls = feats_crop["x_norm_clstoken"]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            dinov2_cls_cropped[m] = cls
    del dinov2_model
    torch.cuda.empty_cache()

    tau = args.patch_coverage_tau
    patchwise_pairs = []
    view_fallbacks = 0
    for a, b in itertools.combinations(members, 2):
        per_view = []
        for v, (pa, pb) in enumerate(zip(dinov2_patch[a], dinov2_patch[b])):
            d_all = 1.0 - (pa * pb).sum(dim=-1)
            valid = ((patch_cov[a][v] >= tau) & (patch_cov[b][v] >= tau)).reshape(-1)
            valid_t = torch.from_numpy(valid).to(d_all.device)
            if valid_t.any():
                per_view.append(float(d_all[valid_t].mean()))
            else:
                view_fallbacks += 1
                per_view.append(float(d_all.mean()))
        patchwise_pairs.append(float(np.mean(per_view)))
    if view_fallbacks:
        print(f"note: {view_fallbacks} member-pair views had zero patches clearing "
              f"--patch-coverage-tau={tau} in both members — used all patches for those views")

    member_cls = torch.stack([dinov2_cls_cropped[m].mean(dim=0) for m in members])
    member_cls = member_cls / member_cls.norm(dim=-1, keepdim=True)
    dinov2_sim = (member_cls @ member_cls.T).cpu().numpy()

    lowres = {m: [low_res_l2(r) for r in cropped_rgbs[m]] for m in members}
    lowres_pairs = []
    for a, b in itertools.combinations(members, 2):
        ds = [float(np.linalg.norm(x - y)) for x, y in zip(lowres[a], lowres[b])]
        lowres_pairs.append(float(np.mean(ds)))

    results = {
        "crowd_prompt": prompt,
        "num_members_evaluated": len(members),
        "skipped_members": skipped,
        "frames": args.frames,
        "patch_coverage_tau": tau,
        "shape_diversity": {
            "dinov2_patchwise_pairwise": pairwise_stats(patchwise_pairs),
            "dreamsim_pairwise": pairwise_stats(dreamsim_pairs),
            "lpips_pairwise": pairwise_stats(lpips_pairs),
            "pixel_l2_32_pairwise": pairwise_stats(lowres_pairs),
            "vendi_score_dinov2": vendi_score(dinov2_sim),
            "dpp_score_dinov2": dpp_score(dinov2_sim),
        },
    }
    out = args.out or (args.crowd_dir / "shape_metrics.json")
    out.write_text(json.dumps(results, indent=2))

    d = results["shape_diversity"]
    print(f"\n=== {prompt} shape-only (n={len(members)}) ===")
    print(f"DINOv2 patchwise    mean {d['dinov2_patchwise_pairwise']['mean']:.4f}  "
          f"min-NN {d['dinov2_patchwise_pairwise']['min']:.4f}  std {d['dinov2_patchwise_pairwise']['std']:.4f}")
    print(f"DreamSim pairwise   mean {d['dreamsim_pairwise']['mean']:.4f}  "
          f"min-NN {d['dreamsim_pairwise']['min']:.4f}  std {d['dreamsim_pairwise']['std']:.4f}")
    print(f"LPIPS pairwise      mean {d['lpips_pairwise']['mean']:.4f}  "
          f"min-NN {d['lpips_pairwise']['min']:.4f}  std {d['lpips_pairwise']['std']:.4f}")
    print(f"Pixel-L2(32) pairwise mean {d['pixel_l2_32_pairwise']['mean']:.4f}  "
          f"min-NN {d['pixel_l2_32_pairwise']['min']:.4f}  std {d['pixel_l2_32_pairwise']['std']:.4f}")
    print(f"Vendi score (DINOv2) {d['vendi_score_dinov2']:.2f} effective members")
    print(f"DPP score (DINOv2)   {d['dpp_score_dinov2']:.4g}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
