"""Shape/silhouette-only diversity evaluation (dissertation eval Part 1).

Companion to evaluate_crowd.py, scoped to what mesh SELECTION alone can
change -- run against the untextured turntables from generate_shape_eval.py
(member_XX/shape_eval/render_rgb.mp4 + render_all.mp4), not the final
textured renders. Same reported battery as evaluate_crowd.py (Vendi/DPP over
a DINOv2 CLS kernel, DreamSim; trimmed 2026-07-17 -- see its docstring) plus
pairwise silhouette 1-IoU (pure geometry, no pretrained model), same
background-masking approach (z-normals panel of render_all.mp4).

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
BG_GRAY_THRESH = 250
BBOX_MARGIN_FRAC = 0.02
MODEL_CACHE_DIR = "./models"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crowd-dir", type=Path, required=True, help="one prompt dir with manifest.json")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 15, 30, 45])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    # (LPIPS, pixel-L2 and DINOv2-patchwise cut 2026-07-17, same reasoning as
    # evaluate_crowd.py: unreported metrics were pure compute waste.)
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

    # --- theme coherence (CLIPScore) + quality (HPSv2) on the UNTEXTURED
    # renders (added 2026-07-17: the mesh stage previously had no coherence
    # axis at all). Caveats, to be repeated wherever these numbers are read:
    # CLIP's signal on grey shading-only renders is weak (its texture/colour
    # cues are absent), and although mesh sampling also uses CLIP, sampling
    # anchors on textured garment PREVIEW renders while this scores the grey
    # merged-outfit render -- indirect circularity, not none. A supporting
    # axis, not standalone validation.
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

    try:
        import hpsv2
        hps = {}
        for m in members:
            scores = hpsv2.score([str(p) for p in views[m]], prompt, hps_version="v2.1")
            hps[m] = float(np.mean([float(s) for s in scores]))
    except Exception as e:
        print(f"warning: HPSv2 scoring unavailable ({e}) -- quality unscored")
        hps = None

    dinov2_model = load_dinov2(device)
    dinov2_cls_cropped = {}
    with torch.no_grad():
        for m in members:
            batch_crop = torch.stack([dinov2_transform(r) for r in cropped_rgbs[m]]).to(device)
            feats_crop = dinov2_model.forward_features(batch_crop)
            cls = feats_crop["x_norm_clstoken"]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            dinov2_cls_cropped[m] = cls
    del dinov2_model
    torch.cuda.empty_cache()

    member_cls = torch.stack([dinov2_cls_cropped[m].mean(dim=0) for m in members])
    member_cls = member_cls / member_cls.norm(dim=-1, keepdim=True)
    dinov2_sim = (member_cls @ member_cls.T).cpu().numpy()

    # --- silhouette IoU: pure-geometry pairwise diversity, no pretrained
    # model in the loop. Masks are pixel-aligned across members (identical
    # camera path at each frame index), so 1 - IoU per view, averaged over
    # views, is a direct outline-overlap distance. Members without a real
    # z-normals-derived mask are excluded: their all-ones fallback masks
    # would score IoU 1.0 (distance 0) against anything, silently deflating
    # the crowd's stats.
    iou_members = [m for m in members if masks[m] is not None]
    iou_pairs = []
    for a, b in itertools.combinations(iou_members, 2):
        ds = []
        for ma, mb in zip(fg_masks[a], fg_masks[b]):
            union = np.logical_or(ma, mb).sum()
            if union:
                ds.append(1.0 - np.logical_and(ma, mb).sum() / union)
        if ds:
            iou_pairs.append(float(np.mean(ds)))
    if len(iou_members) < len(members):
        print(f"note: silhouette IoU computed over {len(iou_members)}/{len(members)} "
              f"members (rest lack a render_all.mp4 foreground mask)")

    results = {
        "crowd_prompt": prompt,
        "num_members_evaluated": len(members),
        "skipped_members": skipped,
        "frames": args.frames,
        "shape_diversity": {
            "dreamsim_pairwise": pairwise_stats(dreamsim_pairs),
            "silhouette_iou_dist_pairwise": pairwise_stats(iou_pairs) if iou_pairs else None,
            "silhouette_iou_members": len(iou_members),
            "vendi_score_dinov2": vendi_score(dinov2_sim),
            "dpp_score_dinov2": dpp_score(dinov2_sim),
        },
        "coherence": {
            "clipscore_per_member": coherence,
            "clipscore_crowd_mean": float(np.mean(list(coherence.values()))),
            "clipscore_crowd_std": float(np.std(list(coherence.values()))),
        },
        "quality": {
            "hpsv2_per_member": hps,
            "hpsv2_crowd_mean": float(np.mean(list(hps.values()))) if hps else None,
            "hpsv2_crowd_std": float(np.std(list(hps.values()))) if hps else None,
        },
    }
    out = args.out or (args.crowd_dir / "shape_metrics.json")
    out.write_text(json.dumps(results, indent=2))

    d = results["shape_diversity"]
    print(f"\n=== {prompt} shape-only (n={len(members)}) ===")
    print(f"DreamSim pairwise   mean {d['dreamsim_pairwise']['mean']:.4f}  "
          f"min-NN {d['dreamsim_pairwise']['min']:.4f}  std {d['dreamsim_pairwise']['std']:.4f}")
    if d["silhouette_iou_dist_pairwise"]:
        print(f"Silhouette 1-IoU    mean {d['silhouette_iou_dist_pairwise']['mean']:.4f}  "
              f"min-NN {d['silhouette_iou_dist_pairwise']['min']:.4f}  std {d['silhouette_iou_dist_pairwise']['std']:.4f}")
    print(f"Vendi score (DINOv2) {d['vendi_score_dinov2']:.2f} effective members")
    print(f"DPP score (DINOv2)   {d['dpp_score_dinov2']:.4g}")
    c, q = results["coherence"], results["quality"]
    print(f"Theme coherence     crowd mean {c['clipscore_crowd_mean']:.4f}  std {c['clipscore_crowd_std']:.4f}  (untextured renders)")
    if q["hpsv2_crowd_mean"] is not None:
        print(f"Quality (HPSv2)     crowd mean {q['hpsv2_crowd_mean']:.4f}  std {q['hpsv2_crowd_std']:.4f}  (untextured renders)")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
