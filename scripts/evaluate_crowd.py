"""Crowd evaluation harness.

Computes the dissertation's metrics for one generated crowd (one prompt dir
produced by generate_crowd_baseline.py). Axis design follows "It's Never Too
Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models",
whose human studies favor SET-LEVEL diversity metrics -- trimmed 2026-07-17
to only what is reported (the pairwise LPIPS / pixel-L2 / DINOv2-patchwise
battery was cut outright; computing unreported metrics was pure cost):

  Diversity   - Vendi score over a DINOv2 [CLS] cosine-similarity kernel
                (effective number of distinct members)
              - DPP score: n-th-rooted determinant (geometric mean of
                eigenvalues) of that same DINOv2 [CLS] kernel
              - DreamSim pairwise distance (human-perception-aligned)
              - pairwise hue-histogram chi-square distance (color diversity)
  Coherence   - CLIPScore-style cosine similarity between the crowd prompt and
                each member's rendered views (per member + crowd mean)
  Quality     - HPSv2 (v2.1) human-preference score per member + crowd mean

Every pairwise statistic reports mean, min (nearest-neighbour / clone
detection), and std across the contributing member pairs.

All appearance metrics (LPIPS, DreamSim, pixel-L2, DINOv2 CLS/Vendi/DPP) are
computed on a fixed set of frames from each member's stage-2 turntable render
(render_rgb.mp4) -- or, for a --skip-stage2 crowd, a turntable of stage1's own
unrefined texture instead (RENDER_REL_STAGE1 fallback; color diversity comes
from stage1's albedo via a derived placeholder mask, flagged
color_source="stage1_derived_mask" -- see uv_hue_histogram) -- cropped to that frame's
foreground bounding box (see
"Background handling" below) -- NOT on raw UV maps (see plan doc: UV layouts
are mostly padding and, more importantly, aren't spatially comparable across
different garment meshes). Frames at the same index correspond to the same
camera angle across members.

Background handling (added 2026-07-07, replacing an HSV "near-white" pixel
heuristic that risked misclassifying light/cream garments as background):
Paint3D's renderer (external/paint3d/paint3d/models/render.py) rasterizes a
true per-pixel foreground mask every frame (`face_idx > -1`) and bakes it
into the z-normals panel of render_all.mp4 by forcing background pixels to
exactly white before that panel is written -- see trainer.py's dr_eval(). We
re-derive a binary foreground mask from that panel (thresholding near-white)
for every frame already on disk, no re-rendering required:
  - pixel-L2 / DreamSim / LPIPS / DINOv2-CLS (Vendi, DPP): each view is
    cropped to the mask's bounding box (small margin, padded to a centered
    square with white fill so we don't introduce a new aspect-ratio
    confound) before the metric's existing preprocessing. This removes a
    real confound where a member's incidental on-screen silhouette size
    (long dress vs. cropped top) changed how much of the frame was shared
    identical-white background, diluting/distorting these whole-image
    metrics independent of actual appearance diversity.
  - Color diversity: switched from a per-frame render-based hue histogram to
    the member's stage-2 UV albedo texture (stage2/albedo.png), masked by
    stage2/mask.png -- the *exact* boolean "which texels got painted" mask
    Paint3D itself uses to drive inpainting (not a heuristic), giving a
    correct mask with zero light-garment blind spot and coverage of the
    whole texture rather than 4 sampled camera angles. Note: must use the
    coarse-stage stage2/mask.png + stage2/albedo.png pair, NOT
    stage2/tile_res_0/albedo.png -- verified empirically that the tiling
    refinement pass overwrites the unpainted-texel placeholder color, so the
    coarse mask no longer lines up with the tile-refined albedo.
  - Theme coherence (CLIPScore): deliberately left uncropped/unmasked --
    it's measuring whole-scene prompt alignment, not fine-grained
    appearance, and there's no silhouette-size confound to fix since it
    isn't a diversity metric.

Usage (inside WSL, "eval" conda env -- see .claude/setup/install_eval_env.sh,
kept separate from the "paint3d" env so upgrading eval deps can't break
generation):
    conda activate eval
    python scripts/evaluate_crowd.py --crowd-dir outputs/<batch>/<prompt_slug> \
        [--frames 0 15 30 45] [--device cuda] [--patch-coverage-tau 0.5]

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
RENDER_ALL_REL = "stage2/tile_res_0/render_all.mp4"  # rgb|depth|z_normals panels, same frame index as RENDER_REL
# Coarse-stage fallbacks (added 2026-07-09): the 2026-07-08 crash left 3
# factorial7 members with a finished tile_res_0/albedo.png but no tile_res_0
# renders -- the coarse stage2/ renders exist and use the same camera path, so
# fall back to them (scored slightly pre-refinement) rather than dropping the
# member from every pairwise metric. Cached separately (eval_views_coarse) so
# a later re-render of tile_res_0 can't collide with stale coarse frames.
RENDER_REL_COARSE = "stage2/render_rgb.mp4"
RENDER_ALL_REL_COARSE = "stage2/render_all.mp4"
# --skip-stage2 crowds (generate_crowd_baseline.py) have no stage2/ at all --
# fall back further to a turntable of stage1's own (unrefined) texture.
# Color diversity falls back too: stage1 never writes a painted-texel
# mask.png, but its placeholder texels hold exactly one value, so
# uv_hue_histogram() derives an equivalent mask from the albedo itself and
# the result is flagged color_source="stage1_derived_mask" in metrics.json
# (verified 2026-07-15: zero pixels near-but-not-exactly the placeholder, and
# the eval turntable frames contain no placeholder pixels at all -- so the
# appearance metrics need no masking change).
RENDER_REL_STAGE1 = "stage1_render/render_rgb.mp4"
RENDER_ALL_REL_STAGE1 = "stage1_render/render_all.mp4"
STAGE1_UV_ALBEDO_REL = "stage1/res-0/albedo.png"  # coarse texture, unpainted texels = placeholder below
STAGE1_PLACEHOLDER_BGR = np.array([204, 25, 204], dtype=np.uint8)  # texture_default_color as uint8; see uv_hue_histogram
UV_ALBEDO_REL = "stage2/albedo.png"  # coarse stage -- see module docstring, don't swap for tile_res_0's copy
UV_MASK_REL = "stage2/mask.png"      # matches UV_ALBEDO_REL only; 0=painted, 255=unpainted (magic-pink) placeholder
VIEWS_CACHE = "eval_views"
MASK_CACHE = "eval_masks"
DINOV2_INPUT_SIZE = 224  # 16x16 = 256 patches at patch size 14
BG_GRAY_THRESH = 250     # z-normals panel: background is forced to exactly 255 pre-compression; small margin for mp4 lossiness
BBOX_MARGIN_FRAC = 0.02  # extra margin around the tight foreground bbox, as a fraction of bbox size, before the square pad

# DreamSim defaults its weight cache to "./models" (relative to cwd) and, as
# a side effect, calls torch.hub.set_dir() globally -- so torch.hub.load()
# for DINOv2 below lands in that same folder too. Left at the default
# location (repo root, gitignored) rather than redirected elsewhere, since
# it's just a downloaded-weights cache, not repo content.
MODEL_CACHE_DIR = "./models"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crowd-dir", type=Path, required=True, help="one prompt dir with manifest.json")
    p.add_argument("--frames", type=int, nargs="+", default=[0, 50],
                   help="frame indices to sample from each turntable render (same angle across members). "
                        "Default is front (0) + back (50 of 100) only: for a --skip-stage2 crowd, stage1's "
                        "few actual camera views land near front/back, so side angles fall in the gaps "
                        "between them and show the worst unfilled (magenta/white-seam) UV -- see "
                        "RENDER_REL_STAGE1's comment. Pass e.g. --frames 0 15 30 45 to restore the old "
                        "4-angle sampling for a fully stage2-textured crowd.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="default: <crowd-dir>/metrics.json")
    return p.parse_args()


def extract_views(member_dir, frame_ids):
    """Grab fixed frames from the member's turntable video; cache as PNGs.
    Falls back to the coarse-stage render (see RENDER_REL_COARSE) when the
    tile_res_0 render is missing."""
    video, cache_name = member_dir / RENDER_REL, VIEWS_CACHE
    if not video.exists():
        video, cache_name = member_dir / RENDER_REL_COARSE, VIEWS_CACHE + "_coarse"
    if not video.exists():
        video, cache_name = member_dir / RENDER_REL_STAGE1, VIEWS_CACHE + "_stage1"
    if not video.exists():
        return None
    cache = member_dir / cache_name
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


def extract_mask_views(member_dir, frame_ids):
    """Grab the z-normals panel (rightmost third) of the matching render_all.mp4
    frames, threshold to a binary foreground mask, cache as PNGs. Returns None
    if render_all.mp4 is missing (older data) -- callers fall back to
    unmasked behaviour in that case."""
    video, cache_name = member_dir / RENDER_ALL_REL, MASK_CACHE
    if not video.exists():
        video, cache_name = member_dir / RENDER_ALL_REL_COARSE, MASK_CACHE + "_coarse"
    if not video.exists():
        video, cache_name = member_dir / RENDER_ALL_REL_STAGE1, MASK_CACHE + "_stage1"
    if not video.exists():
        return None
    cache = member_dir / cache_name
    cache.mkdir(exist_ok=True)
    paths = [cache / f"mask_{i:03d}.png" for i in frame_ids]
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


def load_rgb(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return img


def load_mask(path):
    """Boolean (H, W) foreground mask, True = garment."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127


def bbox_from_mask(mask, margin_frac=BBOX_MARGIN_FRAC):
    """Tight foreground bounding box with a small margin. None if degenerate
    (near-empty mask -- e.g. a view where the garment barely enters frame)."""
    if mask.sum() < 100:
        return None
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = mask.shape
    my = int((y1 - y0) * margin_frac)
    mx = int((x1 - x0) * margin_frac)
    return max(0, y0 - my), min(h, y1 + my), max(0, x0 - mx), min(w, x1 + mx)


def crop_pad_square(rgb, bbox):
    """Crop to bbox, then pad the shorter side with white to make it square
    (centered) -- avoids stretching while still normalizing away incidental
    on-screen silhouette size across members."""
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




def hue_histogram(rgb, fg_mask):
    mask = fg_mask.astype(np.uint8)
    if mask.sum() < 100:  # nearly empty — degenerate
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    return (hist / max(hist.sum(), 1e-8)).flatten()


def uv_hue_histogram(member_dir):
    """Color diversity straight from the painted UV texture, masked by
    Paint3D's own unpainted-texel mask (see module docstring). Returns
    (histogram, source) where source records which mask was used --
    "stage2_mask" (Paint3D's own mask.png) or "stage1_derived_mask" (the
    --skip-stage2 fallback below) -- so stage1- and stage2-scored color
    numbers can be kept apart / combined deliberately downstream."""
    albedo_path, mask_path = member_dir / UV_ALBEDO_REL, member_dir / UV_MASK_REL
    if albedo_path.exists() and mask_path.exists():
        albedo = cv2.cvtColor(cv2.imread(str(albedo_path)), cv2.COLOR_BGR2RGB)
        uv_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        fg = uv_mask == 0  # 0 = painted texel, 255 = unpainted magic-pink placeholder
        return hue_histogram(albedo, fg), "stage2_mask"
    # --skip-stage2 fallback: stage1 writes no painted-texel mask.png, but its
    # unpainted texels hold exactly one value -- texture_default_color
    # [0.8, 0.1, 0.8] (external/paint3d/paint3d/config/train_config_paint3d.py)
    # as uint8 (204, 25, 204). Verified empirically (2026-07-15, this batch):
    # the placeholder occurs ONLY as that exact value (zero near-miss pixels
    # within tol 60, no dilation/AA bleed), so exact matching reproduces the
    # stage2 mask's painted-texel semantics rather than approximating them.
    s1_albedo_path = member_dir / STAGE1_UV_ALBEDO_REL
    if not s1_albedo_path.exists():
        return None, None
    bgr = cv2.imread(str(s1_albedo_path))
    fg = (bgr != STAGE1_PLACEHOLDER_BGR).any(axis=-1)
    if fg.mean() > 0.8:
        # the placeholder covers UV padding at minimum (~30%+ of the map in
        # practice); almost-no-placeholder means texture_default_color has
        # changed and the derived mask can't be trusted -- leave unscored.
        print(f"warning: <20% placeholder texels in {s1_albedo_path} -- "
              f"texture_default_color convention changed? color unscored for this member")
        return None, None
    return hue_histogram(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fg), "stage1_derived_mask"


def chi_square(h1, h2):
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-8)))




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
    & Taskar 2011 via [31] in the collapse-recovery paper), n-th-rooted (n =
    number of members) to report the geometric mean of the kernel's
    eigenvalues rather than their raw product. Same DINOv2 [CLS] kernel as
    the Vendi score above -- near 0 means near-duplicate members, near 1
    means a maximally spread-out set. The un-rooted determinant is a product
    of n eigenvalues that sum to n (unit diagonal), so it collapses toward 0
    almost regardless of actual diversity as n grows -- e.g. an n=10 crowd
    with Vendi ~4.7 (moderately diverse) has a raw determinant ~0.0008, which
    reads as "all clones" despite the n-th root of that same number being
    ~0.49, in line with Vendi. Rooting makes the score comparable across
    crowds of different size, matching how Vendi is already normalized."""
    n = sim_matrix.shape[0]
    sign, logdet = np.linalg.slogdet(sim_matrix)
    return float(sign * np.exp(logdet / n)) if sign > 0 else 0.0


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

    # --- collect views + masks per completed member
    members, views, masks, skipped, coarse_members = [], {}, {}, [], []
    mask_fallback = 0
    for m in manifest["members"]:
        mdir = args.crowd_dir / m["member"]
        paths = extract_views(mdir, args.frames)
        if paths is None:
            skipped.append(m["member"])
            continue
        if not (mdir / RENDER_REL).exists():
            coarse_members.append(m["member"])
        mask_paths = extract_mask_views(mdir, args.frames)
        if mask_paths is None:
            mask_fallback += 1
        members.append(m["member"])
        views[m["member"]] = paths
        masks[m["member"]] = mask_paths
    if len(members) < 2:
        sys.exit(f"only {len(members)} member(s) with renders — need >= 2 for pairwise metrics "
                 f"(skipped: {skipped})")
    print(f"crowd '{prompt}': {len(members)} members evaluated, {len(skipped)} skipped {skipped}")
    if coarse_members:
        print(f"note: {len(coarse_members)} member(s) scored on the coarse-stage render "
              f"(tile_res_0 render missing): {coarse_members}")
    if mask_fallback:
        print(f"warning: {mask_fallback} member(s) missing render_all.mp4 (older data) — "
              f"background masking disabled for them (full-frame fallback)")

    device = torch.device(args.device)
    rgbs = {m: [load_rgb(p) for p in views[m]] for m in members}

    # per-view boolean foreground masks (1024-res) and derived bbox/patch-coverage.
    # falls back to "everything is foreground" when render_all.mp4 wasn't available.
    fg_masks = {}
    for m in members:
        if masks[m] is None:
            fg_masks[m] = [np.ones(rgb.shape[:2], dtype=bool) for rgb in rgbs[m]]
        else:
            fg_masks[m] = [load_mask(p) for p in masks[m]]

    bboxes = {m: [bbox_from_mask(fg) for fg in fg_masks[m]] for m in members}
    cropped_rgbs = {m: [crop_pad_square(rgb, bbox) for rgb, bbox in zip(rgbs[m], bboxes[m])]
                    for m in members}
    # (LPIPS, pixel-L2 and DINOv2-patchwise were cut 2026-07-17 -- reported
    # metrics are the set-level pair (Vendi/DPP), DreamSim, color chi2,
    # CLIPScore and HPSv2; computing the rest was pure waste.)

    # --- DreamSim (view-matched pairwise, averaged over views) -- bbox-cropped
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

    # --- CLIP embeddings (open_clip, ViT-B-32) for coherence only -- intentionally
    # uncropped/unmasked, see module docstring.
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

    # --- quality: HPSv2 (v2.1) human-preference score, restored 2026-07-16.
    # Harrington et al. report set-level diversity (Vendi/DPP) alongside
    # CLIPScore coherence and HPSv2 quality; we follow that axis design (the
    # 2026-07-06 "not worth the dependency" call is reversed -- see
    # install_eval_env.sh for the packaging fix the dependency needs).
    # Scored on the same uncropped eval frames as CLIPScore, averaged per
    # member. Guarded: an env without hpsv2 or its checkpoint still writes
    # metrics.json, with quality nulled.
    try:
        import hpsv2
        hps = {}
        for m in members:
            scores = hpsv2.score([str(p) for p in views[m]], prompt, hps_version="v2.1")
            hps[m] = float(np.mean([float(s) for s in scores]))
    except Exception as e:
        print(f"warning: HPSv2 scoring unavailable ({e}) -- quality unscored")
        hps = None
    del clip_model, clip_view_embs
    torch.cuda.empty_cache()

    # --- DINOv2: Vendi + DPP over a [CLS] kernel built from the bbox-CROPPED
    # frames (see module docstring). The uncropped patch-token pass that fed
    # the patchwise metric is gone with it -- halves the DINOv2 cost.
    dinov2_model = load_dinov2(device)
    dinov2_cls_cropped = {}  # member -> (num_views, D) CLS embeddings, bbox-cropped
    with torch.no_grad():
        for m in members:
            batch_crop = torch.stack([dinov2_transform(r) for r in cropped_rgbs[m]]).to(device)
            feats_crop = dinov2_model.forward_features(batch_crop)
            cls = feats_crop["x_norm_clstoken"]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            dinov2_cls_cropped[m] = cls
    del dinov2_model
    torch.cuda.empty_cache()

    # member-level [CLS] embedding (mean over cropped views, re-normalized) for
    # the Vendi / DPP kernel.
    member_cls = torch.stack([dinov2_cls_cropped[m].mean(dim=0) for m in members])
    member_cls = member_cls / member_cls.norm(dim=-1, keepdim=True)
    dinov2_sim = (member_cls @ member_cls.T).cpu().numpy()

    # --- color diversity: UV albedo texture, masked by Paint3D's own unpainted-texel
    # mask (stage2) or the placeholder-derived equivalent (stage1) -- see uv_hue_histogram
    hists, color_sources = {}, set()
    for m in members:
        hists[m], src = uv_hue_histogram(args.crowd_dir / m)
        if src is not None:
            color_sources.add(src)
    if "stage1_derived_mask" in color_sources:
        print("color diversity: stage1 placeholder-derived mask in use (no stage2/mask.png)")
    color_pairs = []
    for a, b in itertools.combinations(members, 2):
        if hists[a] is not None and hists[b] is not None:
            color_pairs.append(chi_square(hists[a], hists[b]))

    results = {
        "crowd_prompt": prompt,
        "num_members_evaluated": len(members),
        "skipped_members": skipped,
        "coarse_render_members": coarse_members,
        "frames": args.frames,
        "diversity": {
            "dreamsim_pairwise": pairwise_stats(dreamsim_pairs),
            "color_chi2_pairwise": pairwise_stats(color_pairs) if color_pairs else None,
            "color_source": "+".join(sorted(color_sources)) if color_sources else None,
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
    out = args.out or (args.crowd_dir / "metrics.json")
    out.write_text(json.dumps(results, indent=2))

    d, c = results["diversity"], results["coherence"]
    print(f"\n=== {prompt} (n={len(members)}) ===")
    print(f"DreamSim pairwise   mean {d['dreamsim_pairwise']['mean']:.4f}  "
          f"min-NN {d['dreamsim_pairwise']['min']:.4f}  std {d['dreamsim_pairwise']['std']:.4f}")
    if d["color_chi2_pairwise"]:
        print(f"Color chi2 pairwise mean {d['color_chi2_pairwise']['mean']:.4f}  std {d['color_chi2_pairwise']['std']:.4f}")
    print(f"Vendi score (DINOv2) {d['vendi_score_dinov2']:.2f} effective members")
    print(f"DPP score (DINOv2)   {d['dpp_score_dinov2']:.4g}")
    print(f"Theme coherence     crowd mean {c['clipscore_crowd_mean']:.4f}  std {c['clipscore_crowd_std']:.4f}")
    q = results["quality"]
    if q["hpsv2_crowd_mean"] is not None:
        print(f"Quality (HPSv2)     crowd mean {q['hpsv2_crowd_mean']:.4f}  std {q['hpsv2_crowd_std']:.4f}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
