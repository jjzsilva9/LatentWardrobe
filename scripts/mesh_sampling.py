"""Theme-aware garment sampling weights (Method 1a) — render-based.

Each garment is represented by CLIP **image** embeddings of its untextured
front/back renders (produced by scripts/render_garments.py; garment embedding
= mean of the two views, renormalized). The crowd theme prompt is embedded
with CLIP's text encoder, and image<->text cosine similarity — CLIP's native
training regime — scores each garment. Scores become per-slot sampling
weights via a softmax with temperature tau:

    tau -> infinity : uniform sampling (the baseline, exactly)
    tau small       : near-argmax on the best-matching garments

tau is therefore the coherence <-> mesh-diversity knob (see plan doc).

Renders are the ONLY garment representation, by design (decided 2026-07-03):
filenames/text descriptions of these assets mostly describe the ORIGINAL
textures ("polkadot", "disco", "medieval"), which we strip — Paint3D repaints
every garment to match the theme, so shape is the only property a mesh
actually contributes, and shape is exactly what the renders capture.

Preview CLI (safe on CPU while the GPU is busy generating):
    python scripts/mesh_sampling.py --prompt "medieval villagers" --tau 0.02
"""

import argparse
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = REPO_ROOT / "assets" / "meshes"
RENDER_DIR = MESH_DIR / "renders"
CACHE_PATH = MESH_DIR / "garment_clip_cache.pt"

# Raw theme prompts ("aliens on the moon") carry mostly non-clothing
# semantics; anchor the text side to clothing.
THEME_TEMPLATE = "clothing worn by {prompt}"

# Zero-shot garment category labels, assigned from the render CLIP embedding
# already computed for mesh sampling (Method 1a) — reused, at no extra render
# cost, to give Paint3D's prompt a structural hint ("shirt" vs "trousers")
# independent of the theme it's being repainted for. Chosen to cover the
# mesh library's shapes (see assets/meshes/README.md) without leaking
# original-texture words (no "disco", "medieval", ...).
CATEGORY_LABELS = [
    "t-shirt", "tank top", "button-up shirt", "blouse", "sweater", "hoodie",
    "jacket", "tunic", "crop top", "halter top",
    "trousers", "jeans", "shorts", "skirt", "mini skirt", "leggings",
    "dress", "gown", "jumpsuit", "suit", "robe", "uniform", "wetsuit", "apron",
]
CATEGORY_TEMPLATE = "a photo of a {label}"


def _clip(device):
    import open_clip  # env pins open_clip 2.0.2: module-level tokenize(), no get_tokenizer()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.to(device).eval(), open_clip.tokenize, preprocess


def _render_paths(key):
    slot, stem = key.split("/", 1)
    return (RENDER_DIR / slot / f"{stem}_front.png",
            RENDER_DIR / slot / f"{stem}_back.png")


def garment_keys():
    """All garments that have both an .obj and rendered views."""
    keys, missing = [], []
    for slot in ("tops", "bottoms", "fullbody"):
        for obj in sorted((MESH_DIR / slot).glob("*.obj")):
            key = f"{slot}/{obj.stem}"
            front, back = _render_paths(key)
            (keys if front.exists() and back.exists() else missing).append(key)
    return keys, missing


def _render_fingerprint(keys):
    return {k: max(p.stat().st_mtime for p in _render_paths(k)) for k in keys}


def _classify_categories(emb, model, tokenize, device="cpu"):
    """Zero-shot argmax category per garment, from its (already-computed)
    render embedding against CATEGORY_LABELS."""
    with torch.no_grad():
        tokens = tokenize([CATEGORY_TEMPLATE.format(label=l) for l in CATEGORY_LABELS]).to(device)
        text_emb = model.encode_text(tokens).float()
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    sims = emb.to(device) @ text_emb.T.to(device)
    idx = sims.argmax(dim=1).cpu().tolist()
    return [CATEGORY_LABELS[i] for i in idx]


def garment_embeddings(device="cpu"):
    """CLIP image embeddings + zero-shot category per rendered garment;
    disk-cached, refreshed per-garment when its renders change."""
    keys, missing = garment_keys()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} garment(s) lack renders (run scripts/render_garments.py): {missing[:5]} ...")
    fp = _render_fingerprint(keys)
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, map_location="cpu")
        if cache.get("fingerprint") == fp and cache.get("keys") == keys and "categories" in cache:
            return keys, cache["embeddings"], cache["categories"]

    from PIL import Image
    model, tokenize, preprocess = _clip(device)
    embs = []
    with torch.no_grad():
        for k in keys:
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                for p in _render_paths(k)]).to(device)
            e = model.encode_image(imgs).float()
            e /= e.norm(dim=-1, keepdim=True)
            e = e.mean(dim=0)
            embs.append(e / e.norm())
    emb = torch.stack(embs).cpu()
    categories = _classify_categories(emb, model, tokenize, device)
    torch.save({"fingerprint": fp, "keys": keys, "embeddings": emb, "categories": categories}, CACHE_PATH)
    return keys, emb, categories


def theme_weights(prompt, tau, device="cpu"):
    """Per-slot sampling weights for a theme prompt.

    Returns {slot: (obj_paths, weights)} with weights summing to 1 per slot.
    """
    keys, emb, _categories = garment_embeddings(device)
    model, tokenize, _ = _clip(device)
    with torch.no_grad():
        t = model.encode_text(tokenize([THEME_TEMPLATE.format(prompt=prompt)]).to(device)).float()
        t /= t.norm(dim=-1, keepdim=True)
    sims = (emb @ t.cpu().T).squeeze(1).numpy()

    out = {}
    for slot in ("tops", "bottoms", "fullbody"):
        idx = [i for i, k in enumerate(keys) if k.startswith(slot + "/")]
        if not idx:
            continue
        s = sims[idx] / tau
        w = np.exp(s - s.max())
        w /= w.sum()
        paths = [MESH_DIR / (keys[i] + ".obj") for i in idx]
        out[slot] = (paths, w)
    return out


def garment_categories(device="cpu"):
    """{key: zero-shot category label} for every rendered garment."""
    keys, _emb, categories = garment_embeddings(device)
    return dict(zip(keys, categories))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", required=False)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--top", type=int, default=8, help="show top N and bottom 3 per slot")
    p.add_argument("--device", default="cpu")
    p.add_argument("--show-categories", action="store_true",
                   help="print the zero-shot category assigned to every garment, and exit")
    args = p.parse_args()

    if args.show_categories:
        for key, cat in sorted(garment_categories(args.device).items()):
            print(f"{cat:20s} {key}")
        return

    if not args.prompt:
        p.error("--prompt is required unless --show-categories is given")

    for slot, (paths, w) in theme_weights(args.prompt, args.tau, args.device).items():
        order = np.argsort(-w)
        print(f"\n=== {slot} (n={len(paths)}, tau={args.tau}) — '{args.prompt}'")
        for i in order[:args.top]:
            print(f"  {w[i]:6.1%}  {paths[i].stem}")
        print("  ...")
        for i in order[-3:]:
            print(f"  {w[i]:6.1%}  {paths[i].stem}")


if __name__ == "__main__":
    main()
