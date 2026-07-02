# Garment mesh templates

Source: MakeHuman community asset packs (CC0) — https://static.makehumancommunity.org/assets/assetpacks/index.html

## Structure

One subfolder per outfit slot; one garment per `.obj` file (flat, no per-garment subfolders):

- `tops/` — shirts, t-shirts, polos, tube tops, sweaters
- `bottoms/` — jeans, trousers, skirts, shorts
- `outerwear/` — jackets, blazers, coats (optional extra layer; may be empty)
- `fullbody/` — dresses, gowns, one-pieces (sampled INSTEAD of top+bottom)

Per crowd member the generator samples: (1 top + 1 bottom) or (1 fullbody), later optionally + 1 outerwear. The sampled garments are merged into a single mesh with a shared repacked UV atlas and textured in ONE Paint3D run, so each outfit is generated coherently (same prompt/seed/views) rather than garment-by-garment.

## Export conventions (from MakeHuman → here)

1. **One garment per file.** Load the garment on the default MakeHuman body, export, and strip the body (e.g. delete body mesh in Blender), keeping only the garment.
2. **Keep the default pose and scale identical across ALL exports.** All MakeHuman garments are fitted to the same base body — that's what makes top+bottom combine without any fitting. Don't rotate/scale/move anything.
3. **Keep UVs.** Every garment must export with its UV coordinates intact (Paint3D stage 2 inpaints in UV space; a mesh without UVs is unusable). Wavefront `.obj`; `.mtl`/texture files may sit alongside but aren't required — textures get generated, not loaded.
4. Filenames: lowercase snake_case, descriptive (`jeans.obj`, `skirt_tiered.obj`). The filename becomes the garment's ID in run manifests.

Run `python scripts/validate_meshes.py` (to be written when first meshes land) to check UVs/scale/naming before use.
