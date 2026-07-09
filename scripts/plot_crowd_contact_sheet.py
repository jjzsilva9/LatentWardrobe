"""Contact-sheet figure: one row per method, one column per crowd member.

Each tile is a fixed turntable frame of that member's stage-2 render, pulled
via evaluate_crowd's cached extractors (eval_views/ + eval_masks/ -- created
on demand from render_rgb.mp4 / render_all.mp4 if the eval hasn't run yet)
and cropped to the foreground bbox exactly like the appearance metrics are,
so garments sit at comparable scale regardless of on-screen silhouette size.
Rows stack in the given --methods order, so noise pairs / prompt variants
can be read straight down a column (members are seed-matched within a
white/pink pair but NOT across prompt methods -- see manifest sd_seeds).

Usage (inside WSL, "eval" conda env):
    python scripts/plot_crowd_contact_sheet.py \
        --batch-dir outputs/factorial7_20260706 \
        --methods 1_random_white_plain 2_clip_white_plain ... \
        [--prompt-slug medieval_villagers] [--frame 0] [--no-crop] [--out ...]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_crowd as ec
from plot_crowd_comparison import METHOD_LABELS, SURFACE, INK, INK_MUTED

TILE_IN = 1.12      # inches per member tile
LABEL_BAND_IN = 0.34  # vertical space above each row for its method label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", required=True, help="row order, top to bottom")
    p.add_argument("--prompt-slug", default=None,
                   help="prompt dir name; default: the single prompt dir found in the first method")
    p.add_argument("--frame", type=int, default=0, help="turntable frame index (same angle across members)")
    p.add_argument("--no-crop", action="store_true",
                   help="show full frames instead of foreground-bbox square crops")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <batch-dir>/plots/contact_sheet_frame<frame>.png")
    return p.parse_args()


def find_prompt_slug(batch_dir, methods):
    for m in methods:
        dirs = [d for d in sorted((batch_dir / m).iterdir()) if (d / "manifest.json").exists()]
        if dirs:
            return dirs[0].name
    raise SystemExit(f"no prompt dir with manifest.json under {batch_dir} for {methods}")


def member_tile(member_dir, frame, crop=True):
    """RGB array for one member at the given frame, or None if no render."""
    views = ec.extract_views(member_dir, [frame])
    if views is None:
        return None
    rgb = ec.load_rgb(views[0])
    if crop:
        masks = ec.extract_mask_views(member_dir, [frame])
        if masks is not None:
            rgb = ec.crop_pad_square(rgb, ec.bbox_from_mask(ec.load_mask(masks[0])))
    return rgb


def main():
    args = parse_args()
    slug = args.prompt_slug or find_prompt_slug(args.batch_dir, args.methods)
    out = args.out or (args.batch_dir / "plots" / f"contact_sheet_frame{args.frame:03d}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # member list per method, from each manifest (order = member_00..)
    rows = []
    for method in args.methods:
        mpath = args.batch_dir / method / slug / "manifest.json"
        if not mpath.exists():
            print(f"[contact_sheet] {method}: no manifest yet, skipping row")
            continue
        manifest = json.loads(mpath.read_text())
        rows.append((method, [m["member"] for m in manifest["members"]]))
    if not rows:
        raise SystemExit("no methods with a manifest -- nothing to plot")
    n_cols = max(len(members) for _, members in rows)

    fig_w = n_cols * TILE_IN + 0.3
    fig_h = len(rows) * (TILE_IN + LABEL_BAND_IN) + 0.35
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.12 / fig_w, right=1 - 0.12 / fig_w,
                        top=1 - 0.52 / fig_h, bottom=0.06 / fig_h,
                        hspace=LABEL_BAND_IN / TILE_IN, wspace=0.04)

    for r, (method, members) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_axis_off()
            if c >= len(members):
                continue
            tile = member_tile(args.batch_dir / method / slug / members[c], args.frame,
                               crop=not args.no_crop)
            if tile is None:
                ax.text(0.5, 0.5, "no\nrender", ha="center", va="center",
                        fontsize=7, color=INK_MUTED, transform=ax.transAxes)
                continue
            ax.imshow(tile)
            if r == 0:
                ax.set_title(members[c].replace("member_", ""), fontsize=7, color=INK_MUTED, pad=2)
        # method label above its row, aligned with the grid's left edge
        # 0.17in offset clears the member-number column headers (row 0 only,
        # ~0.12in tall at pad=2) that sit between the label and the tiles.
        pos = axes[r][0].get_position()
        fig.text(pos.x0, pos.y1 + 0.17 / fig_h, METHOD_LABELS.get(method, method),
                 fontsize=9, color=INK, va="bottom")

    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
