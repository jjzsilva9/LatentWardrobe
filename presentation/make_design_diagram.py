"""Draw the Design-slide pipeline figure with real intermediate images.

Composes the LatentWardrobe system diagram (WordRobe-Fig-2 style): theme
prompt -> per-member prompt (3 variants, WANDER highlighted) -> CLIP-weighted
garment sampling -> Paint3D 2-stage texturing (white/pink initial latent) ->
textured member -> metrics. All embedded images come from the factorial7
outputs (cell 8, medieval villagers), so every pixel in the figure is real
system output. Run inside WSL ("eval" env):

    cd "/mnt/u/My Documents/LatentWardrobe"
    python presentation/make_design_diagram.py

Writes presentation/design_pipeline.png (+ .svg for editing).
"""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
CROWD = ROOT / "outputs/factorial7_20260706/8_clip_white_llm6/medieval_villagers"
OUT = ROOT / "presentation"

# chrome tokens (match plot_crowd_comparison.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
RULE = "#c3c2b7"
BLUE = "#2a78d6"    # factor: noise (white) / general accent
PINK = "#e87ba4"    # factor: noise (pink)
GREEN = "#1baf7a"   # factor: prompt
ORANGE = "#eb6834"  # factor: mesh

FW, FH = 16.0, 6.2  # figure inches


def load_rgb(p):
    img = cv2.imread(str(p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_square(img, pad_color=255):
    """center square crop-or-pad on white."""
    h, w = img.shape[:2]
    side = max(h, w)
    sq = np.full((side, side, 3), pad_color, np.uint8)
    y, x = (side - h) // 2, (side - w) // 2
    sq[y:y + h, x:x + w] = img
    return sq


def fg_crop(img, thresh=250, margin=0.04):
    """tight crop to non-white foreground."""
    gray = img.mean(axis=2)
    ys, xs = np.where(gray < thresh)
    if len(ys) < 50:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    return img[max(0, y0 - my):y1 + my, max(0, x0 - mx):x1 + mx]


def noise_patch(pink=False, n=96, alpha=0.2, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n))
    if pink:
        f = np.fft.fftfreq(n)
        fx, fy = np.meshgrid(f, f)
        radial = np.sqrt(fx ** 2 + fy ** 2) * n
        weight = 1.0 / (1.0 + radial) ** alpha
        z = np.real(np.fft.ifft2(np.fft.fft2(z) * weight))
        z = (z - z.mean()) / z.std()
    z = np.clip((z + 3) / 6, 0, 1)
    return (plt.cm.gray(z)[..., :3] * 255).astype(np.uint8)


def box(ax, x, y, w, h, color, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                                fill=True, facecolor="white", edgecolor=color,
                                linewidth=lw, mutation_aspect=FW / FH, zorder=2))


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.6, style="-|>", con="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, color=color,
                                 linewidth=lw, mutation_scale=14,
                                 connectionstyle=con, zorder=4))


def thumb(fig, img, x, y, w, h, border=RULE):
    a = fig.add_axes([x, y, w, h], zorder=3)
    a.imshow(img)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(border); s.set_linewidth(0.8)
    return a


def main():
    member = CROWD / "member_00"
    desc = json.loads((CROWD / "llm_members.json").read_text())["members"][0]["llm"]["clothing_desc"]

    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    band_y, band_h = 0.16, 0.66   # main pipeline band

    # ---- theme prompt ----------------------------------------------------
    ax.text(0.055, 0.72, "theme prompt", ha="center", fontsize=10, color=INK, weight="bold")
    box(ax, 0.015, 0.52, 0.08, 0.14, RULE)
    ax.text(0.055, 0.59, '"medieval\nvillagers"', ha="center", va="center", fontsize=10,
            color=INK, style="italic")

    # ---- factor 2: per-member prompt --------------------------------------
    px, pw = 0.125, 0.185
    ax.text(px + pw / 2, 0.88, "FACTOR — per-member prompt", ha="center", fontsize=10,
            color=GREEN, weight="bold")
    box(ax, px, band_y, pw, band_h, GREEN)
    ax.text(px + 0.012, 0.70, "○  plain theme", fontsize=9.5, color=INK2)
    ax.text(px + 0.012, 0.63, "○  + random colour word", fontsize=9.5, color=INK2)
    ax.text(px + 0.012, 0.56, "●  WANDER-style evolution", fontsize=9.5, color=INK, weight="bold")
    ax.text(px + 0.026, 0.505, "6 clothing axes · LLM mutations\n· CLIP-text novelty search",
            fontsize=8, color=INK2)
    # evolved description sample
    box(ax, px + 0.012, 0.30, pw - 0.024, 0.13, GREEN, lw=0.9)
    ax.text(px + pw / 2, 0.365, f'"{desc}"', ha="center", va="center", fontsize=8.2,
            color=INK, style="italic", wrap=True,
            bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"))
    ax.text(px + pw / 2, 0.265, "evolved member description", ha="center", fontsize=7.5, color=MUTED)

    # ---- factor 1: mesh sampling ------------------------------------------
    mx, mw = 0.345, 0.20
    ax.text(mx + mw / 2, 0.88, "FACTOR — garment mesh selection", ha="center", fontsize=10,
            color=ORANGE, weight="bold")
    box(ax, mx, band_y, mw, band_h, ORANGE)
    ax.text(mx + mw / 2, 0.775, "library: 143 CC0 garment meshes", ha="center", fontsize=8.5, color=INK2)
    # silhouette thumbs: depth renders from 4 members of the same crowd
    sil_members = ["member_00", "member_02", "member_03", "member_07"]
    weights = [0.34, 0.27, 0.22, 0.17]
    tw = 0.040
    for i, (mname, wgt) in enumerate(zip(sil_members, weights)):
        p = CROWD / mname / "stage1" / "init_depth_render.png"
        img = fg_crop(load_rgb(p))
        x = mx + 0.010 + i * (tw + 0.007)
        thumb(fig, crop_square(img), x, 0.47, tw, tw * FW / FH)
        # CLIP-similarity bar under each thumb
        ax.add_patch(plt.Rectangle((x, 0.435), tw, 0.018, fc="#eceef5", ec="none", zorder=3))
        ax.add_patch(plt.Rectangle((x, 0.435), tw * wgt / max(weights), 0.018, fc=ORANGE, ec="none", zorder=4))
    ax.text(mx + mw / 2, 0.395, "CLIP cosine( text description , garment render )",
            ha="center", fontsize=8, color=INK2)
    ax.text(mx + mw / 2, 0.30, "softmax(τ) → sample outfit\nτ dial:  coherent ◄──► diverse",
            ha="center", fontsize=8.5, color=INK)
    ax.text(mx + mw / 2, 0.21, "uniform-random = baseline (τ→∞)", ha="center", fontsize=7.5, color=MUTED)

    # ---- factor 3 + Paint3D ------------------------------------------------
    gx, gw = 0.585, 0.21
    ax.text(gx + gw / 2, 0.88, "Paint3D 2-stage texturing", ha="center", fontsize=10,
            color=INK, weight="bold")
    box(ax, gx, band_y, gw, band_h, BLUE)
    ax.text(gx + gw / 2, 0.775, "SD 1.5 + depth ControlNet", ha="center", fontsize=8.5, color=INK2)
    # noise factor chip
    ax.text(gx + 0.048, 0.71, "FACTOR — initial latent", fontsize=8.5, color=BLUE, weight="bold")
    thumb(fig, noise_patch(False), gx + 0.012, 0.53, 0.036, 0.036 * FW / FH)
    thumb(fig, noise_patch(True), gx + 0.058, 0.53, 0.036, 0.036 * FW / FH, border=PINK)
    ax.text(gx + 0.030, 0.505, "white", ha="center", fontsize=7.5, color=INK2)
    ax.text(gx + 0.076, 0.505, "pink 1/(1+f)^α", ha="center", fontsize=7.5, color=PINK)
    # stage outputs
    s1 = load_rgb(member / "stage1" / "init-img-0.png")
    thumb(fig, s1, gx + 0.115, 0.50, 0.082, 0.082 * FW / FH * (s1.shape[0] / s1.shape[1]))
    ax.text(gx + 0.156, 0.475, "stage 1: depth-cond. views", ha="center", fontsize=7.5, color=MUTED)
    uv = load_rgb(member / "stage2" / "albedo.png")
    thumb(fig, uv, gx + 0.062, 0.20, 0.075, 0.075 * FW / FH)
    ax.text(gx + 0.0995, 0.175, "stage 2: UV inpainted texture", ha="center", fontsize=7.5, color=MUTED)

    # ---- output member ------------------------------------------------------
    ox = 0.83
    frame = fg_crop(load_rgb(member / "eval_views" / "frame_000.png"))
    thumb(fig, crop_square(frame), ox, 0.34, 0.085, 0.085 * FW / FH, border=INK2)
    ax.text(ox + 0.0425, 0.75, "textured crowd\nmember", ha="center", fontsize=10, color=INK,
            weight="bold")
    ax.text(ox + 0.0425, 0.285, "engine-ready mesh + 2K texture", ha="center", fontsize=7.5,
            color=MUTED)
    ax.text(ox + 0.0425, 0.20, "× N members → crowd\n↓\n7 diversity metrics + CLIP coherence",
            ha="center", fontsize=8.5, color=INK2)

    # ---- arrows -------------------------------------------------------------
    arrow(ax, 0.097, 0.59, 0.123, 0.59)                     # theme -> prompt
    arrow(ax, px + pw, 0.59, mx - 0.003, 0.59)              # prompt -> mesh (desc as CLIP anchor)
    ax.text((px + pw + mx) / 2, 0.635, "CLIP\nanchor", ha="center", va="bottom",
            fontsize=7.5, color=GREEN)
    arrow(ax, mx + mw, 0.59, gx - 0.003, 0.59)              # mesh -> paint3d
    ax.text((mx + mw + gx) / 2, 0.635, "outfit\nmesh", ha="center", va="bottom",
            fontsize=7.5, color=INK2)
    # prompt text also feeds Paint3D: elbow routed through the empty bottom margin
    x_a, x_b, y_low = px + pw * 0.5, gx + gw * 0.5, 0.085
    ax.plot([x_a, x_a, x_b], [band_y - 0.005, y_low, y_low], color=GREEN, lw=1.6,
            solid_joinstyle="miter", zorder=4)
    arrow(ax, x_b, y_low, x_b, band_y - 0.008, color=GREEN)
    ax.text((x_a + x_b) / 2, y_low + 0.022, "prompt text", ha="center",
            fontsize=7.5, color=GREEN)
    arrow(ax, gx + gw, 0.50, ox - 0.004, 0.50)              # paint3d -> member

    fig.savefig(OUT / "design_pipeline.png", dpi=220, facecolor=SURFACE)
    fig.savefig(OUT / "design_pipeline.svg", facecolor=SURFACE)
    print("written:", OUT / "design_pipeline.png", "and .svg")


if __name__ == "__main__":
    main()
