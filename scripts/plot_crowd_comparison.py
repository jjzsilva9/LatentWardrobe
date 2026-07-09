"""Per-metric comparison charts for crowd-generation methods.

Reads metrics.json (written by evaluate_crowd.py) from each <batch>/<method>/<prompt>/
dir and writes one figure per metric: horizontal bars, one row per method, so method
identity is carried by the row label (no legend color-matching needed). Multiple
prompts stack as small multiples sharing the x-axis.

Color encodes the factorial's *noise factor*, not method identity (which the row
labels already carry): gray = random-mesh baseline, blue = white noise,
pink = pink noise. The blue/pink pair was checked with the dataviz palette
validator (worst CVD pair ΔE 41.3 under protanopia, target >= 12; the pink sits at
2.62:1 contrast on the surface, which obligates the visible value labels every bar
carries). Text always uses ink tokens, never the series color.

Usage:
    python scripts/plot_crowd_comparison.py --batch-dir outputs/weekend_20260703 \
        --methods baseline clip_tau002 [--out-dir outputs/weekend_20260703/plots]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.path import Path as MplPath

# --- chart chrome tokens (light surface; values from the dataviz reference palette)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"           # titles, value labels
INK_SECONDARY = "#52514e"  # subtitle, row labels, whiskers
INK_MUTED = "#898781"      # axis tick labels
GRID = "#e1e0d9"           # hairline gridlines, solid
AXIS = "#c3c2b7"           # baseline rule

# Color follows the noise factor (the factorial's manipulated variable), fixed per
# method -- never reassigned when a subset of methods is plotted.
METHOD_GROUPS = {
    "1_random_white_plain": "baseline",
    "2_clip_white_plain": "white",
    "3_clip_pink_plain": "pink",
    "4_clip_white_color": "white",
    "5_clip_pink_color": "pink",
    "6_clip_white_llm": "white",
    "7_clip_pink_llm": "pink",
    "8_clip_white_llm6": "white",
    "9_clip_pink_llm6": "pink",
    # weekend_20260703 legacy pair: two unrelated methods, two categorical slots
    "baseline": "legacy_baseline",
    "clip_tau002": "legacy_clip",
}
GROUP_COLORS = {
    "baseline": "#898781",       # de-emphasis gray: context, not a condition
    "white": "#2a78d6",          # categorical slot 1 (blue)
    "pink": "#e87ba4",           # categorical slot 7 (magenta) -- mnemonic for pink noise
    "legacy_baseline": "#2a78d6",
    "legacy_clip": "#eb6834",    # categorical slot 8 (orange)
}
# Groups that get a legend entry (the noise-factor encoding needs explaining;
# legacy methods are fully named by their row labels).
GROUP_LEGEND = {
    "baseline": "random-mesh baseline",
    "white": "white noise",
    "pink": "pink noise",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "clip_tau002": "CLIP τ=0.02",
    "1_random_white_plain": "1: random mesh, white noise",
    "2_clip_white_plain": "2: CLIP mesh, white noise",
    "3_clip_pink_plain": "3: CLIP mesh, pink noise",
    "4_clip_white_color": "4: CLIP mesh, white noise, colour prompt",
    "5_clip_pink_color": "5: CLIP mesh, pink noise, colour prompt",
    "6_clip_white_llm": "6: CLIP mesh, white noise, WANDER prompt",
    "7_clip_pink_llm": "7: CLIP mesh, pink noise, WANDER prompt",
    "8_clip_white_llm6": "8: CLIP mesh, white noise, WANDER prompt (6-axis)",
    "9_clip_pink_llm6": "9: CLIP mesh, pink noise, WANDER prompt (6-axis)",
}

METRICS = [
    # (dotted_key, slug, title, subtitle, std_key, log_scale)
    ("diversity.dinov2_patchwise_pairwise.mean", "dinov2_patchwise_mean", "DINOv2 patchwise distance (mean)", "higher = more diverse", "diversity.dinov2_patchwise_pairwise.std", False),
    ("diversity.dreamsim_pairwise.mean", "dreamsim_mean", "DreamSim pairwise distance (mean)", "higher = more diverse", "diversity.dreamsim_pairwise.std", False),
    ("diversity.lpips_pairwise.mean", "lpips_mean", "LPIPS pairwise (mean)", "higher = more diverse", "diversity.lpips_pairwise.std", False),
    ("diversity.pixel_l2_32_pairwise.mean", "pixel_l2_32_mean", "Low-res (32x32) pixel L2 pairwise (mean)", "higher = more diverse", "diversity.pixel_l2_32_pairwise.std", False),
    ("diversity.color_chi2_pairwise.mean", "color_chi2_mean", "Color χ² pairwise (mean)", "higher = more diverse", "diversity.color_chi2_pairwise.std", False),
    ("diversity.vendi_score_dinov2", "vendi_score_dinov2", "Vendi score (DINOv2)", "higher = more effective members", None, False),
    # DPP score (a kernel determinant) spans many orders of magnitude across
    # themes/methods -- a linear axis makes every bar but the largest read as
    # zero. Log-scale trades the usual "bar length = value" honesty for
    # legibility here, which is the standard convention for determinant-like
    # quantities; the printed value label on each bar keeps the exact number
    # readable regardless.
    ("diversity.dpp_score_dinov2", "dpp_score_dinov2", "DPP score (DINOv2)", "higher = more spread out; log scale", None, True),
    ("coherence.clipscore_crowd_mean", "clipscore_crowd_mean", "Theme coherence (CLIPScore, crowd mean)", "higher = more on-theme", "coherence.clipscore_crowd_std", False),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-dir", type=Path, required=True, help="e.g. outputs/weekend_20260703")
    p.add_argument("--methods", nargs="+", default=["baseline", "clip_tau002"])
    p.add_argument("--out-dir", type=Path, default=None, help="default: <batch-dir>/plots")
    return p.parse_args()


def get_path(d, dotted_key):
    val = d
    for part in dotted_key.split("."):
        if val is None:
            return None
        val = val[part]
    return val


def load_metrics(batch_dir, methods):
    """metrics[method][prompt_slug] = metrics dict; prompts = ordered slug list."""
    metrics, prompts = {}, []
    for method in methods:
        method_dir = batch_dir / method
        metrics[method] = {}
        if not method_dir.is_dir():
            continue
        for prompt_dir in sorted(method_dir.iterdir()):
            mfile = prompt_dir / "metrics.json"
            if not mfile.exists():
                continue
            metrics[method][prompt_dir.name] = json.loads(mfile.read_text())
            if prompt_dir.name not in prompts:
                prompts.append(prompt_dir.name)
    return metrics, prompts


def method_color(method, i):
    group = METHOD_GROUPS.get(method)
    if group in GROUP_COLORS:
        return GROUP_COLORS[group]
    # unknown method: walk the fixed categorical order, never generate a hue
    fallback = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
    return fallback[i % len(fallback)]


def fmt_value(v, scale_max):
    """Consistent decimal places within one chart, chosen from the chart's scale."""
    if scale_max < 1e-3:
        return f"{v:.2e}"
    if scale_max < 2:
        return f"{v:.3f}"
    if scale_max < 20:
        return f"{v:.2f}"
    return f"{v:.1f}"


def rounded_barh(ax, y, width, height, color, radius_px=4.0):
    """Horizontal bar with a rounded data-end and a square baseline end.

    Draws a FancyBboxPatch extended past x=0 by the corner radius, then clips it
    back to [0, width] so only the data-end corners survive rounding. Radii are
    converted from pixels to data units per axis (mutation_aspect squares up the
    y-radius), so corners render circular regardless of the axes' aspect."""
    if not np.isfinite(width) or width <= 0:
        return
    # pixels-per-data-unit on each axis (xlim/ylim must be final before calling)
    origin = ax.transData.transform((0.0, 0.0))
    unit = ax.transData.transform((1.0, 1.0))
    sx = radius_px / abs(unit[0] - origin[0])   # radius in x data units
    sy = radius_px / abs(unit[1] - origin[1])   # radius in y data units
    r = min(sx, width / 2.0)
    y0, y1 = y - height / 2.0, y + height / 2.0
    if r <= 0:
        ax.barh(y, width, height=height, color=color, zorder=3)
        return
    patch = FancyBboxPatch(
        (-r, y0), width + r, height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=sy / sx,
        facecolor=color, linewidth=0, zorder=3, transform=ax.transData)
    clip = MplPath([(0, y0), (width, y0), (width, y1), (0, y1), (0, y0)])
    patch.set_clip_path(clip, ax.transData)
    ax.add_patch(patch)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0, labelsize=8, labelcolor=INK_MUTED)


def plot_metric(metrics, prompts, methods, dotted_key, title, subtitle, out_path, std_key=None, log_scale=False):
    n = len(methods)
    row_h = 0.34                                   # inches per bar row: thin marks, generous air
    facet_h = row_h * n + (0.34 if len(prompts) > 1 else 0.12)
    header_h = 0.86
    fig_h = header_h + facet_h * len(prompts) + 0.42
    fig, axes = plt.subplots(len(prompts), 1, figsize=(9.2, fig_h), sharex=True, squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)

    # gather all values first so every facet shares one honest x-scale
    vals = {(m, p): get_path(metrics.get(m, {}).get(p), dotted_key) for m in methods for p in prompts}
    stds = {(m, p): (get_path(metrics.get(m, {}).get(p), std_key) or 0.0) if std_key and metrics.get(m, {}).get(p) else 0.0
            for m in methods for p in prompts}
    finite = [v for v in vals.values() if v is not None and np.isfinite(v)]
    if not finite:
        plt.close(fig)
        return
    scale_max = max(abs(v) + s for v, s in ((vals[k], stds[k]) for k in vals if vals[k] is not None and np.isfinite(vals[k])))

    ys = np.arange(n)[::-1]  # first method on top
    for ax, prompt in zip(axes, prompts):
        style_axes(ax)
        if log_scale:
            positive = [v for v in finite if v > 0]
            ax.set_xscale("log")
            ax.set_xlim(min(positive) / 5 if positive else 1e-3, max(positive) * 1.6 if positive else 1)
        else:
            ax.set_xlim(0, scale_max * 1.22)  # headroom for whisker + end label
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9, color=INK_SECONDARY)
        if not log_scale:
            ax.axvline(0, color=AXIS, linewidth=1, zorder=4)
        if len(prompts) > 1:
            ax.set_title(prompt.replace("_", " "), loc="left", fontsize=9, color=INK_SECONDARY, pad=4)

        for i, (m, y) in enumerate(zip(methods, ys)):
            v, s = vals[(m, prompt)], stds[(m, prompt)]
            if v is None or not np.isfinite(v):
                continue
            color = method_color(m, i)
            if log_scale:
                ax.barh(y, v, height=0.45, color=color, zorder=3)
            else:
                rounded_barh(ax, y, v, 0.45, color)
                if s:
                    ax.errorbar(v, y, xerr=s, fmt="none", ecolor=INK_SECONDARY,
                                elinewidth=1, capsize=2, capthick=1, zorder=4)
            # value at the bar tip (past the whisker), in ink -- never the series color
            pad = scale_max * 0.015 if not log_scale else v * 0.12
            ax.annotate(fmt_value(v, scale_max), (v + s + pad, y), va="center", ha="left",
                        fontsize=8, color=INK, zorder=5)

    # header: left-aligned title + subtitle in ink tokens
    fig.text(0.012, 1 - 0.28 / fig_h, title, fontsize=12, fontweight="semibold", color=INK, va="top")
    fig.text(0.012, 1 - 0.53 / fig_h, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")

    # legend: only the noise-factor groups actually plotted, and only if >1
    groups = [METHOD_GROUPS.get(m) for m in methods]
    legend_groups = [g for g in dict.fromkeys(groups) if g in GROUP_LEGEND]
    if len(legend_groups) > 1:
        handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_LEGEND[g]) for g in legend_groups]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1 - 0.12 / fig_h),
                   ncol=len(handles), frameon=False, fontsize=8.5, handlelength=1.1,
                   handleheight=1.1, labelcolor=INK_SECONDARY)

    fig.subplots_adjust(top=1 - header_h / fig_h, bottom=0.42 / fig_h, left=0.335, right=0.985)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.batch_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, prompts = load_metrics(args.batch_dir, args.methods)
    if not prompts:
        raise SystemExit(f"no metrics.json found under {args.batch_dir} for methods {args.methods}")

    for dotted_key, slug, title, subtitle, std_key, log_scale in METRICS:
        out_path = out_dir / f"{slug}.png"
        plot_metric(metrics, prompts, args.methods, dotted_key, title, subtitle, out_path,
                   std_key=std_key, log_scale=log_scale)
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
