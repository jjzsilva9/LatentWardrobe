"""Grouped bar charts comparing crowd-generation methods across prompts.

Reads metrics.json (written by evaluate_crowd.py) from each <batch>/<method>/<prompt>/
dir and plots one grouped-bar figure per metric: prompts on the x-axis, one bar
per method, fixed method->color mapping so colors mean the same thing across
figures.

Usage:
    python scripts/plot_crowd_comparison.py --batch-dir outputs/weekend_20260703 \
        --methods baseline clip_tau002 [--out-dir outputs/weekend_20260703/plots]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed method -> color mapping (categorical, colorblind-safe pair, consistent order).
METHOD_COLORS = {
    "baseline": "#4C72B0",
    "clip_tau002": "#DD8452",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "clip_tau002": "CLIP τ=0.02",
}

METRICS = [
    ("diversity.lpips_pairwise.mean", "lpips_mean", "LPIPS pairwise (mean)", "higher = more diverse"),
    ("diversity.clip_distance_pairwise.mean", "clip_distance_mean", "CLIP distance pairwise (mean)", "higher = more diverse"),
    ("diversity.color_chi2_pairwise.mean", "color_chi2_mean", "Color χ² pairwise (mean)", "higher = more diverse"),
    ("diversity.vendi_score_clip", "vendi_score_clip", "Vendi score (CLIP)", "higher = more effective members"),
    ("coherence.clipscore_crowd_mean", "clipscore_crowd_mean", "Theme coherence (CLIPScore, crowd mean)", "higher = more on-theme"),
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


def plot_metric(metrics, prompts, methods, dotted_key, title, subtitle, out_path):
    n_methods = len(methods)
    x = np.arange(len(prompts))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(1.6 * len(prompts) + 3, 4.5))
    for i, method in enumerate(methods):
        vals = []
        for prompt in prompts:
            m = metrics.get(method, {}).get(prompt)
            vals.append(get_path(m, dotted_key) if m else np.nan)
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, f"C{i}"))
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.margins(y=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in prompts], rotation=15, ha="right")
    ax.set_ylabel(title)
    ax.set_title(f"{title}\n({subtitle})", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.batch_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, prompts = load_metrics(args.batch_dir, args.methods)
    if not prompts:
        raise SystemExit(f"no metrics.json found under {args.batch_dir} for methods {args.methods}")

    for dotted_key, slug, title, subtitle in METRICS:
        out_path = out_dir / f"{slug}.png"
        plot_metric(metrics, prompts, args.methods, dotted_key, title, subtitle, out_path)
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
