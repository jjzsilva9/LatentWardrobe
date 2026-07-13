#!/bin/bash
# Comparison plots for the multitheme batch -- thin wrapper around
# plot_crowd_comparison.py (generic, takes --batch-dir/--methods) with this
# batch's 7 cells, mirroring run_multitheme_eval.sh's CELLS list. Run after
# run_multitheme_eval.sh has produced metrics.json for the cells you want
# plotted (missing cells are skipped by load_metrics, not fatal).
#
# One figure per metric under $OUT/plots/, each with a row per cell/method
# and one facet per prompt (superheroes, business_formalwear) -- so both
# comparisons the loop needs (across methods, across prompts) are in every
# PNG at once.
#
# Launch (visible window, from Windows PowerShell):
#   Start-Process wsl.exe -ArgumentList '-u root bash -c "mkdir -p /mnt/u; mountpoint -q /mnt/u || mount -t drvfs U: /mnt/u; bash \"/mnt/u/My Documents/LatentWardrobe/scripts/run_multitheme_plot.sh\""'
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eval
cd "/mnt/u/My Documents/LatentWardrobe"

OUT=outputs/multitheme_20260710
FACTORIAL7=outputs/factorial7_20260706
CELLS=(1_random_white_plain 2_clip_white_plain 3_clip_pink_plain 4_clip_white_color \
       5_clip_pink_color 8_clip_white_llm6 9_clip_pink_llm6)

python -u scripts/plot_crowd_comparison.py --batch-dir "$OUT" --methods "${CELLS[@]}" \
  --out-dir "$OUT/plots" 2>&1 | tee -a "$OUT/plot.log"
rc=${PIPESTATUS[0]}
echo "[run_multitheme_plot] plot_exit=$rc" | tee -a "$OUT/plot.log"

# Same 7 cells, but adding medieval_villagers (factorial7_20260706, cells 6/7
# excluded there too) as a third theme alongside superheroes/business_formalwear.
python -u scripts/plot_crowd_comparison.py --batch-dir "$OUT" --extra-batch-dir "$FACTORIAL7" \
  --methods "${CELLS[@]}" --out-dir "$OUT/plots_3themes" 2>&1 | tee -a "$OUT/plot.log"
rc=${PIPESTATUS[0]}
echo "[run_multitheme_plot] plot_3themes_exit=$rc" | tee -a "$OUT/plot.log"
