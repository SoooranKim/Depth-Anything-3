#!/usr/bin/env bash
set -euo pipefail

# Nerfbusters dataset runner (AUTO mode on train images) for DA3 -> 3DGS PLY
#
# NOTE:
# - `da3 auto` estimates poses in its own coordinate system from ONLY the input images.
# - If you later want to render using dataset eval poses (COLMAP/transforms),
#   you generally need an additional alignment step (or use `da3 colmap` instead).
#
# Usage:
#   ./run_nb.sh                  # run all scenes, conf percentile = 0
#   ./run_nb.sh 30               # set --conf-thresh-percentile 30

conf_percentile="${1:-0}"

export_root="/media/ssd1/users/sooran/Depth-Anything-3/output/nb-0-gs"
model_dir="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
dataset_root="/media/ssd1/users/sooran/dataset/nerfbusters-dataset"
max_images=230

for scene in aloe art car century flowers garbage picnic pipe plant roses table; do
  scene_root="$dataset_root/$scene"
  train_dir="$scene_root/train"
  export_dir="$export_root/$scene"

  if [[ ! -d "$train_dir" ]]; then
    echo "[ERROR] missing train dir: $train_dir" >&2
    exit 1
  fi

  mkdir -p "$export_dir"
  da3 auto "$train_dir" \
    --conf-thresh-percentile "$conf_percentile" \
    --model-dir "$model_dir" \
    --export-dir "$export_dir" \
    --export-format gs_ply \
    --max-images "$max_images" \
    --auto-cleanup

  echo "[OK] $scene -> $export_dir/gs_ply/"
done

