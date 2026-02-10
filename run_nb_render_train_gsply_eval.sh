#!/usr/bin/env bash
set -euo pipefail

# Render TRAIN gs_ply using EVAL poses (nerfbusters).
#
# This renders the *already exported* DA3 gaussian PLY from train:
#   <EXPORT_ROOT>/<scene>__train/gs_ply/0000.ply
# using eval camera poses from the dataset COLMAP model:
#   <DATASET_ROOT>/<scene>/colmap/sparse/<SPARSE_SUBDIR>/
#
# We select "eval" poses by filtering COLMAP image names via regex.
# By default this matches names like: frame_00001.png (NOT frame_1_00001.png).
#
# Usage:
#   ./run_nb_render_train_gsply_eval.sh
#   ./run_nb_render_train_gsply_eval.sh aloe car
#   SCENES="aloe car" ./run_nb_render_train_gsply_eval.sh
#
# Optional env:
#   DATASET_ROOT="/media/ssd1/users/sooran/dataset/nerfbusters-dataset"
#   EXPORT_ROOT="/media/ssd1/users/sooran/Depth-Anything-3/output/nb-colmap-gs"
#   SPARSE_SUBDIR="0"
#   DEVICE="cuda"
#   CHUNK_SIZE="4"
#   VIDEO_QUALITY="high"
#   IMAGE_NAME_REGEX="^frame_[0-9]+\\.(png|jpg|jpeg)$"
#   TRJ_MODE="original"   # original | smooth | ... (see tool help)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${DATASET_ROOT:-/media/ssd1/users/sooran/dataset/nerfbusters-dataset}"
export_root="${EXPORT_ROOT:-$repo_root/output/nb-colmap-gs}"
sparse_subdir="${SPARSE_SUBDIR:-0}"
device="${DEVICE:-cuda}"
chunk_size="${CHUNK_SIZE:-4}"
video_quality="${VIDEO_QUALITY:-high}"
image_name_regex="${IMAGE_NAME_REGEX:-^frame_[0-9]+\\.(png|jpg|jpeg)$}"
trj_mode="${TRJ_MODE:-original}"

if [[ ! -d "$dataset_root" ]]; then
  echo "[ERROR] missing dataset root: $dataset_root" >&2
  exit 1
fi
if [[ ! -d "$export_root" ]]; then
  echo "[ERROR] missing export root: $export_root" >&2
  exit 1
fi

declare -a scenes=()
if [[ $# -gt 0 ]]; then
  scenes=("$@")
elif [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=(${SCENES})
else
  scenes=(aloe art car century flowers garbage picnic pipe plant roses table)
fi

for scene in "${scenes[@]}"; do
  gsply="$export_root/${scene}__train/gs_ply/0000.ply"
  colmap_dir="$dataset_root/$scene/colmap"
  out_dir="$export_root/${scene}__train/gs_video_eval"
  pose_norm_json="$export_root/${scene}__train/pose_norm.json"

  if [[ ! -f "$gsply" ]]; then
    echo "[ERROR] missing train gsply: $gsply" >&2
    exit 1
  fi
  if [[ ! -d "$colmap_dir/sparse/$sparse_subdir" ]]; then
    echo "[ERROR] missing COLMAP sparse: $colmap_dir/sparse/$sparse_subdir" >&2
    exit 1
  fi

  echo "[RUN] scene=$scene render_poses=eval gsply=$gsply -> $out_dir"
  python3 "$repo_root/tools/render_gsply_with_eval_poses.py" \
    --gsply "$gsply" \
    --colmap-dir "$colmap_dir" \
    --sparse-subdir "$sparse_subdir" \
    --image-name-regex "$image_name_regex" \
    --out-dir "$out_dir" \
    --output-name "${scene}__train__eval" \
    --chunk-size "$chunk_size" \
    --pose-norm-json "$pose_norm_json" \
    --trj-mode "$trj_mode" \
    --vis-depth none \
    --video-quality "$video_quality" \
    --device "$device"
done

