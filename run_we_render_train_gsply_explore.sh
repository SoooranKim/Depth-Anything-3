#!/usr/bin/env bash
set -euo pipefail

# Render TRAIN gs_ply using EXPLORE COLMAP poses (wild-explore).
#
# This renders the *already exported* DA3 gaussian PLY from train:
#   <EXPORT_ROOT>/<scene>_train/gs_ply/0000.ply
# using camera poses from:
#   <dataset>/<scene>/colmap/sparse/0/model_<explore1|explore2>/
#
# Usage:
#   ./run_we_render_train_gsply_explore.sh explore1 bicycle
#   ./run_we_render_train_gsply_explore.sh explore2 bicycle
#   SCENES="bicycle hat" ./run_we_render_train_gsply_explore.sh explore1
#
# Optional env:
#   DATASET_ROOT="/media/ssd1/users/sooran/dataset/wild-explore"
#   EXPORT_ROOT="/media/ssd1/users/sooran/Depth-Anything-3/output/we-colmap-gs"
#   SPARSE_SUBDIR="0"
#   DEVICE="cuda"
#   CHUNK_SIZE="4"
#   VIDEO_QUALITY="high"

split="${1:-}"
shift || true

if [[ -z "$split" || "$split" == "-h" || "$split" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run_we_render_train_gsply_explore.sh <explore1|explore2> [scene1 scene2 ...]
  SCENES="bicycle hat" ./run_we_render_train_gsply_explore.sh explore1
EOF
  exit 0
fi

case "$split" in
  explore1|explore2) ;;
  *)
    echo "[ERROR] split must be explore1 or explore2 (got: $split)" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dataset_root="${DATASET_ROOT:-/media/ssd1/users/sooran/dataset/wild-explore}"
export_root="${EXPORT_ROOT:-$repo_root/output/we-colmap-gs}"
sparse_subdir="${SPARSE_SUBDIR:-0}"
device="${DEVICE:-cuda}"
chunk_size="${CHUNK_SIZE:-4}"
video_quality="${VIDEO_QUALITY:-high}"

declare -a scenes=()
if [[ $# -gt 0 ]]; then
  scenes=("$@")
elif [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=(${SCENES})
else
  scenes=(bicycle hat meetingroom2 print starbucks2 study)
fi

for scene in "${scenes[@]}"; do
  gsply="$export_root/${scene}_train/gs_ply/0000.ply"
  colmap_dir="$dataset_root/$scene/colmap/sparse/0/model_${split}"
  out_dir="$export_root/${scene}_train/gs_video_${split}"
  pose_norm_json="$export_root/${scene}_train/pose_norm.json"

  if [[ ! -f "$gsply" ]]; then
    echo "[ERROR] missing train gsply: $gsply" >&2
    exit 1
  fi
  if [[ ! -d "$colmap_dir/sparse/$sparse_subdir" ]]; then
    echo "[ERROR] missing COLMAP sparse: $colmap_dir/sparse/$sparse_subdir" >&2
    exit 1
  fi

  echo "[RUN] scene=$scene render_poses=$split gsply=$gsply -> $out_dir"
  python3 "$repo_root/tools/render_gsply_with_colmap_poses.py" \
    --gsply "$gsply" \
    --colmap-dir "$colmap_dir" \
    --sparse-subdir "$sparse_subdir" \
    --out-dir "$out_dir" \
    --output-name "${scene}_train__${split}" \
    --chunk-size "$chunk_size" \
    --pose-norm-json "$pose_norm_json" \
    --trj-mode original \
    --vis-depth none \
    --video-quality "$video_quality" \
    --device "$device"
done

