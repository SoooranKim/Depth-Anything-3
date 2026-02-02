#!/usr/bin/env bash
set -euo pipefail

# Wild-Explore dataset runner (COLMAP mode) for DA3 -> exports (default: 3DGS PLY)
#
# This script runs pose-conditioned inference using the dataset's COLMAP reconstruction.
# It uses the per-scene COLMAP bundle at:
#   <scene>/colmap/sparse/0/model_{train,explore1,explore2}/
# which already contains:
#   images/
#   sparse/0/{cameras,images,points3D}.{bin|txt}
#
# Usage:
#   ./run_we_colmap.sh
#   ./run_we_colmap.sh train
#   ./run_we_colmap.sh explore1
#   ./run_we_colmap.sh explore2
#   ./run_we_colmap.sh --help
#
# Optional env vars:
#   EXPORT_ROOT="/abs/path"            # default: ./output/we-colmap-gs
#   MODEL_DIR="depth-anything/..."     # default: depth-anything/DA3NESTED-GIANT-LARGE-1.1
#   EXPORT_FORMAT="gs_ply"             # e.g., "glb", "mini_npz-glb", "gs_ply"
#   SPARSE_SUBDIR="0"                  # default: 0 (maps to sparse/0)
#   CONF_THRESH_PERCENTILE="0"         # only relevant for GLB exports
#   POSE_NORM_MODE="recenter_scale"    # recenter_scale | scale | none (default: recenter_scale)
#   GS_VIDEO_USE_INPUT_NORM_POSES="0|1" # default: 1. If 1, render gs_video using normalized input poses (ex_t_norm)
#   USE_BACKEND="0|1"                  # default: 0
#   BACKEND_URL="http://localhost:8008"
#   AUTO_CLEANUP="0|1"                 # default: 1

usage() {
  cat <<'EOF'
Wild-Explore DA3 COLMAP runner

Usage:
  ./run_we_colmap.sh [train|explore1|explore2] [scene1 scene2 ...]
  SCENES="bicycle hat" ./run_we_colmap.sh train

Optional env vars:
  EXPORT_ROOT="/abs/path"            # default: ./output/we-colmap-gs
  MODEL_DIR="depth-anything/..."     # default: depth-anything/DA3NESTED-GIANT-LARGE-1.1
  EXPORT_FORMAT="gs_ply"             # e.g., "glb", "mini_npz-glb", "gs_ply"
  SPARSE_SUBDIR="0"                  # default: 0 (maps to sparse/0)
  CONF_THRESH_PERCENTILE="0"         # only relevant for GLB exports
  POSE_NORM_MODE="recenter_scale"    # recenter_scale | scale | none (default: recenter_scale)
  GS_VIDEO_USE_INPUT_NORM_POSES="0|1" # default: 1. If 1, render gs_video using normalized input poses (ex_t_norm)
  USE_BACKEND="0|1"                  # default: 0
  BACKEND_URL="http://localhost:8008"
  AUTO_CLEANUP="0|1"                 # default: 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

we_split="${1:-train}" # train | explore1 | explore2
shift $(( $# > 0 ? 1 : 0 ))

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_root="/media/ssd1/users/sooran/dataset/wild-explore"
export_root="${EXPORT_ROOT:-"$repo_root/output/we-colmap-gs"}"
model_dir="${MODEL_DIR:-"depth-anything/DA3NESTED-GIANT-LARGE-1.1"}"
export_format="${EXPORT_FORMAT:-gs_ply}"
sparse_subdir="${SPARSE_SUBDIR:-0}"
conf_percentile="${CONF_THRESH_PERCENTILE:-0}"
pose_norm_mode="${POSE_NORM_MODE:-recenter_scale}"
gs_video_use_input_norm_poses="${GS_VIDEO_USE_INPUT_NORM_POSES:-1}"
use_backend="${USE_BACKEND:-0}"
backend_url="${BACKEND_URL:-http://localhost:8008}"
auto_cleanup="${AUTO_CLEANUP:-1}"

case "$we_split" in
  train|explore1|explore2) ;;
  *)
    echo "[ERROR] invalid split: '$we_split' (expected: train|explore1|explore2)" >&2
    exit 2
    ;;
esac

if [[ ! -d "$dataset_root" ]]; then
  echo "[ERROR] missing dataset root: $dataset_root" >&2
  exit 1
fi

mkdir -p "$export_root"

declare -a scenes=()
if [[ $# -gt 0 ]]; then
  scenes=("$@")
elif [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  scenes=(${SCENES})
else
  scenes=(
    bicycle
    hat
    meetingroom2
    print
    starbucks2
    study
  )
fi

for scene in "${scenes[@]}"; do
  scene_root="$dataset_root/$scene"
  colmap_bundle="$scene_root/colmap/sparse/0/model_${we_split}"
  export_dir="$export_root/${scene}_${we_split}"

  if [[ ! -d "$scene_root" ]]; then
    echo "[WARN] missing scene dir, skipping: $scene_root" >&2
    continue
  fi
  if [[ ! -d "$colmap_bundle/images" ]]; then
    echo "[ERROR] missing COLMAP images dir: $colmap_bundle/images" >&2
    exit 1
  fi
  if [[ ! -d "$colmap_bundle/sparse/$sparse_subdir" ]]; then
    echo "[ERROR] missing COLMAP sparse dir: $colmap_bundle/sparse/$sparse_subdir" >&2
    exit 1
  fi

  mkdir -p "$export_dir"

  args=(
    colmap "$colmap_bundle"
    --sparse-subdir "$sparse_subdir"
    --model-dir "$model_dir"
    --export-dir "$export_dir"
    --export-format "$export_format"
    --conf-thresh-percentile "$conf_percentile"
    --pose-norm-mode "$pose_norm_mode"
  )

  if [[ "$gs_video_use_input_norm_poses" == "1" ]]; then
    args+=(--gs-video-use-input-norm-poses)
  fi

  if [[ "$use_backend" == "1" ]]; then
    args+=(--use-backend --backend-url "$backend_url")
  fi
  if [[ "$auto_cleanup" == "1" ]]; then
    args+=(--auto-cleanup)
  fi

  echo "[RUN] scene=$scene split=$we_split colmap=$colmap_bundle -> $export_dir"
  da3 "${args[@]}"
  echo "[OK] $scene -> $export_dir/$export_format/"
done

