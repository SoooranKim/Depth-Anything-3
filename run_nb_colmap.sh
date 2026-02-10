#!/usr/bin/env bash
set -euo pipefail

# Nerfbusters dataset runner (COLMAP mode) for DA3 -> exports (default: 3DGS PLY)
#
# This script runs pose-conditioned inference using the dataset's COLMAP reconstruction,
# using a selected image split:
# - train (default): uses <scene>/train images, linking them to COLMAP-friendly names.
#          By default, we link each _train_1_XXXXX.png as:
#            - frame_1_XXXXX.png
#          This avoids accidentally using both "frame_" and "frame_1_" pose tracks at once
#          (which would apply two different camera poses to the same underlying image).
#          Set TRAIN_LINK_BOTH=1 to also link frame_XXXXX.png (not recommended unless you know you need it).
#          Sparse model is taken from <scene>/colmap/sparse/<SPARSE_SUBDIR>.
# - eval:  uses <scene>/eval with <scene>/colmap/sparse/0 (renames _eval_XXXXX.png -> frame_XXXXX.png)
# - auto: prefer train; otherwise fall back to eval split
#
# Implementation detail:
# - DA3 expects a COLMAP bundle directory with:
#     <bundle>/images/    (actual image files)
#     <bundle>/sparse/0/  (COLMAP model files)
# - Nerfbusters stores COLMAP model under <scene>/colmap/sparse/0 without an images/ dir.
# - We therefore create a temporary bundle per scene:
#     images/  -> symlinks to <scene>/train/*
#     sparse/  -> symlink to <scene>/colmap/sparse
#   ColmapHandler will automatically ignore any COLMAP entries whose image file is absent,
#   so putting only train images in images/ effectively filters the used views.
#
# Usage:
#   ./run_nb_colmap.sh
#   ./run_nb_colmap.sh auto
#   ./run_nb_colmap.sh eval aloe
#   ./run_nb_colmap.sh train flowers
#   ./run_nb_colmap.sh aloe car
#   SCENES="aloe car" ./run_nb_colmap.sh auto
#   ./run_nb_colmap.sh --help
#
# Optional env vars:
#   NB_MODE="train|auto|eval"          # default: train
#   TRAIN_LINK_BOTH="0|1"              # default: 0 (train mode only)
#   DATASET_ROOT="/abs/path"           # default: /media/ssd1/users/sooran/dataset/nerfbusters-dataset
#   EXPORT_ROOT="/abs/path"            # default: ./output/nb-colmap-gs
#   MODEL_DIR="depth-anything/..."     # default: depth-anything/DA3NESTED-GIANT-LARGE-1.1
#   EXPORT_FORMAT="gs_ply"             # e.g., "glb", "mini_npz-glb", "gs_ply"
#   PROCESS_RES="504"                  # DA3 processing resolution (smaller => less VRAM)
#   PROCESS_RES_METHOD="upper_bound_resize"  # DA3 resize method
#   MAX_IMAGES="230"                   # max COLMAP views to process (<=0 disables capping)
#   SPARSE_SUBDIR="0"                  # default: 0 (maps to sparse/0)
#   CONF_THRESH_PERCENTILE="0"         # only relevant for GLB exports
#   POSE_NORM_MODE="recenter_scale"    # recenter_scale | scale | none (default: recenter_scale)
#   GS_VIDEO_USE_INPUT_NORM_POSES="0|1" # default: 1. If 1, render gs_video using normalized input poses (ex_t_norm)
#   USE_BACKEND="0|1"                  # default: 0
#   BACKEND_URL="http://localhost:8008"
#   AUTO_CLEANUP="0|1"                 # default: 1

usage() {
  cat <<'EOF'
Nerfbusters DA3 COLMAP runner
Supports: auto/train/eval image sources

Usage:
  ./run_nb_colmap.sh [auto|train|eval] [scene1 scene2 ...]
  ./run_nb_colmap.sh aloe car
  SCENES="aloe car" ./run_nb_colmap.sh auto

Optional env vars:
  NB_MODE="train|auto|eval"          # default: train
  TRAIN_LINK_BOTH="0|1"              # default: 0 (train mode only)
  DATASET_ROOT="/abs/path"           # default: /media/ssd1/users/sooran/dataset/nerfbusters-dataset
  EXPORT_ROOT="/abs/path"            # default: ./output/nb-colmap-gs
  MODEL_DIR="depth-anything/..."     # default: depth-anything/DA3NESTED-GIANT-LARGE-1.1
  EXPORT_FORMAT="gs_ply"             # e.g., "glb", "mini_npz-glb", "gs_ply"
  PROCESS_RES="504"                  # DA3 processing resolution (smaller => less VRAM)
  PROCESS_RES_METHOD="upper_bound_resize"  # DA3 resize method
  MAX_IMAGES="230"                   # max COLMAP views to process (<=0 disables capping)
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

nb_mode="${NB_MODE:-train}" # train | auto | eval
case "$nb_mode" in
  auto|train|eval) ;;
  *)
    echo "[ERROR] invalid NB_MODE: '$nb_mode' (expected: auto|train|eval)" >&2
    exit 2
    ;;
esac

# Optional first arg can override NB_MODE
if [[ "${1:-}" == "auto" || "${1:-}" == "train" || "${1:-}" == "eval" ]]; then
  nb_mode="$1"
  shift
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dataset_root="${DATASET_ROOT:-/media/ssd1/users/sooran/dataset/nerfbusters-dataset}"
export_root="${EXPORT_ROOT:-"$repo_root/output/nb-colmap-gs"}"
model_dir="${MODEL_DIR:-"depth-anything/DA3NESTED-GIANT-LARGE-1.1"}"
export_format="${EXPORT_FORMAT:-gs_ply}"
process_res="${PROCESS_RES:-504}"
process_res_method="${PROCESS_RES_METHOD:-upper_bound_resize}"
max_images="${MAX_IMAGES:-230}"
sparse_subdir="${SPARSE_SUBDIR:-0}"
conf_percentile="${CONF_THRESH_PERCENTILE:-0}"
pose_norm_mode="${POSE_NORM_MODE:-recenter_scale}"
gs_video_use_input_norm_poses="${GS_VIDEO_USE_INPUT_NORM_POSES:-1}"
use_backend="${USE_BACKEND:-0}"
backend_url="${BACKEND_URL:-http://localhost:8008}"
auto_cleanup="${AUTO_CLEANUP:-1}"
train_link_both="${TRAIN_LINK_BOTH:-0}"

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
    aloe
    art
    car
    century
    flowers
    garbage
    picnic
    pipe
    plant
    roses
    table
  )
fi

tmp_root="$(mktemp -d -t da3_nb_colmap_XXXXXX)"
cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

for scene in "${scenes[@]}"; do
  scene_root="$dataset_root/$scene"
  export_dir_base="$export_root/$scene"

  if [[ ! -d "$scene_root" ]]; then
    echo "[WARN] missing scene dir, skipping: $scene_root" >&2
    continue
  fi

  # Decide which split sources to use.
  # train: <scene>/train + <scene>/colmap/sparse/<SPARSE_SUBDIR> (renaming links)
  # eval:  <scene>/eval + <scene>/colmap/sparse/<SPARSE_SUBDIR> (renaming links)
  selected_mode=""
  images_source=""
  sparse_source=""
  image_rename_mode="" # "none" | "eval_to_frame" | "train_to_frame"

  train_dir="$scene_root/train"
  eval_dir="$scene_root/eval"
  colmap_sparse_dir="$scene_root/colmap/sparse/$sparse_subdir"

  if [[ "$nb_mode" == "train" || "$nb_mode" == "auto" ]]; then
    if [[ -d "$train_dir" && -d "$colmap_sparse_dir" ]]; then
      selected_mode="train"
      images_source="$train_dir"
      sparse_source="$colmap_sparse_dir"
      image_rename_mode="train_to_frame"
    elif [[ "$nb_mode" == "train" ]]; then
      echo "[ERROR] scene '$scene' does not have train COLMAP sources:" >&2
      echo "        missing: $train_dir and/or $colmap_sparse_dir" >&2
      echo "        Hint: try eval mode (uses <scene>/eval + <scene>/colmap) or run_nb.sh (auto poses)." >&2
      exit 1
    fi
  fi

  if [[ -z "$selected_mode" ]]; then
    # eval mode (or auto fallback)
    if [[ -d "$eval_dir" && -d "$colmap_sparse_dir" ]]; then
      selected_mode="eval"
      images_source="$eval_dir"
      sparse_source="$colmap_sparse_dir"
      image_rename_mode="eval_to_frame"
    else
      echo "[ERROR] scene '$scene' does not have eval COLMAP sources:" >&2
      echo "        missing: $eval_dir and/or $colmap_sparse_dir" >&2
      exit 1
    fi
  fi

  export_dir="${export_dir_base}__${selected_mode}"
  mkdir -p "$export_dir"

  # Build a temporary COLMAP bundle directory with required structure.
  bundle_dir="$tmp_root/${scene}_bundle"
  rm -rf "$bundle_dir"
  mkdir -p "$bundle_dir/images"

  # Provide sparse/<SPARSE_SUBDIR>/ by linking the chosen sparse source dir.
  mkdir -p "$bundle_dir/sparse"
  ln -s "$sparse_source" "$bundle_dir/sparse/$sparse_subdir"

  # Provide images/ with filenames matching the COLMAP model's image names.
  # - eval_to_frame: eval uses _eval_XXXXX.png -> link as frame_XXXXX.png.
  # - train_to_frame:train uses _train_*_XXXXX.png -> link as frame_1_XXXXX.png by default
  shopt -s nullglob
  linked_count=0
  if [[ "$image_rename_mode" == "eval_to_frame" ]]; then
    for img in "$images_source"/_eval_*.{png,jpg,jpeg,PNG,JPG,JPEG,webp,WEBP}; do
      bn="$(basename "$img")"           # _eval_00257.png
      num="${bn#_eval_}"                # 00257.png
      num="${num%.*}"                   # 00257
      ln -sf "$img" "$bundle_dir/images/frame_${num}.png"
      linked_count=$((linked_count + 1))
    done
  else
    # train_to_frame
    for img in "$images_source"/_train_*_*.{png,jpg,jpeg,PNG,JPG,JPEG,webp,WEBP}; do
      bn="$(basename "$img")"           # _train_1_00001.png
      base="${bn%.*}"                   # _train_1_00001
      num="${base##*_}"                 # 00001
      if [[ "$num" =~ ^[0-9]+$ ]]; then
        ln -sf "$img" "$bundle_dir/images/frame_1_${num}.png"
        if [[ "$train_link_both" == "1" ]]; then
          ln -sf "$img" "$bundle_dir/images/frame_${num}.png"
        fi
        linked_count=$((linked_count + 1))
      fi
    done
  fi
  shopt -u nullglob

  if [[ "$linked_count" -le 0 ]]; then
    echo "[ERROR] scene=$scene mode=$selected_mode: no images linked from $images_source" >&2
    exit 1
  fi

  args=(
    colmap "$bundle_dir"
    --sparse-subdir "$sparse_subdir"
    --model-dir "$model_dir"
    --export-dir "$export_dir"
    --export-format "$export_format"
    --process-res "$process_res"
    --process-res-method "$process_res_method"
    --max-images "$max_images"
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

  echo "[RUN] scene=$scene mode=$selected_mode images=$images_source sparse=$sparse_source -> $export_dir"
  da3 "${args[@]}"
  echo "[OK] $scene -> $export_dir/$export_format/"
done

