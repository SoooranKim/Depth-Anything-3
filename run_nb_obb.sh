#!/usr/bin/env bash
set -euo pipefail

# Nerfbusters OBB batch runner for already-exported GLBs.
#
# Looks for:
#   /media/ssd1/users/sooran/Depth-Anything-3/output/nb-0/<scene>/scene.glb
# and writes:
#   .../<scene>/obb.json
# Optionally also writes an overlay GLB:
#   .../<scene>/obb_<style>.glb
#
# Usage:
#   ./run_nb_obb.sh                 # expand ratio = 0.0, json only
#   ./run_nb_obb.sh 0.2             # expand extents by +20% (center stays fixed)
#   ./run_nb_obb.sh 0.2 overlay     # also export overlay GLB (wire by default)
#
# Notes:
# - `--camera-padding-radius` is a RATIO (e.g. 0.2 => extents *= 1.2).
# - In `overlay` mode, camera centers are included by default.
#   To override: set INCLUDE_CAMERA_CENTERS=0/1 explicitly.

expand_ratio="${1:-0.0}"
mode="${2:-json}"

root="/media/ssd1/users/sooran/Depth-Anything-3/output/nb-0"
style="${OBB_STYLE:-wire}"
alpha="${OBB_ALPHA:-80}"

if [[ ! -d "$root" ]]; then
  echo "[ERROR] missing root dir: $root" >&2
  exit 1
fi

shopt -s nullglob
glbs=("$root"/*/scene.glb)
shopt -u nullglob

if [[ ${#glbs[@]} -eq 0 ]]; then
  echo "[ERROR] no scene.glb found under: $root/*/scene.glb" >&2
  exit 1
fi

for glb in "${glbs[@]}"; do
  scene_dir="$(dirname "$glb")"
  scene="$(basename "$scene_dir")"

  out_json="$scene_dir/obb.json"
  out_glb=""
  if [[ "$mode" == "overlay" ]]; then
    out_glb="$scene_dir/obb_${style}.glb"
  fi

  args=(obb "$glb" --out-json "$out_json" --camera-padding-radius "$expand_ratio")

  include_cam_centers="${INCLUDE_CAMERA_CENTERS:-}"
  if [[ -z "$include_cam_centers" ]]; then
    if [[ "$mode" == "overlay" ]]; then
      include_cam_centers="1"
    else
      include_cam_centers="0"
    fi
  fi

  if [[ "$include_cam_centers" == "1" ]]; then
    args+=(--include-camera-centers)
  fi

  if [[ -n "$out_glb" ]]; then
    args+=(--out-glb "$out_glb" --obb-style "$style" --obb-alpha "$alpha")
  fi

  da3 "${args[@]}"
  echo "[OK] $scene -> $out_json${out_glb:+, $out_glb}"
done

