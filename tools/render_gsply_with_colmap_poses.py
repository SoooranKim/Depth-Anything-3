#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData

from depth_anything_3.specs import Gaussians, Prediction
from depth_anything_3.utils.export.gs import export_to_gs_video
from depth_anything_3.utils.read_write_model import read_model


def _load_da3_gsply(path: str, device: str) -> Gaussians:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY missing 'vertex' element: {path}")
    v = ply["vertex"].data

    def col(name: str) -> np.ndarray:
        if name not in v.dtype.names:
            raise ValueError(f"PLY missing field '{name}': {path}")
        return np.asarray(v[name])

    means = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)  # (N, 3)

    # DA3 saves log(scales) as scale_*
    scales = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1).astype(np.float32)
    scales = np.exp(scales)

    # DA3 saves world_quat_wxyz as rot_0..rot_3
    rotations = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1).astype(
        np.float32
    )

    # DA3 saves inverse_sigmoid(opacity) by default
    opacities_logits = col("opacity").astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacities_logits))

    # SH DC only by default: f_dc_0..2
    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1).astype(np.float32)
    harmonics = f_dc[:, :, None]  # (N, 3, 1) => degree 0

    # Optional: f_rest_* (if present)
    f_rest_names = [n for n in (v.dtype.names or []) if n.startswith("f_rest_")]
    if f_rest_names:
        # Keep deterministic ordering: f_rest_0, f_rest_1, ...
        f_rest_names = sorted(f_rest_names, key=lambda s: int(s.split("_")[-1]))
        f_rest = np.stack([col(n) for n in f_rest_names], axis=1).astype(np.float32)  # (N, K)
        # f_rest is flattened over xyz, so reshape back to (N, 3, d_sh-1)
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected f_rest dims (not divisible by 3): {f_rest.shape}")
        d_sh_minus_1 = f_rest.shape[1] // 3
        f_rest = f_rest.reshape(-1, 3, d_sh_minus_1)
        harmonics = np.concatenate([harmonics, f_rest], axis=2)  # (N, 3, d_sh)

    # Add batch dim expected by renderer: (B=1, N, ...)
    to_t = lambda x: torch.from_numpy(x).to(device=device)
    return Gaussians(
        means=to_t(means)[None, ...],
        scales=to_t(scales)[None, ...],
        rotations=to_t(rotations)[None, ...],
        harmonics=to_t(harmonics)[None, ...],
        opacities=to_t(opacities)[None, ...],
    )


def _load_colmap_poses(colmap_dir: str, sparse_subdir: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    images_dir = os.path.join(colmap_dir, "images")
    sparse_dir = os.path.join(colmap_dir, "sparse", sparse_subdir) if sparse_subdir else os.path.join(colmap_dir, "sparse")

    cameras, images, _points3d = read_model(sparse_dir)
    if not images:
        raise ValueError(f"No images in COLMAP model: {sparse_dir}")

    # Deterministic order by filename
    extrinsics = []
    intrinsics = []
    hw: Tuple[int, int] | None = None

    for _image_id, image_data in sorted(images.items(), key=lambda kv: kv[1].name):
        cam = cameras[image_data.camera_id]

        # Camera size (assume consistent)
        if hw is None:
            hw = (int(cam.height), int(cam.width))

        R = image_data.qvec2rotmat()
        t = image_data.tvec
        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = R.astype(np.float32)
        extr[:3, 3] = t.astype(np.float32)
        extrinsics.append(extr)

        if cam.model == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx = fy = f
        else:
            # fallback approximation
            fx = fy = cam.params[0] if len(cam.params) > 0 else 1000.0
            cx = cam.width / 2.0
            cy = cam.height / 2.0

        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(intr)

    assert hw is not None
    return np.stack(extrinsics, axis=0), np.stack(intrinsics, axis=0), hw


def _scale_w2c_extrinsics(extrs_w2c: np.ndarray, pose_norm_scale: float) -> np.ndarray:
    """Scale-only normalization for COLMAP w2c extrinsics.

    We interpret scale normalization as: take camera centers in world space (c2w translation),
    divide by `pose_norm_scale`, then invert back to w2c.
    """
    if pose_norm_scale <= 0:
        raise ValueError(f"pose_norm_scale must be > 0, got {pose_norm_scale}")
    c2ws = np.linalg.inv(extrs_w2c)
    c2ws[:, :3, 3] = c2ws[:, :3, 3] / float(pose_norm_scale)
    return np.linalg.inv(c2ws).astype(np.float32)


def _recenter_and_scale_w2c_extrinsics(
    extrs_w2c: np.ndarray, pose_norm_transform_c2w: np.ndarray, pose_norm_scale: float
) -> np.ndarray:
    """Recenter+scale normalization matching DepthAnything3._normalize_extrinsics(mode='recenter_scale').

    - Recenter (and reorient) using the stored transform (c2w): ex' = ex @ transform
    - Scale by dividing camera-center translations (in c2w of ex') by pose_norm_scale
    """
    if pose_norm_transform_c2w.shape != (4, 4):
        raise ValueError(f"pose_norm_transform must be 4x4, got {pose_norm_transform_c2w.shape}")
    if pose_norm_scale <= 0:
        raise ValueError(f"pose_norm_scale must be > 0, got {pose_norm_scale}")

    ex_rec = (extrs_w2c @ pose_norm_transform_c2w[None, ...]).astype(np.float64)
    c2ws = np.linalg.inv(ex_rec)
    c2ws[:, :3, 3] = c2ws[:, :3, 3] / float(pose_norm_scale)
    return np.linalg.inv(c2ws).astype(np.float32)


def _try_load_pose_norm_scale(pose_norm_json: str) -> float | None:
    if not pose_norm_json:
        return None
    if not os.path.isfile(pose_norm_json):
        return None
    with open(pose_norm_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Primary key used by our pipeline
    if "pose_norm_scale" in data and data["pose_norm_scale"] is not None:
        return float(data["pose_norm_scale"])
    # Backward/alt keys (just in case)
    for k in ("median_dist", "scale", "pose_scale"):
        if k in data and data[k] is not None:
            return float(data[k])
    return None


def _try_load_pose_norm(pose_norm_json: str) -> tuple[str | None, np.ndarray | None, float | None]:
    """Load (mode, transform, scale) from pose_norm.json.

    Returns:
        (pose_norm_mode, pose_norm_transform_c2w_4x4, pose_norm_scale)
    """
    if not pose_norm_json:
        return None, None, None
    if not os.path.isfile(pose_norm_json):
        return None, None, None
    with open(pose_norm_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mode = data.get("pose_norm_mode", None)

    transform = data.get("pose_norm_transform", None)
    transform_np = None
    if transform is not None:
        transform_np = np.array(transform, dtype=np.float32)

    scale = None
    if "pose_norm_scale" in data and data["pose_norm_scale"] is not None:
        scale = float(data["pose_norm_scale"])
    else:
        # Backward/alt keys (just in case)
        for k in ("median_dist", "scale", "pose_scale"):
            if k in data and data[k] is not None:
                scale = float(data[k])
                break

    return mode, transform_np, scale


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a DA3-exported gs_ply PLY using COLMAP poses (explore1/2 etc.)"
    )
    p.add_argument("--gsply", required=True, help="Path to DA3 gs_ply/0000.ply")
    p.add_argument(
        "--colmap-dir",
        required=True,
        help="COLMAP bundle dir containing images/ and sparse/ (e.g., .../model_explore1)",
    )
    p.add_argument("--sparse-subdir", default="0", help="Sparse subdir (default: 0 for sparse/0)")
    p.add_argument("--out-dir", default="", help="Output directory (default: alongside gsply)")
    p.add_argument("--output-name", default="", help="Output mp4 base name (without .mp4)")
    p.add_argument("--chunk-size", type=int, default=4, help="Rendering chunk size (gsplat batch)")
    p.add_argument(
        "--pose-norm-json",
        default="",
        help=(
            "Optional path to pose_norm.json produced by DA3 export (contains pose_norm_scale). "
            "If provided (or auto-found), the loaded COLMAP poses will be normalized before rendering."
        ),
    )
    p.add_argument(
        "--pose-scale",
        type=float,
        default=0.0,
        help="Optional explicit pose scale normalization factor. Overrides --pose-norm-json if > 0.",
    )
    p.add_argument(
        "--trj-mode",
        default="original",
        choices=[
            "original",
            "smooth",
            "interpolate",
            "interpolate_smooth",
            "wander",
            "dolly_zoom",
            "extend",
            "wobble_inter",
        ],
        help="Trajectory mode; use 'original' to render exactly provided poses",
    )
    p.add_argument("--vis-depth", default="none", choices=["none", "hcat", "vcat"])
    p.add_argument("--video-quality", default="high", choices=["low", "medium", "high"])
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    gaussians = _load_da3_gsply(args.gsply, device=device)
    extrs_np, intrs_np, (H, W) = _load_colmap_poses(args.colmap_dir, sparse_subdir=args.sparse_subdir)

    # Pose scaling: explicit > json > auto (next to gsply)
    pose_scale = float(args.pose_scale) if args.pose_scale and args.pose_scale > 0 else None
    if pose_scale is None:
        pose_norm_json = args.pose_norm_json
        if not pose_norm_json:
            # Auto-find: <export_dir>/pose_norm.json where export_dir is parent of gs_ply/
            pose_norm_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.gsply))), "pose_norm.json")
        pose_mode, pose_transform, pose_scale = _try_load_pose_norm(pose_norm_json)
    else:
        pose_mode, pose_transform = None, None

    if pose_scale is not None:
        if pose_transform is not None and (pose_mode == "recenter_scale" or pose_mode is None):
            extrs_np = _recenter_and_scale_w2c_extrinsics(
                extrs_np, pose_norm_transform_c2w=pose_transform, pose_norm_scale=pose_scale
            )
        else:
            extrs_np = _scale_w2c_extrinsics(extrs_np, pose_norm_scale=pose_scale)

    extrs = torch.from_numpy(extrs_np).to(device=device)[None, ...]  # (1, V, 4, 4)
    intrs = torch.from_numpy(intrs_np).to(device=device)[None, ...]  # (1, V, 3, 3)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.gsply), "..", "gs_video_colmap")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    vis_depth = None if args.vis_depth == "none" else args.vis_depth
    output_name = args.output_name or "colmap_render"

    # Minimal Prediction for export_to_gs_video
    pred = Prediction(depth=np.zeros((1, 1, 1), dtype=np.float32), is_metric=0, gaussians=gaussians)

    export_to_gs_video(
        prediction=pred,
        export_dir=out_dir,
        extrinsics=extrs,
        intrinsics=intrs,
        out_image_hw=(H, W),
        chunk_size=args.chunk_size,
        trj_mode=args.trj_mode,
        vis_depth=vis_depth,
        output_name=output_name,
        video_quality=args.video_quality,
    )

    print(f"[OK] wrote video under: {out_dir}")


if __name__ == "__main__":
    main()

