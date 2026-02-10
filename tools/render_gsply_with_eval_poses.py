#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
        f_rest_names = sorted(f_rest_names, key=lambda s: int(s.split("_")[-1]))
        f_rest = np.stack([col(n) for n in f_rest_names], axis=1).astype(np.float32)  # (N, K)
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Unexpected f_rest dims (not divisible by 3): {f_rest.shape}")
        d_sh_minus_1 = f_rest.shape[1] // 3
        f_rest = f_rest.reshape(-1, 3, d_sh_minus_1)
        harmonics = np.concatenate([harmonics, f_rest], axis=2)  # (N, 3, d_sh)

    to_t = lambda x: torch.from_numpy(x).to(device=device)
    return Gaussians(
        means=to_t(means)[None, ...],
        scales=to_t(scales)[None, ...],
        rotations=to_t(rotations)[None, ...],
        harmonics=to_t(harmonics)[None, ...],
        opacities=to_t(opacities)[None, ...],
    )


def _try_load_pose_norm(pose_norm_json: str) -> tuple[str | None, np.ndarray | None, float | None]:
    if not pose_norm_json:
        return None, None, None
    if not os.path.isfile(pose_norm_json):
        return None, None, None
    with open(pose_norm_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    mode = data.get("pose_norm_mode", None)
    transform = data.get("pose_norm_transform", None)
    transform_np = np.array(transform, dtype=np.float32) if transform is not None else None

    scale = None
    if "pose_norm_scale" in data and data["pose_norm_scale"] is not None:
        scale = float(data["pose_norm_scale"])
    else:
        for k in ("median_dist", "scale", "pose_scale"):
            if k in data and data[k] is not None:
                scale = float(data[k])
                break
    return mode, transform_np, scale


def _scale_w2c_extrinsics(extrs_w2c: np.ndarray, pose_norm_scale: float) -> np.ndarray:
    if pose_norm_scale <= 0:
        raise ValueError(f"pose_norm_scale must be > 0, got {pose_norm_scale}")
    c2ws = np.linalg.inv(extrs_w2c)
    c2ws[:, :3, 3] = c2ws[:, :3, 3] / float(pose_norm_scale)
    return np.linalg.inv(c2ws).astype(np.float32)


def _recenter_and_scale_w2c_extrinsics(
    extrs_w2c: np.ndarray, pose_norm_transform_c2w: np.ndarray, pose_norm_scale: float
) -> np.ndarray:
    if pose_norm_transform_c2w.shape != (4, 4):
        raise ValueError(f"pose_norm_transform must be 4x4, got {pose_norm_transform_c2w.shape}")
    if pose_norm_scale <= 0:
        raise ValueError(f"pose_norm_scale must be > 0, got {pose_norm_scale}")
    ex_rec = (extrs_w2c @ pose_norm_transform_c2w[None, ...]).astype(np.float64)
    c2ws = np.linalg.inv(ex_rec)
    c2ws[:, :3, 3] = c2ws[:, :3, 3] / float(pose_norm_scale)
    return np.linalg.inv(c2ws).astype(np.float32)


def _load_colmap_eval_poses(
    colmap_dir: str, sparse_subdir: str, image_name_regex: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], list[str]]:
    sparse_dir = (
        os.path.join(colmap_dir, "sparse", sparse_subdir)
        if sparse_subdir
        else os.path.join(colmap_dir, "sparse")
    )

    cameras, images, _points3d = read_model(sparse_dir)
    if not images:
        raise ValueError(f"No images in COLMAP model: {sparse_dir}")

    pat = re.compile(image_name_regex)

    extrinsics = []
    intrinsics = []
    names = []
    hw: Tuple[int, int] | None = None

    for _image_id, image_data in sorted(images.items(), key=lambda kv: kv[1].name):
        name = image_data.name
        if not pat.search(name):
            continue

        cam = cameras[image_data.camera_id]
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
            fx = fy = cam.params[0] if len(cam.params) > 0 else 1000.0
            cx = cam.width / 2.0
            cy = cam.height / 2.0

        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        intrinsics.append(intr)
        names.append(name)

    if not extrinsics:
        raise ValueError(
            f"No COLMAP images matched image_name_regex='{image_name_regex}' under: {sparse_dir}"
        )
    assert hw is not None
    return np.stack(extrinsics, axis=0), np.stack(intrinsics, axis=0), hw, names


def _fov_to_intrinsics(fov_deg: float, W: int, H: int, fov_axis: str) -> np.ndarray:
    fov = float(fov_deg) * np.pi / 180.0
    if fov <= 0 or fov >= np.pi:
        raise ValueError(f"Invalid fov (deg): {fov_deg}")

    if fov_axis == "vertical":
        fy = 0.5 * H / np.tan(0.5 * fov)
        fx = fy * (W / float(H))
    elif fov_axis == "horizontal":
        fx = 0.5 * W / np.tan(0.5 * fov)
        fy = fx * (H / float(W))
    else:
        raise ValueError(f"Invalid fov_axis: {fov_axis}")

    cx = W / 2.0
    cy = H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _load_camera_path_json(
    camera_path_json: str, fov_axis: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    with open(camera_path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    W = int(data.get("render_width", 0) or 0)
    H = int(data.get("render_height", 0) or 0)
    if W <= 0 or H <= 0:
        raise ValueError(f"camera_path_json missing render_width/render_height: {camera_path_json}")

    cams = data.get("camera_path", None)
    if not isinstance(cams, list) or not cams:
        raise ValueError(f"camera_path_json missing camera_path list: {camera_path_json}")

    extrs_w2c = []
    intrs = []
    for cam in cams:
        c2w_flat = cam.get("camera_to_world", None)
        if not isinstance(c2w_flat, list) or len(c2w_flat) != 16:
            raise ValueError("camera_to_world must be a 16-float list")
        c2w = np.array(c2w_flat, dtype=np.float32).reshape(4, 4)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        extrs_w2c.append(w2c)

        fov = cam.get("fov", None)
        if fov is None:
            fov = data.get("fov", None)
        if fov is None:
            raise ValueError("Missing fov in camera_path entry (and no top-level fov)")
        intrs.append(_fov_to_intrinsics(float(fov), W=W, H=H, fov_axis=fov_axis))

    return np.stack(extrs_w2c, axis=0), np.stack(intrs, axis=0), (H, W)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Render a DA3-exported gs_ply PLY using eval poses. "
            "Primary: COLMAP poses filtered by image-name regex. "
            "Fallback: nerfstudio-style camera_path.json."
        )
    )
    p.add_argument("--gsply", required=True, help="Path to DA3 gs_ply/0000.ply")

    p.add_argument("--colmap-dir", default="", help="COLMAP root dir containing sparse/<subdir>/")
    p.add_argument("--sparse-subdir", default="0", help="Sparse subdir (default: 0 for sparse/0)")
    p.add_argument(
        "--image-name-regex",
        default=r"^frame_[0-9]+\.(png|jpg|jpeg)$",
        help="Regex to select eval images inside COLMAP model (default matches frame_00001.png style).",
    )

    p.add_argument(
        "--camera-path-json",
        default="",
        help="Optional nerfstudio-style camera_paths/<scene>-eval.json (used if COLMAP selection is empty).",
    )
    p.add_argument(
        "--fov-axis",
        default="vertical",
        choices=["vertical", "horizontal"],
        help="Interpretation of fov in camera_path.json (default: vertical).",
    )

    p.add_argument("--out-dir", default="", help="Output directory (default: alongside gsply)")
    p.add_argument("--output-name", default="", help="Output mp4 base name (without .mp4)")
    p.add_argument("--chunk-size", type=int, default=4, help="Rendering chunk size (gsplat batch)")
    p.add_argument(
        "--pose-norm-json",
        default="",
        help=(
            "Optional path to pose_norm.json produced by DA3 export. "
            "If provided (or auto-found), poses will be normalized before rendering."
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

    extrs_np = None
    intrs_np = None
    H = W = None

    if args.colmap_dir:
        try:
            extrs_np, intrs_np, (H, W), _names = _load_colmap_eval_poses(
                args.colmap_dir, sparse_subdir=args.sparse_subdir, image_name_regex=args.image_name_regex
            )
        except Exception as e:
            # Soft-fail: allow fallback to camera-path json if provided.
            if not args.camera_path_json:
                raise
            print(f"[WARN] COLMAP eval pose load failed, falling back to camera_path_json: {e}")

    if extrs_np is None:
        if not args.camera_path_json:
            raise ValueError("No poses loaded: provide --colmap-dir or --camera-path-json")
        extrs_np, intrs_np, (H, W) = _load_camera_path_json(args.camera_path_json, fov_axis=args.fov_axis)

    assert extrs_np is not None and intrs_np is not None and H is not None and W is not None

    # Pose scaling: explicit > json > auto (next to gsply)
    pose_scale = float(args.pose_scale) if args.pose_scale and args.pose_scale > 0 else None
    if pose_scale is None:
        pose_norm_json = args.pose_norm_json
        if not pose_norm_json:
            pose_norm_json = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(args.gsply))), "pose_norm.json"
            )
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

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.gsply), "..", "gs_video_eval")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    vis_depth = None if args.vis_depth == "none" else args.vis_depth
    output_name = args.output_name or "eval_render"

    pred = Prediction(depth=np.zeros((1, 1, 1), dtype=np.float32), is_metric=0, gaussians=gaussians)

    export_to_gs_video(
        prediction=pred,
        export_dir=out_dir,
        extrinsics=extrs,
        intrinsics=intrs,
        out_image_hw=(int(H), int(W)),
        chunk_size=args.chunk_size,
        trj_mode=args.trj_mode,
        vis_depth=vis_depth,
        output_name=output_name,
        video_quality=args.video_quality,
    )

    print(f"[OK] wrote video under: {out_dir}")


if __name__ == "__main__":
    main()

