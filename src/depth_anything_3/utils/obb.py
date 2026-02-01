from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class OrientedBBox:
    """
    Oriented bounding box (OBB) in the same coordinate frame as the input points.

    Conventions:
    - `center`: box center in world coordinates.
    - `axes`: 3x3 rotation matrix whose columns are the box local X/Y/Z axes in world coords.
    - `extents`: lengths along local X/Y/Z axes (positive).
    - `from_origin`: 4x4 transform mapping local box coords -> world coords (origin at box center).
    - `to_origin`: inverse of `from_origin` (world -> local box coords).
    """

    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3,3)
    extents: np.ndarray  # (3,)
    from_origin: np.ndarray  # (4,4)
    to_origin: np.ndarray  # (4,4)
    corners: np.ndarray  # (8,3)
    num_points: int
    source_geometry: str

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "extents": self.extents.tolist(),
            "from_origin": self.from_origin.tolist(),
            "to_origin": self.to_origin.tolist(),
            "corners": self.corners.tolist(),
            "num_points": int(self.num_points),
            "source_geometry": self.source_geometry,
        }


def _as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsics, return (4,4) homogeneous matrix.
    """
    ext = np.asarray(ext)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def load_camera_centers_from_npz(npz_path: str) -> np.ndarray:
    """
    Load camera centers (N,3) from an export npz containing `extrinsics` (world-to-cam).
    """
    data = np.load(npz_path)
    if "extrinsics" not in data:
        raise ValueError(f"NPZ does not contain 'extrinsics': {npz_path}")
    ext_w2c = np.asarray(data["extrinsics"])
    if ext_w2c.ndim != 3 or ext_w2c.shape[-2:] not in {(4, 4), (3, 4)}:
        raise ValueError(f"extrinsics must be (N,4,4) or (N,3,4); got {ext_w2c.shape}")

    centers = []
    for i in range(ext_w2c.shape[0]):
        w2c = _as_homogeneous44(ext_w2c[i]).astype(np.float64, copy=False)
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    centers = np.asarray(centers, dtype=np.float64)
    return _filter_finite(centers)


def _filter_finite(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def _downsample(points: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0:
        return points
    if points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def load_points_from_glb(
    glb_path: str,
    geometry_name: str | None = None,
    max_points: int | None = None,
    include_cameras: bool = False,
    exclude_geometry_names: set[str] | None = None,
) -> tuple[np.ndarray, str]:
    """
    Load vertices from a .glb and return (points, chosen_geometry_name).

    Heuristic:
    - If `geometry_name` is given, pick an exact match if possible, else substring match.
    - Otherwise pick the geometry with the largest vertex count.
    """
    scene = trimesh.load(glb_path, force="scene")
    if not hasattr(scene, "geometry") or scene.geometry is None or len(scene.geometry) == 0:
        raise ValueError(f"No geometry found in GLB: {glb_path}")

    if exclude_geometry_names is None:
        exclude_geometry_names = {"obb", "obb_wire"}

    def _geom_vertices(name: str, geom) -> np.ndarray | None:
        if name in exclude_geometry_names:
            return None
        verts = getattr(geom, "vertices", None)
        if verts is None:
            return None
        verts = np.asarray(verts)
        if verts.ndim != 2 or verts.shape[1] != 3:
            return None
        if verts.shape[0] == 0:
            return None
        return verts

    candidates: list[tuple[str, np.ndarray]] = []
    path_vertices: list[np.ndarray] = []
    for name, geom in scene.geometry.items():
        verts = _geom_vertices(name, geom)
        if verts is None:
            continue
        candidates.append((name, verts))
        if include_cameras and type(geom).__name__ == "Path3D":
            path_vertices.append(verts)

    if len(candidates) == 0:
        raise ValueError(f"No vertex geometry found in GLB: {glb_path}")

    chosen_name: str
    chosen_points: np.ndarray

    if geometry_name:
        # exact match first
        exact = [c for c in candidates if c[0] == geometry_name]
        if exact:
            chosen_name, chosen_points = exact[0]
        else:
            sub = [c for c in candidates if geometry_name in c[0]]
            if not sub:
                available = ", ".join([c[0] for c in candidates])
                raise ValueError(
                    f"Requested geometry '{geometry_name}' not found. Available: {available}"
                )
            chosen_name, chosen_points = max(sub, key=lambda x: x[1].shape[0])
    else:
        chosen_name, chosen_points = max(candidates, key=lambda x: x[1].shape[0])

    points = chosen_points.astype(np.float64, copy=False)
    points = _filter_finite(points)
    if include_cameras and len(path_vertices) > 0:
        cam_pts = np.concatenate([v.astype(np.float64, copy=False) for v in path_vertices], axis=0)
        cam_pts = _filter_finite(cam_pts)
        if cam_pts.shape[0] > 0:
            points = np.concatenate([points, cam_pts], axis=0)
    points = _downsample(points, max_points=max_points)
    return points, chosen_name


def compute_obb_from_points(
    points: np.ndarray,
    clip_percentile: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OBB using trimesh's oriented bounds.

    Returns:
        (to_origin, extents)
    where `to_origin` maps world points into the OBB local frame, centered at origin.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3); got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points to compute OBB; got {points.shape[0]}")

    pts = _filter_finite(points)
    if pts.shape[0] < 3:
        raise ValueError("Not enough finite points to compute OBB")

    if clip_percentile and clip_percentile > 0:
        p = float(clip_percentile)
        if not (0.0 < p < 50.0):
            raise ValueError("clip_percentile must be in (0, 50)")
        lo = np.percentile(pts, p, axis=0)
        hi = np.percentile(pts, 100.0 - p, axis=0)
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts = pts[mask]
        if pts.shape[0] < 3:
            raise ValueError("Not enough points left after clipping to compute OBB")

    to_origin, extents = trimesh.bounds.oriented_bounds(pts)
    extents = np.asarray(extents, dtype=np.float64)
    to_origin = np.asarray(to_origin, dtype=np.float64)
    return to_origin, extents


def obb_from_glb(
    glb_path: str,
    geometry_name: str | None = None,
    max_points: int | None = 2_000_000,
    clip_percentile: float = 0.0,
    include_cameras: bool = False,
    camera_padding_radius: float = 0.0,
    camera_sphere_radius: float | None = None,
    include_camera_centers: bool = False,
    camera_npz_path: str | None = None,
) -> OrientedBBox:
    points, chosen = load_points_from_glb(
        glb_path,
        geometry_name=geometry_name,
        max_points=max_points,
        include_cameras=include_cameras,
    )

    # Option: expand the final box extents by a ratio p (keeps center fixed).
    # Example: p=0.2 => extents *= 1.2 (+20% along each box axis).
    # Backward compatibility: `camera_sphere_radius` (old name) overrides if explicitly provided.
    if camera_sphere_radius is not None:
        p = float(camera_sphere_radius or 0.0)
    else:
        p = float(camera_padding_radius or 0.0)
    if p < 0:
        raise ValueError("camera_padding_radius must be >= 0 (ratio, e.g. 0.2 for +20%)")

    # Option: include camera centers (requires extrinsics npz).
    if include_camera_centers:
        if camera_npz_path is None:
            glb_dir = os.path.dirname(glb_path)
            candidates = [
                os.path.join(glb_dir, "exports", "npz", "results.npz"),
                os.path.join(glb_dir, "exports", "mini_npz", "results.npz"),
            ]
            camera_npz_path = next((p for p in candidates if os.path.exists(p)), None)
        if camera_npz_path is None:
            raise ValueError(
                "include_camera_centers=True but no camera NPZ was found. "
                "Export npz alongside glb (e.g., --export-format glb-npz), "
                "or pass camera_npz_path explicitly."
            )

        cam_centers = load_camera_centers_from_npz(camera_npz_path)
        if cam_centers.shape[0] > 0:
            # Map camera centers to the GLB coordinate system (same as point cloud) using hf_alignment.
            try:
                scene = trimesh.load(glb_path, force="scene")
                A = scene.metadata.get("hf_alignment", None) if getattr(scene, "metadata", None) else None
                A = np.asarray(A, dtype=np.float64) if A is not None else np.eye(4, dtype=np.float64)
            except Exception:
                A = np.eye(4, dtype=np.float64)
            cam_centers = trimesh.transform_points(cam_centers, A)
            points = np.concatenate([points, cam_centers], axis=0)

    to_origin, extents = compute_obb_from_points(points, clip_percentile=clip_percentile)
    if p > 0:
        extents = np.asarray(extents, dtype=np.float64) * (1.0 + p)
    from_origin = np.linalg.inv(to_origin)

    axes = from_origin[:3, :3].copy()
    # Ensure a right-handed frame (det > 0). If not, flip Z.
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0
        from_origin[:3, :3] = axes
        to_origin = np.linalg.inv(from_origin)

    center = from_origin[:3, 3].copy()

    # 8 corners in local box coordinates (centered at origin)
    half = extents * 0.5
    corners_local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], +half[2]],
            [-half[0], +half[1], -half[2]],
            [-half[0], +half[1], +half[2]],
            [+half[0], -half[1], -half[2]],
            [+half[0], -half[1], +half[2]],
            [+half[0], +half[1], -half[2]],
            [+half[0], +half[1], +half[2]],
        ],
        dtype=np.float64,
    )
    corners_world = trimesh.transform_points(corners_local, from_origin)

    return OrientedBBox(
        center=center,
        axes=axes,
        extents=extents,
        from_origin=from_origin,
        to_origin=to_origin,
        corners=corners_world,
        num_points=int(points.shape[0]),
        source_geometry=str(chosen),
    )


def save_obb_json(obb: OrientedBBox, out_json: str) -> str:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(obb.to_dict(), f, indent=2)
    return out_json


def export_glb_with_obb(
    src_glb_path: str,
    obb: OrientedBBox,
    out_glb_path: str,
    style: str = "wire",
    rgba: tuple[int, int, int, int] = (255, 0, 0, 80),
) -> str:
    """
    Export a .glb containing the original scene plus an OBB visualization.

    Args:
        style: "wire" (outline only) or "solid" (box mesh).
    """
    scene = trimesh.load(src_glb_path, force="scene")

    style = (style or "wire").lower().strip()
    if style not in {"wire", "solid"}:
        raise ValueError(f"style must be 'wire' or 'solid', got: {style}")

    if style == "solid":
        box = trimesh.creation.box(extents=obb.extents, transform=obb.from_origin)
        # Best-effort: set a transparent PBR material so glTF viewers render alpha.
        try:
            from trimesh.visual.material import PBRMaterial

            r, g, b, a = [int(x) for x in rgba]
            box.visual.material = PBRMaterial(
                baseColorFactor=[r / 255.0, g / 255.0, b / 255.0, a / 255.0],
                alphaMode="BLEND",
                metallicFactor=0.0,
                roughnessFactor=1.0,
                doubleSided=True,
            )
        except Exception:
            # Fallback: vertex/face colors (some exporters/viewers may still treat as opaque)
            try:
                box.visual.face_colors = np.array(rgba, dtype=np.uint8)
            except Exception:
                pass
        scene.add_geometry(box, geom_name="obb")
    else:
        # Wireframe edges as a path (12 segments).
        edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (7, 3),
            (7, 5),
            (7, 6),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (4, 5),
            (4, 6),
        ]
        segs = np.stack(
            [np.stack([obb.corners[a], obb.corners[b]], axis=0) for a, b in edges], axis=0
        )
        path = trimesh.load_path(segs)
        try:
            color = np.array(rgba, dtype=np.uint8)
            if hasattr(path, "colors"):
                path.colors = np.tile(color, (len(path.entities), 1))
        except Exception:
            pass
        scene.add_geometry(path, geom_name="obb_wire")
    os.makedirs(os.path.dirname(out_glb_path) or ".", exist_ok=True)
    scene.export(out_glb_path)
    return out_glb_path

