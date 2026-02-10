#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class FrameMetric:
    idx: int
    png: str
    psnr: float
    ssim: float
    lpips: float


def _sorted_eval_pngs(eval_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No eval PNGs matched: {os.path.join(eval_dir, pattern)}")
    return files


def _png_hw(path: str) -> Tuple[int, int]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return h, w


def _count_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    n = 0
    while True:
        ok, _fr = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def _read_png_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _compute_means_for_pair(
    *,
    pngs: List[str],
    video_path: str,
    lpips_model: lpips.LPIPS,
    device: str,
    save_per_frame: bool,
) -> Dict[str, Any]:
    H, W = _png_hw(pngs[0])

    video_decoded_frames = _count_video_frames(video_path)

    png_count = len(pngs)
    extra_trailing = video_decoded_frames == png_count + 1
    compared_frames = min(png_count, video_decoded_frames - 1 if extra_trailing else video_decoded_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    per_frame: List[FrameMetric] = []

    for i in range(compared_frames):
        ok, fr_bgr = cap.read()
        if not ok:
            compared_frames = i
            break
        pr = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        if pr.shape[0] != H or pr.shape[1] != W:
            pr = cv2.resize(pr, (W, H), interpolation=cv2.INTER_AREA)

        gt = _read_png_rgb(pngs[i])
        if gt.shape[0] != H or gt.shape[1] != W:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)

        psnr = float(peak_signal_noise_ratio(gt, pr, data_range=255))
        ssim = float(structural_similarity(gt, pr, channel_axis=2, data_range=255))

        gt_t = torch.from_numpy(gt).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        pr_t = torch.from_numpy(pr).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        with torch.no_grad():
            lp = float(lpips_model(gt_t, pr_t).item())

        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lpips_vals.append(lp)

        if save_per_frame:
            per_frame.append(
                FrameMetric(idx=i, png=os.path.basename(pngs[i]), psnr=psnr, ssim=ssim, lpips=lp)
            )

    cap.release()

    if compared_frames <= 0:
        raise RuntimeError("No overlapping frames to compare.")

    out: Dict[str, Any] = {
        "png_count": png_count,
        "video_decoded_frames": video_decoded_frames,
        "compared_frames": compared_frames,
        "png_resolution": [H, W],
        "alignment": {
            "policy": "compare first N frames; if video_len == png_len + 1, treat last frame as extra",
            "video_has_extra_trailing_frame": bool(extra_trailing),
        },
        "mean": {
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(np.mean(lpips_vals)),
        },
    }
    if save_per_frame:
        out["per_frame"] = [asdict(x) for x in per_frame]
    return out


def _default_scenes() -> List[str]:
    return [
        "aloe",
        "art",
        "car",
        "century",
        "flowers",
        "garbage",
        "picnic",
        "pipe",
        "plant",
        "roses",
        "table",
    ]

def _discover_scenes(dataset_root: str) -> List[str]:
    scenes: List[str] = []
    for name in sorted(os.listdir(dataset_root)):
        if name.startswith("."):
            continue
        p = os.path.join(dataset_root, name)
        if not os.path.isdir(p):
            continue
        eval_dir = os.path.join(p, "eval")
        if os.path.isdir(eval_dir):
            scenes.append(name)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute scene-mean PSNR/SSIM/LPIPS between Nerfbusters eval PNGs and DA3 gs_video_eval mp4."
    )
    ap.add_argument(
        "--dataset-root",
        default="/media/ssd1/users/sooran/dataset/nerfbusters-dataset",
        help="Root containing {scene}/eval/_eval_*.png",
    )
    ap.add_argument(
        "--export-root",
        default="/media/ssd1/users/sooran/Depth-Anything-3/output/nb-colmap-gs",
        help="Root containing {scene}/gs_video_eval/gs_video/{scene}__eval.mp4",
    )
    ap.add_argument("--scenes", nargs="*", default=[], help="Scenes to evaluate (default: common nerfbusters list)")
    ap.add_argument(
        "--export-scene-template",
        default="{scene}__train",
        help=(
            "Template for export scene directory name under --export-root. "
            "Example: '{scene}__train' -> 'aloe__train'. Use '{scene}' if your export dir is just the scene name."
        ),
    )
    ap.add_argument(
        "--eval-png-glob",
        default="_eval_*.png",
        help="Glob pattern under {scene}/eval/ for GT images (default: _eval_*.png)",
    )
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="If set, skip scenes missing eval dir or video instead of raising.",
    )
    ap.add_argument(
        "--save-per-frame",
        action="store_true",
        help="If set, store per-frame metrics in JSON (large). By default only means are stored.",
    )
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.scenes:
        scenes = args.scenes
    else:
        # Prefer discovery so we match the dataset root contents.
        scenes = _discover_scenes(args.dataset_root) or _default_scenes()

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device=device)
    lpips_model.eval()

    out: Dict[str, Any] = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "export_root": os.path.abspath(args.export_root),
        "device": device,
        "lpips_net": args.lpips_net,
        "eval_png_glob": args.eval_png_glob,
        "scenes": {},
    }

    # Also compute overall mean (weighted by frames)
    total_frames = 0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    for scene in scenes:
        eval_dir = os.path.join(args.dataset_root, scene, "eval")
        export_scene = str(args.export_scene_template).format(scene=scene)
        video_path = os.path.join(
            args.export_root,
            export_scene,
            "gs_video_eval",
            "gs_video",
            f"{export_scene}__eval.mp4",
        )

        if not os.path.isdir(eval_dir):
            if args.skip_missing:
                print(f"[WARN] missing eval dir, skipping: {eval_dir}")
                continue
            raise FileNotFoundError(f"Missing eval dir: {eval_dir}")
        if not os.path.isfile(video_path):
            if args.skip_missing:
                print(f"[WARN] missing rendered eval video, skipping: {video_path}")
                continue
            raise FileNotFoundError(f"Missing rendered eval video: {video_path}")

        pngs = _sorted_eval_pngs(eval_dir, args.eval_png_glob)
        stats = _compute_means_for_pair(
            pngs=pngs,
            video_path=video_path,
            lpips_model=lpips_model,
            device=device,
            save_per_frame=bool(args.save_per_frame),
        )

        out["scenes"][scene] = {
            "eval_dir": os.path.abspath(eval_dir),
            "video_path": os.path.abspath(video_path),
            "export_scene": export_scene,
            **stats,
        }

        n = int(stats["compared_frames"])
        total_frames += n
        psnr_sum += float(stats["mean"]["psnr"]) * n
        ssim_sum += float(stats["mean"]["ssim"]) * n
        lpips_sum += float(stats["mean"]["lpips"]) * n

    if total_frames > 0:
        out["overall_mean"] = {
            "weighted_by": "compared_frames",
            "total_frames": int(total_frames),
            "psnr": psnr_sum / total_frames,
            "ssim": ssim_sum / total_frames,
            "lpips": lpips_sum / total_frames,
        }

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

