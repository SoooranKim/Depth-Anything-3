#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _sorted_pngs(eval_dir: str, prefix: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(eval_dir, f"{prefix}*.png")))
    if not files:
        raise FileNotFoundError(f"No PNGs matched: {os.path.join(eval_dir, f'{prefix}*.png')}")
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
) -> Dict[str, Any]:
    # Determine target resolution from PNGs
    H, W = _png_hw(pngs[0])

    # Count decoded video frames (robust)
    video_decoded_frames = _count_video_frames(video_path)

    # Decide how many video frames to use: for our rendered mp4s we often get +1 trailing frame.
    # We align by comparing the first N = len(pngs) frames. If video has exactly png+1, we treat last as extra.
    png_count = len(pngs)
    extra_trailing = (video_decoded_frames == png_count + 1)
    compared_frames = min(png_count, video_decoded_frames - 1 if extra_trailing else video_decoded_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0

    for i in range(compared_frames):
        ok, fr_bgr = cap.read()
        if not ok:
            compared_frames = i
            break
        fr_rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        if fr_rgb.shape[0] != H or fr_rgb.shape[1] != W:
            fr_rgb = cv2.resize(fr_rgb, (W, H), interpolation=cv2.INTER_AREA)

        gt = _read_png_rgb(pngs[i])
        if gt.shape[0] != H or gt.shape[1] != W:
            # Shouldn't happen, but be defensive
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)

        psnr = float(peak_signal_noise_ratio(gt, fr_rgb, data_range=255))
        ssim = float(structural_similarity(gt, fr_rgb, channel_axis=2, data_range=255))

        gt_t = torch.from_numpy(gt).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        fr_t = torch.from_numpy(fr_rgb).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        with torch.no_grad():
            lp = float(lpips_model(gt_t, fr_t).item())

        psnr_sum += psnr
        ssim_sum += ssim
        lpips_sum += lp

    cap.release()
    if compared_frames <= 0:
        raise RuntimeError("No overlapping frames to compare.")

    return {
        "png_count": png_count,
        "video_decoded_frames": video_decoded_frames,
        "compared_frames": compared_frames,
        "png_resolution": [H, W],
        "alignment": {
            "policy": "compare first N frames; if video_len == png_len + 1, treat last frame as extra",
            "video_has_extra_trailing_frame": bool(extra_trailing),
        },
        "mean": {
            "psnr": psnr_sum / compared_frames,
            "ssim": ssim_sum / compared_frames,
            "lpips": lpips_sum / compared_frames,
        },
    }


def _scene_video_path(export_root: str, scene: str, split: str) -> str:
    return os.path.join(
        export_root,
        f"{scene}_train",
        f"gs_video_{split}",
        "gs_video",
        f"{scene}_train__{split}.mp4",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/media/ssd1/users/sooran/dataset/wild-explore")
    ap.add_argument("--export-root", default="/media/ssd1/users/sooran/Depth-Anything-3/output/we-colmap-gs")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device=device)
    lpips_model.eval()

    out: Dict[str, Any] = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "export_root": os.path.abspath(args.export_root),
        "device": device,
        "lpips_net": args.lpips_net,
        "scenes": {},
    }

    for scene in args.scenes:
        eval_dir = os.path.join(args.dataset_root, scene, "eval")
        scene_out: Dict[str, Any] = {"eval_dir": os.path.abspath(eval_dir), "splits": {}}
        for split in ["explore1", "explore2"]:
            pngs = _sorted_pngs(eval_dir, split)
            vid = _scene_video_path(args.export_root, scene, split)
            if not os.path.isfile(vid):
                raise FileNotFoundError(f"Missing video for scene={scene} split={split}: {vid}")
            scene_out["splits"][split] = {
                "video_path": os.path.abspath(vid),
                **_compute_means_for_pair(pngs=pngs, video_path=vid, lpips_model=lpips_model, device=device),
            }
        out["scenes"][scene] = scene_out

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

