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


def _sorted_pngs(eval_dir: str, prefix: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(eval_dir, f"{prefix}*.png")))
    if not files:
        raise FileNotFoundError(f"No PNGs matched: {os.path.join(eval_dir, f'{prefix}*.png')}")
    return files


def _read_png_rgb(path: str) -> np.ndarray:
    # RGB uint8 HxWx3
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_video_frames_rgb(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, fr_bgr = cap.read()
        if not ok:
            break
        fr_rgb = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB)
        frames.append(fr_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"Decoded 0 frames from: {path}")
    return frames


def _maybe_drop_trailing_dup(video_frames: List[np.ndarray], target_len: int) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    meta: Dict[str, Any] = {"dropped_last_frame": False, "drop_reason": None}
    if len(video_frames) == target_len + 1 and len(video_frames) >= 2:
        # Heuristic: these rendered mp4s often have 1 extra trailing frame that is nearly identical to prev.
        last = video_frames[-1].astype(np.float32)
        prev = video_frames[-2].astype(np.float32)
        mse = float(np.mean((last - prev) ** 2))
        # MSE threshold: ~1.0 is very small difference for uint8 images; tuned to be conservative.
        if mse < 1.0:
            meta["dropped_last_frame"] = True
            meta["drop_reason"] = f"video_len == png_len + 1 and trailing MSE={mse:.6f} < 1.0"
            return video_frames[:-1], meta
        # Even if not tiny, still align by length to avoid shifting all frames.
        meta["dropped_last_frame"] = True
        meta["drop_reason"] = f"video_len == png_len + 1 (forced drop); trailing MSE={mse:.6f}"
        return video_frames[:-1], meta
    return video_frames, meta


def _compute_metrics_for_split(
    *,
    eval_dir: str,
    prefix: str,
    video_path: str,
    lpips_model: lpips.LPIPS,
    device: str,
) -> Dict[str, Any]:
    pngs = _sorted_pngs(eval_dir, prefix)
    png0 = _read_png_rgb(pngs[0])
    H, W = png0.shape[0], png0.shape[1]

    video_frames = _read_video_frames_rgb(video_path)
    video_frames, drop_meta = _maybe_drop_trailing_dup(video_frames, target_len=len(pngs))

    n = min(len(pngs), len(video_frames))
    if n <= 0:
        raise RuntimeError("No overlapping frames to compare.")

    per_frame: List[FrameMetric] = []
    lpips_vals: List[float] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []

    for i in range(n):
        gt = _read_png_rgb(pngs[i])  # HxWx3 uint8
        pr = video_frames[i]  # HxWx3 uint8 (expected)

        if pr.shape[0] != H or pr.shape[1] != W:
            pr = cv2.resize(pr, (W, H), interpolation=cv2.INTER_AREA)

        psnr = float(peak_signal_noise_ratio(gt, pr, data_range=255))
        ssim = float(structural_similarity(gt, pr, channel_axis=2, data_range=255))

        # LPIPS expects [-1, 1], float32, NCHW
        gt_t = torch.from_numpy(gt).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        pr_t = torch.from_numpy(pr).to(device=device).float().permute(2, 0, 1)[None, ...] / 127.5 - 1.0
        with torch.no_grad():
            lp = float(lpips_model(gt_t, pr_t).item())

        per_frame.append(
            FrameMetric(
                idx=i,
                png=os.path.basename(pngs[i]),
                psnr=psnr,
                ssim=ssim,
                lpips=lp,
            )
        )
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        lpips_vals.append(lp)

    return {
        "prefix": prefix,
        "eval_dir": os.path.abspath(eval_dir),
        "video_path": os.path.abspath(video_path),
        "png_count": len(pngs),
        "video_decoded_frames": len(_read_video_frames_rgb(video_path)),
        "video_frames_used": len(video_frames),
        "compared_frames": n,
        "png_resolution": [H, W],
        "drop_alignment": drop_meta,
        "mean": {
            "psnr": float(np.mean(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "lpips": float(np.mean(lpips_vals)),
        },
        "per_frame": [asdict(x) for x in per_frame],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--explore1-video", required=True)
    ap.add_argument("--explore2-video", required=True)
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
        "device": device,
        "lpips_net": args.lpips_net,
        "splits": {},
    }

    out["splits"]["explore1"] = _compute_metrics_for_split(
        eval_dir=args.eval_dir,
        prefix="explore1",
        video_path=args.explore1_video,
        lpips_model=lpips_model,
        device=device,
    )
    out["splits"]["explore2"] = _compute_metrics_for_split(
        eval_dir=args.eval_dir,
        prefix="explore2",
        video_path=args.explore2_video,
        lpips_model=lpips_model,
        device=device,
    )

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

