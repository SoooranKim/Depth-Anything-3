#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict


def _wavg(a: float, wa: int, b: float, wb: int) -> float:
    denom = wa + wb
    if denom <= 0:
        raise ValueError("Total weight must be > 0")
    return (a * wa + b * wb) / denom


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine explore1/explore2 means by frame-count-weighted average.")
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--backup", action="store_true", help="If overwriting, create .bak copy first")
    args = ap.parse_args()

    in_path = os.path.abspath(args.in_json)
    out_path = os.path.abspath(args.out_json)

    if args.backup and os.path.exists(out_path) and os.path.samefile(in_path, out_path):
        bak = out_path + ".bak"
        shutil.copy2(out_path, bak)
        print(f"[OK] backup: {bak}")

    with open(in_path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = json.load(f)

    scenes = d.get("scenes", {})
    combined_scenes: Dict[str, Any] = {}

    for scene, scene_data in scenes.items():
        splits = (scene_data or {}).get("splits", {})
        e1 = splits.get("explore1")
        e2 = splits.get("explore2")

        def get_mean(x: Any, k: str) -> float:
            return float(x["mean"][k])

        def get_w(x: Any) -> int:
            return int(x.get("compared_frames", 0))

        if e1 is None and e2 is None:
            continue

        if e1 is None:
            w = get_w(e2)
            combined = {"psnr": get_mean(e2, "psnr"), "ssim": get_mean(e2, "ssim"), "lpips": get_mean(e2, "lpips")}
        elif e2 is None:
            w = get_w(e1)
            combined = {"psnr": get_mean(e1, "psnr"), "ssim": get_mean(e1, "ssim"), "lpips": get_mean(e1, "lpips")}
        else:
            w1 = get_w(e1)
            w2 = get_w(e2)
            w = w1 + w2
            combined = {
                "psnr": _wavg(get_mean(e1, "psnr"), w1, get_mean(e2, "psnr"), w2),
                "ssim": _wavg(get_mean(e1, "ssim"), w1, get_mean(e2, "ssim"), w2),
                "lpips": _wavg(get_mean(e1, "lpips"), w1, get_mean(e2, "lpips"), w2),
            }

        combined_scenes[scene] = {
            "eval_dir": scene_data.get("eval_dir"),
            "compared_frames_total": int(w),
            "weights": {
                "explore1_frames": 0 if e1 is None else int(get_w(e1)),
                "explore2_frames": 0 if e2 is None else int(get_w(e2)),
            },
            "mean": combined,
        }

    out = {
        "dataset_root": d.get("dataset_root"),
        "export_root": d.get("export_root"),
        "device": d.get("device"),
        "lpips_net": d.get("lpips_net"),
        "note": "Per-scene means are frame-count-weighted average of explore1/explore2 means.",
        "scenes": combined_scenes,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()

