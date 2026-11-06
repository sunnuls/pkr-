"""Crop face-up cards from raw frames and sort them into folders by predicted label."""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.vision.yolo_engine import VisionEngine  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="dataset/raw/e2d0507")
    p.add_argument("--since", type=int, default=0, help="min frame epoch ms")
    p.add_argument("--out", default="dataset/cards_prelabel")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--dup", type=float, default=10.0, help="thumb diff below = duplicate")
    p.add_argument("--hero-min-w", type=int, default=0,
                   help="skip hero crops narrower than this (overlapped card)")
    return p.parse_args(argv)


def _thumb(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 48)).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.disable(logging.WARNING)
    engine = VisionEngine.from_config(
        yaml.safe_load(open("config/bots.yaml", encoding="utf-8"))
    )
    frames = [
        f for f in sorted(glob.glob(f"{args.raw}/*.png"))
        if int(Path(f).stem) >= args.since
    ]
    out = Path(args.out)
    kept: dict[str, list[np.ndarray]] = {}
    saved = skipped_dup = skipped_small = 0
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        for det in engine.detector.detect(img):
            kind = det["cls"]
            if kind not in ("hero_card", "board_card") or det["conf"] < args.conf:
                continue
            nx1, ny1, nx2, ny2 = det["box_norm"]
            x1, y1 = max(0, int(nx1 * w) - 2), max(0, int(ny1 * h) - 2)
            x2, y2 = min(w, int(nx2 * w) + 2), min(h, int(ny2 * h) + 2)
            if x2 - x1 < 24 or y2 - y1 < 36:
                skipped_small += 1
                continue
            if kind == "hero_card" and x2 - x1 < args.hero_min_w:
                skipped_small += 1
                continue
            crop = img[y1:y2, x1:x2]
            label, conf = engine.classifier.classify(crop)
            label = label or "unknown"
            t = _thumb(crop)
            pool = kept.setdefault(label, [])
            if any(float(np.mean(np.abs(t - o))) < args.dup for o in pool):
                skipped_dup += 1
                continue
            pool.append(t)
            dest = out / label
            dest.mkdir(parents=True, exist_ok=True)
            name = f"{kind.split('_')[0]}_{Path(f).stem}_{x1}_{int(conf * 100):02d}.png"
            cv2.imwrite(str(dest / name), crop)
            saved += 1
    print(f"frames {len(frames)} saved {saved} dup {skipped_dup} small {skipped_small}")
    for label in sorted(kept):
        print(label, len(kept[label]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
