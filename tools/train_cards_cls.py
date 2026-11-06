"""Train the card rank+suit classifier from a Roboflow folder export."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset/cards_cls_all")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=128)
    p.add_argument("--name", default="cards_v2")
    p.add_argument("--out", default="models/yolo/cls_best.pt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=0,
        imgsz=args.imgsz,
        batch=16,
        fliplr=0.0,
        # keep the rank/suit corner: no random erasing, crops of at least 90% of the image
        erasing=0.0,
        scale=0.1,
        device="cpu",
        project="runs/cls",
        name=args.name,
        exist_ok=True,
        workers=0,
        verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"copied {best} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
