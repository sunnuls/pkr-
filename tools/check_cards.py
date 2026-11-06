"""Print card labels read from table frames and save annotated previews."""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.vision.yolo_engine import VisionEngine  # noqa: E402


def main() -> int:
    logging.disable(logging.WARNING)
    cfg = yaml.safe_load(open("config/bots.yaml", encoding="utf-8"))
    engine = VisionEngine.from_config(cfg)
    print("classifier stub:", engine.classifier._stub)
    out = Path("runs/check_cards")
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(glob.glob("dataset/train_sorted/buttons/*.png"))[::4]
    frames += sorted(glob.glob("dataset/train_sorted/table/*.png"))[::12]
    for f in frames:
        img = cv2.imread(f)
        state = engine.get_game_state(img)
        name = Path(f).name
        print(name, "hero", state.hero_cards, "board", state.board_cards)
        h, w = img.shape[:2]
        for det in engine.detector.detect(img):
            if det["cls"] not in ("hero_card", "board_card"):
                continue
            nx1, ny1, nx2, ny2 = det["box_norm"]
            x1, y1, x2, y2 = int(nx1 * w), int(ny1 * h), int(nx2 * w), int(ny2 * h)
            label, conf = engine.classifier.classify(img[y1:y2, x1:x2])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out / name), img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
