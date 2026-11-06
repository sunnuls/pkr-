"""Dry-run: show which button the solo vision bot would tap on saved frames."""

from __future__ import annotations

import glob
import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.vision.yolo_engine import VisionEngine  # noqa: E402
from sim_engine.vision_decision import decide, legal_actions  # noqa: E402


def main() -> int:
    logging.disable(logging.WARNING)
    pattern = sys.argv[1] if len(sys.argv) > 1 else "dataset/train_sorted/buttons/*.png"
    engine = VisionEngine.from_config(
        yaml.safe_load(open("config/bots.yaml", encoding="utf-8"))
    )
    engine.fast_text = True
    for f in sorted(glob.glob(pattern)):
        t0 = time.time()
        state = engine.get_game_state(cv2.imread(f))
        choice = decide(state)
        took = time.time() - t0
        tap = f"{choice.button} {choice.xy} {choice.decision.action.value}" if choice else "-"
        print(
            Path(f).name, f"{took:.1f}s",
            "hero", state.hero_cards, "board", state.board_cards,
            "pot", state.pot, "call", state.call_amount,
            "legal", legal_actions(state), "->", tap,
            "|", choice.decision.reasoning[:70] if choice else "",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
