"""Number every crop under a folder and render numbered sheets for manual labeling."""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

COLS, ROWS, W, H = 10, 4, 80, 120


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset/cards_prelabel"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/check_cards/sheets"
    files = sorted(glob.glob(f"{root}/*/*.png"))
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/index.json", "w", encoding="utf-8") as fh:
        json.dump(files, fh, indent=0)
    per = COLS * ROWS
    for s in range(0, len(files), per):
        tiles = []
        for i in range(s, s + per):
            tile = np.full((H + 22, W, 3), 255, np.uint8)
            if i < len(files):
                tile[22:, :] = cv2.resize(cv2.imread(files[i]), (W, H))
                cv2.putText(tile, str(i), (2, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)
            tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0)))
        grid = np.vstack([np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        cv2.imwrite(f"{out}/sheet_{s // per:02d}.png", grid)
    print(len(files), "crops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
