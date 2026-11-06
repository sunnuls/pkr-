"""Build a contact sheet: one row per label folder, first N crops each."""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset/cards_prelabel"
    out = sys.argv[2] if len(sys.argv) > 2 else "runs/check_cards/prelabel_sheet.png"
    n = 14
    rows = []
    for d in sorted(os.listdir(root)):
        files = sorted(glob.glob(f"{root}/{d}/*.png"))[:n]
        ims = [cv2.resize(cv2.imread(f), (40, 60)) for f in files]
        ims += [np.zeros((60, 40, 3), np.uint8)] * (n - len(ims))
        lab = np.full((60, 60, 3), 255, np.uint8)
        cv2.putText(lab, d, (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        rows.append(np.hstack([lab] + ims))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.vstack(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
