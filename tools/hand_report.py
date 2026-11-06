"""Hand-by-hand results from the phone bot console log.

A hand is a run of decisions with the same hole cards. Its result is the stack at the
first decision of the next hand minus the stack at the first decision of this one
(blinds included, so it is approximate). Jumps over half the stack are treated as OCR
misreads and skipped.

  python tools/hand_report.py [runs/live_play/console.log] [--csv runs/live_play/hands.csv]
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from pathlib import Path

LINE = re.compile(
    r"^#\d+ (?P<time>\d\d:\d\d:\d\d) hero=(?P<hero>\[.*?\]) board=(?P<board>\[.*?\]) "
    r"stack=(?P<stack>\d+) pot=\d+ call=(?P<call>\d+) -> (?P<action>[A-Z]+)"
)
ORDER = "23456789TJQKA"


def category(hero: list[str]) -> str:
    if len(hero) != 2:
        return "unknown"
    (r1, s1), (r2, s2) = hero
    hi, lo = sorted((ORDER.index(r1), ORDER.index(r2)), reverse=True)
    if hi == lo:
        return "pair"
    suited = "suited" if s1 == s2 else "offsuit"
    if lo >= ORDER.index("T"):
        return f"broadway {suited}"
    if hi == ORDER.index("A"):
        return f"ace-x {suited}"
    if hi - lo <= 2:
        return f"connector {suited}"
    return f"junk {suited}"


def parse(path: Path) -> list[dict]:
    hands: list[dict] = []
    data = path.read_bytes()
    # PowerShell Tee-Object writes UTF-16
    text = data.decode("utf-16", errors="ignore") if data[:2] == b"\xff\xfe" else data.decode("utf-8", "ignore")
    for raw in text.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        hero = ast.literal_eval(m["hero"])
        board = ast.literal_eval(m["board"])
        stack = float(m["stack"])
        if hands and hands[-1]["hero"] == hero:
            h = hands[-1]
        else:
            h = {"time": m["time"], "hero": hero, "stack": stack, "board": [], "actions": []}
            hands.append(h)
        if len(board) >= len(h["board"]):
            h["board"] = board
        street = {0: "pre", 3: "flop", 4: "turn", 5: "river"}.get(len(board), "?")
        h["actions"].append(f"{street}:{m['action'].lower()}")
    for h, nxt in zip(hands, hands[1:]):
        delta = nxt["stack"] - h["stack"]
        ok = h["stack"] > 0 and abs(delta) <= 0.5 * h["stack"]
        h["delta"] = delta if ok else None
    if hands:
        hands[-1]["delta"] = None
    return hands


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", nargs="?", default="runs/live_play/console.log")
    p.add_argument("--csv", default="runs/live_play/hands.csv")
    args = p.parse_args(argv)

    hands = parse(Path(args.log))
    scored = [h for h in hands if h["delta"] is not None]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "hero", "board", "actions", "stack", "result"])
        for h in hands:
            w.writerow([h["time"], " ".join(h["hero"]), " ".join(h["board"]),
                        " ".join(h["actions"]), int(h["stack"]),
                        "" if h["delta"] is None else int(h["delta"])])

    total = sum(h["delta"] for h in scored)
    print(f"hands {len(hands)}, scored {len(scored)}, total {total:+.0f}")

    by_cat: dict[str, list[float]] = defaultdict(list)
    by_end: dict[str, list[float]] = defaultdict(list)
    for h in scored:
        by_cat[category(h["hero"])].append(h["delta"])
        by_end[h["actions"][-1]].append(h["delta"])
    print("\nby hole cards:")
    for k, v in sorted(by_cat.items(), key=lambda kv: sum(kv[1])):
        print(f"  {k:20s} hands {len(v):3d}  total {sum(v):+8.0f}  avg {sum(v) / len(v):+7.0f}")
    print("\nby last action:")
    for k, v in sorted(by_end.items(), key=lambda kv: sum(kv[1])):
        print(f"  {k:20s} hands {len(v):3d}  total {sum(v):+8.0f}")
    print("\nbiggest losses:")
    for h in sorted(scored, key=lambda h: h["delta"])[:6]:
        print(f"  {h['time']} {' '.join(h['hero']):6s} board {' '.join(h['board']):15s} "
              f"{' '.join(h['actions'])}  {h['delta']:+.0f}")
    print("\nbiggest wins:")
    for h in sorted(scored, key=lambda h: -h["delta"])[:6]:
        print(f"  {h['time']} {' '.join(h['hero']):6s} board {' '.join(h['board']):15s} "
              f"{' '.join(h['actions'])}  {h['delta']:+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
