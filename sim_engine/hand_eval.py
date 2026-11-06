"""Exact Texas Hold'em hand ranking and Monte Carlo equity vs random hands."""

from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS]
_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}


def _rank5(cards: Sequence[str]) -> Tuple[int, ...]:
    values = sorted((_VALUE[c[0]] for c in cards), reverse=True)
    counts = Counter(values)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5
    if straight_high and flush:
        return (8, straight_high)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    ordered = [v for v, _ in groups]
    if shape[0] == 4:
        return (7, *ordered)
    if shape[:2] == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *values)
    if straight_high:
        return (4, straight_high)
    if shape[0] == 3:
        return (3, *ordered)
    if shape[:2] == [2, 2]:
        return (2, *ordered)
    if shape[0] == 2:
        return (1, *ordered)
    return (0, *values)


def best_rank(cards: Sequence[str]) -> Tuple[int, ...]:
    """Best 5-card rank tuple from 5–7 cards; higher tuple wins."""
    return max(_rank5(combo) for combo in combinations(cards, 5))


def _preflop_score(hole: Sequence[str]) -> Tuple[int, ...]:
    hi, lo = sorted((_VALUE[hole[0][0]], _VALUE[hole[1][0]]), reverse=True)
    if hi == lo:
        return (40 + 2 * hi,)
    suited = 4 if hole[0][1] == hole[1][1] else 0
    gap = hi - lo
    return (2 * hi + lo + suited - (0 if gap == 1 else min(gap, 5)),)


def _has_draw(cards: Sequence[str]) -> bool:
    """Flush draw (4 of a suit) or open-ended straight draw among the cards."""
    if max(Counter(c[1] for c in cards).values()) == 4:
        return True
    vals = {_VALUE[c[0]] for c in cards}
    if 14 in vals:
        vals.add(1)
    return any(all(v + k in vals for k in range(4)) for v in range(2, 11))


def hand_strength(hole: Sequence[str], board: Sequence[str]) -> Tuple[int, ...]:
    """Comparable strength of a holding on the current street (draws count as a middle pair)."""
    if len(board) < 3:
        return _preflop_score(hole)
    made = best_rank(list(hole) + list(board))
    if len(board) < 5 and made[0] < 1 and _has_draw(list(hole) + list(board)):
        return (1, 9)
    return made


def _range_threshold(
    board: Sequence[str], deck: Sequence[str], top: float, rng: random.Random, samples: int = 300
) -> Tuple[int, ...]:
    scores = sorted(hand_strength(rng.sample(deck, 2), board) for _ in range(samples))
    return scores[min(len(scores) - 1, int((1.0 - top) * len(scores)))]


def equity(
    hero: Sequence[str],
    board: Sequence[str],
    opponents: int = 1,
    iterations: int = 500,
    rng: Optional[random.Random] = None,
    range_top: float = 1.0,
) -> float:
    """Share of the pot hero wins (ties split).

    ``range_top`` < 1 restricts opponents to the strongest share of holdings on the
    current street (e.g. 0.3 = top 30%), modelling players who bet or call.
    ``hero`` may hold fewer than 2 cards; missing hole cards are drawn at random.
    """
    rng = rng or random.Random()
    known = set(hero) | set(board)
    deck = [c for c in DECK if c not in known]
    missing = 2 - len(hero)
    need = 5 - len(board)
    threshold = _range_threshold(board, deck, range_top, rng) if range_top < 1.0 else None
    total = 0.0
    for _ in range(iterations):
        pool = deck[:]
        rng.shuffle(pool)
        mine_hole = list(hero) + pool[:missing]
        pool = pool[missing:]
        opp_holes = []
        for _i in range(opponents):
            hole = pool[:2]
            if threshold is not None:
                for _try in range(60):
                    if hand_strength(hole, board) >= threshold:
                        break
                    rng.shuffle(pool)
                    hole = pool[:2]
            opp_holes.append(hole)
            pool = pool[2:]
        full = list(board) + pool[:need]
        mine = best_rank(mine_hole + full)
        best_opp = max(best_rank(h + full) for h in opp_holes)
        if mine > best_opp:
            total += 1.0
        elif mine == best_opp:
            total += 0.5
    return total / iterations


def made_hand_name(cards: List[str]) -> str:
    names = ["high card", "pair", "two pair", "trips", "straight",
             "flush", "full house", "quads", "straight flush"]
    return names[best_rank(cards)[0]]
