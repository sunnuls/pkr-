"""Single-bot decision from a VisionEngine GameState (mobile table screenshot).

Uses only what the bot itself sees: its own hole cards, the board, pot,
stacks, how many opponents still hold cards, and the action buttons on
screen. Returns which on-screen button to tap.

Strategy:
- free to continue → Bet a pot fraction with a strong hand, otherwise Check;
- facing a bet → Raise with a very strong hand, Call when Monte Carlo equity
  against the opponents still in the hand beats the price (pot odds plus a
  margin), else Fold;
- a hole card read with low confidence is treated as unknown and drawn at
  random in the simulation instead of trusting a likely misread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sim_engine.hand_eval import best_rank
from sim_engine.hand_eval import equity as mc_equity
from sim_engine.poker_ai import AIDecision, PokerAction

logger = logging.getLogger(__name__)

MIN_CARD_CONF = 0.70


def _bettor_range(to_call: float, pot: float, board_len: int) -> float:
    """Share of holdings a player who just bet `to_call` into `pot` plausibly has:
    bigger bets and later streets mean a narrower, stronger range."""
    before = max(pot - to_call, 1.0)
    ratio = min(to_call / before, 1.5)
    base = {0: 0.45, 3: 0.55, 4: 0.45}.get(board_len, 0.40)
    return max(0.12, base - 0.20 * ratio)


# Range of hands that call a bet, by board size; we value-bet when ahead of it.
CALLING_RANGE = {3: 0.50, 4: 0.45, 5: 0.40}


@dataclass
class VisionAction:
    """Button to tap and the decision behind it."""

    button: str
    xy: Tuple[int, int]
    decision: AIDecision
    # Bet panel preset to pick before confirming ("third", "half", "three_quarter", "pot").
    size: Optional[str] = None
    # The check/call label was not read; the caller should re-read before acting on it.
    unread: bool = False


def _bet_size(eq: float, opponents: int, board_len: int, facing_bet: bool, pos: str = "") -> Optional[str]:
    """Pot fraction to bet/raise, or None to just check/call.

    Preflop eq is vs random hands; postflop eq is vs the range that would call (when
    checking to us) or the bettor's range (when facing a bet).
    """
    if board_len < 3:
        strong = min(0.62, max(0.30, 2.0 / (opponents + 1)))
        if eq >= strong * (1.15 if facing_bet else 1.0):
            return "half"
        return None
    # Small blind is first to act after the flop, so it bets only a stronger hand.
    # The button acts last and can bet thinner.
    half, big = 0.55, 0.70
    if pos == "sb":
        half, big = 0.64, 0.78
    elif pos == "btn":
        half, big = 0.50, 0.68
    if facing_bet:
        return "three_quarter" if eq >= big else None
    if eq >= big:
        return "three_quarter"
    if eq >= half:
        return "half"
    return None


def legal_actions(state: Any) -> List[str]:
    buttons = getattr(state, "buttons", {}) or {}
    text = {k: (v or "").lower() for k, v in (getattr(state, "button_text", {}) or {}).items()}
    legal: List[str] = []
    if "fold" in buttons:
        legal.append("fold")
    if "check_call" in buttons:
        cc = text.get("check_call", "")
        if "call" in cc or any(ch.isdigit() for ch in cc) or getattr(state, "call_amount", 0) > 0:
            legal.append("call")
        elif "chec" in cc or "heck" in cc:
            legal.append("check")
        else:
            # unread label: tapping it could be a big call, never assume a free check
            legal.append("unknown")
    if "bet" in buttons:
        if _is_allin_call(state):
            legal.append("call")
        else:
            legal.append("raise" if "raise" in text.get("bet", "") else "bet")
    return legal


def _is_allin_call(state: Any) -> bool:
    """Facing an all-in: no Check/Call button, the yellow button reads All-in."""
    buttons = getattr(state, "buttons", {}) or {}
    bet_text = ((getattr(state, "button_text", {}) or {}).get("bet") or "").lower()
    return "check_call" not in buttons and "all" in bet_text


def _trusted_hole_cards(state: Any, board: List[str]) -> List[str]:
    cards = list(getattr(state, "hero_cards", []) or [])
    confs = list(getattr(state, "hero_card_conf", []) or [])
    kept = [
        c for i, c in enumerate(cards[:2])
        if (confs[i] if i < len(confs) else 1.0) >= MIN_CARD_CONF and c not in board
    ]
    return kept if len(set(kept)) == len(kept) else kept[:1]


def _playable_preflop(hero: List[str]) -> bool:
    """Pairs, two broadways, suited aces, suited connectors (one-gappers from 7 up)."""
    order = "23456789TJQKA"
    if len(hero) != 2:
        return False
    (r1, s1), (r2, s2) = hero[0], hero[1]
    hi, lo = sorted((order.index(r1), order.index(r2)), reverse=True)
    suited = s1 == s2
    if hi == lo or lo >= order.index("T"):
        return True
    if suited and (hi == order.index("A") or (hi - lo <= 2 and lo >= order.index("5"))):
        return True
    return hi == order.index("A") and lo >= order.index("9")


def _bb_defend(hero: List[str]) -> bool:
    """Big blind already paid; defend suited hands, any ace, pairs and broadways."""
    if _playable_preflop(hero):
        return True
    if len(hero) != 2:
        return False
    (r1, s1), (r2, s2) = hero[0], hero[1]
    return s1 == s2 or "A" in (r1, r2)


def _preflop_call_limit(hero: List[str], stack: float, pot: float, pos: str = "") -> float:
    """Largest preflop call taken without an equity check (cards are cheap to see a flop)."""
    if stack <= 0:
        return pot
    if pos == "sb":
        # completing the small blind with junk is a losing play; raise or fold instead
        return 0.04 * stack if _playable_preflop(hero) else 0.0
    if pos == "bb" and _bb_defend(hero):
        return 0.10 * stack
    if _playable_preflop(hero):
        return 0.10 * stack
    if len(hero) == 1 and hero[0][0] in "AK":
        return 0.05 * stack
    return 0.03 * stack if pos != "bb" else 0.05 * stack


def _required_equity(pot_odds: float, to_call: float, stack: float, preflop: bool, allin: bool) -> float:
    margin = 0.02 if preflop else 0.04
    if allin or (stack > 0 and to_call > 0.3 * stack):
        margin += 0.06
    return pot_odds + margin


def decide(state: Any, iterations: int = 300) -> Optional[VisionAction]:
    """Pick a button for the current frame; None when it is not our turn."""
    buttons = getattr(state, "buttons", {}) or {}
    legal = legal_actions(state)
    allin_call = _is_allin_call(state)
    if not legal or "fold" not in buttons or ("check_call" not in buttons and not allin_call):
        return None
    call_unread = "call" in legal and not allin_call and not getattr(state, "call_amount", 0)
    if "unknown" in legal or call_unread:
        return VisionAction(
            button="fold",
            xy=tuple(buttons["fold"]),
            decision=AIDecision(action=PokerAction.FOLD, reasoning="unread check/call label"),
            unread=True,
        )

    board = list(getattr(state, "board_cards", []) or [])[:5]
    slots = int(getattr(state, "board_slots", 0) or 0)
    # 1–2 cards, or boxes we have not read yet, means the flop is still arriving.
    # Deciding then is a preflop fold of a hand that is about to be dealt.
    if slots in (1, 2) or len(board) in (1, 2) or (slots >= 3 and len(board) < slots):
        return None
    if len(set(board)) != len(board):
        return None
    hero = _trusted_hole_cards(state, board)
    pos = (getattr(state, "hero_blind", "") or "").lower()
    spot = {"sb": "малый блайнд", "bb": "большой блайнд", "btn": "батон"}.get(pos, "без блайнда")
    pot = float(getattr(state, "pot", 0.0) or 0.0)
    stack = float(getattr(state, "hero_stack", 0.0) or 0.0)
    opponents = max(1, min(5, int(getattr(state, "opponents_in_hand", 0) or 0)))

    preflop = not board
    known = "".join(hero) or "??"
    cards = f"cards={known}{'' if len(hero) == 2 else '+?'}"
    can_bet = "bet" in buttons and not allin_call and len(hero) == 2
    raw_eq = mc_equity(hero, board, opponents=opponents, iterations=iterations) if hero else 0.0

    if "check" in legal:
        bet_eq = raw_eq
        if can_bet and not preflop:
            calling = CALLING_RANGE.get(len(board), 0.40)
            bet_eq = mc_equity(hero, board, opponents=opponents, iterations=iterations, range_top=calling)
        size = _bet_size(bet_eq, opponents, len(board), facing_bet=False, pos=pos) if can_bet else None
        if size:
            return VisionAction(
                button="bet",
                xy=tuple(buttons["bet"]),
                decision=AIDecision(
                    action=PokerAction.BET,
                    reasoning=f"bet {size} eq={bet_eq:.0%} vs {opponents} callers {cards}",
                    confidence=bet_eq,
                ),
                size=size,
            )
        return VisionAction(
            button="check_call",
            xy=tuple(buttons["check_call"]),
            decision=AIDecision(
                action=PokerAction.CHECK,
                reasoning=f"free check {spot} eq={bet_eq:.0%} vs {opponents} {cards}",
            ),
        )

    if allin_call:
        to_call = stack or pot or 1.0
    else:
        to_call = float(getattr(state, "call_amount", 0.0) or 0.0)
    if pot < to_call:
        # pot label read before the latest bet was added to it
        pot += to_call
    pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 1.0

    top = 0.12 if allin_call else _bettor_range(to_call, pot, len(board))
    reraised = bool(getattr(state, "we_were_aggressor", False))
    if reraised:
        # we bet or raised this street and got raised back: their range is strong
        top = max(0.08, top * 0.5)
        can_bet = False
    eq = mc_equity(hero, board, opponents=opponents, iterations=iterations, range_top=top) if hero else 0.0
    need = _required_equity(pot_odds, to_call, stack, preflop=preflop, allin=allin_call)
    if not preflop and pos == "sb":
        need += 0.05
    elif not preflop and pos == "btn":
        need = max(0.0, need - 0.03)
    reason = (
        f"{spot} eq={eq:.0%} need={need:.0%} (odds {pot_odds:.0%}) vs {opponents} top{top:.0%} {cards}"
        + (" reraised" if reraised else "")
    )

    size = _bet_size(raw_eq if preflop else eq, opponents, len(board), facing_bet=True, pos=pos) if can_bet else None
    if size and eq >= need:
        return VisionAction(
            button="bet",
            xy=tuple(buttons["bet"]),
            decision=AIDecision(action=PokerAction.RAISE, reasoning=f"raise {size} " + reason, confidence=eq),
            size=size,
        )

    if (
        preflop and not allin_call and not reraised
        and to_call <= _preflop_call_limit(hero, stack, pot, pos)
        and to_call <= 0.5 * pot
    ):
        reason = f"cheap flop ({to_call:.0f} <= limit) " + reason
        eq = max(eq, need)

    if (
        len(board) >= 3 and len(hero) == 2 and not allin_call
        and best_rank(hero + board)[0] >= 3
        and (stack <= 0 or to_call < 0.45 * stack)
    ):
        # trips or better: a normal bet is a call, the hand is not a fold
        eq = max(eq, need)
        reason = "trips+ " + reason
    if eq >= need:
        button = "bet" if allin_call else "check_call"
        return VisionAction(
            button=button,
            xy=tuple(buttons[button]),
            decision=AIDecision(action=PokerAction.CALL, reasoning="call " + reason, confidence=eq),
        )
    return VisionAction(
        button="fold",
        xy=tuple(buttons["fold"]),
        decision=AIDecision(action=PokerAction.FOLD, reasoning="fold " + reason, confidence=1 - eq),
    )
