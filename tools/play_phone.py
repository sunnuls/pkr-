"""Play one table on a phone over ADB with the solo vision bot.

You open the table yourself. The script screenshots the phone, reads the
table and, when action buttons appear, picks fold / check / call and taps it.

Without --tap it only prints what it would press.

  python tools/play_phone.py --serial e2d0507          # watch only
  python tools/play_phone.py --serial e2d0507 --tap    # really taps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge.emulator.adb_backend import ADBBackend  # noqa: E402
from bridge.safety import SafetyConfig, SafetyFramework, SafetyMode  # noqa: E402
from bridge.vision.yolo_engine import VisionEngine, find_words  # noqa: E402
from sim_engine.vision_decision import decide  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", required=True)
    p.add_argument("--tap", action="store_true", help="really tap the phone")
    p.add_argument("--interval", type=float, default=0.2)
    p.add_argument("--save-dir", default="", help="save a screenshot of every decision")
    p.add_argument("--log", default="", help="write warnings and errors to this file")
    p.add_argument("--stop-loss", type=float, default=0.5,
                   help="stop when the stack falls below this share of the session start")
    p.add_argument("--take-profit", type=float, default=1.5,
                   help="stop when the stack reaches this multiple of the session start")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global STOP_FILE
    args = parse_args(argv)
    if args.save_dir:
        STOP_FILE = Path(args.save_dir) / "STOPPED"
    if STOP_FILE.exists():
        print(f"{STOP_FILE} exists ({STOP_FILE.read_text().strip()}) - delete it to play again")
        return 3
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        filename=args.log or None,
    )
    mode = SafetyMode.UNSAFE if args.tap else SafetyMode.DRY_RUN
    safety = SafetyFramework(SafetyConfig(mode=mode, enable_kill_switch=False))
    phone = ADBBackend(serial=args.serial, safety=safety)
    engine = VisionEngine.from_config(
        yaml.safe_load(open("config/bots.yaml", encoding="utf-8"))
    )
    engine.fast_text = True
    if engine.detector._stub or engine.classifier._stub:
        # a failed torch load leaves the bot blind (no turns → sat out for inactivity)
        print("model load failed - exiting so the runner restarts", flush=True)
        return 1
    print(f"phone={args.serial} tap={'ON' if args.tap else 'off (watch only)'}  Ctrl+C to stop")
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    last_key = None
    turn = 0
    while True:
        try:
            last_key, turn = _step(args, phone, engine, save_dir, last_key, turn)
        except (Exception, SystemExit):
            print("step error:\n" + traceback.format_exc(), flush=True)
            time.sleep(2)


session = time.strftime("%m%d_%H%M")
# No turn for this long usually means a win/info popup covers the table.
IDLE_TAP_SEC = 60
JOIN_CHECK_SEC = 20
last_action_at = time.time()
last_join_check = 0.0
unread_retries = 0
# Written on stop-loss / take-profit; the runner script will not restart while it exists.
STOP_FILE = Path("runs/live_play/STOPPED")
session_start_stack = 0.0

# Bet panel presets (bottom row after tapping Bet), as fractions of the screen.
PRESET_X = {"third": 0.121, "half": 0.281, "three_quarter": 0.432, "pot": 0.585}
PRESET_Y = 0.936


def _is_preaction(image, xy) -> bool:
    """Pre-action buttons (X/F, Call N, Call Any) shown during others' turns are dark grey;
    our real Fold button is red."""
    x, y = xy
    patch = image[max(0, y - 12): y + 12, max(0, x - 30): x + 30]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) < 60


hand_pot = {"hero": None, "board": 0, "pot": 0.0}
# (hole cards, board size) where we already bet or raised
aggressor_streets: set = set()


def _fix_pot(state):
    """Within one hand the pot only grows; a smaller OCR read is a misread."""
    hero = tuple(state.hero_cards)
    board = len(state.board_cards)
    same_hand = hero and hero == hand_pot["hero"] and board >= hand_pot["board"]
    prev = hand_pot["pot"]
    jump_limit = 4 * prev + 2 * state.call_amount + 2000
    if same_hand and (state.pot < prev or state.pot > jump_limit):
        state.pot = prev
    stack = state.hero_stack or hand_pot.get("stack", 0.0)
    if not same_hand and stack and state.pot > 3 * stack + 2 * state.call_amount:
        # new hand: a pot several times our stack is a misread (missing decimal point, merged digits)
        state.pot = 0.0
    if hero:
        hand_pot.update(hero=hero, board=board, pot=max(state.pot, hand_pot["pot"] if same_hand else 0.0))
    # A zero or a sudden drop below 20% is usually an OCR slip. Keep the last real stack
    # so the session total does not fall to 0 between reads.
    last = hand_pot.get("stack", 0.0)
    if last and state.hero_stack <= 0:
        state.hero_stack = last
    elif last and 0 < state.hero_stack < 0.2 * last and hand_pot.get("low_reads", 0) < 3:
        hand_pot["low_reads"] = hand_pot.get("low_reads", 0) + 1
        state.hero_stack = last
    elif state.hero_stack > 0:
        hand_pot.update(stack=state.hero_stack, low_reads=0)


REPORT_EVERY = 5
hands: list = []  # finished hands: {"hero", "board", "actions", "start", "end"}
current_hand: dict = {}


def _track_hand(state, choice, save_dir):
    """Group decisions into hands; every REPORT_EVERY finished hands print and log a report.
    A hand's result is the stack at our first decision of the next hand minus at this one."""
    hero = tuple(state.hero_cards)
    if not hero:
        return
    if current_hand.get("hero") != hero:
        if current_hand and state.hero_stack > 0:
            hands.append(dict(current_hand, end=state.hero_stack))
            if len(hands) % REPORT_EVERY == 0:
                _report(save_dir)
        current_hand.clear()
        current_hand.update(hero=hero, board=(), actions=[], start=state.hero_stack)
    if len(state.board_cards) >= len(current_hand["board"]):
        current_hand["board"] = tuple(state.board_cards)
    street = {0: "pre", 3: "flop", 4: "turn", 5: "river"}.get(len(state.board_cards), "?")
    act = f"{street}:{choice.decision.action.value}"
    if not current_hand["actions"] or current_hand["actions"][-1] != act:
        current_hand["actions"].append(act)


def _report(save_dir):
    last = hands[-REPORT_EVERY:]
    delta = [h["end"] - h["start"] for h in last]
    lines = [
        f"=== report {time.strftime('%H:%M:%S')}: {len(hands)} hands, "
        f"session {hands[-1]['end'] - session_start_stack:+.0f} "
        f"(stack {hands[-1]['end']:.0f} from {session_start_stack:.0f}), last {REPORT_EVERY}: {sum(delta):+.0f}"
    ]
    for h, d in zip(last, delta):
        lines.append(
            f"   {' '.join(h['hero']):6s} board {' '.join(h['board']):15s} "
            f"{' '.join(h['actions']):40s} {d:+.0f}"
        )
    text = "\n".join(lines)
    print(text, flush=True)
    if save_dir:
        with open(save_dir / "report.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")


last_decision = ""


def _publish(args, save_dir, image, state, choice, paused):
    """Latest frame and status for the dashboard (written atomically)."""
    global last_decision
    if not save_dir:
        return
    if choice is not None:
        last_decision = (
            f"{time.strftime('%H:%M:%S')} {choice.decision.action.value.upper()} "
            f"| {choice.decision.reasoning[:70]}"
        )
    h, w = image.shape[:2]
    small = cv2.resize(image, (360, int(h * 360 / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        tmp = save_dir / "latest.tmp.jpg"
        tmp.write_bytes(buf.tobytes())
        tmp.replace(save_dir / "latest.jpg")
    status = {
        "serial": args.serial,
        "time": time.time(),
        "tap": args.tap,
        "paused": paused,
        "hero": state.hero_cards,
        "board": state.board_cards,
        "position": getattr(state, "hero_blind", "") or "",
        "pot": state.pot,
        "stack": state.hero_stack,
        "session_start": session_start_stack,
        "hands": len(hands),
        "our_turn": choice is not None,
        "last_decision": last_decision,
    }
    tmp = save_dir / "status.tmp.json"
    tmp.write_text(json.dumps(status), encoding="utf-8")
    tmp.replace(save_dir / "status.json")


def _bet(phone, shape, choice):
    """Open the bet panel, pick a pot-fraction preset, confirm with the same button."""
    h, w = shape[:2]
    preset = (int(w * PRESET_X[choice.size]), int(h * PRESET_Y))
    phone.click(*choice.xy)
    time.sleep(0.35)
    phone.click(*preset)
    time.sleep(0.2)
    ok = phone.click(*choice.xy)
    print(f"   bet {choice.size}: open {choice.xy} preset {preset} confirm -> {ok}", flush=True)


def _read_frame(phone, engine):
    image = phone.capture()
    if image is None:
        return None, None
    state = engine.get_game_state(image)
    if "fold" in state.buttons and _is_preaction(image, state.buttons["fold"]):
        state.buttons = {}
    _fix_pot(state)
    return image, state


def _step(args, phone, engine, save_dir, last_key, turn):
    """One screenshot → decision → tap. Returns updated (last_key, turn)."""
    image, state = _read_frame(phone, engine)
    if image is None:
        print("no screenshot — is the phone connected?", flush=True)
        time.sleep(3)
        return last_key, turn
    paused = bool(save_dir and (save_dir / "PAUSE").exists())
    if paused:
        # paused from the dashboard: keep watching, never tap
        state.buttons = {}
    street = (tuple(state.hero_cards), len(state.board_cards))
    state.we_were_aggressor = street in aggressor_streets
    choice = decide(state)
    if choice is not None and choice.size:
        aggressor_streets.add(street)
    if choice is None:
        global last_action_at, last_join_check
        if state.buttons:
            # buttons are up: never poke the table center while a decision is due
            last_action_at = time.time()
        idle = time.time() - last_action_at
        if args.tap and idle > JOIN_CHECK_SEC and time.time() - last_join_check > JOIN_CHECK_SEC:
            last_join_check = time.time()
            # sat out for inactivity: the table shows a "Join now" button
            # ("Join Similar" sits in the top bar and is excluded by the height check)
            xy = find_words(image, ("join", "rejoin"))
            if xy and xy[1] > image.shape[0] * 0.15:
                if save_dir:
                    cv2.imwrite(str(save_dir / f"join_{int(time.time())}.png"), image)
                phone.click(*xy)
                print(f"   sat out: tapped join at {xy}", flush=True)
                time.sleep(2)
                return None, turn
        if args.tap and idle > IDLE_TAP_SEC:
            h, w = image.shape[:2]
            if save_dir:
                cv2.imwrite(str(save_dir / f"idle_{int(time.time())}.png"), image)
            phone.click(w // 2, int(h * 0.45))
            print(f"   idle {IDLE_TAP_SEC}s: tapped table center to close popups", flush=True)
            last_action_at = time.time()
        time.sleep(0.05)
        try:
            _publish(args, save_dir, image, state, None, paused)
        except OSError:
            pass
        return None, turn
    global unread_retries, session_start_stack
    if choice.unread and unread_retries < 1:
        unread_retries += 1
        print(f"   check/call label unread, re-reading ({unread_retries})", flush=True)
        return last_key, turn
    unread_retries = 0

    if state.hero_stack > 0:
        if not session_start_stack:
            session_start_stack = state.hero_stack
        elif state.hero_stack > session_start_stack * 5:
            # the first read was a fragment (198 instead of 196000), not a win
            session_start_stack = state.hero_stack
        ratio = state.hero_stack / session_start_stack
        if ratio < args.stop_loss or ratio >= args.take_profit:
            kind = "STOP-LOSS" if ratio < args.stop_loss else "TAKE-PROFIT"
            msg = f"{kind}: stack {state.hero_stack:.0f} from session start {session_start_stack:.0f} ({ratio:.0%})"
            STOP_FILE.write_text(msg + "\n")
            print(msg + " - stopping, leave the table", flush=True)
            raise KeyboardInterrupt

    # Press before any screenshot save. The timer does not wait for the log.
    last_action_at = time.time()
    if args.tap and choice.size:
        _bet(phone, image.shape, choice)
    elif args.tap:
        ok = phone.click(*choice.xy)
        print(f"   tap {choice.button} {choice.xy} -> {ok}", flush=True)

    _track_hand(state, choice, save_dir)
    key = (tuple(state.hero_cards), tuple(state.board_cards), state.call_amount)
    if key != last_key:
        turn += 1
        if save_dir:
            cv2.imwrite(
                str(save_dir / f"{session}_{turn:04d}_{choice.decision.action.value}.png"), image
            )
        print(
            f"#{turn} {time.strftime('%H:%M:%S')} hero={state.hero_cards} "
            f"board={state.board_cards} stack={state.hero_stack:.0f} pot={state.pot:.0f} "
            f"call={state.call_amount:.0f} -> "
            f"{choice.decision.action.value.upper()} [{choice.button} {choice.xy}] "
            f"| {choice.decision.reasoning[:80]}",
            flush=True,
        )
    try:
        _publish(args, save_dir, image, state, choice, paused)
    except OSError:
        pass
    if args.tap:
        time.sleep(0.15)
    return key, turn


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("stopped")
