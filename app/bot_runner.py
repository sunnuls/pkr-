"""BotRunner — runs tools/play_phone.py per phone for the Poker Bots app.

Each phone gets its own process (a crash on one phone does not touch the others) and
its own folder runs/live_play/<serial> with console.log, report.log, status.json,
latest.jpg and the PAUSE / STOPPED flags. A crashed bot is restarted after 3 s unless
it was stopped by hand or by its stop-loss / take-profit (exit code 3 or STOPPED).
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Deque, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "live_play"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _hidden_kwargs() -> dict:
    if sys.platform != "win32":
        return {}
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": NO_WINDOW, "startupinfo": info}


def _python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe":
        windowless = exe.with_name("pythonw.exe")
        if windowless.exists():
            return str(windowless)
    return str(exe)
LOG_KEEP = 600
RESTART_DELAY = 3.0


class _Bot:
    def __init__(self, serial: str):
        self.serial = serial
        self.dir = RUNS / serial
        self.dir.mkdir(parents=True, exist_ok=True)
        self.proc: Optional[subprocess.Popen] = None
        self.want_running = False
        self.tap = True
        self.restart_at = 0.0
        self.started_at = 0.0
        self.lines: Deque[tuple[int, str]] = collections.deque(maxlen=LOG_KEEP)
        self.seq = 0
        self.lock = threading.Lock()

    def log(self, line: str) -> None:
        with self.lock:
            self.seq += 1
            self.lines.append((self.seq, line))

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class BotRunner:
    def __init__(self) -> None:
        self._bots: Dict[str, _Bot] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._watch, daemon=True).start()

    def _bot(self, serial: str) -> _Bot:
        with self._lock:
            if serial not in self._bots:
                self._bots[serial] = _Bot(serial)
            return self._bots[serial]

    # --- control ---------------------------------------------------------
    def start(self, serial: str, tap: bool, reset_stop: bool = False) -> dict:
        bot = self._bot(serial)
        stopped = bot.dir / "STOPPED"
        if stopped.exists():
            if not reset_stop:
                return {"ok": False, "stopped": stopped.read_text(encoding="utf-8", errors="ignore").strip()}
            stopped.unlink()
        self.stop(serial)
        bot.tap = tap
        bot.want_running = True
        self._spawn(bot)
        return {"ok": True}

    def stop(self, serial: str) -> dict:
        bot = self._bot(serial)
        bot.want_running = False
        bot.restart_at = 0.0
        if bot.running:
            bot.proc.kill()
            bot.log(f"--- остановлен {time.strftime('%H:%M:%S')}")
        bot.proc = None
        return {"ok": True}

    def toggle_pause(self, serial: str) -> dict:
        flag = self._bot(serial).dir / "PAUSE"
        if flag.exists():
            flag.unlink()
            return {"ok": True, "paused": False}
        flag.write_text("paused from Poker Bots app\n", encoding="utf-8")
        return {"ok": True, "paused": True}

    def stop_all(self) -> None:
        for serial in list(self._bots):
            self.stop(serial)

    def _spawn(self, bot: _Bot) -> None:
        cmd = [_python(), "-u", str(ROOT / "tools" / "play_phone.py"), "--serial", bot.serial,
               "--save-dir", str(bot.dir), "--log", str(bot.dir / "bot.log"),
               "--interval", "0.2"]
        if bot.tap:
            cmd.append("--tap")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        bot.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", env=env,
                                    **_hidden_kwargs())
        bot.started_at = time.time()
        bot.log(f"--- запущен ({'играет' if bot.tap else 'только смотрит'}) {time.strftime('%H:%M:%S')}")
        threading.Thread(target=self._read, args=(bot, bot.proc), daemon=True).start()

    def _read(self, bot: _Bot, proc: subprocess.Popen) -> None:
        with open(bot.dir / "console.log", "a", encoding="utf-8") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                bot.log(line.rstrip("\n"))

    def _watch(self) -> None:
        while True:
            time.sleep(1.0)
            for bot in list(self._bots.values()):
                if bot.want_running and bot.proc is not None and bot.proc.poll() is not None:
                    code = bot.proc.returncode
                    bot.proc = None
                    stopped = bot.dir / "STOPPED"
                    if code == 3 or stopped.exists():
                        bot.want_running = False
                        reason = stopped.read_text(encoding="utf-8", errors="ignore").strip() if stopped.exists() else ""
                        bot.log(f"--- STOP: {reason}")
                    else:
                        bot.log(f"--- упал (код {code}), перезапуск через {RESTART_DELAY:.0f} с")
                        bot.restart_at = time.time() + RESTART_DELAY
                if bot.want_running and bot.proc is None and bot.restart_at and time.time() >= bot.restart_at:
                    bot.restart_at = 0.0
                    self._spawn(bot)

    # --- status ----------------------------------------------------------
    def status(self, serial: str) -> dict:
        bot = self._bot(serial)
        try:
            st = json.loads((bot.dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        stopped = bot.dir / "STOPPED"
        stack, start = st.get("stack") or 0, st.get("session_start") or 0
        if bot.running:
            age = time.time() - st.get("time", 0)
            if (bot.dir / "PAUSE").exists():
                state = "paused"
            elif age > 15 and time.time() - bot.started_at < 60:
                state = "starting"
            elif age > 15:
                state = "no_frames"
            elif st.get("our_turn"):
                state = "our_turn"
            else:
                state = "playing" if bot.tap else "watching"
        elif bot.want_running:
            state = "restarting"
        else:
            state = "stopped_limit" if stopped.exists() else "stopped"
        return {
            "serial": serial,
            "state": state,
            "tap": bot.tap,
            "paused": (bot.dir / "PAUSE").exists(),
            "stop_reason": stopped.read_text(encoding="utf-8", errors="ignore").strip() if stopped.exists() else "",
            "stack": stack,
            "session_start": start,
            "session_result": (stack - start) if stack and start else 0,
            "hands": st.get("hands", 0),
            "hero": st.get("hero", []),
            "board": st.get("board", []),
            "position": st.get("position", ""),
            "pot": st.get("pot", 0),
            "last_decision": st.get("last_decision", ""),
            "has_frame": (bot.dir / "latest.jpg").exists(),
            "frame_time": st.get("time", 0),
        }

    def log_since(self, serial: str, since: int) -> dict:
        bot = self._bot(serial)
        with bot.lock:
            lines = [(n, s) for n, s in bot.lines if n > since]
        if since == 0 and not lines:
            tail = _tail(bot.dir / "console.log", 120)
            lines = [(0, s) for s in tail]
        return {"seq": bot.seq, "lines": [s for _, s in lines]}

    def frame_path(self, serial: str) -> Path:
        return self._bot(serial).dir / "latest.jpg"


def _tail(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 64_000))
            data = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    return data.splitlines()[-n:]
