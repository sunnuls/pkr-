"""Poker Bots — desktop app to run and watch the phone bots.

  python app/main.py        (or double-click "Poker Bots.bat")

One card per connected Android phone: live screen (click it to tap the phone), bot
state, stack and session result, current hand, last decision, log and 5-hand reports.
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import webview

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.bot_runner import NO_WINDOW, BotRunner  # noqa: E402

UI = Path(__file__).resolve().parent / "ui" / "index.html"
PREVIEW_EVERY = 2.0  # seconds between screenshots of phones whose bot is not running


def _adb(*args: str, timeout: float = 8.0, binary: bool = False):
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    try:
        out = subprocess.run(["adb", *args], capture_output=True, timeout=timeout,
                             creationflags=NO_WINDOW, startupinfo=info).stdout
    except (OSError, subprocess.SubprocessError):
        return b"" if binary else ""
    return out if binary else out.decode("utf-8", errors="replace")


class Phones:
    """Connected phones, their model / resolution and preview frames."""

    def __init__(self, runner: BotRunner):
        self.runner = runner
        self.info: Dict[str, dict] = {}
        self.preview: Dict[str, tuple[float, bytes]] = {}
        self.serials: list[str] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        last_list = 0.0
        while True:
            if time.time() - last_list > 4:
                last_list = time.time()
                out = _adb("devices")
                self.serials = [l.split()[0] for l in out.splitlines()[1:] if l.strip().endswith("device")]
                for s in self.serials:
                    if s not in self.info:
                        size = re.search(r"(\d+)x(\d+)", _adb("-s", s, "shell", "wm", "size"))
                        self.info[s] = {
                            "model": _adb("-s", s, "shell", "getprop", "ro.product.model").strip() or s,
                            "w": int(size.group(1)) if size else 0,
                            "h": int(size.group(2)) if size else 0,
                        }
            for s in list(self.serials):
                if self.runner.status(s)["state"] in ("stopped", "stopped_limit"):
                    ts, _ = self.preview.get(s, (0.0, b""))
                    if time.time() - ts > PREVIEW_EVERY:
                        png = _adb("-s", s, "exec-out", "screencap", "-p", binary=True, timeout=6)
                        if png:
                            self.preview[s] = (time.time(), _to_jpeg(png))
            time.sleep(0.3)


def _to_jpeg(png: bytes) -> bytes:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return b""
    h, w = img.shape[:2]
    img = cv2.resize(img, (360, int(h * 360 / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b""


class Api:
    """Methods callable from the UI as window.pywebview.api.<name>()."""

    def __init__(self):
        self.runner = BotRunner()
        self.phones = Phones(self.runner)

    def overview(self) -> list:
        rows = []
        for s in self.phones.serials:
            row = self.runner.status(s)
            row.update(self.phones.info.get(s, {"model": s, "w": 0, "h": 0}))
            rows.append(row)
        return rows

    def frame(self, serial: str, known: float = 0.0) -> Optional[dict]:
        """Newest screen as a data URL, or None when unchanged since `known`."""
        st = self.runner.status(serial)
        if st["state"] not in ("stopped", "stopped_limit"):
            path = self.runner.frame_path(serial)
            try:
                ts = path.stat().st_mtime
                if ts <= known:
                    return None
                data = path.read_bytes()
            except OSError:
                return None
        else:
            ts, data = self.phones.preview.get(serial, (0.0, b""))
            if not data or ts <= known:
                return None
        return {"ts": ts, "src": "data:image/jpeg;base64," + base64.b64encode(data).decode()}

    def logs(self, serial: str, since: int = 0) -> dict:
        return self.runner.log_since(serial, since)

    def reports(self, serial: str) -> str:
        path = self.runner.frame_path(serial).parent / "report.log"
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[-20000:]
        except OSError:
            return ""

    def start(self, serial: str, tap: bool = True, reset_stop: bool = False) -> dict:
        return self.runner.start(serial, tap=tap, reset_stop=reset_stop)

    def stop(self, serial: str) -> dict:
        return self.runner.stop(serial)

    def pause(self, serial: str) -> dict:
        return self.runner.toggle_pause(serial)

    def start_all(self) -> dict:
        # The NoviSea board is Android 6 and cannot run the table app.
        serials = [s for s in self.phones.serials if s != "0123456789ABCDEF"]
        return {s: self.runner.start(s, tap=True) for s in serials}

    def stop_all(self) -> dict:
        self.runner.stop_all()
        return {"ok": True}

    def tap(self, serial: str, fx: float, fy: float) -> dict:
        """Tap the phone at a fraction of its screen (clicked on the preview)."""
        info = self.phones.info.get(serial) or {}
        if not info.get("w"):
            return {"ok": False}
        x, y = int(fx * info["w"]), int(fy * info["h"])
        _adb("-s", serial, "shell", "input", "tap", str(x), str(y))
        return {"ok": True, "x": x, "y": y}


PLAYABLE = {"e2d0507", "b89dde68"}


def _watch_ready(api: Api) -> None:
    """Start each working phone in watch mode once, so two tables never share a screen."""
    started = set()
    while len(started) < len(PLAYABLE):
        for serial in list(api.phones.serials):
            if serial in PLAYABLE and serial not in started:
                api.runner.start(serial, tap=True)
                started.add(serial)
        time.sleep(1.0)


def main() -> None:
    api = Api()
    threading.Thread(target=_watch_ready, args=(api,), daemon=True).start()
    window = webview.create_window(
        "Poker Bots", str(UI), js_api=api, width=1400, height=860, min_size=(900, 600),
        background_color="#0a0d14", confirm_close=True, maximized=True,
    )
    window.events.closed += api.runner.stop_all
    webview.start()


if __name__ == "__main__":
    main()
