"""
YOLO vision engine — resolution-independent detection pipeline.

Stage 1: YOLODetector (object detection in letterboxed 640 space)
Stage 2: RankSuitClassifier (52-class card crop classification)
Facade:  VisionEngine (capture → detect → classify → GameState)

No absolute ROI / template matching. Coordinates map via LetterboxMapper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DETECT_CLASSES = [
    "hero_card",
    "board_card",
    "fold_button",
    "call_button",
    "raise_button",
    "check_button",
    "pot",
    "stack",
]


@dataclass
class GameState:
    """Structured vision output for one frame."""

    hero_cards: List[str] = field(default_factory=list)
    hero_card_conf: List[float] = field(default_factory=list)
    board_cards: List[str] = field(default_factory=list)
    buttons: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    pot_box: Optional[Tuple[int, int, int, int]] = None
    stack_box: Optional[Tuple[int, int, int, int]] = None
    enemy_stack_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    dealer_box: Optional[Tuple[int, int, int, int]] = None
    board_slots: int = 0
    # "sb", "bb", "btn" or "" when the hero is not in a blind and not on the button
    hero_blind: str = ""
    lobby_rows: List[dict] = field(default_factory=list)
    resolution: Tuple[int, int] = (0, 0)
    # Compatibility with BotInstance decision path
    is_bots_turn: bool = False
    pot: float = 0.0
    hero_stack: float = 0.0
    enemy_stacks: List[float] = field(default_factory=list)
    button_text: Dict[str, str] = field(default_factory=dict)
    call_amount: float = 0.0
    opponents_in_hand: int = 0
    hand_number: int = 0
    players: list = field(default_factory=list)

    def get_hero_cards(self) -> List[str]:
        return list(self.hero_cards)

    def empty(self) -> bool:
        return (
            not self.hero_cards
            and not self.board_cards
            and not self.buttons
            and self.pot_box is None
            and not self.lobby_rows
        )


class LetterboxMapper:
    """Map boxes from letterboxed model space (e.g. 640×640) back to native pixels."""

    def __init__(self, native_w: int, native_h: int, model_size: int = 640):
        if native_w <= 0 or native_h <= 0:
            raise ValueError(f"invalid native size {native_w}x{native_h}")
        if model_size <= 0:
            raise ValueError(f"invalid model_size {model_size}")
        self.native_w = int(native_w)
        self.native_h = int(native_h)
        self.model_size = int(model_size)
        self.scale = min(model_size / native_w, model_size / native_h)
        self.new_w = native_w * self.scale
        self.new_h = native_h * self.scale
        self.pad_x = (model_size - self.new_w) / 2.0
        self.pad_y = (model_size - self.new_h) / 2.0

    def to_native(self, box_xyxy_norm: Sequence[float]) -> Tuple[int, int, int, int]:
        """Normalize box in model canvas [0,1] → native pixel xyxy (clamped)."""
        x1, y1, x2, y2 = [float(v) for v in box_xyxy_norm]
        mx1, my1 = x1 * self.model_size, y1 * self.model_size
        mx2, my2 = x2 * self.model_size, y2 * self.model_size
        nx1 = int(round((mx1 - self.pad_x) / self.scale))
        ny1 = int(round((my1 - self.pad_y) / self.scale))
        nx2 = int(round((mx2 - self.pad_x) / self.scale))
        ny2 = int(round((my2 - self.pad_y) / self.scale))
        nx1 = max(0, min(self.native_w - 1, nx1))
        ny1 = max(0, min(self.native_h - 1, ny1))
        nx2 = max(0, min(self.native_w, nx2))
        ny2 = max(0, min(self.native_h, ny2))
        if nx2 < nx1:
            nx1, nx2 = nx2, nx1
        if ny2 < ny1:
            ny1, ny2 = ny2, ny1
        return nx1, ny1, nx2, ny2

    def center(self, box_xyxy_norm: Sequence[float]) -> Tuple[int, int]:
        """Native pixel center of box — intended for adb tap."""
        x1, y1, x2, y2 = self.to_native(box_xyxy_norm)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx = max(0, min(self.native_w - 1, cx))
        cy = max(0, min(self.native_h - 1, cy))
        return cx, cy

    def native_point_to_norm(self, x: float, y: float) -> Tuple[float, float]:
        """Native pixel → normalized model-canvas coords (for round-trip tests)."""
        mx = x * self.scale + self.pad_x
        my = y * self.scale + self.pad_y
        return mx / self.model_size, my / self.model_size


class YOLODetector:
    """Stage 1 — load detect weights and return normalized boxes."""

    def __init__(self, weights_path: Optional[str], classes: Optional[List[str]] = None):
        self.classes = list(classes or DEFAULT_DETECT_CLASSES)
        self.weights_path = weights_path
        self._model = None
        self._stub = True
        if weights_path and Path(weights_path).is_file():
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(weights_path)
                self._stub = False
                logger.info("YOLODetector loaded weights=%s", weights_path)
            except Exception as exc:
                logger.warning("YOLODetector load failed (%s) — stub mode", exc)
                self._model = None
                self._stub = True
        else:
            logger.warning(
                "YOLODetector stub mode (weights missing: %s)", weights_path
            )

    def detect(self, image: np.ndarray) -> List[dict]:
        """Return list of {cls, box_norm, conf}. Stub → []."""
        if self._stub or self._model is None or image is None:
            return []
        try:
            results = self._model.predict(image, conf=0.15, verbose=False)
            if not results:
                return []
            r0 = results[0]
            out: List[dict] = []
            names = getattr(r0, "names", None) or {}
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                return []
            h, w = image.shape[:2]
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                cls_name = names.get(cls_id, str(cls_id))
                # Normalize against letterboxed model canvas (assume square input)
                # Ultralytics returns coords in original image space — re-norm to [0,1]
                # relative to native image; LetterboxMapper expects model-canvas norms.
                # For Stage-1 scaffold we emit native-relative norms; VisionEngine
                # remaps via LetterboxMapper built from image shape.
                box_norm = (
                    xyxy[0] / max(w, 1),
                    xyxy[1] / max(h, 1),
                    xyxy[2] / max(w, 1),
                    xyxy[3] / max(h, 1),
                )
                out.append({"cls": cls_name, "box_norm": box_norm, "conf": conf})
            return out
        except Exception as exc:
            logger.warning("YOLODetector.detect error: %s", exc)
            return []


def pad_to_square(img: np.ndarray, value: int = 114) -> np.ndarray:
    """Pad to a square canvas, image anchored top-left (where the rank and suit are)."""
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side) + img.shape[2:], value, dtype=img.dtype)
    out[:h, :w] = img
    return out


class RankSuitClassifier:
    """Stage 2 — classify a card crop into 52 classes (As..Kc)."""

    def __init__(self, weights_path: Optional[str], pad_square: bool = False):
        self.weights_path = weights_path
        # The classifier center-crops to a square; narrow overlapped cards lose their corner
        # unless padded first. Must match how the weights were trained.
        self.pad_square = pad_square
        self._model = None
        self._stub = True
        if weights_path and Path(weights_path).is_file():
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(weights_path)
                self._stub = False
                logger.info("RankSuitClassifier loaded weights=%s", weights_path)
            except Exception as exc:
                logger.warning("RankSuitClassifier load failed (%s) — stub", exc)
        else:
            logger.warning(
                "RankSuitClassifier stub mode (weights missing: %s)", weights_path
            )

    def classify(self, crop: np.ndarray) -> Tuple[str, float]:
        if self._stub or self._model is None or crop is None or crop.size == 0:
            return ("", 0.0)
        try:
            if self.pad_square:
                crop = pad_to_square(crop)
            results = self._model.predict(crop, verbose=False)
            if not results:
                return ("", 0.0)
            r0 = results[0]
            probs = getattr(r0, "probs", None)
            names = getattr(r0, "names", {}) or {}
            if probs is None:
                return ("", 0.0)
            top = int(probs.top1)
            conf = float(probs.top1conf)
            label = names.get(top, str(top))
            return (str(label), conf)
        except Exception as exc:
            logger.warning("RankSuitClassifier.classify error: %s", exc)
            return ("", 0.0)


class VisionEngine:
    """Facade: image → GameState via detect + classify."""

    def __init__(
        self,
        detect_weights: Optional[str] = None,
        cls_weights: Optional[str] = None,
        lobby_weights: Optional[str] = None,
        model_size: int = 640,
        classes: Optional[List[str]] = None,
        backend: str = "yolo",
        cls_pad_square: bool = False,
    ):
        self.model_size = model_size
        self.backend = (backend or "yolo").lower()
        use_weights = self.backend != "stub"
        self.detector = YOLODetector(
            detect_weights if use_weights else None,
            classes=classes,
        )
        self.lobby_detector = YOLODetector(
            lobby_weights if use_weights else None,
            classes=classes,
        )
        self.classifier = RankSuitClassifier(
            cls_weights if use_weights else None, pad_square=cls_pad_square
        )
        # Fast play: OCR only on our turn and skip opponents' stacks.
        self.fast_text = False
        self._blind_key: tuple = ()
        self._blind_val = ""

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "VisionEngine":
        vision = (cfg or {}).get("vision", cfg or {})
        return cls(
            detect_weights=vision.get("detect_weights"),
            cls_weights=vision.get("cls_weights"),
            lobby_weights=vision.get("lobby_weights"),
            model_size=int(vision.get("model_size", 640)),
            backend=str(vision.get("backend", "yolo")),
            cls_pad_square=bool(vision.get("cls_pad_square", False)),
        )

    def get_game_state(self, image: Optional[np.ndarray]) -> GameState:
        if image is None or not hasattr(image, "shape") or image.size == 0:
            return GameState()
        h, w = image.shape[:2]
        state = GameState(resolution=(w, h))
        if self.backend == "stub" or (self.detector._stub and self.lobby_detector._stub):
            return state

        detections = [] if self.detector._stub else self.detector.detect(image)
        if not detections and self.lobby_detector._stub:
            return state

        # Detections from Ultralytics are native-relative norms; build mapper
        # and treat box_norm as already in native [0,1] — convert via fake
        # identity letterbox (native==model letterbox of same aspect).
        mapper = LetterboxMapper(w, h, self.model_size)

        hero_crops: List[Tuple[float, Tuple[int, int, int, int]]] = []
        board_crops: List[Tuple[float, Tuple[int, int, int, int]]] = []
        back_centers: List[Tuple[int, int]] = []

        for det in detections:
            cls = str(det.get("cls", "")).lower()
            conf = float(det.get("conf", 0.0))
            box_norm = det.get("box_norm") or (0, 0, 0, 0)
            # Native-relative norm → model-canvas norm for LetterboxMapper
            # Identity path: map native xyxy pixels through letterbox forward.
            x1n, y1n, x2n, y2n = (
                box_norm[0] * w,
                box_norm[1] * h,
                box_norm[2] * w,
                box_norm[3] * h,
            )
            n1 = mapper.native_point_to_norm(x1n, y1n)
            n2 = mapper.native_point_to_norm(x2n, y2n)
            model_norm = (n1[0], n1[1], n2[0], n2[1])
            xyxy = mapper.to_native(model_norm)
            cx, cy = mapper.center(model_norm)

            if cls == "btn_fold" or cls.endswith("fold") or "fold" in cls:
                state.buttons["fold"] = (cx, cy)
                state.is_bots_turn = True
            elif cls == "btn_check_call":
                state.buttons["check_call"] = (cx, cy)
                state.is_bots_turn = True
            elif cls == "btn_bet" or "raise" in cls or cls.endswith("bet"):
                state.buttons["bet"] = (cx, cy)
                state.is_bots_turn = True
            elif "call" in cls or "check" in cls:
                key = "call" if "call" in cls else "check"
                state.buttons[key] = (cx, cy)
                state.is_bots_turn = True
            elif cls == "pot" or cls.endswith("_pot"):
                state.pot_box = xyxy
            elif cls == "stack_own":
                state.stack_box = xyxy
            elif cls == "stack_enemy":
                state.enemy_stack_boxes.append(xyxy)
            elif cls == "dealer_button":
                state.dealer_box = xyxy
            elif cls == "card_back":
                if all(abs(cx - bx) > 40 or abs(cy - by) > 40 for bx, by in back_centers):
                    back_centers.append((cx, cy))
                    state.opponents_in_hand += 1
            elif "hero" in cls and "card" in cls:
                hero_crops.append((conf, xyxy))
            elif "board" in cls and "card" in cls:
                board_crops.append((conf, xyxy))

        hero_crops.sort(key=lambda t: t[1][0])  # left→right
        board_crops.sort(key=lambda t: t[1][0])
        state.board_slots = len(board_crops)
        hero_crops = _clip_overlapped_hero(hero_crops)

        for _, xyxy in hero_crops:
            crop = _safe_crop(image, xyxy)
            label, conf = self.classifier.classify(crop)
            if label:
                state.hero_cards.append(label)
                state.hero_card_conf.append(conf)

        for _, xyxy in board_crops[:5]:
            crop = _safe_crop(image, xyxy)
            label, _c = self.classifier.classify(crop)
            if label:
                state.board_cards.append(label)

        on_table = bool(
            hero_crops
            or board_crops
            or state.buttons
            or state.pot_box is not None
            or state.stack_box is not None
            or state.enemy_stack_boxes
            or state.dealer_box is not None
        )
        if on_table:
            if not self.fast_text or state.buttons:
                _read_table_text(image, state, enemy_stacks=not self.fast_text)
            if state.buttons:
                key = tuple(state.hero_cards)
                if key and key == self._blind_key:
                    state.hero_blind = self._blind_val
                else:
                    state.hero_blind = _hero_blind(image)
                    if not state.hero_blind and _box_on_hero(state.dealer_box, w, h):
                        state.hero_blind = "btn"
                    self._blind_key = key
                    self._blind_val = state.hero_blind
        elif not self.lobby_detector._stub:
            state.lobby_rows = _lobby_rows(self.lobby_detector.detect(image), w, h)
            _read_lobby_text(image, state.lobby_rows)

        return state


def _box_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    cx = (ix1 + ix2) / 2.0
    cy = (iy1 + iy2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _lobby_rows(detections: List[dict], w: int, h: int) -> List[dict]:
    """Group lobby text boxes into the table row that contains them."""
    boxes: Dict[str, List[Tuple[int, int, int, int]]] = {
        "lobby_row": [],
        "lobby_blinds": [],
        "lobby_buyin": [],
        "lobby_seats": [],
    }
    for det in detections:
        cls = str(det.get("cls", ""))
        if cls not in boxes:
            continue
        x1, y1, x2, y2 = det.get("box_norm") or (0, 0, 0, 0)
        boxes[cls].append(
            (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
        )
    rows: List[dict] = []
    for row_box in boxes["lobby_row"]:
        row = {"box": row_box, "blinds": None, "buyin": None, "seats": None}
        for key, cls in (
            ("blinds", "lobby_blinds"),
            ("buyin", "lobby_buyin"),
            ("seats", "lobby_seats"),
        ):
            for box in boxes[cls]:
                if _box_contains(row_box, box):
                    row[key] = box
                    break
        rows.append(row)
    rows.sort(key=lambda row: row["box"][1])
    used = [row["blinds"] for row in rows if row.get("blinds") is not None]
    header = [
        box for box in boxes["lobby_blinds"]
        if box not in used and (not rows or box[3] <= rows[0]["box"][1] + 8)
    ]
    if len(header) == 1:
        for row in rows:
            row["screen_blinds"] = header[0]
    return rows


def _read_table_text(image: np.ndarray, state: GameState, enemy_stacks: bool = True) -> None:
    """Fill pot, stacks and button labels from the boxes the table model found."""
    parser = _numeric_parser()
    if state.pot_box is not None:
        state.pot = _read_amount(image, state.pot_box, parser)
    if state.stack_box is not None:
        state.hero_stack = _read_amount(image, state.stack_box, parser)
    if enemy_stacks:
        state.enemy_stacks = [
            _read_amount(image, box, parser) for box in state.enemy_stack_boxes
        ]
    for key, center in state.buttons.items():
        if not enemy_stacks and (
            key == "fold" or (key == "bet" and "check_call" in state.buttons)
        ):
            continue
        box = _around_center(image, center, 220, 56)
        text = _read_words(image, box)
        if text:
            state.button_text[key] = text
        if key == "check_call":
            state.call_amount = parse_amount(text)


def _read_lobby_text(image: np.ndarray, rows: List[dict]) -> None:
    parser = _numeric_parser()
    for row in rows:
        blinds_box = row.get("blinds") or row.get("screen_blinds")
        if blinds_box is not None:
            raw = _read_words(image, _middle_band(blinds_box))
            row["blinds_text"] = raw
            nums = _numbers_in(raw)
            if len(nums) >= 2:
                row["small_blind"] = nums[0]
                row["big_blind"] = nums[1]
        if row.get("buyin") is not None:
            row["buyin_value"] = _read_number(image, row["buyin"], parser)
        if row.get("seats") is not None:
            raw = _read_words(image, row["seats"])
            row["seats_text"] = raw
            match = re.search(r"(\d+)\s*/\s*(\d+)", raw)
            if match:
                row["seated"] = int(match.group(1))
                row["max_seats"] = int(match.group(2))


def _numeric_parser():
    global _PARSER
    if _PARSER is None:
        from bridge.vision.numeric_parser import NumericParser

        _PARSER = NumericParser(dry_run=False, fallback_to_simulation=False)
    return _PARSER


_PARSER = None


def _middle_band(xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Keep the middle of a tall lobby pill, where the stake numbers sit."""
    x1, y1, x2, y2 = xyxy
    h = max(1, y2 - y1)
    return x1, y1 + int(h * 0.28), x2, y1 + int(h * 0.72)


def _xywh(xyxy: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _around_center(
    image: np.ndarray, center: Tuple[int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    h, w = image.shape[:2]
    cx, cy = center
    x1 = max(0, cx - width // 2)
    y1 = max(0, cy - height // 2)
    x2 = min(w, cx + width // 2)
    y2 = min(h, cy + height // 2)
    return x1, y1, x2, y2


def _box_on_hero(box: Optional[Tuple[int, int, int, int]], w: int, h: int) -> bool:
    if not box or w <= 0 or h <= 0:
        return False
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return cy > 0.72 * h and 0.20 * w < cx < 0.82 * w


def _hero_blind(image: np.ndarray) -> str:
    """SB / BB badge drawn on the hero avatar at the bottom of a CoinPoker table."""
    parser = _numeric_parser()
    if parser._pytesseract is None and not parser._try_load_ocr():
        return ""
    import cv2

    h, w = image.shape[:2]
    crop = image[int(h * 0.70): int(h * 0.93), int(w * 0.12): int(w * 0.72)]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        data = parser._pytesseract.image_to_data(
            gray, config="--psm 11", output_type=parser._pytesseract.Output.DICT
        )
    except Exception as exc:
        logger.debug("blind OCR failed: %s", exc)
        return ""
    for text in data.get("text", []):
        t = "".join(ch for ch in (text or "").lower() if ch.isalpha())
        if t in ("sb", "bb"):
            return t
    return ""


def read_pot_line(image: np.ndarray) -> float:
    """Pot printed above the board. The detector often misses that label."""
    parser = _numeric_parser()
    if image is None or parser._pytesseract is None and not parser._try_load_ocr():
        return 0.0
    import cv2

    h, w = image.shape[:2]
    crop = image[int(h * 0.30): int(h * 0.42), int(w * 0.30): int(w * 0.70)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    try:
        text = parser._pytesseract.image_to_string(bw, config="--psm 6")
    except Exception as exc:
        logger.debug("pot OCR failed: %s", exc)
        return 0.0
    amounts = [parse_amount(line) for line in (text or "").splitlines()]
    amounts = [a for a in amounts if a >= 100]
    return max(amounts) if amounts else 0.0


def find_words(image: np.ndarray, words: Tuple[str, ...]) -> Optional[Tuple[int, int]]:
    """Center of the first on-screen word starting with one of `words` (case-insensitive)."""
    parser = _numeric_parser()
    if parser._pytesseract is None and not parser._try_load_ocr():
        return None
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # light text on coloured buttons and dark text on light dialogs
    for variant in (gray, 255 - gray):
        try:
            data = parser._pytesseract.image_to_data(
                variant, config="--psm 11", output_type=parser._pytesseract.Output.DICT
            )
        except Exception as exc:
            logger.debug("find_words OCR failed: %s", exc)
            return None
        for i, text in enumerate(data.get("text", [])):
            t = (text or "").strip().lower()
            if t and any(t.startswith(w) for w in words):
                x = data["left"][i] + data["width"][i] // 2
                y = data["top"][i] + data["height"][i] // 2
                return int(x), int(y)
    return None


def _read_number(image: np.ndarray, xyxy: Tuple[int, int, int, int], parser) -> float:
    if parser._pytesseract is None and not parser._try_load_ocr():
        return 0.0
    return float(parser._ocr_roi(image, _xywh(xyxy)) or 0.0)


def _read_words(image: np.ndarray, xyxy: Tuple[int, int, int, int]) -> str:
    parser = _numeric_parser()
    if parser._pytesseract is None and not parser._try_load_ocr():
        return ""
    crop = _safe_crop(image, xyxy)
    if crop.size == 0:
        return ""
    try:
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        ch, cw = gray.shape[:2]
        scale = max(2, int(48 / max(ch, 1)))
        gray = cv2.resize(gray, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./,"
        text = parser._pytesseract.image_to_string(bw, config=config).strip()
        return " ".join(text.split())
    except Exception as exc:
        logger.debug("word OCR failed: %s", exc)
        return ""


_AMOUNT_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d{1,2})?)\s*([KkMm]?)")


def parse_amount(text: str) -> float:
    """First chip amount in text: '210,320' → 210320, '0.05' → 0.05, '17.28K' → 17280."""
    match = _AMOUNT_RE.search(text or "")
    if not match:
        return 0.0
    digits, suffix = match.group(1), match.group(2).lower()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", digits) and not suffix:
        value = float(re.sub(r"[.,]", "", digits))
    else:
        value = float(digits.replace(",", "."))
    return value * {"k": 1e3, "m": 1e6}.get(suffix, 1.0)


def _read_amount(image: np.ndarray, xyxy: Tuple[int, int, int, int], parser) -> float:
    """Chip amount; the numeric parser reads '12,640' as 12.64, word OCR fixes that."""
    base = _read_number(image, xyxy, parser)
    words = parse_amount(_read_words(image, xyxy))
    if base and words and abs(words - base * 1000) < 1.0:
        return words
    return base or words


def _numbers_in(text: str) -> List[float]:
    found: List[float] = []
    for match in re.finditer(r"\d+(?:[.,]\d+)?", text or ""):
        try:
            found.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue
    return found


def _safe_crop(image: np.ndarray, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _clip_overlapped_hero(
    crops: List[Tuple[float, Tuple[int, int, int, int]]],
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """Drop hero boxes spanning both overlapped cards, or cut one to the right card's edge."""

    def _contains(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        return a[0] <= b[0] + 4 and a[2] >= b[2] - 4 and (a[2] - a[0]) > (b[2] - b[0]) + 8

    if len(crops) > 2:
        crops = [
            c for c in crops
            if not any(o is not c and _contains(c[1], o[1]) for o in crops)
        ]
    crops = sorted(crops, key=lambda t: t[1][0])[:2]
    if len(crops) < 2:
        return crops
    (lc, (lx1, ly1, lx2, ly2)), right = crops[0], crops[1]
    rx1, _, rx2, _ = right[1]
    if lx2 >= rx2 - 4 and rx1 > lx1 + 8:
        lx2 = rx1 + (rx1 - lx1) // 3
    return [(lc, (lx1, ly1, lx2, ly2)), right]


def load_bots_vision_config(path: str = "config/bots.yaml") -> dict:
    """Load vision section from config/bots.yaml (empty dict on failure)."""
    try:
        import yaml

        p = Path(path)
        if not p.is_file():
            return {}
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("load_bots_vision_config: %s", exc)
        return {}
