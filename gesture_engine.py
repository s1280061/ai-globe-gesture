"""
gesture_engine.py
シンプル両手ジェスチャー検出

ジェスチャー分類は最小限。速度ベースで操作を判断する。
  - 手が見える          : カーソル移動
  - 手を素早く払う      : フリック（地球を回す）
  - 両手の距離変化      : ズーム
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GestureState:
    # 右手
    right_x:       float = 0.5
    right_y:       float = 0.5
    right_visible: bool  = False

    # 左手
    left_x:        float = 0.5
    left_y:        float = 0.5
    left_visible:  bool  = False

    # 両手ズーム
    both_visible:   bool  = False
    hands_distance: float = 0.0

    fps: float = 0.0

    # カメラフレーム共有（録画用）
    cam_frame: object = None       # BGR numpy array or None
    cam_frame_lock: object = field(default_factory=threading.Lock)

    # 後方互換プロパティ
    @property
    def right_gesture(self):    return "point" if self.right_visible else "none"
    @property
    def left_gesture(self):     return "point" if self.left_visible  else "none"
    @property
    def right_pinch_just(self): return False
    @property
    def gesture(self):          return self.right_gesture
    @property
    def cursor_x(self):         return self.right_x
    @property
    def cursor_y(self):         return self.right_y
    @property
    def pinch_just(self):       return False
    @property
    def hand_visible(self):     return self.right_visible or self.left_visible

    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    object.__setattr__(self, k, v)

    def snapshot(self):
        with self.lock:
            return {
                "right_x":        self.right_x,
                "right_y":        self.right_y,
                "right_visible":  self.right_visible,
                "left_x":         self.left_x,
                "left_y":         self.left_y,
                "left_visible":   self.left_visible,
                "both_visible":   self.both_visible,
                "hands_distance": self.hands_distance,
                # 後方互換
                "right_gesture":    self.right_gesture,
                "left_gesture":     self.left_gesture,
                "right_pinch_just": self.right_pinch_just,
                "gesture":          self.gesture,
                "cursor_x":         self.cursor_x,
                "cursor_y":         self.cursor_y,
                "pinch_just":       self.pinch_just,
                "hand_visible":     self.hand_visible,
                "fps":              self.fps,
            }


class GestureEngine(threading.Thread):

    def __init__(self, state: GestureState, cam_id: int = 0, show_window: bool = True):
        super().__init__(daemon=True)
        self.state       = state
        self.cam_id      = cam_id
        self.show_window = show_window
        self._stop_event = threading.Event()

        # 平滑化
        self._rx, self._ry = 0.5, 0.5
        self._lx, self._ly = 0.5, 0.5
        self.SMOOTH = 0.4

    def stop(self):
        self._stop_event.set()

    def run(self):
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_t = time.time()

        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now

                right_visible = False
                left_visible  = False
                both_visible  = False
                hands_distance = 0.0

                if res.multi_hand_landmarks and res.multi_handedness:
                    for hand_lm, hand_info in zip(
                        res.multi_hand_landmarks, res.multi_handedness
                    ):
                        label = hand_info.classification[0].label
                        lm    = hand_lm.landmark

                        # 手首(0)を基準座標に使用（安定している）
                        wx, wy = lm[0].x, lm[0].y

                        if label == "Right":
                            right_visible = True
                            self._rx += (wx - self._rx) * (1 - self.SMOOTH)
                            self._ry += (wy - self._ry) * (1 - self.SMOOTH)
                        else:
                            left_visible = True
                            self._lx += (wx - self._lx) * (1 - self.SMOOTH)
                            self._ly += (wy - self._ly) * (1 - self.SMOOTH)

                        if self.show_window:
                            mp_draw.draw_landmarks(
                                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                                mp_draw.DrawingSpec(
                                    color=(0,220,100) if label=="Right" else (60,120,255),
                                    thickness=2, circle_radius=4),
                                mp_draw.DrawingSpec(
                                    color=(0,160,70) if label=="Right" else (40,80,200),
                                    thickness=2),
                            )

                if right_visible and left_visible:
                    both_visible   = True
                    hands_distance = np.hypot(
                        self._rx - self._lx,
                        self._ry - self._ly
                    )

                self.state.update(
                    right_x       = float(np.clip(self._rx, 0, 1)),
                    right_y       = float(np.clip(self._ry, 0, 1)),
                    right_visible = right_visible,
                    left_x        = float(np.clip(self._lx, 0, 1)),
                    left_y        = float(np.clip(self._ly, 0, 1)),
                    left_visible  = left_visible,
                    both_visible  = both_visible,
                    hands_distance= float(hands_distance),
                    fps           = fps,
                )

                if self.show_window:
                    self._draw_overlay(frame, right_visible, left_visible, fps)
                    cv2.imshow("Hand Tracking", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self._stop_event.set()

                # カメラフレームを共有バッファへ（録画用）
                with self.state.cam_frame_lock:
                    self.state.cam_frame = frame.copy()

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

    def _draw_overlay(self, frame, rv, lv, fps):
        h, w = frame.shape[:2]

        # 右手パネル
        cv2.rectangle(frame, (0, 0), (200, 50), (10, 30, 15), -1)
        rc = (0, 220, 100) if rv else (50, 60, 50)
        cv2.putText(frame, "RIGHT HAND", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1)
        cv2.putText(frame, "ACTIVE" if rv else "---", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2)

        # 左手パネル
        cv2.rectangle(frame, (w-200, 0), (w, 50), (10, 15, 30), -1)
        lc = (60, 120, 255) if lv else (50, 55, 70)
        cv2.putText(frame, "LEFT HAND", (w-190, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 1)
        cv2.putText(frame, "ACTIVE" if lv else "---", (w-190, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lc, 2)

        # FPS
        cv2.putText(frame, f"FPS {fps:.0f}", (w//2-28, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1)

        # 右手カーソルドット
        if rv:
            cx = int(self._rx * w)
            cy = int(self._ry * h)
            cv2.circle(frame, (cx, cy), 12, (0, 220, 100), 2)
            cv2.line(frame, (cx-16, cy), (cx+16, cy), (0,220,100), 1)
            cv2.line(frame, (cx, cy-16), (cx, cy+16), (0,220,100), 1)
