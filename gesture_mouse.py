"""
Hand Gesture Mouse Controller
カメラの手のジェスチャーでマウスを操作するスクリプト

ジェスチャー一覧:
  ポイント (人差し指だけ立てる) : カーソル移動
  ピンチ (親指+人差し指を近づける) : 左クリック
  ダブルピンチ (素早く2回ピンチ)  : ダブルクリック
  グー (全指を曲げる)             : 左ボタン押しっぱなし (ドラッグ)
  ピース (人差し指+中指)          : スクロール (上下に動かす)
  パー (全指を伸ばす)             : 何もしない (ニュートラル)
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# フェイルセーフ無効化（画面四隅でも止まらないようにする）
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---- 定数 ----
SCREEN_W, SCREEN_H = pyautogui.size()
CAM_ID = 0
SMOOTHING = 0.3          # カーソル平滑化係数 (0=即時, 1=全く動かない)
PINCH_THRESHOLD = 0.05   # ピンチ判定の距離しきい値 (正規化座標)
SCROLL_SPEED = 30         # スクロール量 (px)
CLICK_COOLDOWN = 0.4     # クリック間隔 (秒)
MARGIN = 0.15            # カメラ映像の端のマージン (ジェスチャー検出エリア外)


def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def finger_up(lm, tip, pip):
    """指が伸びているか判定（tip の y が pip の y より上）"""
    return lm[tip].y < lm[pip].y


def classify_gesture(lm):
    """
    ランドマークからジェスチャーを分類する
    戻り値: 'point' | 'pinch' | 'fist' | 'peace' | 'open' | 'unknown'
    """
    # 各指の伸び状態
    thumb_up  = lm[4].x < lm[3].x  # 右手の場合、親指は横方向で判定
    index_up  = finger_up(lm, 8, 6)
    middle_up = finger_up(lm, 12, 10)
    ring_up   = finger_up(lm, 16, 14)
    pinky_up  = finger_up(lm, 20, 18)

    # ピンチ: 親指と人差し指の距離
    pinch_dist = dist(lm[4], lm[8])

    up_count = sum([index_up, middle_up, ring_up, pinky_up])

    if pinch_dist < PINCH_THRESHOLD:
        return 'pinch'
    elif up_count == 0:
        return 'fist'
    elif index_up and middle_up and not ring_up and not pinky_up:
        return 'peace'
    elif index_up and not middle_up and not ring_up and not pinky_up:
        return 'point'
    elif up_count >= 3:
        return 'open'
    else:
        return 'unknown'


def cam_to_screen(x, y):
    """
    カメラの正規化座標 → スクリーン座標へ変換
    マージンを除いた内側エリアをスクリーン全体にマッピング
    """
    x = np.clip((x - MARGIN) / (1 - 2 * MARGIN), 0, 1)
    y = np.clip((y - MARGIN) / (1 - 2 * MARGIN), 0, 1)
    return int(x * SCREEN_W), int(y * SCREEN_H)


def draw_ui(frame, gesture, cursor_pos, fps):
    h, w = frame.shape[:2]

    # ジェスチャーごとの色と表示名
    color_map = {
        'point':   ((0, 255, 0),   'POINT  - Move cursor'),
        'pinch':   ((0, 100, 255), 'PINCH  - Click'),
        'fist':    ((0, 0, 255),   'FIST   - Drag'),
        'peace':   ((255, 200, 0), 'PEACE  - Scroll'),
        'open':    ((200, 200, 200), 'OPEN  - Neutral'),
        'unknown': ((150, 150, 150), 'UNKNOWN'),
    }
    color, label = color_map.get(gesture, ((150, 150, 150), gesture))

    # 有効エリアの枠
    mx1 = int(MARGIN * w)
    my1 = int(MARGIN * h)
    mx2 = int((1 - MARGIN) * w)
    my2 = int((1 - MARGIN) * h)
    cv2.rectangle(frame, (mx1, my1), (mx2, my2), (80, 80, 80), 1)

    # ジェスチャーラベル
    cv2.rectangle(frame, (0, 0), (320, 36), (30, 30, 30), -1)
    cv2.putText(frame, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS
    cv2.putText(frame, f'FPS: {fps:.0f}', (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # カーソル位置
    cv2.putText(frame, f'Screen: {cursor_pos}', (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # 操作説明 (右下)
    guide = [
        'Q: quit',
        'Esc: quit',
    ]
    for i, g in enumerate(guide):
        cv2.putText(frame, g, (w - 120, h - 10 - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main():
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 平滑化用の前フレームのスクリーン座標
    prev_sx, prev_sy = SCREEN_W // 2, SCREEN_H // 2

    # 状態管理
    drag_active = False
    last_click_time = 0
    prev_gesture = None
    pinch_times = []   # ダブルクリック検出用タイムスタンプ

    # ピース時のスクロール基準 y
    peace_ref_y = None

    prev_time = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 左右反転（鏡モード）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = 'unknown'
            cursor_pos = (prev_sx, prev_sy)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark

                # ランドマーク描画
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

                gesture = classify_gesture(lm)

                # 人差し指の先端を基準座標に使用
                ix, iy = lm[8].x, lm[8].y
                tx, ty = cam_to_screen(ix, iy)

                # スムージング
                sx = int(prev_sx + (tx - prev_sx) * (1 - SMOOTHING))
                sy = int(prev_sy + (ty - prev_sy) * (1 - SMOOTHING))
                # 画面端に張り付かないよう5px内側に制限
                sx = max(5, min(SCREEN_W - 5, sx))
                sy = max(5, min(SCREEN_H - 5, sy))
                prev_sx, prev_sy = sx, sy
                cursor_pos = (sx, sy)

                now = time.time()

                # ---- ジェスチャー処理 ----

                if gesture == 'point':
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    pyautogui.moveTo(sx, sy)

                elif gesture == 'pinch':
                    pyautogui.moveTo(sx, sy)
                    if now - last_click_time > CLICK_COOLDOWN:
                        # ダブルクリック判定 (0.5秒以内に2回ピンチ)
                        pinch_times.append(now)
                        pinch_times = [t for t in pinch_times if now - t < 0.5]
                        if len(pinch_times) >= 2:
                            pyautogui.doubleClick()
                            pinch_times = []
                        else:
                            pyautogui.click()
                        last_click_time = now

                elif gesture == 'fist':
                    if not drag_active:
                        pyautogui.mouseDown()
                        drag_active = True
                    pyautogui.moveTo(sx, sy)

                elif gesture == 'peace':
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    # スクロール: 基準位置からの y 差分で制御
                    if prev_gesture != 'peace':
                        peace_ref_y = iy
                    if peace_ref_y is not None:
                        delta_y = iy - peace_ref_y
                        if abs(delta_y) > 0.01:
                            scroll_amount = int(-delta_y * SCROLL_SPEED * 10)
                            pyautogui.scroll(scroll_amount)
                            peace_ref_y = iy  # 基準を更新（連続スクロール）

                elif gesture == 'open':
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    peace_ref_y = None

                else:
                    peace_ref_y = None

                prev_gesture = gesture

            else:
                # 手が検出されない場合はドラッグ解除
                if drag_active:
                    pyautogui.mouseUp()
                    drag_active = False
                prev_gesture = None
                peace_ref_y = None

            # FPS 計算
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_ui(frame, gesture, cursor_pos, fps)

            cv2.imshow('Gesture Mouse Controller', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # Q or Esc
                break

    if drag_active:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
