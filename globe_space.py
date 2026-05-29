"""
globe_space.py
ジェスチャーで回す 3D 地球儀 + AI Knowledge Space

操作:
  ✊ グー + 左右/上下  : 地球儀を回す
  ☝ ポイント           : カーソル移動・都市ハイライト
  🤌 ピンチ            : 都市/地域を選択 → AI が解説ノードを展開
  ✌ ピース            : ズームイン/アウト（上下）
  🖐 パー              : ニュートラル
"""

import pygame
import math
import time
import threading
import random
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import anthropic
from ambient_music import start_bgm, stop_bgm, set_volume
from recorder import Recorder

# ============================================================
# 定数
# ============================================================
WIN_W, WIN_H = 1400, 800
BG_COLOR  = (2, 3, 8)          # ほぼ真っ黒・宇宙

GLOBE_R   = 260          # 地球儀の半径(px)
OCEAN_C   = (10,  35,  80)      # 深い海色
LAND_C    = (30,  65,  40)
GRID_C    = (60,  100, 160)
GLOW_C    = (60, 130, 255)
ATMO_C    = (80, 160, 255)      # 大気の色
FLARE_C   = (200, 220, 255)     # レンズフレア

AI_COLOR  = (255, 220,  60)
NODE_COLORS = [
    (80,  200, 255),
    (120, 255, 160),
    (255, 160,  80),
    (220,  80, 200),
    (160, 120, 255),
    (80,  255, 220),
]


# ============================================================
# 都市/地域データ (lat, lon, name, region_color_idx)
# ============================================================
LOCATIONS = [
    # アジア
    ( 35.7,  139.7, "東京",       0),
    ( 37.6,  126.9, "ソウル",     0),
    ( 39.9,  116.4, "北京",       0),
    ( 31.2,  121.5, "上海",       0),
    ( 22.3,  114.2, "香港",       0),
    ( 28.6,   77.2, "ニューデリー",0),
    ( 13.8,  100.5, "バンコク",   0),
    (  1.3,  103.8, "シンガポール",0),
    ( 35.7,   51.4, "テヘラン",   0),
    # ヨーロッパ
    ( 51.5,   -0.1, "ロンドン",   1),
    ( 48.9,    2.3, "パリ",       1),
    ( 52.5,   13.4, "ベルリン",   1),
    ( 41.9,   12.5, "ローマ",     1),
    ( 40.4,   -3.7, "マドリード", 1),
    ( 55.8,   37.6, "モスクワ",   1),
    ( 59.9,   10.7, "オスロ",     1),
    # アメリカ
    ( 40.7,  -74.0, "ニューヨーク",2),
    ( 34.1, -118.2, "ロサンゼルス",2),
    ( 41.9,  -87.6, "シカゴ",     2),
    ( 19.4,  -99.1, "メキシコシティ",2),
    (-23.5,  -46.6, "サンパウロ", 2),
    (-34.6,  -58.4, "ブエノスアイレス",2),
    ( 45.4,  -75.7, "オタワ",     2),
    # アフリカ
    ( 30.0,   31.2, "カイロ",     3),
    (-33.9,   18.4, "ケープタウン",3),
    (  6.5,    3.4, "ラゴス",     3),
    (-1.3,   36.8, "ナイロビ",   3),
    ( 36.8,    3.1, "アルジェ",   3),
    # オセアニア
    (-33.9,  151.2, "シドニー",   4),
    (-37.8,  144.9, "メルボルン", 4),
    (-36.9,  174.8, "オークランド",4),
    # 中東
    ( 25.2,   55.3, "ドバイ",     5),
    ( 31.8,   35.2, "エルサレム", 5),
    ( 24.7,   46.7, "リヤド",     5),
]


# ============================================================
# 3D 数学ユーティリティ
# ============================================================

def latlon_to_xyz(lat_deg, lon_deg, r=1.0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.sin(lat)
    z = r * math.cos(lat) * math.sin(lon)
    return np.array([x, y, z])


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)


def apply_rot(R, v):
    return R @ v


def project(v, scale, cx, cy):
    """正射影（奥行きなし・シンプル）"""
    return int(cx + v[0] * scale), int(cy - v[1] * scale)


# ============================================================
# AI 解説生成
# ============================================================
class GlobeAI:
    def __init__(self):
        self.client  = anthropic.Anthropic()
        self._result = None
        self._lock   = threading.Lock()
        self._thread = None
        self._topic  = ""

    @property
    def is_thinking(self):
        return self._thread is not None and self._thread.is_alive()

    def ask(self, location: str):
        if self.is_thinking:
            return
        self._topic  = location
        self._result = None
        self._thread = threading.Thread(target=self._call, daemon=True)
        self._thread.start()

    def pop(self):
        with self._lock:
            r, self._result = self._result, None
            return r

    def _call(self):
        prompt = (
            f"「{self._topic}」について、以下の観点で各10文字以内の日本語キーワードを5つ挙げてください。\n"
            "観点: 文化・歴史・経済・自然・食文化\n"
            "形式: 観点:キーワード の形で5行。説明不要。\n"
            "例:\n文化:茶道\n歴史:江戸幕府\n経済:自動車産業\n自然:富士山\n食文化:寿司"
        )
        try:
            msg = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            lines = msg.content[0].text.strip().splitlines()
            result = []
            for line in lines:
                line = line.strip()
                if ":" in line or "：" in line:
                    parts = line.replace("：", ":").split(":", 1)
                    if len(parts) == 2:
                        result.append((parts[0].strip(), parts[1].strip()))
            with self._lock:
                self._result = result[:5]
        except Exception as e:
            with self._lock:
                self._result = [("エラー", str(e)[:15])]


# ============================================================
# 知識ノード（地球儀の周囲に浮かぶ）
# ============================================================
@dataclass
class FloatNode:
    label: str
    category: str
    angle: float      # 公転角度
    radius: float     # 地球儀からの距離
    height: float     # y オフセット
    color: tuple
    scale: float = 0.0
    pulse: float = 0.0


# ============================================================
# メインクラス
# ============================================================
class GlobeSpace:

    def __init__(self, gesture_state):
        self.gs = gesture_state

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("AI Globe - Gesture Knowledge Space")
        self.clock  = pygame.font.init() or pygame.time.Clock()
        self.clock  = pygame.time.Clock()

        font_path = "C:/Windows/Fonts/meiryo.ttc"
        if os.path.exists(font_path):
            self.font_lg = pygame.font.Font(font_path, 20)
            self.font_md = pygame.font.Font(font_path, 15)
            self.font_sm = pygame.font.Font(font_path, 12)
        else:
            self.font_lg = pygame.font.SysFont("arial", 20)
            self.font_md = pygame.font.SysFont("arial", 15)
            self.font_sm = pygame.font.SysFont("arial", 12)

        # 地球儀の中心
        self.globe_cx = WIN_W // 2
        self.globe_cy = WIN_H // 2

        # 回転角
        self.rot_lon = 0.0   # 経度方向
        self.rot_lat = 0.2   # 緯度方向（少し傾ける）

        # ズーム
        self.zoom = 1.0

        # 慣性
        self.vel_lon = 0.0
        self.vel_lat = 0.0

        # AI
        self.ai = GlobeAI()
        self._expanding_loc = None

        # 知識ノード
        self.float_nodes: list[FloatNode] = []

        # ジェスチャー
        self._prev_right_gesture = "none"
        self._prev_left_gesture  = "none"
        self._prev_rx = 0.5; self._prev_ry = 0.5
        self._prev_lx = 0.5; self._prev_ly = 0.5
        self._cursor  = (WIN_W // 2, WIN_H // 2)
        self._hover_loc_idx: Optional[int] = None
        self._selected_loc: Optional[str]  = None
        # 両手ズーム
        self._prev_hands_dist = 0.0
        self._zoom_active = False

        # ── ドウェル選択（ホバーで自動選択）──
        self._dwell_loc_idx: Optional[int] = None
        self._dwell_start:   float = 0.0
        self._dwell_time:    float = 1.5
        self._dwell_progress: float = 0.0

        # ── グラブ（掴む）状態 ──
        self._grab_active: bool = False   # 右手グーで掴んでいる
        self._both_grab_active: bool = False  # 両手グーで移動

        # ── 手首角度（前フレーム値）──
        self._prev_right_wrist: float = 0.0
        self._prev_left_wrist:  float = 0.0

        # ── リセットタイマー（両手を上げて2秒）──
        self._reset_timer: float = 0.0
        self._globe_cx_default = WIN_W // 2
        self._globe_cy_default = WIN_H // 2
        self._reset_flash: float = 0.0

        # ── ピンチ選択モード ──
        self._pinch_active: bool = False

        # ── フリック検出（後方互換、今はオープンハンドで代用）──
        self._rx_history: list = []   # (time, x, y) の直近履歴
        self._flick_flash: float = 0.0

        # 緯度・経度ライン（事前計算）
        self._grid_lats = range(-75, 90, 15)
        self._grid_lons = range(-180, 181, 20)

        # ── テクスチャ球面マッピング ──
        self._tex_rgb   = self._load_texture()   # (H, W, 3) numpy RGB
        self._tex_cache_rot  = None              # キャッシュ済み回転行列
        self._tex_cache_zoom = None
        self._tex_surface: Optional[pygame.Surface] = None  # キャッシュ済みサーフェス

        # 自動ゆっくり回転（デフォルトOFF）
        self._auto_spin = False

        # ── BGM 起動 ──
        start_bgm()
        self._bgm_volume = 1.0

        # ── 録画 ──
        self._recorder = Recorder(
            output_dir=r"C:\Users\s1280\Desktop\gestures\recordings",
            with_bgm=True,
            with_mic=False,
            gesture_state=gesture_state,  # カメラフレーム共有
        )

        # ── シネマ用サーフェス（起動時に一度だけ生成） ──
        self._vignette_surf = self._make_vignette()
        self._nebula_surf   = self._make_nebula()
        self._star_surf     = self._make_stars()
        self._grain_surf    = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

    # ---- メインループ ----

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._auto_spin = not self._auto_spin
                    elif event.key == pygame.K_m:
                        self._bgm_volume = 0.0 if self._bgm_volume > 0 else 1.0
                        set_volume(self._bgm_volume)
                    elif event.key == pygame.K_UP:
                        self._bgm_volume = min(1.0, self._bgm_volume + 0.1)
                        set_volume(self._bgm_volume)
                    elif event.key == pygame.K_DOWN:
                        self._bgm_volume = max(0.0, self._bgm_volume - 0.1)
                        set_volume(self._bgm_volume)
                    elif event.key == pygame.K_r:
                        # R: ウィンドウ録画トグル
                        self._recorder.toggle_window(self.screen)
                    elif event.key == pygame.K_t:
                        # T: フルスクリーン録画トグル
                        self._recorder.toggle_screen()

            snap = self.gs.snapshot()
            self._process_gesture(snap, dt)
            self._update(dt)
            self._check_ai()
            self._draw()

            # ウィンドウ録画フレームキャプチャ
            self._recorder.capture_frame(self.screen)

            pygame.display.flip()

        self._recorder.stop()
        pygame.quit()

    # ---- ジェスチャー処理（シンプル両手対応）----

    def _process_gesture(self, snap, dt):
        rv = snap["right_visible"]
        lv = snap["left_visible"]
        bv = snap["both_visible"]
        rx, ry = snap["right_x"], snap["right_y"]
        lx, ly = snap["left_x"],  snap["left_y"]
        hands_dist = snap["hands_distance"]

        right_fist  = snap.get("right_fist",  False)
        left_fist   = snap.get("left_fist",   False)
        right_pinch = snap.get("right_pinch", False)
        right_wrist = snap.get("right_wrist_angle", 0.0)

        drx = rx - self._prev_rx
        dry = ry - self._prev_ry
        dlx = lx - self._prev_lx
        dly = ly - self._prev_ly

        # ── カーソル（常に右手位置）──
        if rv:
            self._cursor = (int(rx * WIN_W), int(ry * WIN_H))

        # ────────────────────────────────────────────────────
        # 優先度1: リセット（両手を高く上げて2秒維持）
        # ────────────────────────────────────────────────────
        if bv and ry < 0.28 and ly < 0.28:
            self._reset_timer += dt
            if self._reset_timer >= 2.0:
                self._do_reset()
                self._reset_flash = 0.5
                self._reset_timer = 0.0
        else:
            self._reset_timer = max(0.0, self._reset_timer - dt * 0.5)

        if self._reset_flash > 0:
            self._reset_flash = max(0.0, self._reset_flash - dt)

        # ────────────────────────────────────────────────────
        # 優先度2: 両手グー → 地球儀を画面内で移動
        # ────────────────────────────────────────────────────
        if bv and right_fist and left_fist:
            self._both_grab_active = True
            self._grab_active      = False
            self._zoom_active      = False
            avg_dx = (drx + dlx) * 0.5
            avg_dy = (dry + dly) * 0.5
            self.globe_cx = int(max(200, min(WIN_W - 200,
                                            self.globe_cx + avg_dx * WIN_W)))
            self.globe_cy = int(max(150, min(WIN_H - 150,
                                            self.globe_cy + avg_dy * WIN_H)))

        # ────────────────────────────────────────────────────
        # 優先度3: 両手オープン → ズーム（Google Earthスタイル）
        # ────────────────────────────────────────────────────
        elif bv and not (right_fist and left_fist):
            self._both_grab_active = False
            self._grab_active      = False
            if not self._zoom_active:
                self._prev_hands_dist = hands_dist
                self._zoom_active = True
            else:
                delta = hands_dist - self._prev_hands_dist
                self.zoom += delta * 4.5
                self.zoom = max(0.4, min(3.0, self.zoom))
                self._prev_hands_dist = hands_dist

        else:
            self._zoom_active      = False
            self._both_grab_active = False

            # ────────────────────────────────────────────────
            # 優先度4: 右手グー → 地球儀をつかんで移動 ＋ 手首でスピン
            # ────────────────────────────────────────────────
            if rv and right_fist:
                self._grab_active = True
                self._auto_spin   = False

                # 位置移動（直接ドラッグ）
                self.globe_cx = int(max(200, min(WIN_W - 200,
                                                self.globe_cx + drx * WIN_W)))
                self.globe_cy = int(max(150, min(WIN_H - 150,
                                                self.globe_cy + dry * WIN_H)))

                # 手首のひねり → 地球儀スピン（角度差分）
                d_wrist = right_wrist - self._prev_right_wrist
                # 角度のラップアラウンド補正
                if d_wrist >  math.pi: d_wrist -= 2 * math.pi
                if d_wrist < -math.pi: d_wrist += 2 * math.pi
                self.vel_lon += d_wrist * 4.0   # 感度調整

            # ────────────────────────────────────────────────
            # 優先度5: 右手オープン → 地球儀を直接なぞって回転
            # ────────────────────────────────────────────────
            elif rv and not right_fist:
                self._grab_active = False

                # 直接回転（なぞる感覚）
                # 手を右 → 地球儀を右回転（rot_lon 増加）
                # 手を下 → 地球儀を下回転（rot_lat 増加）
                SENS = 6.0
                self.rot_lon += drx * SENS
                self.rot_lat += dry * SENS
                self.rot_lat  = max(-1.4, min(1.4, self.rot_lat))

            else:
                self._grab_active = False

        # ── ピンチ → 選択モード ──
        if rv and right_pinch and not self._pinch_active:
            self._pinch_active = True
            hit = self._hit_location(*self._cursor)
            if hit is not None:
                name = LOCATIONS[hit][2]
                if name != self._selected_loc or not self.float_nodes:
                    self._selected_loc  = name
                    self._expanding_loc = name
                    self.float_nodes.clear()
                    self.ai.ask(name)
        elif not right_pinch:
            self._pinch_active = False

        # ── ドウェル選択（右手オープン＆静止）──
        if not right_fist and not right_pinch:
            self._update_dwell(rv, "open", dt)
        else:
            self._dwell_loc_idx  = None
            self._dwell_progress = 0.0

        # フラッシュのカウントダウン
        if self._flick_flash > 0:
            self._flick_flash = max(0.0, self._flick_flash - dt)

        self._prev_rx = rx; self._prev_ry = ry
        self._prev_lx = lx; self._prev_ly = ly
        self._prev_right_wrist = right_wrist

    def _do_reset(self):
        """地球儀の位置・回転・ズームを初期化"""
        self.globe_cx  = self._globe_cx_default
        self.globe_cy  = self._globe_cy_default
        self.rot_lon   = 0.0
        self.rot_lat   = 0.2
        self.zoom      = 1.0
        self.vel_lon   = 0.0
        self.vel_lat   = 0.0
        self._auto_spin = False
        print("[GLOBE] リセット完了")

    def _update_dwell(self, rv: bool, rg: str, dt: float):
        """右手が都市の上で静止している時間を計測し、閾値で自動選択"""
        if not rv:
            self._dwell_loc_idx  = None
            self._dwell_progress = 0.0
            return

        hit = self._hit_location(*self._cursor)

        if hit is None:
            # 都市の上にいない → リセット
            self._dwell_loc_idx  = None
            self._dwell_progress = 0.0
            return

        if hit != self._dwell_loc_idx:
            # 別の都市に移った → タイマーリセット
            self._dwell_loc_idx = hit
            self._dwell_start   = time.time()
            self._dwell_progress = 0.0
            return

        # 同じ都市でホバー中 → 進捗を更新
        elapsed = time.time() - self._dwell_start
        self._dwell_progress = min(elapsed / self._dwell_time, 1.0)

        # 閾値到達 → 選択確定
        if self._dwell_progress >= 1.0 and not self.ai.is_thinking:
            name = LOCATIONS[hit][2]
            if name != self._selected_loc or not self.float_nodes:
                self._selected_loc  = name
                self._expanding_loc = name
                self.float_nodes.clear()
                self.ai.ask(name)
            # タイマーをリセット（連続トリガー防止）
            self._dwell_start    = time.time() + 99999
            self._dwell_progress = 0.0

    def _update(self, dt):
        # 慣性回転
        self.rot_lon += self.vel_lon * dt
        self.rot_lat += self.vel_lat * dt
        self.vel_lon *= 0.80
        self.vel_lat *= 0.80
        self.rot_lat = max(-1.4, min(1.4, self.rot_lat))

        # 自動回転
        if self._auto_spin:
            self.rot_lon += 0.05 * dt

        # ノードアニメ
        for node in self.float_nodes:
            node.scale = min(1.0, node.scale + dt * 2.5)
            # angle は固定（公転しない）
            node.pulse  = (node.pulse + dt * 3) % (2 * math.pi)

    def _check_ai(self):
        result = self.ai.pop()
        if result is None:
            return
        self.float_nodes.clear()
        category_colors = {
            "文化":  NODE_COLORS[0],
            "歴史":  NODE_COLORS[1],
            "経済":  NODE_COLORS[2],
            "自然":  NODE_COLORS[3],
            "食文化":NODE_COLORS[4],
        }
        for i, (cat, kw) in enumerate(result):
            angle  = 2 * math.pi * i / len(result) + random.uniform(-0.3, 0.3)
            radius = GLOBE_R * self.zoom + random.uniform(60, 120)
            height = random.uniform(-80, 80)
            color  = category_colors.get(cat, NODE_COLORS[i % len(NODE_COLORS)])
            node   = FloatNode(
                label=kw, category=cat,
                angle=angle, radius=radius, height=height,
                color=color, scale=0.0,
            )
            self.float_nodes.append(node)
        self._expanding_loc = None

    # ---- ヒットテスト ----

    def _location_screen_pos(self, idx):
        lat, lon, name, ci = LOCATIONS[idx]
        v = latlon_to_xyz(lat, lon)
        R = rot_x(self.rot_lat) @ rot_y(self.rot_lon)
        rv = apply_rot(R, v)
        if rv[2] < 0:        # 裏側
            return None, False
        scale = GLOBE_R * self.zoom
        sx, sy = project(rv, scale, self.globe_cx, self.globe_cy)
        return (sx, sy), True

    def _hit_location(self, sx, sy):
        best, best_d = None, 28
        for i in range(len(LOCATIONS)):
            pos, visible = self._location_screen_pos(i)
            if not visible or pos is None:
                continue
            d = math.hypot(pos[0] - sx, pos[1] - sy)
            if d < best_d:
                best_d, best = d, i
        return best

    # ---- テクスチャ球面マッピング ----

    def _load_texture(self) -> Optional[np.ndarray]:
        """世界地図テクスチャを読み込む"""
        path = os.path.join(os.path.dirname(__file__), "world_map.jpg")
        if not os.path.exists(path):
            print("[GLOBE] world_map.jpg が見つかりません。テクスチャなしで動作します。")
            return None
        import cv2 as cv
        img = cv.imread(path)
        if img is None:
            return None
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        print(f"[GLOBE] テクスチャ読み込み完了: {img.shape[1]}x{img.shape[0]}")
        return img

    def _render_textured_globe(self, R: np.ndarray) -> Optional[pygame.Surface]:
        """
        回転行列 R を使って球面テクスチャをマッピングしたサーフェスを返す。
        回転・ズームが変わった時だけ再計算（キャッシュ）。
        """
        if self._tex_rgb is None:
            return None

        scale = int(GLOBE_R * self.zoom)
        size  = scale * 2 + 2

        # 回転が僅かな変化なら前回キャッシュを返す
        R_key = (round(self.rot_lon, 3), round(self.rot_lat, 3), round(self.zoom, 2))
        if R_key == self._tex_cache_rot:
            return self._tex_surface

        tex_h, tex_w = self._tex_rgb.shape[:2]

        # ── numpy ベクトル化球面マッピング ──
        # 解像度を半分にしてから拡大（高速化）
        half = max(64, size // 2)
        r    = half

        # グリッド生成 (-1 〜 1)
        y_idx, x_idx = np.mgrid[0:half*2, 0:half*2]
        nx = (x_idx - r + 0.5) / r
        ny = (y_idx - r + 0.5) / r
        r2 = nx*nx + ny*ny
        mask = r2 <= 1.0

        # 球面座標 (カメラ方向 = +Z)
        nz        = np.zeros_like(nx)
        nz[mask]  = np.sqrt(np.clip(1.0 - r2[mask], 0, 1))

        # カメラ空間 → ワールド空間（逆回転）
        # ※ スクリーンのY軸は下向き、ワールドのY軸は上向きなので ny を反転
        R_inv  = R.T
        ny_w   = -ny   # スクリーン下向き → ワールド上向きに変換
        wx = R_inv[0,0]*nx + R_inv[0,1]*ny_w + R_inv[0,2]*nz
        wy = R_inv[1,0]*nx + R_inv[1,1]*ny_w + R_inv[1,2]*nz
        wz = R_inv[2,0]*nx + R_inv[2,1]*ny_w + R_inv[2,2]*nz

        # ワールド座標 → UV（正距円筒図法）
        lat = np.arcsin(np.clip(wy, -1, 1))    # -π/2 〜 π/2
        lon = np.arctan2(wz, wx)               # -π 〜 π

        u = ((lon / (2*np.pi)) + 0.5) % 1.0   # 0=180°W, 0.5=0°, 1=180°E
        v = 0.5 - lat / np.pi                  # 0=北極, 0.5=赤道, 1=南極

        # UV → テクスチャ座標
        tx = np.clip((u * tex_w).astype(np.int32), 0, tex_w - 1)
        ty = np.clip((v * tex_h).astype(np.int32), 0, tex_h - 1)

        # RGBA 出力バッファ
        out = np.zeros((half*2, half*2, 4), dtype=np.uint8)
        out[mask, :3] = self._tex_rgb[ty[mask], tx[mask]]
        out[mask,  3] = 255  # alpha

        # 球面の縁を少しフェード（大気光との馴染み）
        edge = np.clip((1.0 - r2) * 6, 0, 1)
        out[:,:,3] = np.where(mask, (edge * 255).astype(np.uint8), 0)

        # pygame Surface に変換
        surf_half = pygame.surfarray.make_surface(
            np.transpose(out[:,:,:3], (1,0,2))
        )
        alpha_arr = np.transpose(out[:,:,3])
        pygame.surfarray.pixels_alpha(
            surf_half.convert_alpha()
        )
        # alpha 付きで再構築
        surf_a = pygame.Surface((half*2, half*2), pygame.SRCALPHA)
        surf_a.blit(surf_half, (0,0))
        # alpha チャンネルを設定
        pa = pygame.surfarray.pixels_alpha(surf_a)
        pa[:] = np.transpose(out[:,:,3])
        del pa

        # フルサイズにスケール
        full = pygame.transform.smoothscale(surf_a, (size, size))

        self._tex_cache_rot  = R_key
        self._tex_surface    = full
        return full

    # ---- シネマサーフェス生成（起動時1回） ----

    def _make_vignette(self):
        """画面周囲を暗くするビネット"""
        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        cx, cy = WIN_W // 2, WIN_H // 2
        max_r  = math.hypot(cx, cy)
        for step in range(80, 0, -4):
            r_px  = int(max_r * step / 80)
            alpha = int(180 * (1 - step / 80) ** 2)
            pygame.draw.ellipse(surf, (0, 0, 0, alpha),
                                (cx - r_px, cy - r_px * WIN_H // WIN_W,
                                 r_px * 2,  r_px * 2 * WIN_H // WIN_W))
        return surf

    def _make_nebula(self):
        """星雲っぽいぼんやりした光のクラウド"""
        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        rng  = random.Random(7)
        # 数個の大きなぼんやりした楕円
        blobs = [
            (200,  150, 320, 180, (30, 20, 80),  35),
            (900,  600, 280, 140, (10, 40, 90),  30),
            (1100, 200, 200, 120, (60, 20, 80),  25),
            (400,  500, 240, 160, (20, 50, 100), 20),
            (1300, 400, 180, 100, (40, 15, 70),  18),
        ]
        for (bx, by, bw, bh, bc, layers) in blobs:
            for l in range(layers, 0, -2):
                a = int(12 * (l / layers))
                lsurf = pygame.Surface((bw, bh), pygame.SRCALPHA)
                pygame.draw.ellipse(lsurf, (*bc, a), (0, 0, bw, bh))
                # ガウスっぽいぼかし代わりに段階縮小
                sw = int(bw * l / layers)
                sh = int(bh * l / layers)
                if sw > 0 and sh > 0:
                    scaled = pygame.transform.smoothscale(lsurf, (sw, sh))
                    surf.blit(scaled, (bx - sw // 2, by - sh // 2))
        return surf

    def _make_stars(self):
        """シネマ品質の星空（輝度・サイズ・色にばらつき）"""
        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        rng  = random.Random(42)
        # 暗い小さな星（多数）
        for _ in range(600):
            x  = rng.randint(0, WIN_W)
            y  = rng.randint(0, WIN_H)
            br = rng.randint(60, 160)
            pygame.draw.circle(surf, (br, br, br + rng.randint(0, 30), 255), (x, y), 1)
        # 中程度の星
        for _ in range(80):
            x  = rng.randint(0, WIN_W)
            y  = rng.randint(0, WIN_H)
            br = rng.randint(160, 230)
            pygame.draw.circle(surf, (br, br, min(255, br + 40), 220), (x, y), 1)
        # 明るい星（少数）+ 十字フレア
        for _ in range(18):
            x  = rng.randint(0, WIN_W)
            y  = rng.randint(0, WIN_H)
            br = rng.randint(210, 255)
            tint = rng.choice([(255,255,255),(200,220,255),(255,240,200)])
            pygame.draw.circle(surf, (*tint, 255), (x, y), 2)
            # 十字フレア
            for l in range(1, 10):
                a = int(180 / l)
                pygame.draw.line(surf, (*tint, a), (x-l*3, y), (x+l*3, y), 1)
                pygame.draw.line(surf, (*tint, a), (x, y-l*3), (x, y+l*3), 1)
        return surf

    # ---- 描画 ----

    def _draw(self):
        # 真っ黒背景
        self.screen.fill(BG_COLOR)

        # 星雲
        self.screen.blit(self._nebula_surf, (0, 0))

        # 星空
        self.screen.blit(self._star_surf, (0, 0))

        # 地球儀
        self._draw_globe()

        # 知識ノード
        self._draw_float_nodes()

        # フリックフラッシュ（後方互換）
        if self._flick_flash > 0:
            alpha = int(self._flick_flash / 0.25 * 60)
            flash = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            flash.fill((120, 200, 255, alpha))
            self.screen.blit(flash, (0, 0))

        # リセットフラッシュ（白く光る）
        if self._reset_flash > 0:
            alpha = int(self._reset_flash / 0.5 * 120)
            flash = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            flash.fill((220, 240, 255, alpha))
            self.screen.blit(flash, (0, 0))

        # フィルムグレイン
        self._draw_grain()

        # ビネット（最後に重ねる）
        self.screen.blit(self._vignette_surf, (0, 0))

        # カーソル・HUD（ビネットの上）
        self._draw_cursor()
        self._draw_hud()

    def _draw_grain(self):
        """フィルムグレイン（毎フレーム微妙に変化）"""
        self._grain_surf.fill((0, 0, 0, 0))
        rng = random.Random(int(time.time() * 30))
        for _ in range(2000):
            x = rng.randint(0, WIN_W - 1)
            y = rng.randint(0, WIN_H - 1)
            v = rng.randint(0, 40)
            self._grain_surf.set_at((x, y), (v, v, v, rng.randint(10, 35)))
        self.screen.blit(self._grain_surf, (0, 0))

    def _draw_globe(self):
        scale = GLOBE_R * self.zoom
        cx, cy = self.globe_cx, self.globe_cy
        R = rot_x(self.rot_lat) @ rot_y(self.rot_lon)

        # ── 大気光（複数層のハロー）──
        atmo_layers = [
            (scale * 1.18, (30,  80, 180, 18)),
            (scale * 1.10, (50, 110, 220, 28)),
            (scale * 1.05, (70, 140, 255, 40)),
            (scale * 1.02, (90, 160, 255, 55)),
        ]
        atmo_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for r, c in atmo_layers:
            pygame.draw.circle(atmo_surf, c, (cx, cy), int(r))
        self.screen.blit(atmo_surf, (0, 0))

        # ── 球体ベース（海の色）──
        pygame.draw.circle(self.screen, OCEAN_C, (cx, cy), int(scale))

        # ── テクスチャ球面マッピング ──
        tex_surf = self._render_textured_globe(R)
        if tex_surf is not None:
            tw = tex_surf.get_width()
            self.screen.blit(tex_surf, (cx - tw//2, cy - tw//2))

        # ── 球面グラデーション（ハイライト：左上から光が当たる感じ）──
        grad_surf = pygame.Surface((int(scale)*2+4, int(scale)*2+4), pygame.SRCALPHA)
        for i in range(int(scale), 0, -4):
            # 右下を暗く
            offset_x = int(scale * 0.25)
            offset_y = int(scale * 0.15)
            dx = offset_x
            dy = offset_y
            ratio = i / scale
            a = int(28 * (1 - ratio) ** 1.5)
            pygame.draw.circle(grad_surf, (0, 10, 40, a),
                               (int(scale)+4 + dx, int(scale)+4 + dy), i)
        # 左上ハイライト
        for i in range(int(scale * 0.55), 0, -6):
            ox = -int(scale * 0.28)
            oy = -int(scale * 0.22)
            ratio = i / (scale * 0.55)
            a = int(22 * (1 - ratio) ** 1.8)
            pygame.draw.circle(grad_surf, (120, 180, 255, a),
                               (int(scale)+4 + ox, int(scale)+4 + oy), i)
        self.screen.blit(grad_surf, (cx - int(scale) - 4, cy - int(scale) - 4))

        # 緯度・経度線（半透明サーフェスで重ねる）
        grid_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        GRID_A = (*GRID_C, 55)   # 半透明

        for lat in self._grid_lats:
            points = []
            for lon in range(-180, 181, 4):
                v  = latlon_to_xyz(lat, lon)
                rv = apply_rot(R, v)
                if rv[2] < 0: continue
                sx, sy = project(rv, scale, cx, cy)
                points.append((sx, sy))
            for i in range(len(points) - 1):
                pygame.draw.line(grid_surf, GRID_A, points[i], points[i+1], 1)

        for lon in self._grid_lons:
            points = []
            for lat in range(-90, 91, 3):
                v  = latlon_to_xyz(lat, lon)
                rv = apply_rot(R, v)
                if rv[2] < 0: continue
                sx, sy = project(rv, scale, cx, cy)
                points.append((sx, sy))
            for i in range(len(points) - 1):
                pygame.draw.line(grid_surf, GRID_A, points[i], points[i+1], 1)

        self.screen.blit(grid_surf, (0, 0))

        # 縁取り（細くほのかに）
        rim_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.circle(rim_surf, (*GLOW_C, 120), (cx, cy), int(scale), 2)
        pygame.draw.circle(rim_surf, (200, 230, 255, 60), (cx, cy), int(scale)+1, 1)
        self.screen.blit(rim_surf, (0, 0))

        # ── レンズフレア（左上の光源） ──
        self._draw_lens_flare(cx, cy, scale)

        # 都市マーカー
        self._draw_locations(R, scale, cx, cy)

    def _draw_lens_flare(self, cx, cy, scale):
        """インターステラー風レンズフレア（左上の光源から）"""
        # 光源位置（左上）
        lx = cx - scale * 0.5
        ly = cy - scale * 0.45

        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)

        # 光源の輝点
        for r, a in [(40, 8), (20, 18), (10, 35), (5, 60), (2, 100)]:
            pygame.draw.circle(surf, (200, 220, 255, a), (int(lx), int(ly)), r)

        # 中心へのフレア列（光軸）
        axis_x = cx - lx
        axis_y = cy - ly
        flares = [
            (0.3,  12, (180, 200, 255, 25)),
            (0.5,   8, (200, 210, 255, 20)),
            (0.7,  18, (160, 190, 255, 15)),
            (1.0,   6, (210, 225, 255, 30)),
            (1.3,  14, (170, 200, 255, 18)),
            (1.6,   5, (220, 230, 255, 22)),
            (1.9,  10, (180, 210, 255, 14)),
        ]
        for t, r, c in flares:
            fx = int(lx + axis_x * t)
            fy = int(ly + axis_y * t)
            pygame.draw.circle(surf, c, (fx, fy), r)

        # 光源から放射する細い光芒（スターバースト）
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            length = scale * 0.6
            ex = lx + math.cos(angle) * length
            ey = ly + math.sin(angle) * length
            for step in range(10):
                t   = step / 10
                px  = int(lx + (ex - lx) * t)
                py  = int(ly + (ey - ly) * t)
                a   = int(20 * (1 - t) ** 2)
                surf.set_at((max(0, min(WIN_W-1, px)),
                             max(0, min(WIN_H-1, py))), (200, 220, 255, a))

        self.screen.blit(surf, (0, 0))

    def _draw_locations(self, R, scale, cx, cy):
        hover_idx = self._hover_loc_idx
        selected  = self._selected_loc

        for i, (lat, lon, name, ci) in enumerate(LOCATIONS):
            v  = latlon_to_xyz(lat, lon)
            rv = apply_rot(R, v)
            if rv[2] < 0:   # 裏側は描画しない
                continue

            sx, sy = project(rv, scale, cx, cy)
            color  = NODE_COLORS[ci % len(NODE_COLORS)]
            is_hover    = (i == hover_idx)
            is_selected = (name == selected)

            # グロー
            if is_hover or is_selected:
                for gr in range(18, 0, -4):
                    a_ = int(80 * (1 - gr / 18))
                    gs_ = pygame.Surface((gr*2, gr*2), pygame.SRCALPHA)
                    pygame.draw.circle(gs_, (*color, a_), (gr, gr), gr)
                    self.screen.blit(gs_, (sx - gr, sy - gr))

            r = 7 if is_selected else (5 if is_hover else 4)
            pygame.draw.circle(self.screen, color, (sx, sy), r)
            pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), r, 1)

            # ラベル
            if is_hover or is_selected:
                lsurf = self.font_md.render(name, True, (255, 255, 255))
                lw, lh = lsurf.get_size()
                bg = pygame.Surface((lw + 8, lh + 4), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 180))
                self.screen.blit(bg, (sx - lw // 2 - 4, sy - lh - 12))
                self.screen.blit(lsurf, (sx - lw // 2, sy - lh - 10))

    def _draw_float_nodes(self):
        if not self.float_nodes:
            return

        cx, cy = self.globe_cx, self.globe_cy

        # 選択都市名バナー
        if self._selected_loc:
            title = f"📍 {self._selected_loc}"
            ts = self.font_lg.render(title, True, (255, 240, 200))
            tw = ts.get_width()
            bg = pygame.Surface((tw + 20, ts.get_height() + 8), pygame.SRCALPHA)
            bg.fill((20, 15, 5, 200))
            self.screen.blit(bg, (cx - tw // 2 - 10, cy - GLOBE_R * self.zoom - 70))
            self.screen.blit(ts, (cx - tw // 2, cy - GLOBE_R * self.zoom - 66))

        for node in self.float_nodes:
            if node.scale < 0.02:
                continue

            nx = cx + math.cos(node.angle) * node.radius
            ny = cy + node.height + math.sin(node.pulse * 0.5) * 8   # ふわふわ

            # 接続線（地球儀の中心から）
            line_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            alpha_line = int(80 * node.scale)
            pygame.draw.line(line_surf, (*node.color, alpha_line),
                             (cx, cy), (int(nx), int(ny)), 1)
            self.screen.blit(line_surf, (0, 0))

            # ノードグロー
            gr = int(30 * node.scale)
            if gr > 0:
                gs_ = pygame.Surface((gr*2, gr*2), pygame.SRCALPHA)
                pygame.draw.circle(gs_, (*node.color, 50), (gr, gr), gr)
                self.screen.blit(gs_, (int(nx) - gr, int(ny) - gr))

            # ノード本体
            nr = int(18 * node.scale)
            if nr > 2:
                pygame.draw.circle(self.screen, node.color, (int(nx), int(ny)), nr)
                pygame.draw.circle(self.screen, (255,255,255), (int(nx), int(ny)), nr, 1)

            # カテゴリ
            cat_s = self.font_sm.render(node.category, True, (180, 200, 255))
            self.screen.blit(cat_s, (int(nx) - cat_s.get_width()//2, int(ny) - nr - 28))

            # キーワード
            kw_s  = self.font_md.render(node.label, True, (255, 255, 255))
            kw_bg = pygame.Surface((kw_s.get_width()+8, kw_s.get_height()+4), pygame.SRCALPHA)
            kw_bg.fill((0,0,0,160))
            self.screen.blit(kw_bg, (int(nx) - kw_s.get_width()//2 - 4, int(ny) + nr + 4))
            self.screen.blit(kw_s,  (int(nx) - kw_s.get_width()//2,     int(ny) + nr + 6))

    def _draw_cursor(self):
        snap = self.gs.snapshot()
        rv   = snap["right_visible"]
        rf   = snap.get("right_fist",  False)
        rp   = snap.get("right_pinch", False)
        cx, cy = self._cursor

        if not rv:
            return  # 手が見えていなければ描画しない

        # ジェスチャーで色・形変更
        if rf:
            c = (100, 120, 255)  # 青紫：グラブ中
            # グラブ時は閉じた手のアイコン（塗りつぶし円）
            pygame.draw.circle(self.screen, c, (cx, cy), 8)
            pygame.draw.circle(self.screen, (200, 210, 255), (cx, cy), 8, 2)
        elif rp:
            c = (0, 220, 220)    # シアン：ピンチ
            pygame.draw.circle(self.screen, c, (cx, cy), 6)
            pygame.draw.circle(self.screen, (200, 255, 255), (cx, cy), 6, 2)
        elif self._dwell_loc_idx is not None:
            c = (80, 220, 255)   # 水色：ホバー中
            pygame.draw.line(self.screen, c, (cx-14, cy), (cx+14, cy), 1)
            pygame.draw.line(self.screen, c, (cx, cy-14), (cx, cy+14), 1)
            pygame.draw.circle(self.screen, c, (cx, cy), 5, 1)
        else:
            c = (200, 220, 255)  # 白：通常（オープンハンド）
            pygame.draw.line(self.screen, c, (cx-14, cy), (cx+14, cy), 1)
            pygame.draw.line(self.screen, c, (cx, cy-14), (cx, cy+14), 1)
            pygame.draw.circle(self.screen, c, (cx, cy), 5, 1)

        # リセットタイマーUI（両手が高い位置にある時）
        if self._reset_timer > 0:
            prog = min(self._reset_timer / 2.0, 1.0)
            r = 40
            bar_w = int(160 * prog)
            pygame.draw.rect(self.screen, (60, 80, 110),
                             (WIN_W//2 - 80, WIN_H//2 + 60, 160, 12), border_radius=6)
            pygame.draw.rect(self.screen, (100, 200, 255),
                             (WIN_W//2 - 80, WIN_H//2 + 60, bar_w, 12), border_radius=6)
            msg = self.font_md.render("RESET...", True, (100, 200, 255))
            self.screen.blit(msg, (WIN_W//2 - msg.get_width()//2, WIN_H//2 + 78))

        # ── ドウェルプログレスリング ──
        if self._dwell_progress > 0.0:
            r    = 26
            prog = self._dwell_progress
            # 背景リング
            pygame.draw.circle(self.screen, (60, 80, 100), (cx, cy), r, 2)
            # 進捗弧（扇形サーフェス）
            surf  = pygame.Surface((r*2+4, r*2+4), pygame.SRCALPHA)
            steps = 60
            pts   = [(r+2, r+2)]
            for i in range(int(steps * prog) + 1):
                angle = math.radians(-90 + 360 * i / steps)
                px = r + 2 + math.cos(angle) * r
                py = r + 2 + math.sin(angle) * r
                pts.append((int(px), int(py)))
            if len(pts) >= 3:
                pygame.draw.polygon(surf, (80, 220, 255, 60), pts)
            # 先端の輝点
            end_angle = math.radians(-90 + 360 * prog)
            ex = int(r + 2 + math.cos(end_angle) * r)
            ey = int(r + 2 + math.sin(end_angle) * r)
            pygame.draw.circle(surf, (150, 240, 255, 220), (ex, ey), 4)
            self.screen.blit(surf, (cx - r - 2, cy - r - 2))

            # 残り時間テキスト
            remain = self._dwell_time * (1 - prog)
            if remain > 0.05:
                ts = self.font_sm.render(f"{remain:.1f}s", True, (180, 240, 255))
                self.screen.blit(ts, (cx - ts.get_width()//2, cy + r + 6))

    def _draw_hud(self):
        snap   = self.gs.snapshot()
        rg     = snap["right_gesture"]
        lg     = snap["left_gesture"]
        rv     = snap["right_visible"]
        lv     = snap["left_visible"]
        bv     = snap["both_visible"]
        fps_c  = snap["fps"]
        fps_a  = self.clock.get_fps()

        # 操作ガイド（左上）
        snap2   = self.gs.snapshot()
        rf2     = snap2.get("right_fist", False)
        lf2     = snap2.get("left_fist",  False)
        rp2     = snap2.get("right_pinch", False)
        dwell_active = self._dwell_loc_idx is not None
        grab_active  = self._grab_active
        zoom_active  = self._zoom_active
        reset_active = self._reset_timer > 0.1
        guide = [
            ("🖐 手を開いて動かす",  "地球儀を回す",       (200, 220, 255), rv and not rf2 and not bv),
            ("✊ グーで動かす",      "地球儀を移動",       (100, 120, 255), grab_active),
            ("↩ グー＋手首ひねり",  "地球儀をスピン",     (160, 140, 255), grab_active),
            ("🤲 両手を広げ縮める", "ズーム",             (255, 220, 50),  zoom_active),
            ("✊✊ 両手グー移動",    "地球儀を移動",       (200, 160, 255), self._both_grab_active),
            ("👆 ピンチ",           "都市を選択",         (0,   220, 220), rp2),
            ("🎯 都市の上で止まる",  "1.5秒 → AI解説",    (80,  200, 255), dwell_active),
            ("🙌 両手を高く2秒",    "リセット",           (255, 120, 80),  reset_active),
        ]
        panel_h = len(guide) * 24 + 16
        pygame.draw.rect(self.screen, (10, 12, 30), (10, 10, 300, panel_h), border_radius=8)

        for i, (icon, desc, c, active) in enumerate(guide):
            fc = c if active else (60, 70, 95)
            prefix = "▶ " if active else "  "
            s = self.font_md.render(f"{prefix}{icon}  {desc}", True, fc)
            self.screen.blit(s, (14, 18 + i * 24))

        # 手の検出状態バッジ
        badge_y = panel_h + 18
        for label, visible, bg_c in [
            ("右手", rv, (0,   220, 100)),
            ("左手", lv, (60,  120, 255)),
        ]:
            bx = 10 if label == "右手" else 105
            badge_surf = pygame.Surface((88, 26), pygame.SRCALPHA)
            alpha = 200 if visible else 60
            badge_surf.fill((*bg_c, alpha // 5))
            pygame.draw.rect(badge_surf, (*bg_c, alpha), (0,0,88,26), 1, border_radius=4)
            bs = self.font_sm.render(f"{label}: {'✓' if visible else '---'}", True, bg_c)
            badge_surf.blit(bs, (6, 5))
            self.screen.blit(badge_surf, (bx, badge_y))

        # FPS
        f = self.font_sm.render(f"App {fps_a:.0f} | Cam {fps_c:.0f} fps", True, (60, 80, 110))
        self.screen.blit(f, (WIN_W - f.get_width() - 12, 12))

        # ズームレベル
        zs = self.font_sm.render(f"Zoom {self.zoom:.1f}x", True, (70, 90, 120))
        self.screen.blit(zs, (WIN_W - zs.get_width() - 12, 34))

        # AI 思考中バナー
        if self.ai.is_thinking and self._expanding_loc:
            t    = time.time()
            dots = "." * (int(t * 2) % 4)
            msg  = f"AI が「{self._expanding_loc}」を解析中{dots}"
            ms   = self.font_lg.render(msg, True, AI_COLOR)
            mw   = ms.get_width()
            bx   = WIN_W // 2 - mw // 2 - 10
            by   = WIN_H - 54
            pygame.draw.rect(self.screen, (30, 26, 8), (bx - 2, by - 2, mw + 24, ms.get_height() + 14), border_radius=8)
            pygame.draw.rect(self.screen, AI_COLOR,    (bx, by, mw + 20, ms.get_height() + 10), 1, border_radius=8)
            self.screen.blit(ms, (bx + 10, by + 5))

        # 自動回転インジケータ
        spin_c = (80, 180, 80) if self._auto_spin else (80, 80, 80)
        spin_s = self.font_sm.render("SPIN: " + ("ON [SPACE]" if self._auto_spin else "OFF [SPACE]"), True, spin_c)
        self.screen.blit(spin_s, (WIN_W - spin_s.get_width() - 12, WIN_H - 62))

        # 録画インジケータ
        rec = self._recorder
        if rec.is_recording:
            # 点滅する赤丸
            blink = int(time.time() * 2) % 2 == 0
            if blink:
                pygame.draw.circle(self.screen, (255, 40, 40),
                                   (WIN_W - 18, WIN_H - 100), 7)
            mode_label = "WIN" if rec.mode == "window" else "SCR"
            t = rec.elapsed
            rec_s = self.font_sm.render(
                f"● REC [{mode_label}]  {int(t//60):02d}:{int(t%60):02d}",
                True, (255, 80, 80))
            self.screen.blit(rec_s, (WIN_W - rec_s.get_width() - 12, WIN_H - 100))
        else:
            hint = self.font_sm.render("R: 地球儀＋カメラ録画  T: 全画面録画", True, (55, 65, 85))
            self.screen.blit(hint, (WIN_W - hint.get_width() - 12, WIN_H - 100))

        # 音量インジケータ
        vol_pct = int(self._bgm_volume * 100)
        muted   = vol_pct == 0
        vol_c   = (80, 80, 80) if muted else (100, 180, 255)
        vol_label = "BGM: MUTE" if muted else f"BGM: {'▮' * (vol_pct // 10)}{'▯' * (10 - vol_pct // 10)} {vol_pct}%"
        vol_s   = self.font_sm.render(vol_label, True, vol_c)
        self.screen.blit(vol_s, (WIN_W - vol_s.get_width() - 12, WIN_H - 24))
