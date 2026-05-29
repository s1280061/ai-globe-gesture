"""
knowledge_space.py
3D ノード空間の描画・操作・AI連携
"""

import pygame
import math
import random
import time
import threading
import os
from dataclasses import dataclass, field
from typing import Optional
import anthropic


# ============================================================
# 定数
# ============================================================
WIN_W, WIN_H = 1280, 720

# 色
BG_COLOR     = (8,   10,  20)
GRID_COLOR   = (25,  30,  55)
NODE_COLORS  = [
    (80,  160, 255),   # 青
    (120, 220, 160),   # 緑
    (255, 160,  80),   # 橙
    (220,  80, 180),   # ピンク
    (160, 120, 255),   # 紫
    (80,  220, 220),   # シアン
]
EDGE_COLOR        = (50,  70, 110)
EDGE_ACTIVE_COLOR = (100, 160, 255)
CURSOR_COLOR      = (255, 255, 255)
TEXT_COLOR        = (220, 230, 255)
AI_THINKING_COLOR = (255, 220,  60)

# カメラ / 3D
FOV        = 600.0   # 透視投影の焦点距離
CAMERA_Z   = 5.0     # カメラの Z 位置
NODE_RADIUS = 22


# ============================================================
# データクラス
# ============================================================
@dataclass
class Node3D:
    id: int
    label: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    color_idx: int = 0
    parent_id: Optional[int] = None
    scale: float = 0.0          # アニメ用 (0→1)
    selected: bool = False
    pulse: float = 0.0           # 選択時のパルスアニメ

    @property
    def color(self):
        return NODE_COLORS[self.color_idx % len(NODE_COLORS)]

    def project(self, cam_x, cam_y, cam_z):
        """3D→2Dスクリーン座標に透視投影"""
        dz = self.z - cam_z
        if dz >= 0:
            dz = -0.1
        scale = FOV / (-dz)
        sx = WIN_W // 2 + (self.x - cam_x) * scale
        sy = WIN_H // 2 + (self.y - cam_y) * scale
        return sx, sy, scale


@dataclass
class Edge:
    src: int
    dst: int


# ============================================================
# AI ノード生成
# ============================================================
class AIExpander:
    """バックグラウンドスレッドで Claude API を呼び出しノードを生成する"""

    def __init__(self):
        self.client   = anthropic.Anthropic()
        self._pending = threading.Event()
        self._result: Optional[list] = None
        self._topic  = ""
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_thinking(self):
        return self._thread is not None and self._thread.is_alive()

    def expand(self, topic: str):
        """topic に関連する概念を非同期で生成開始"""
        if self.is_thinking:
            return
        self._topic  = topic
        self._result = None
        self._thread = threading.Thread(target=self._call_api, daemon=True)
        self._thread.start()

    def pop_result(self):
        """結果があれば取り出して None にリセット"""
        with self._lock:
            r = self._result
            self._result = None
            return r

    def _call_api(self):
        prompt = (
            f"「{self._topic}」に密接に関連する概念・キーワードを5つ挙げてください。\n"
            "各概念は10文字以内の日本語で。\n"
            "箇条書きで概念名だけを返してください。説明は不要です。\n"
            "例:\n- 機械学習\n- ニューラルネット\n..."
        )
        try:
            msg = self.client.messages.create(
                model="claude-opus-4-5",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            concepts = []
            for line in text.splitlines():
                line = line.strip().lstrip("-・• ").strip()
                if line and len(line) <= 20:
                    concepts.append(line)
            with self._lock:
                self._result = concepts[:5]
        except Exception as e:
            with self._lock:
                self._result = [f"Error: {e}"[:20]]


# ============================================================
# メインスペース
# ============================================================
class KnowledgeSpace:

    def __init__(self, gesture_state):
        self.gs     = gesture_state
        self.ai     = AIExpander()

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("AI Knowledge Space")
        self.clock  = pygame.time.Clock()

        # フォント
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        if os.path.exists(font_path):
            self.font_lg = pygame.font.Font(font_path, 18)
            self.font_sm = pygame.font.Font(font_path, 13)
            self.font_ui = pygame.font.Font(font_path, 14)
        else:
            self.font_lg = pygame.font.SysFont("arial", 18)
            self.font_sm = pygame.font.SysFont("arial", 13)
            self.font_ui = pygame.font.SysFont("arial", 14)

        # カメラ位置
        self.cam_x, self.cam_y, self.cam_z = 0.0, 0.0, CAMERA_Z

        # ノード & エッジ
        self.nodes: dict[int, Node3D] = {}
        self.edges: list[Edge]        = []
        self._next_id = 0

        # 初期ノード
        root = self._add_node("AI Knowledge", x=0, y=0, z=0)
        root.scale = 1.0

        # ジェスチャー状態
        self._prev_gesture  = "none"
        self._drag_node_id: Optional[int] = None
        self._hover_node_id: Optional[int] = None
        self._selected_node_id: Optional[int] = None

        # ズーム用
        self._peace_ref_y = 0.5

        # AI 展開中のノード
        self._expanding_node_id: Optional[int] = None

        # カーソル（スクリーン座標）
        self._cursor = (WIN_W // 2, WIN_H // 2)

    # ---- ノード管理 ----

    def _add_node(self, label, x=None, y=None, z=None, parent_id=None, color_idx=None) -> Node3D:
        nid = self._next_id
        self._next_id += 1
        if x is None: x = random.uniform(-3, 3)
        if y is None: y = random.uniform(-2, 2)
        if z is None: z = random.uniform(-4, 4)
        if color_idx is None: color_idx = random.randint(0, len(NODE_COLORS) - 1)

        node = Node3D(id=nid, label=label, x=x, y=y, z=z,
                      color_idx=color_idx, parent_id=parent_id)
        self.nodes[nid] = node
        if parent_id is not None and parent_id in self.nodes:
            self.edges.append(Edge(parent_id, nid))
        return node

    def _spawn_children(self, parent: Node3D, concepts: list[str]):
        """AI が生成した概念をノードとして parent の周囲に配置"""
        n = len(concepts)
        for i, concept in enumerate(concepts):
            angle = 2 * math.pi * i / n
            r = 2.2
            cx = parent.x + r * math.cos(angle)
            cy = parent.y + r * math.sin(angle)
            cz = parent.z + random.uniform(-1.5, 1.5)
            child = self._add_node(concept, x=cx, y=cy, z=cz,
                                   parent_id=parent.id,
                                   color_idx=(parent.color_idx + i + 1) % len(NODE_COLORS))
            # 初期スケール 0 でアニメ開始
            child.scale = 0.0

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

            snap = self.gs.snapshot()
            self._process_gesture(snap, dt)
            self._update_physics(dt)
            self._check_ai_result()
            self._draw()
            pygame.display.flip()

        pygame.quit()

    # ---- ジェスチャー処理 ----

    def _process_gesture(self, snap: dict, dt: float):
        gesture     = snap["gesture"]
        cx          = snap["cursor_x"]
        cy          = snap["cursor_y"]
        pinch_just  = snap["pinch_just"]
        hand_visible = snap["hand_visible"]

        # カーソルをスクリーン座標に変換
        self._cursor = (int(cx * WIN_W), int(cy * WIN_H))

        # ホバー判定
        self._hover_node_id = self._hit_test(*self._cursor)

        if gesture == "point":
            # カーソル移動のみ
            self._drag_node_id = None

        elif gesture == "pinch":
            if pinch_just:
                # ノードを選択 → AI 展開
                hid = self._hover_node_id
                if hid is not None:
                    self._selected_node_id = hid
                    node = self.nodes[hid]
                    node.selected = True
                    if not self.ai.is_thinking:
                        self._expanding_node_id = hid
                        self.ai.expand(node.label)

        elif gesture == "fist":
            # ドラッグ開始
            if self._prev_gesture != "fist":
                self._drag_node_id = self._hover_node_id
            if self._drag_node_id is not None:
                self._drag_node_to_cursor(self._drag_node_id, cx, cy)

        elif gesture == "peace":
            # ズーム: ピース開始時の y を基準にする
            if self._prev_gesture != "peace":
                self._peace_ref_y = cy
            delta = cy - self._peace_ref_y
            self.cam_z += delta * 6 * dt
            self.cam_z = max(-8, min(12, self.cam_z))
            self._peace_ref_y = cy

        elif gesture in ("open", "none"):
            self._drag_node_id = None

        self._prev_gesture = gesture

    def _drag_node_to_cursor(self, nid: int, cx: float, cy: float):
        """カーソル位置にノードを追随させる (Z は固定)"""
        node = self.nodes.get(nid)
        if node is None:
            return
        dz = node.z - self.cam_z
        if dz >= 0:
            dz = -0.1
        scale = FOV / (-dz)
        node.x = (cx * WIN_W - WIN_W // 2) / scale + self.cam_x
        node.y = (cy * WIN_H - WIN_H // 2) / scale + self.cam_y

    def _hit_test(self, sx: int, sy: int) -> Optional[int]:
        """スクリーン座標 (sx,sy) に最も近いノードの id を返す"""
        best_id   = None
        best_dist = NODE_RADIUS * 1.5
        for nid, node in self.nodes.items():
            if node.scale < 0.3:
                continue
            px, py, _ = node.project(self.cam_x, self.cam_y, self.cam_z)
            d = math.hypot(px - sx, py - sy)
            if d < best_dist:
                best_dist = d
                best_id   = nid
        return best_id

    # ---- 物理シミュレーション ----

    def _update_physics(self, dt: float):
        nodes = list(self.nodes.values())

        # ノードのスケールアニメ
        for node in nodes:
            if node.scale < 1.0:
                node.scale = min(1.0, node.scale + dt * 2.0)

            # 選択パルス
            if node.selected:
                node.pulse = (node.pulse + dt * 4) % (2 * math.pi)

        # 斥力（ノード同士が重ならないよう）
        REPULSE = 1.5
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
                dist2 = dx*dx + dy*dy + dz*dz + 0.01
                f = REPULSE / dist2
                a.vx += dx * f * dt;  b.vx -= dx * f * dt
                a.vy += dy * f * dt;  b.vy -= dy * f * dt
                a.vz += dz * f * dt;  b.vz -= dz * f * dt

        # 引力（エッジ）
        ATTRACT = 0.8
        IDEAL   = 2.5
        for edge in self.edges:
            a = self.nodes.get(edge.src)
            b = self.nodes.get(edge.dst)
            if a is None or b is None:
                continue
            dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
            dist_ = math.sqrt(dx*dx + dy*dy + dz*dz + 0.01)
            f = ATTRACT * (dist_ - IDEAL) / dist_
            a.vx += dx * f * dt;  b.vx -= dx * f * dt
            a.vy += dy * f * dt;  b.vy -= dy * f * dt
            a.vz += dz * f * dt;  b.vz -= dz * f * dt

        # 減衰＋位置更新
        DAMP = 0.85
        for node in nodes:
            if node.id == self._drag_node_id:
                node.vx = node.vy = node.vz = 0
                continue
            node.vx *= DAMP;  node.vy *= DAMP;  node.vz *= DAMP
            node.x  += node.vx * dt
            node.y  += node.vy * dt
            node.z  += node.vz * dt
            # 空間の中心に軽く引き戻す
            node.vx -= node.x * 0.05 * dt
            node.vy -= node.y * 0.05 * dt
            node.vz -= node.z * 0.05 * dt

    # ---- AI 結果の反映 ----

    def _check_ai_result(self):
        result = self.ai.pop_result()
        if result is None:
            return
        parent = self.nodes.get(self._expanding_node_id)
        if parent is None:
            return
        self._spawn_children(parent, result)
        self._expanding_node_id = None

    # ---- 描画 ----

    def _draw(self):
        self.screen.fill(BG_COLOR)
        self._draw_grid()
        self._draw_edges()
        self._draw_nodes()
        self._draw_cursor()
        self._draw_hud()

    def _draw_grid(self):
        """奥行き感のある格子線"""
        surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for i in range(-8, 9):
            for z_val in [-6, -2, 2]:
                dz = z_val - self.cam_z
                if dz >= 0: continue
                sc = FOV / (-dz)
                x0 = WIN_W // 2 + (i * 1.5 - self.cam_x) * sc
                y0 = WIN_H // 2 + (-6 - self.cam_y) * sc
                x1 = WIN_W // 2 + (i * 1.5 - self.cam_x) * sc
                y1 = WIN_H // 2 + (6  - self.cam_y) * sc
                alpha = max(0, min(80, int(60 + dz * 5)))
                pygame.draw.line(surf, (*GRID_COLOR, alpha),
                                 (int(x0), int(y0)), (int(x1), int(y1)), 1)
        self.screen.blit(surf, (0, 0))

    def _draw_edges(self):
        for edge in self.edges:
            a = self.nodes.get(edge.src)
            b = self.nodes.get(edge.dst)
            if a is None or b is None: continue
            if a.scale < 0.1 or b.scale < 0.1: continue

            ax, ay, asc = a.project(self.cam_x, self.cam_y, self.cam_z)
            bx, by, bsc = b.project(self.cam_x, self.cam_y, self.cam_z)

            active = (a.id in (self._hover_node_id, self._selected_node_id) or
                      b.id in (self._hover_node_id, self._selected_node_id))
            color = EDGE_ACTIVE_COLOR if active else EDGE_COLOR
            alpha = int(180 * min(a.scale, b.scale))
            surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.line(surf, (*color, alpha),
                             (int(ax), int(ay)), (int(bx), int(by)), 1)
            self.screen.blit(surf, (0, 0))

    def _draw_nodes(self):
        # Z でソート（遠いものを先に描画）
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.z)

        for node in sorted_nodes:
            if node.scale <= 0.01: continue
            sx, sy, sc = node.project(self.cam_x, self.cam_y, self.cam_z)
            if not (-100 < sx < WIN_W + 100 and -100 < sy < WIN_H + 100):
                continue

            r = int(NODE_RADIUS * node.scale * min(1.5, max(0.3, sc / 60)))
            if r < 2: continue

            c = node.color
            hover   = node.id == self._hover_node_id
            sel     = node.id == self._selected_node_id
            expand  = node.id == self._expanding_node_id

            # グロー
            if hover or sel or expand:
                glow_r = r + 14
                glow_c = AI_THINKING_COLOR if expand else (200, 220, 255)
                glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
                for gr in range(glow_r, 0, -3):
                    a_ = int(60 * (gr / glow_r))
                    pygame.draw.circle(glow_surf, (*glow_c, a_),
                                       (glow_r, glow_r), gr)
                self.screen.blit(glow_surf, (int(sx) - glow_r, int(sy) - glow_r))

            # パルスリング（選択中）
            if sel and not expand:
                pr = r + int(6 + 4 * math.sin(node.pulse))
                pygame.draw.circle(self.screen, (180, 210, 255), (int(sx), int(sy)), pr, 1)

            # AI 展開中スピナー
            if expand:
                t = time.time()
                for k in range(8):
                    ang = 2 * math.pi * k / 8 + t * 3
                    ex  = int(sx + (r + 10) * math.cos(ang))
                    ey  = int(sy + (r + 10) * math.sin(ang))
                    pa  = int(255 * (k / 8))
                    pygame.draw.circle(self.screen, (*AI_THINKING_COLOR, pa), (ex, ey), 2)

            # ノード本体
            pygame.draw.circle(self.screen, c, (int(sx), int(sy)), r)
            border_c = (255, 255, 255) if hover else tuple(min(255, v + 60) for v in c)
            pygame.draw.circle(self.screen, border_c, (int(sx), int(sy)), r, 2)

            # ラベル
            if r >= 8:
                label_surf = self.font_sm.render(node.label, True, TEXT_COLOR)
                lw, lh = label_surf.get_size()
                # 背景
                bg = pygame.Surface((lw + 6, lh + 4), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 140))
                self.screen.blit(bg, (int(sx) - lw // 2 - 3, int(sy) + r + 4))
                self.screen.blit(label_surf, (int(sx) - lw // 2, int(sy) + r + 6))

    def _draw_cursor(self):
        snap = self.gs.snapshot()
        gesture = snap["gesture"]
        cx, cy  = self._cursor

        color_map = {
            "point":   (0, 255, 100),
            "pinch":   (80, 160, 255),
            "fist":    (255, 80,  80),
            "peace":   (255, 220, 50),
            "open":    (200, 200, 200),
            "none":    (80,  80,  80),
            "unknown": (120, 120, 120),
        }
        c = color_map.get(gesture, (180, 180, 180))

        # 十字カーソル
        pygame.draw.line(self.screen, c, (cx - 14, cy), (cx + 14, cy), 1)
        pygame.draw.line(self.screen, c, (cx, cy - 14), (cx, cy + 14), 1)
        pygame.draw.circle(self.screen, c, (cx, cy), 6, 2)

    def _draw_hud(self):
        snap = self.gs.snapshot()
        gesture = snap["gesture"]
        fps_cam = snap["fps"]
        fps_app = self.clock.get_fps()

        # ジェスチャーガイド（左上）
        guide = [
            ("POINT",  "カーソル移動",   (0, 255, 100)),
            ("PINCH",  "ノード選択 / AI展開", (80, 160, 255)),
            ("FIST",   "ノードをドラッグ", (255, 80,  80)),
            ("PEACE",  "ズーム（上下）",  (255, 220, 50)),
            ("OPEN",   "ニュートラル",   (200, 200, 200)),
        ]
        pygame.draw.rect(self.screen, (15, 18, 35), (10, 10, 230, len(guide) * 24 + 12), border_radius=8)
        for i, (g, desc, c) in enumerate(guide):
            active = gesture.upper() == g
            bc = c if active else (80, 85, 100)
            s = self.font_ui.render(f"{'▶ ' if active else '  '}{g}  {desc}", True, bc)
            self.screen.blit(s, (18, 18 + i * 24))

        # FPS（右上）
        fps_s = self.font_sm.render(f"App {fps_app:.0f} fps  Cam {fps_cam:.0f} fps", True, (80, 90, 120))
        self.screen.blit(fps_s, (WIN_W - fps_s.get_width() - 14, 14))

        # AI 思考中バナー
        if self.ai.is_thinking:
            node = self.nodes.get(self._expanding_node_id)
            label = node.label if node else "..."
            t = time.time()
            dots = "." * (int(t * 2) % 4)
            msg = f"AI が「{label}」を展開中{dots}"
            s = self.font_lg.render(msg, True, AI_THINKING_COLOR)
            bw, bh = s.get_width() + 20, s.get_height() + 10
            bx = WIN_W // 2 - bw // 2
            by = WIN_H - 50
            pygame.draw.rect(self.screen, (30, 28, 10), (bx - 2, by - 2, bw + 4, bh + 4), border_radius=8)
            pygame.draw.rect(self.screen, AI_THINKING_COLOR, (bx, by, bw, bh), 1, border_radius=8)
            self.screen.blit(s, (bx + 10, by + 5))

        # ノード数（右下）
        info = self.font_sm.render(f"Nodes: {len(self.nodes)}   Edges: {len(self.edges)}", True, (60, 70, 100))
        self.screen.blit(info, (WIN_W - info.get_width() - 14, WIN_H - 24))
