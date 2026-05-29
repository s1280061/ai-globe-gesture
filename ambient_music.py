"""
ambient_music.py
インターステラー風アンビエントBGM をリアルタイム合成して再生する

Hans Zimmer スタイル:
  - 教会オルガン風ドローン
  - ゆっくり変化するコード進行
  - 深い低音ベース
  - 倍音・トレモロによる厚み
"""

import numpy as np
import pygame
import threading
import time
import math


SAMPLE_RATE = 44100
CHUNK       = 2048       # 1チャンク分のサンプル数
CHANNELS    = 2          # ステレオ
MASTER_VOL  = 0.18       # 全体音量（0.0〜1.0）


# ============================================================
# コード進行（インターステラー風 Dm 系）
# 各コード: [(倍率, 音量比), ...] で構成
# 基準周波数 D3 = 146.83 Hz
# ============================================================
D3  = 146.83
A3  = 220.00
F3  = 174.61
C3  = 130.81
Bb2 = 116.54
G3  = 196.00
E3  = 164.81

# コード進行と持続時間(秒)
CHORD_SEQUENCE = [
    # (ノート周波数リスト,  持続秒)
    ([D3, A3, F3*2],          16.0),   # Dm
    ([Bb2*2, F3, D3*2],       16.0),   # Bb
    ([F3, C3*2, A3],          16.0),   # F
    ([C3, G3, E3],            16.0),   # C
    ([D3, A3, F3*2, C3*2],    20.0),   # Dm add9 （広がり）
    ([Bb2*2, F3, D3*2, A3],   16.0),   # Bb maj7
]


def organ_wave(freq: float, t: np.ndarray, detune: float = 0.0) -> np.ndarray:
    """
    教会オルガン風の波形
    基音 + 倍音（2f, 3f, 4f, 8f）を合成
    """
    f = freq * (1 + detune)
    wave  = 0.55 * np.sin(2 * np.pi * f       * t)
    wave += 0.25 * np.sin(2 * np.pi * f * 2   * t)
    wave += 0.12 * np.sin(2 * np.pi * f * 3   * t)
    wave += 0.06 * np.sin(2 * np.pi * f * 4   * t)
    wave += 0.02 * np.sin(2 * np.pi * f * 8   * t)
    return wave


def pad_wave(freq: float, t: np.ndarray, lfo_rate: float = 0.12) -> np.ndarray:
    """
    ゆっくり揺れるパッドシンセ（トレモロ + ビブラート）
    """
    lfo_amp   = 0.006
    vibrato   = 1 + lfo_amp * np.sin(2 * np.pi * lfo_rate * t)
    wave      = np.sin(2 * np.pi * freq * vibrato * t)
    tremolo   = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t)
    return wave * tremolo


def render_chord(freqs: list, duration: float,
                 fade_in: float = 3.0, fade_out: float = 4.0) -> np.ndarray:
    """
    コードを duration 秒分レンダリングしてステレオ numpy 配列を返す
    """
    n   = int(duration * SAMPLE_RATE)
    t   = np.linspace(0, duration, n, endpoint=False)
    buf = np.zeros(n, dtype=np.float32)

    for i, freq in enumerate(freqs):
        detune = (i % 3 - 1) * 0.0015   # 微妙なデチューン
        lfo    = 0.08 + i * 0.03

        # オルガン層
        buf += organ_wave(freq,       t, detune)       * 0.5
        buf += organ_wave(freq * 0.5, t, detune * 2)   * 0.25   # サブオクターブ

        # パッド層（少し遅れて加算）
        buf += pad_wave(freq, t, lfo) * 0.3

    # フェードイン/アウトエンベロープ
    fi = min(int(fade_in  * SAMPLE_RATE), n // 3)
    fo = min(int(fade_out * SAMPLE_RATE), n // 3)
    env = np.ones(n, dtype=np.float32)
    env[:fi] = np.linspace(0, 1, fi)
    env[-fo:] = np.linspace(1, 0, fo)
    buf *= env

    # ステレオ化（L/R に僅かな時間差でコーラス感）
    shift = 256   # サンプル数のシフト
    left  = buf
    right = np.roll(buf, shift)
    right[:shift] = 0
    stereo = np.stack([left, right], axis=1)

    # 正規化
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo /= peak

    stereo *= MASTER_VOL
    return (stereo * 32767).astype(np.int16)


class AmbientPlayer(threading.Thread):
    """
    バックグラウンドスレッドでアンビエント BGM を合成・ループ再生
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event  = threading.Event()
        self._volume      = 1.0
        self.is_ready     = False

        # 録音バッファ
        self._recording      = False
        self._record_buf:list = []   # int16 チャンクのリスト
        self._record_lock    = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def set_volume(self, v: float):
        self._volume = max(0.0, min(1.0, v))

    # ── 録音制御 ──
    def start_recording(self):
        with self._record_lock:
            self._record_buf.clear()
            self._recording = True

    def stop_recording(self) -> np.ndarray:
        """録音停止。int16 ステレオ配列を返す"""
        with self._record_lock:
            self._recording = False
            if not self._record_buf:
                return np.zeros((0, 2), dtype=np.int16)
            return np.concatenate(self._record_buf, axis=0)

    def run(self):
        # pygame.mixer の初期化
        pygame.mixer.pre_init(SAMPLE_RATE, -16, CHANNELS, CHUNK)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Sound チャンネルを2つ使ってクロスフェード
        ch0 = pygame.mixer.Channel(6)
        ch1 = pygame.mixer.Channel(7)
        channels = [ch0, ch1]
        active   = 0

        print("[BGM] アンビエント合成中...", flush=True)

        chord_idx = 0
        while not self._stop_event.is_set():
            freqs, duration = CHORD_SEQUENCE[chord_idx % len(CHORD_SEQUENCE)]

            # 次のコードも先読み（クロスフェード用）
            next_freqs, _ = CHORD_SEQUENCE[(chord_idx + 1) % len(CHORD_SEQUENCE)]

            # レンダリング
            data   = render_chord(freqs, duration, fade_in=3.0, fade_out=4.5)
            sound  = pygame.sndarray.make_sound(data)

            # 録音バッファへ追記
            with self._record_lock:
                if self._recording:
                    self._record_buf.append(data.copy())

            # 現在のチャンネルで再生
            cur_ch = channels[active]
            cur_ch.play(sound)
            cur_ch.set_volume(self._volume)

            if not self.is_ready:
                self.is_ready = True
                print("[BGM] 再生開始", flush=True)

            # duration 秒待つ（停止シグナルを監視しながら）
            end_t = time.time() + duration - 3.0   # 3秒前から次を準備
            while time.time() < end_t:
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

            active    = 1 - active
            chord_idx += 1

        # フェードアウト
        for ch in channels:
            ch.fadeout(3000)
        time.sleep(3)


# ============================================================
# 使いやすいシングルトン関数
# ============================================================
_player: AmbientPlayer | None = None


def start_bgm():
    global _player
    if _player is not None and _player.is_alive():
        return
    _player = AmbientPlayer()
    _player.start()


def stop_bgm():
    global _player
    if _player:
        _player.stop()


def set_volume(v: float):
    global _player
    if _player:
        _player._volume = v


def start_bgm_recording():
    global _player
    if _player:
        _player.start_recording()


def stop_bgm_recording() -> np.ndarray:
    """BGM録音停止。int16ステレオ配列を返す"""
    global _player
    if _player:
        return _player.stop_recording()
    return np.zeros((0, 2), dtype=np.int16)


if __name__ == "__main__":
    # 単体テスト
    pygame.init()
    pygame.mixer.init(SAMPLE_RATE, -16, CHANNELS, CHUNK)
    print("BGM テスト再生（Ctrl+C で終了）")
    start_bgm()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_bgm()
        print("停止")
