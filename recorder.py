"""
recorder.py
画面・ウィンドウ録画モジュール

使い方:
  recorder = Recorder(output_dir="recordings")

  # Pygame ウィンドウ録画
  recorder.start_window(pygame_surface)
  recorder.capture_frame(pygame_surface)   # 毎フレーム呼ぶ
  recorder.stop()

  # フルスクリーン録画
  recorder.start_screen()
  recorder.stop()
"""

import cv2
import numpy as np
import threading
import time
import os
import wave
import subprocess
import pygame
from datetime import datetime
from typing import Optional

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

SAMPLE_RATE = 44100


class Recorder:

    def __init__(self, output_dir: str = "recordings", fps: int = 30,
                 with_bgm: bool = True, with_mic: bool = False,
                 gesture_state=None):
        self.output_dir    = output_dir
        self.fps           = fps
        self.with_bgm      = with_bgm
        self.with_mic      = with_mic and HAS_SD
        self.gesture_state = gesture_state   # カメラフレーム取得用
        os.makedirs(output_dir, exist_ok=True)

        self._writer:    Optional[cv2.VideoWriter] = None
        self._mode:      str   = "none"
        self._running:   bool  = False
        self._thread:    Optional[threading.Thread] = None
        self._lock       = threading.Lock()
        self.last_file:  str   = ""

        # 音声
        self._mic_buf:   list  = []
        self._mic_thread: Optional[threading.Thread] = None
        self._video_tmp:  str  = ""

        # 録画時間計測
        self._start_time: float = 0.0

        # 合成モード（globe＋cam を横並び）
        self._combined_size: Optional[tuple] = None

    # ──────────────────────────────────────────
    # Pygame ウィンドウ録画
    # ──────────────────────────────────────────

    def start_window(self, surface: pygame.Surface) -> str:
        """Pygame ウィンドウの録画を開始。ファイルパスを返す"""
        if self._running:
            self.stop()

        w, h = surface.get_size()
        # 音声付きの場合は映像を一時ファイルに書く
        final_path = self._make_path("window")
        if self.with_bgm or self.with_mic:
            self._video_tmp  = final_path.replace(".mp4", "_tmp.mp4")
            write_path = self._video_tmp
        else:
            self._video_tmp  = ""
            write_path = final_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(write_path, fourcc, self.fps, (w, h))
        self._mode   = "window"
        self._running = True
        self._start_time = time.time()
        self.last_file = final_path

        self._start_audio()
        print(f"[REC] ウィンドウ録画開始 → {final_path}")
        return final_path

    def capture_frame(self, surface: pygame.Surface):
        """Pygame ウィンドウ ＋ カメラ映像を横並びで1フレームに合成して書き込む"""
        if not self._running or self._mode != "window" or self._writer is None:
            return

        # ── Pygame → BGR numpy ──
        arr = pygame.surfarray.array3d(surface)     # (W, H, 3) RGB
        arr = np.transpose(arr, (1, 0, 2))           # → (H, W, 3)
        globe_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gh, gw = globe_bgr.shape[:2]

        # ── カメラフレーム取得 ──
        cam_bgr = None
        if self.gesture_state is not None:
            with self.gesture_state.cam_frame_lock:
                if self.gesture_state.cam_frame is not None:
                    cam_bgr = self.gesture_state.cam_frame.copy()

        if cam_bgr is not None:
            # カメラをGlobeの高さに合わせてリサイズ
            ch, cw = cam_bgr.shape[:2]
            new_cw = int(cw * gh / ch)
            cam_resized = cv2.resize(cam_bgr, (new_cw, gh))

            # 横並び合成（Globe | Camera）
            combined = np.concatenate([globe_bgr, cam_resized], axis=1)
        else:
            combined = globe_bgr

        # サイズが変わったらWriterを再作成
        ch_, cw_ = combined.shape[:2]
        if self._combined_size != (cw_, ch_):
            self._combined_size = (cw_, ch_)
            # Writerを正しいサイズで作り直す
            if self._writer:
                self._writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            path = self._video_tmp if self._video_tmp else self.last_file
            self._writer = cv2.VideoWriter(path, fourcc, self.fps, (cw_, ch_))

        with self._lock:
            self._writer.write(combined)

    # ──────────────────────────────────────────
    # フルスクリーン録画
    # ──────────────────────────────────────────

    def start_screen(self, monitor_idx: int = 1) -> str:
        """フルスクリーン録画を開始（別スレッド）。ファイルパスを返す"""
        if not HAS_MSS:
            print("[REC] mss が見つかりません。")
            return ""
        if self._running:
            self.stop()

        final_path = self._make_path("screen")
        if self.with_bgm or self.with_mic:
            self._video_tmp = final_path.replace(".mp4", "_tmp.mp4")
            write_path = self._video_tmp
        else:
            self._video_tmp = ""
            write_path = final_path

        self._mode    = "screen"
        self._running = True
        self._start_time = time.time()
        self.last_file = final_path
        self._thread = threading.Thread(
            target=self._screen_loop,
            args=(write_path, monitor_idx),
            daemon=True,
        )
        self._thread.start()
        self._start_audio()
        print(f"[REC] スクリーン録画開始 → {final_path}")
        return final_path

    def _screen_loop(self, path: str, monitor_idx: int):
        with mss.mss() as sct:
            mon = sct.monitors[monitor_idx]
            w, h = mon["width"], mon["height"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
            interval = 1.0 / self.fps
            while self._running:
                t0  = time.time()
                img = sct.grab(mon)
                arr = np.array(img)
                bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                with self._lock:
                    writer.write(bgr)
                elapsed = time.time() - t0
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
            writer.release()

    # ──────────────────────────────────────────
    # 停止・共通
    # ──────────────────────────────────────────

    # ── 音声関連 ──

    def _start_audio(self):
        """BGM録音開始 ＋ マイク録音開始"""
        if self.with_bgm:
            try:
                from ambient_music import start_bgm_recording
                start_bgm_recording()
            except Exception as e:
                print(f"[REC] BGM録音開始失敗: {e}")

        if self.with_mic and HAS_SD:
            self._mic_buf.clear()
            self._mic_thread = threading.Thread(
                target=self._mic_loop, daemon=True)
            self._mic_thread.start()

    def _mic_loop(self):
        """マイク音声を録音するループ"""
        def callback(indata, frames, time_, status):
            if self._running:
                self._mic_buf.append(indata.copy())
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=2,
                            dtype="int16", callback=callback):
            while self._running:
                time.sleep(0.05)

    def _save_audio_wav(self, audio: np.ndarray, path: str):
        """numpy int16 配列を WAV ファイルに保存"""
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        ch = audio.shape[1] if audio.ndim > 1 else 1
        with wave.open(path, "w") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

    def _mux(self, video_path: str, audio_path: str, out_path: str):
        """ffmpeg で映像 + 音声を合成"""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("[REC] ffmpeg エラー:", result.stderr.decode()[:200])
        else:
            # 一時ファイル削除
            for f in [video_path, audio_path]:
                try: os.remove(f)
                except: pass

    # ─────────────────────────────────────────

    def stop(self):
        """録画を停止してファイルを保存（音声付き合成）"""
        if not self._running:
            return
        self._running = False
        elapsed = time.time() - self._start_time

        # スレッド終了待ち
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._mic_thread:
            self._mic_thread.join(timeout=2)
            self._mic_thread = None

        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None

        # 音声収集
        audio_parts = []

        if self.with_bgm:
            try:
                from ambient_music import stop_bgm_recording
                bgm_data = stop_bgm_recording()
                if len(bgm_data) > 0:
                    audio_parts.append(bgm_data.astype(np.float32))
            except Exception as e:
                print(f"[REC] BGM音声取得失敗: {e}")

        if self.with_mic and self._mic_buf:
            mic_data = np.concatenate(self._mic_buf, axis=0).astype(np.float32)
            audio_parts.append(mic_data)

        # 音声合成 & ffmpeg で合体
        if audio_parts and self._video_tmp and os.path.exists(self._video_tmp):
            # 長さを揃えてミックス
            max_len = max(a.shape[0] for a in audio_parts)
            mixed = np.zeros((max_len, 2), dtype=np.float32)
            for a in audio_parts:
                if a.ndim == 1:
                    a = np.stack([a, a], axis=1)
                n = min(a.shape[0], max_len)
                mixed[:n] += a[:n]

            # クリッピング防止 & int16変換
            peak = np.max(np.abs(mixed))
            if peak > 0:
                mixed = mixed / peak * 32000
            mixed = mixed.astype(np.int16)

            wav_path = self._video_tmp.replace("_tmp.mp4", "_audio.wav")
            self._save_audio_wav(mixed, wav_path)

            print(f"[REC] 音声合成中...")
            self._mux(self._video_tmp, wav_path, self.last_file)
            print(f"[REC] 完成 → {self.last_file}")
        else:
            # 音声なし、一時ファイルをそのままリネーム
            if self._video_tmp and os.path.exists(self._video_tmp):
                os.replace(self._video_tmp, self.last_file)

        print(f"[REC] 録画停止 ({elapsed:.1f}秒)")
        self._mode = "none"

    def toggle_window(self, surface: pygame.Surface) -> str:
        """録画中なら停止、停止中なら開始（トグル）"""
        if self._running and self._mode == "window":
            self.stop()
            return "stopped"
        else:
            return self.start_window(surface)

    def toggle_screen(self) -> str:
        """スクリーン録画トグル"""
        if self._running and self._mode == "screen":
            self.stop()
            return "stopped"
        else:
            return self.start_screen()

    @property
    def is_recording(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def elapsed(self) -> float:
        if not self._running:
            return 0.0
        return time.time() - self._start_time

    def _make_path(self, prefix: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{prefix}_{ts}.mp4")
