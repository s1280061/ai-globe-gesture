"""
face_mask.py
動画内の顔を自動検出してモザイク処理する

使い方:
  python face_mask.py input.mp4
  python face_mask.py input.mp4 --mode blur    # ぼかし（デフォルト）
  python face_mask.py input.mp4 --mode mosaic  # モザイク
  python face_mask.py input.mp4 --mode black   # 黒塗り
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm


def apply_mask(frame, x1, y1, x2, y2, mode, margin=0.15):
    h, w = frame.shape[:2]

    # マージンを追加（顔全体をカバー）
    pw = int((x2 - x1) * margin)
    ph = int((y2 - y1) * margin)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    if mode == "blur":
        # ガウシアンぼかし（強め）
        ksize = max(51, ((x2-x1)//4)*2+1)
        masked = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    elif mode == "mosaic":
        # モザイク
        block = max(8, (x2-x1)//12)
        small = cv2.resize(roi, (max(1,(x2-x1)//block), max(1,(y2-y1)//block)),
                           interpolation=cv2.INTER_LINEAR)
        masked = cv2.resize(small, (x2-x1, y2-y1),
                            interpolation=cv2.INTER_NEAREST)
    elif mode == "black":
        masked = np.zeros_like(roi)
    else:
        masked = roi

    result = frame.copy()
    result[y1:y2, x1:x2] = masked
    return result


def process_video(input_path: str, mode: str = "blur"):
    if not os.path.exists(input_path):
        print(f"ファイルが見つかりません: {input_path}")
        return

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_masked{ext}"

    cap = cv2.VideoCapture(input_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"入力:  {input_path}")
    print(f"解像度: {width}x{height}  FPS: {fps:.0f}  フレーム数: {total}")
    print(f"モード: {mode}")
    print(f"出力:  {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(
        model_selection=1,          # 1=遠距離モデル
        min_detection_confidence=0.4
    ) as detector:

        for _ in tqdm(range(total), desc="処理中", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            if result.detections:
                for det in result.detections:
                    bb  = det.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width)  * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    frame = apply_mask(frame, x1, y1, x2, y2, mode)

            out.write(frame)

    cap.release()
    out.release()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✅ 完了: {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="入力動画ファイル")
    parser.add_argument("--mode", choices=["blur","mosaic","black"],
                        default="blur", help="マスク方法")
    args = parser.parse_args()
    process_video(args.input, args.mode)
