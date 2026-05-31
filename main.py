"""
main.py
AI Knowledge Space 起動エントリーポイント

使い方:
  python main.py          # モード選択メニューが出る
  python main.py globe    # 地球儀モード直接起動
  python main.py nodes    # ノード空間モード直接起動
"""

import os
import sys
from gesture_engine import GestureEngine, GestureState


def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("=" * 55)
        print(" ANTHROPIC_API_KEY が未設定です。")
        print(" → ダミーデータモードで動作します（API不要）")
        print()
        print(" AI機能を有効にする場合:")
        print("   $env:ANTHROPIC_API_KEY='sk-ant-xxxx'  (PowerShell)")
        print("=" * 55)
    else:
        print("[OK] ANTHROPIC_API_KEY 検出 → AI機能が有効です")


def select_mode():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║      AI Knowledge Space              ║")
    print("  ║      ジェスチャー操作モード選択      ║")
    print("  ╠══════════════════════════════════════╣")
    print("  ║  1. 🌍 地球儀モード                 ║")
    print("  ║     地球を回して都市を選択→AI解説   ║")
    print("  ║                                      ║")
    print("  ║  2. 🔵 ノード空間モード             ║")
    print("  ║     3Dノードをピンチ→AI展開        ║")
    print("  ╚══════════════════════════════════════╝")
    print()
    choice = input("  選択 (1 or 2) [デフォルト: 1]: ").strip()
    return "globe" if choice != "2" else "nodes"


def print_guide(mode: str):
    print()
    if mode == "globe":
        print("  🌍 地球儀モード 起動中")
        print()
        print("  ジェスチャー操作:")
        print("  ✊ グー      : 地球儀を回す")
        print("  🤌 ピンチ   : 都市を選択 → AI 解説が展開")
        print("  ✌ ピース   : ズームイン/アウト")
        print("  ☝ ポイント  : カーソル移動・ホバー")
        print("  SPACE キー  : 自動回転 ON/OFF")
    else:
        print("  🔵 ノード空間モード 起動中")
        print()
        print("  ジェスチャー操作:")
        print("  ☝ ポイント  : カーソル移動")
        print("  🤌 ピンチ   : ノード選択 → AI が関連概念を展開")
        print("  ✊ グー      : ノードをドラッグ")
        print("  ✌ ピース   : ズーム")
    print()
    print("  Q / Esc: 終了")
    print()


def main():
    check_api_key()

    # コマンドライン引数でモード指定
    if len(sys.argv) > 1 and sys.argv[1] in ("globe", "nodes"):
        mode = sys.argv[1]
    else:
        mode = select_mode()

    print_guide(mode)

    # ジェスチャー状態（スレッド間共有）
    state  = GestureState()
    engine = GestureEngine(state, cam_id=0, show_window=True)
    engine.start()

    # モードに応じてスペースを起動
    if mode == "globe":
        from globe_space import GlobeSpace
        space = GlobeSpace(state)
    else:
        from knowledge_space import KnowledgeSpace
        space = KnowledgeSpace(state)

    space.run()

    engine.stop()
    engine.join(timeout=2)
    print("終了しました。")


if __name__ == "__main__":
    main()
