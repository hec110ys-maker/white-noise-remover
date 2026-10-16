"""
Desktop GUI application entry point for White Noise Remover.
Launch with:
    python main.py
Or process directly from terminal:
    python main.py "video_file.mp4"
"""
import sys
import traceback
from pathlib import Path


def _setup_console_streams():
    """Packaged GUI builds have no console (stdout is None) and Windows consoles default to cp949,
    so route output to a UTF-8 log file or force UTF-8 to keep emoji/Korean prints from crashing."""
    if sys.stdout is None or sys.stderr is None:
        from utils.paths import user_data_dir
        log = open(user_data_dir() / "app.log", "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _isolate_child_processes():
    """PyInstaller's Qt runtime hook points DYLD_LIBRARY_PATH at the bundle, which makes child ffmpeg
    processes load Qt's stripped FFmpeg libraries and crash. This process is already loaded, so drop it."""
    if getattr(sys, "frozen", False):
        import os
        os.environ.pop("DYLD_LIBRARY_PATH", None)


def main():
    _setup_console_streams()
    _isolate_child_processes()

    # If a video file argument is passed via terminal, route to CLI mode
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        from run_cli import main as cli_main
        cli_main()
        return

    # Otherwise launch Qt Desktop GUI
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("화이트 노이즈 제거 프로그램")
        from utils.version import __version__
        app.setApplicationVersion(__version__)
        app.setStyle("Fusion")

        window = MainWindow()
        window.show()

        sys.exit(app.exec())
    except Exception as e:
        print("\n❌ [오류] GUI 실행 중 예외가 발생했습니다:")
        traceback.print_exc()
        # Also write to crash.log
        try:
            from utils.paths import user_data_dir
            with open(user_data_dir() / "crash.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
