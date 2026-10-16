"""
Headless test of the main window's batch queue: two files in, two '_denoised.mp4' out,
temp work folders removed afterwards.
"""
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE = Path(__file__).resolve().parent.parent / "sample_demo.mp4"


def test_batch_processes_all_files(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warn", a[2])))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.append(("crit", a[2])))

    files = []
    for name in ("영상 하나.mp4", "video_two.mp4"):
        dst = tmp_path / name
        shutil.copy(SAMPLE, dst)
        files.append(dst)

    w = MainWindow()
    w.combo_engine.setCurrentIndex(w.combo_engine.findData("spectral"))
    w.add_files([str(f) for f in files])
    assert w.file_list.count() == 2

    w.start_denoising()
    deadline = time.time() + 120
    while w.batch_index >= 0 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)

    assert w.batch_index == -1, "batch did not finish"
    assert [kind for kind, _ in shown] == ["info"], shown
    for f in files:
        assert (tmp_path / f"{f.stem}_denoised.mp4").stat().st_size > 0
    assert not (tmp_path / ".temp_denoise").exists(), "temp work folder was not cleaned up"


def test_batch_cancel_stops_remaining_files(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)

    files = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        dst = tmp_path / name
        shutil.copy(SAMPLE, dst)
        files.append(dst)

    w = MainWindow()
    w.combo_engine.setCurrentIndex(w.combo_engine.findData("spectral"))
    w.add_files([str(f) for f in files])
    w.start_denoising()
    w.cancel_processing()
    deadline = time.time() + 60
    while w.batch_index >= 0 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)

    assert w.batch_index == -1
    statuses = [w.file_list.item(i).data(0x0100) for i in range(3)]  # Qt.UserRole
    assert statuses[1:] == ["skip", "skip"], statuses
    assert not (tmp_path / "b_denoised.mp4").exists()
    assert w.btn_start.isEnabled()
