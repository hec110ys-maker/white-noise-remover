"""
Headless test of the 30-second A/B preview dialog (extract -> denoise -> wav files).
"""
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE = Path(__file__).resolve().parent.parent / "sample_demo.mp4"


@pytest.mark.parametrize("engine", ["spectral", "rnnoise"])
def test_preview_generates_wavs(engine, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from gui.preview_dialog import PreviewDialog

    app = QApplication.instance() or QApplication([])

    def fail_on_error_box(parent, title, text, *args, **kwargs):
        raise AssertionError(f"{title}: {text}")
    monkeypatch.setattr(QMessageBox, "critical", fail_on_error_box)

    dlg = PreviewDialog(SAMPLE, engine_type=engine)
    dlg.spin_start.setValue(0)
    dlg.generate_preview()

    assert dlg.orig_preview_wav.stat().st_size > 0
    assert dlg.denoised_preview_wav.stat().st_size > 0

    # Regenerating must be able to overwrite the files, and closing removes the temp folder
    dlg.generate_preview()
    preview_dir = dlg.preview_dir
    dlg.reject()
    assert not preview_dir.exists()
    app.processEvents()
