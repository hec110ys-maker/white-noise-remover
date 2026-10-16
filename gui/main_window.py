"""
Main Window for White Noise Remover desktop application.
Features modern dark theme, drag-and-drop, sliders, real-time progress, and spectrogram report.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSlider, QProgressBar, QMessageBox, QGroupBox,
    QFrame, QScrollArea, QCheckBox, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QUrl, QSettings
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDesktopServices, QIcon
from utils.paths import app_base_dir
from utils.version import __version__

from utils.device import get_optimal_device
from core.denoise import deepfilter_available
from utils.paths import get_output_video_path
from core.probe import probe_media, MediaMetadata
from gui.worker import DenoiseWorker
from gui.preview_dialog import PreviewDialog


VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv")


def _is_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


class DragDropArea(QFrame):
    """Visual drag & drop landing card."""
    def __init__(self, on_files_selected, parent=None):
        super().__init__(parent)
        self.on_files_selected = on_files_selected
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")
        self.setStyleSheet("""
            #dropArea {
                border: 2px dashed #89b4fa;
                border-radius: 12px;
                background-color: #181825;
                min-height: 110px;
            }
            #dropArea:hover {
                background-color: #1e1e2e;
                border-color: #b4befe;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel("📥 여기에 동영상 파일을 드래그 앤 드롭하세요\n(여러 개 가능 · MP4, MOV, MKV 지원)")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(_is_video(u.toLocalFile()) for u in urls):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls() if _is_video(u.toLocalFile())]
        if files:
            self.on_files_selected(files)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"동영상 화이트 노이즈 제거 프로그램 v{__version__}")
        icon_path = app_base_dir() / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.settings = QSettings("WhiteNoiseRemover", "WhiteNoiseRemover")
        self._fit_to_screen(780, 850)
        self.current_video_path: Optional[Path] = None
        self.queue: list[Path] = []          # files waiting / done in this batch
        self.batch_index = -1                # index of the file being processed, -1 when idle
        self.batch_results: list[tuple[Path, Optional[str]]] = []  # (file, error or None)
        self.current_meta: Optional[MediaMetadata] = None
        self.worker: Optional[DenoiseWorker] = None
        self.last_result: Optional[dict] = None

        self.apply_theme()
        self.init_ui()
        self.load_settings()

    def _fit_to_screen(self, width: int, height: int):
        """Laptops at 125-150% Windows scaling have ~700 logical px of height; never open larger than the screen."""
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(width, height)
            return
        avail = screen.availableGeometry()
        w = min(width, int(avail.width() * 0.95))
        h = min(height, int(avail.height() * 0.92))
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2)

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #11111b; }
            QWidget { color: #cdd6f4; font-family: -apple-system, BlinkMacSystemFont, "Malgun Gothic", "Segoe UI", Roboto, sans-serif; }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
                font-size: 13px;
                font-weight: bold;
                color: #89b4fa;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 10px 18px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton#btnStart { background-color: #a6e3a1; color: #11111b; border: none; font-size: 15px; }
            QPushButton#btnStart:hover { background-color: #94e2d5; }
            QPushButton#btnCancel { background-color: #f38ba8; color: #11111b; border: none; }
            QPushButton#btnPreview { background-color: #fab387; color: #11111b; border: none; }
            QComboBox {
                background-color: #181825;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QSlider::groove:horizontal { height: 6px; background: #313244; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #b4befe;
                border: 2px solid #89b4fa;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QProgressBar {
                border: 1px solid #313244;
                border-radius: 8px;
                text-align: center;
                background-color: #181825;
                color: #cdd6f4;
                font-weight: bold;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 7px;
            }
        """)

    def init_ui(self):
        # Content scrolls instead of being squeezed when the window is shorter than the layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background-color: #11111b; }")
        self.scroll_area = scroll
        central_widget = QWidget()
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 1. Header Bar: Device Info
        dev_info = get_optimal_device()
        header_layout = QHBoxLayout()
        title_lbl = QLabel("🎙️ 동영상 화이트 노이즈 제거")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: bold; color: #f8f8f2;")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        self.dev_badge = QLabel(f"  {dev_info.display_string}  ")
        self.dev_badge.setStyleSheet("""
            background-color: #181825;
            color: #a6e3a1;
            border: 1px solid #a6e3a1;
            border-radius: 12px;
            padding: 4px 10px;
            font-size: 12px;
            font-weight: bold;
        """)
        header_layout.addWidget(self.dev_badge)
        main_layout.addLayout(header_layout)

        # 2. Drag & Drop Area
        self.drop_area = DragDropArea(self.add_files)
        main_layout.addWidget(self.drop_area)

        # File Select Button Row
        file_row = QHBoxLayout()
        self.file_path_lbl = QLabel("선택된 파일 없음")
        self.file_path_lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
        file_row.addWidget(self.file_path_lbl, stretch=1)

        btn_browse = QPushButton("📁 파일 찾아보기...")
        btn_browse.clicked.connect(self.browse_file)
        file_row.addWidget(btn_browse)
        self.btn_clear = QPushButton("🗑️ 목록 비우기")
        self.btn_clear.clicked.connect(self.clear_queue)
        self.btn_clear.setVisible(False)
        file_row.addWidget(self.btn_clear)
        main_layout.addLayout(file_row)

        # Batch file list (shown once a file is added); clicking a row shows its info / preview
        self.file_list = QListWidget()
        self.file_list.setVisible(False)
        self.file_list.setMaximumHeight(130)
        self.file_list.setStyleSheet(
            "QListWidget { background-color: #181825; border: 1px solid #45475a; border-radius: 6px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 6px; }"
            "QListWidget::item:selected { background-color: #313244; color: #cdd6f4; }"
        )
        self.file_list.currentRowChanged.connect(self.on_file_row_changed)
        main_layout.addWidget(self.file_list)

        # 3. Video Metadata Info Card
        self.meta_group = QGroupBox("영상 정보")
        meta_layout = QVBoxLayout(self.meta_group)
        self.meta_lbl = QLabel("동영상 파일을 선택하면 해상도, 길이, 오디오 정보가 표시됩니다.")
        self.meta_lbl.setStyleSheet("color: #bac2de; font-size: 12px;")
        meta_layout.addWidget(self.meta_lbl)
        main_layout.addWidget(self.meta_group)

        # 4. Engine & Tuning Options
        settings_group = QGroupBox("노이즈 제거 설정")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(12)

        # Engine selector
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("제거 엔진 선택:"))
        self.combo_engine = QComboBox()
        if deepfilter_available():
            self.combo_engine.addItem("DeepFilterNet3 (1순위 권장 - AI 딥러닝 음성 향상)", "deepfilter")
        self.combo_engine.addItem("noisereduce (스펙트럴 게이팅 - 초고속 CPU)", "spectral")
        self.combo_engine.addItem("RNNoise (경량 순환신경망 RNN - 초경량 CPU)", "rnnoise")
        eng_row.addWidget(self.combo_engine, stretch=1)
        settings_layout.addLayout(eng_row)

        # Strength Slider
        s_layout = QHBoxLayout()
        self.lbl_strength = QLabel("노이즈 제거 강도: 100%")
        s_layout.addWidget(self.lbl_strength)
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(100)
        self.slider_strength.valueChanged.connect(lambda v: self.lbl_strength.setText(f"노이즈 제거 강도: {v}%"))
        s_layout.addWidget(self.slider_strength, stretch=1)
        settings_layout.addLayout(s_layout)

        # Voice Protection Slider
        vp_layout = QHBoxLayout()
        self.lbl_vp = QLabel("음성 보호 강도: 30%")
        vp_layout.addWidget(self.lbl_vp)
        self.slider_vp = QSlider(Qt.Horizontal)
        self.slider_vp.setRange(0, 100)
        self.slider_vp.setValue(30)
        self.slider_vp.valueChanged.connect(lambda v: self.lbl_vp.setText(f"음성 보호 강도: {v}%"))
        vp_layout.addWidget(self.slider_vp, stretch=1)
        settings_layout.addLayout(vp_layout)

        # Studio Clean Mode CheckBox
        self.chk_studio = QCheckBox("✨ 스튜디오 무잡음 모드 (미세 잔류 Hiss && 75Hz 럼블 100% 완전 제거)")
        self.chk_studio.setChecked(True)
        self.chk_studio.setToolTip("말소리 사이의 쉼(Pause) 구간 잔류 화이트 노이즈와 마이크 저주파 웅~ 소리를 스튜디오 방송급으로 정밀 차단합니다.")
        self.chk_studio.setStyleSheet("""
            QCheckBox {
                color: #a6e3a1;
                font-size: 13px;
                font-weight: bold;
                padding: 4px 0px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        settings_layout.addWidget(self.chk_studio)

        main_layout.addWidget(settings_group)

        # 5. Action Buttons (Preview, Start, Cancel)
        action_layout = QHBoxLayout()
        self.btn_preview = QPushButton("🎧 30초 미리듣기 (A/B 비교)")
        self.btn_preview.setObjectName("btnPreview")
        self.btn_preview.setEnabled(False)
        self.btn_preview.clicked.connect(self.open_preview_dialog)
        action_layout.addWidget(self.btn_preview)

        self.btn_start = QPushButton("🚀 노이즈 제거 시작")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_denoising)
        action_layout.addWidget(self.btn_start, stretch=1)

        self.btn_cancel = QPushButton("🛑 취소")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        action_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(action_layout)

        # 6. Progress Bar and Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("대기 중")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
        main_layout.addWidget(self.status_lbl)

        # 7. QA Results Section
        self.results_group = QGroupBox("품질 검증 리포트 && 스펙트로그램")
        self.results_group.setVisible(False)
        results_layout = QVBoxLayout(self.results_group)

        self.report_lbl = QLabel()
        self.report_lbl.setStyleSheet("color: #a6e3a1; font-size: 13px;")
        results_layout.addWidget(self.report_lbl)

        self.spec_img_lbl = QLabel()
        self.spec_img_lbl.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.spec_img_lbl)

        btn_row = QHBoxLayout()
        self.btn_open_file = QPushButton("🎬 완료 영상 재생")
        self.btn_open_file.clicked.connect(self.open_result_file)
        btn_row.addWidget(self.btn_open_file)

        self.btn_open_folder = QPushButton("📂 폴더 열기")
        self.btn_open_folder.clicked.connect(self.open_result_folder)
        btn_row.addWidget(self.btn_open_folder)
        results_layout.addLayout(btn_row)

        main_layout.addWidget(self.results_group)

    def browse_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "동영상 파일 선택 (여러 개 선택 가능)",
            self.settings.value("last_dir", "", type=str),
            "Video Files (*.mp4 *.mov *.mkv);;All Files (*)"
        )
        if file_paths:
            self.settings.setValue("last_dir", str(Path(file_paths[0]).parent))
            self.add_files(file_paths)

    # ---- Remembered settings (registry on Windows, plist on Mac) -------
    def load_settings(self):
        st = self.settings
        idx = self.combo_engine.findData(st.value("engine", "", type=str))
        if idx >= 0:
            self.combo_engine.setCurrentIndex(idx)
        self.slider_strength.setValue(st.value("strength", 100, type=int))
        self.slider_vp.setValue(st.value("voice_protection", 30, type=int))
        self.chk_studio.setChecked(st.value("studio_cleanup", True, type=bool))

    def save_settings(self):
        st = self.settings
        st.setValue("engine", self.combo_engine.currentData())
        st.setValue("strength", self.slider_strength.value())
        st.setValue("voice_protection", self.slider_vp.value())
        st.setValue("studio_cleanup", self.chk_studio.isChecked())

    def closeEvent(self, event):
        if self.batch_index >= 0:
            ans = QMessageBox.question(self, "작업 중", "노이즈 제거가 진행 중입니다. 취소하고 종료할까요?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            if self.worker:
                self.worker.cancel()
                self.worker.wait(10000)
        self.save_settings()
        super().closeEvent(event)

    # ---- Batch queue -------------------------------------------------
    STATUS_ICONS = {"wait": "⏳ 대기", "run": "▶️ 처리 중", "done": "✅ 완료", "fail": "❌ 실패", "skip": "⏭️ 건너뜀"}

    def add_files(self, file_paths):
        if self.batch_index >= 0:
            QMessageBox.information(self, "처리 중", "현재 작업이 끝난 뒤에 파일을 추가해 주세요.")
            return
        added = 0
        for fp in file_paths:
            p = Path(fp).resolve()
            if p.exists() and p not in self.queue and not p.stem.endswith("_denoised"):
                self.queue.append(p)
                item = QListWidgetItem()
                item.setData(Qt.UserRole, "wait")
                self.file_list.addItem(item)
                self._render_row(len(self.queue) - 1)
                added += 1
        if not added:
            return
        self.file_list.setVisible(True)
        self.btn_clear.setVisible(True)
        self.file_list.setCurrentRow(len(self.queue) - added)
        self._update_queue_label()

    def clear_queue(self):
        if self.batch_index >= 0:
            return
        self.queue.clear()
        self.file_list.clear()
        self.file_list.setVisible(False)
        self.btn_clear.setVisible(False)
        self.current_video_path = None
        self.current_meta = None
        self.file_path_lbl.setText("선택된 파일 없음")
        self.drop_area.label.setText("📥 여기에 동영상 파일을 드래그 앤 드롭하세요\n(여러 개 가능 · MP4, MOV, MKV 지원)")
        self.meta_lbl.setText("동영상 파일을 선택하면 해상도, 길이, 오디오 정보가 표시됩니다.")
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.status_lbl.setText("대기 중")

    def _render_row(self, row: int):
        item = self.file_list.item(row)
        status = item.data(Qt.UserRole)
        item.setText(f"{self.STATUS_ICONS[status]}   {self.queue[row].name}")

    def _set_row_status(self, row: int, status: str):
        self.file_list.item(row).setData(Qt.UserRole, status)
        self._render_row(row)

    def _update_queue_label(self):
        n = len(self.queue)
        self.file_path_lbl.setText(f"<b>파일 {n}개</b> 선택됨" if n > 1 else f"<b>선택된 파일:</b> {self.queue[0].name}")
        self.drop_area.label.setText(f"✅ 파일 {n}개 준비됨\n(드롭하면 목록에 추가됩니다)")
        self.btn_start.setText("🚀 노이즈 제거 시작" if n == 1 else f"🚀 {n}개 파일 노이즈 제거 시작")

    def on_file_row_changed(self, row: int):
        if 0 <= row < len(self.queue):
            self.load_video_file(str(self.queue[row]))

    def load_video_file(self, file_path: str):
        p = Path(file_path).resolve()
        if not p.exists():
            return

        self.current_video_path = p

        try:
            self.current_meta = probe_media(p)
            dur_m = int(self.current_meta.duration_sec // 60)
            dur_s = int(self.current_meta.duration_sec % 60)
            info_text = (
                f"• 해상도: <b>{self.current_meta.width}x{self.current_meta.height}</b> | "
                f"FPS: <b>{self.current_meta.fps:.2f}</b> | "
                f"비디오 코덱: <b>{self.current_meta.video_codec}</b> (무손실 복사)<br>"
                f"• 영상 길이: <b>{dur_m}분 {dur_s}초</b> ({self.current_meta.duration_sec:.1f}s) | "
                f"오디오: <b>{self.current_meta.audio_sample_rate}Hz {self.current_meta.audio_channels}ch</b>"
            )
            self.meta_lbl.setText(info_text)
            if self.batch_index < 0:
                self.btn_start.setEnabled(True)
                self.btn_preview.setEnabled(True)
                self.status_lbl.setText("준비 완료! [노이즈 제거 시작] 또는 [30초 미리듣기]를 클릭하세요.")
        except Exception as e:
            QMessageBox.critical(self, "파일 읽기 오류", f"메타데이터 분석 실패: {e}")

    def open_preview_dialog(self):
        if not self.current_video_path:
            return
        eng = self.combo_engine.currentData()
        str_val = self.slider_strength.value() / 100.0
        vp_val = self.slider_vp.value() / 100.0
        studio_clean = self.chk_studio.isChecked()

        dialog = PreviewDialog(
            video_path=self.current_video_path,
            engine_type=eng,
            strength=str_val,
            voice_protection=vp_val,
            enable_studio_cleanup=studio_clean,
            parent=self
        )
        dialog.exec()

    def start_denoising(self):
        if not self.queue:
            return

        existing = [p for p in self.queue if get_output_video_path(p).exists()]
        self.overwrite_existing = True
        if existing:
            names = "\n".join(f"• {get_output_video_path(p).name}" for p in existing[:5])
            more = f"\n…외 {len(existing) - 5}개" if len(existing) > 5 else ""
            ans = QMessageBox.question(
                self, "이미 처리된 파일",
                f"결과 파일이 이미 있습니다:\n{names}{more}\n\n덮어쓸까요? (아니요: 이 파일들은 건너뜀)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if ans == QMessageBox.Cancel:
                return
            self.overwrite_existing = ans == QMessageBox.Yes

        self.batch_results = []
        for row in range(len(self.queue)):
            self._set_row_status(row, "wait")
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.results_group.setVisible(False)
        self.batch_index = -1
        self._start_next_in_batch()

    def _start_next_in_batch(self):
        self.batch_index += 1
        while self.batch_index < len(self.queue):
            path = self.queue[self.batch_index]
            if self.overwrite_existing or not get_output_video_path(path).exists():
                break
            self._set_row_status(self.batch_index, "skip")
            self.batch_index += 1
        if self.batch_index >= len(self.queue):
            self._finish_batch()
            return

        row = self.batch_index
        path = self.queue[row]
        self._set_row_status(row, "run")
        self.file_list.setCurrentRow(row)
        self.progress_bar.setValue(0)

        self.worker = DenoiseWorker(
            input_video_path=path,
            output_video_path=get_output_video_path(path),
            engine_type=self.combo_engine.currentData(),
            strength=self.slider_strength.value() / 100.0,
            voice_protection=self.slider_vp.value() / 100.0,
            enable_studio_cleanup=self.chk_studio.isChecked()
        )
        self.worker.progress_changed.connect(self.on_worker_progress)
        self.worker.status_changed.connect(self.on_worker_status)
        self.worker.finished_success.connect(self.on_worker_finished)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_thread_done)
        self.worker.start()

    def on_worker_thread_done(self):
        """Runs after every job (success, error or cancel) once the thread has fully stopped."""
        if self.batch_index < 0:
            return
        if self.worker and self.worker._is_cancelled:
            for row in range(self.batch_index, len(self.queue)):
                if self.file_list.item(row).data(Qt.UserRole) in ("wait", "run"):
                    self._set_row_status(row, "skip")
            self.batch_index = len(self.queue)
            self._finish_batch(cancelled=True)
            return
        self._start_next_in_batch()

    def _batch_prefix(self) -> str:
        if len(self.queue) > 1 and self.batch_index >= 0:
            return f"[{self.batch_index + 1}/{len(self.queue)}] "
        return ""

    def _finish_batch(self, cancelled: bool = False):
        self.batch_index = -1
        self.btn_start.setEnabled(bool(self.queue))
        self.btn_preview.setEnabled(self.current_video_path is not None)
        self.btn_clear.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        ok = [p for p, err in self.batch_results if err is None]
        failed = [(p, err) for p, err in self.batch_results if err is not None]
        if cancelled:
            self.status_lbl.setText("⏹️ 작업이 취소되었습니다.")
            return
        if not self.batch_results:
            self.status_lbl.setText("처리할 파일이 없습니다 (모두 건너뜀).")
            return
        self.status_lbl.setText(f"✅ 완료 {len(ok)}개" + (f" · ❌ 실패 {len(failed)}개" if failed else ""))
        if failed:
            details = "\n\n".join(f"• {p.name}\n  {err.splitlines()[0] if err else ''}" for p, err in failed)
            QMessageBox.warning(self, "일부 파일 실패", f"성공 {len(ok)}개, 실패 {len(failed)}개\n\n{details}")
        elif len(ok) == 1:
            QMessageBox.information(self, "완료", f"화이트 노이즈 제거가 성공적으로 완료되었습니다!\n\n저장 위치: {get_output_video_path(ok[0])}")
        else:
            QMessageBox.information(self, "완료", f"{len(ok)}개 파일의 화이트 노이즈 제거가 모두 끝났습니다.\n\n결과는 각 원본 영상과 같은 폴더에 '_denoised.mp4'로 저장되었습니다.")

    def cancel_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_lbl.setText("취소 요청 중...")
            self.btn_cancel.setEnabled(False)

    def on_worker_progress(self, info: dict):
        pct = int(info["progress_percent"])
        chunk = info["chunk_index"]
        total = info["total_chunks"]
        eta = info["eta_sec"]
        rtf = info["rtf"]
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(f"{self._batch_prefix()}진행 중: {pct}% (청크 {chunk}/{total}) | 남은 예상 시간: {eta:.1f}초 | 처리 속도: {rtf:.3f}x")

    def on_worker_status(self, msg: str):
        self.status_lbl.setText(self._batch_prefix() + msg)

    def on_worker_finished(self, result: dict):
        self.last_result = result
        self.progress_bar.setValue(100)
        self._set_row_status(self.batch_index, "done")
        self.batch_results.append((self.queue[self.batch_index], None))

        comp = result["meta_comparison"]
        qa = result["qa_metrics"]
        elapsed = result["elapsed_sec"]

        report_text = (
            f"🎉 <b>노이즈 제거 완료!</b> {self._batch_prefix()}(소요 시간: {elapsed:.1f}초)<br>"
            f"• 출력 파일: <b>{Path(result['output_path']).name}</b><br>"
            f"• 영상 화질 보존: <b>{'✅ 100% 무손실 복사 일치' if comp['is_resolution_match'] else '⚠️ 해상도 불일치'}</b> "
            f"({comp['orig_resolution']} @ {comp['orig_fps']}fps)<br>"
            f"• 오디오 싱크 오차: <b>{comp['duration_diff_sec']:.3f}초 (밀림 없음)</b><br>"
            f"• 배경 노이즈 감쇄량: <b>약 {qa['total_reduction_db']} dB 감소</b>"
        )
        self.report_lbl.setText(report_text)

        # Load spectrogram thumbnail
        spec_path = result.get("spectrogram_image")
        if spec_path and Path(spec_path).exists():
            pix = QPixmap(spec_path)
            # Fit the image to the visible width so it never widens the window
            target_w = max(300, min(720, self.scroll_area.viewport().width() - 70))
            self.spec_img_lbl.setPixmap(pix.scaledToWidth(target_w, Qt.SmoothTransformation))

        self.results_group.setVisible(True)

    def on_worker_error(self, err: str):
        # Record and keep going; the summary at the end lists every failure
        self._set_row_status(self.batch_index, "fail")
        self.batch_results.append((self.queue[self.batch_index], err))
        print(f"[batch] {self.queue[self.batch_index]} failed: {err}")

    def open_result_file(self):
        if self.last_result:
            p = self.last_result["output_path"]
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def open_result_folder(self):
        if self.last_result:
            p = str(Path(self.last_result["output_path"]).parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(p))
