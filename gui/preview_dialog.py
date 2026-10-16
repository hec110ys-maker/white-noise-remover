"""
Preview Dialog for instant A/B comparison on a user-selected 30-second slice.
"""
from pathlib import Path
from typing import Optional
import soundfile as sf
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QProgressBar, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from core.probe import probe_media, get_ffmpeg_binary
from utils.paths import SUBPROCESS_FLAGS
import subprocess
from core.denoise.spectral import SpectralGatingDenoiser
from core.denoise.rnnoise import RNNoiseDenoiser
from core.vad import extract_noise_sample_from_file


class PreviewDialog(QDialog):
    def __init__(
        self,
        video_path: str | Path,
        engine_type: str = "spectral",
        strength: float = 1.0,
        voice_protection: float = 0.3,
        enable_studio_cleanup: bool = True,
        parent=None
    ):
        super().__init__(parent)
        self.video_path = Path(video_path).resolve()
        self.engine_type = engine_type
        self.strength = strength
        self.voice_protection = voice_protection
        self.enable_studio_cleanup = enable_studio_cleanup

        self.setWindowTitle("🎧 30초 구간 미리듣기 (A/B 비교 청취)")
        self.resize(550, 420)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto; }
            QLabel { color: #cdd6f4; }
            QGroupBox { border: 1px solid #45475a; border-radius: 8px; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #89b4fa; }
            QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 8px 16px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background-color: #45475a; }
            QPushButton#btnGenerate { background-color: #89b4fa; color: #11111b; border: none; }
            QPushButton#btnGenerate:hover { background-color: #b4befe; }
            QPushButton#btnPlayA { background-color: #f38ba8; color: #11111b; border: none; }
            QPushButton#btnPlayB { background-color: #a6e3a1; color: #11111b; border: none; }
            QSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
            QProgressBar { border: 1px solid #45475a; border-radius: 5px; text-align: center; color: #cdd6f4; background-color: #313244; }
            QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }
        """)

        # Media Player & Audio Output
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.9)

        self.orig_preview_wav = Path("/tmp/preview_orig.wav")
        self.denoised_preview_wav = Path("/tmp/preview_denoised.wav")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header Info
        meta = probe_media(self.video_path)
        max_duration = int(meta.duration_sec)
        
        info_lbl = QLabel(f"<b>영상 파일:</b> {self.video_path.name}<br><b>총 재생시간:</b> {max_duration}초 ({max_duration//60}분 {max_duration%60}초)")
        layout.addWidget(info_lbl)

        # Section Selector Group
        sec_group = QGroupBox("구간 선택 (30초)")
        sec_layout = QHBoxLayout(sec_group)
        sec_layout.addWidget(QLabel("시작 시간(초):"))
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, max(0, max_duration - 30))
        self.spin_start.setValue(min(30, max(0, max_duration - 30)))
        sec_layout.addWidget(self.spin_start)

        self.btn_generate = QPushButton("⚡ 30초 미리듣기 생성")
        self.btn_generate.setObjectName("btnGenerate")
        self.btn_generate.clicked.connect(self.generate_preview)
        sec_layout.addWidget(self.btn_generate)
        layout.addWidget(sec_group)

        # Progress / Status
        self.status_lbl = QLabel("시작 시간을 설정하고 '미리듣기 생성' 버튼을 눌러주세요.")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_lbl)

        # A/B Player Group
        player_group = QGroupBox("실시간 A/B 비교 청취")
        player_layout = QVBoxLayout(player_group)
        player_layout.setSpacing(10)

        btns_layout = QHBoxLayout()
        self.btn_play_orig = QPushButton("🔊 [A] 원본 오디오 듣기 (잡음 포함)")
        self.btn_play_orig.setObjectName("btnPlayA")
        self.btn_play_orig.setEnabled(False)
        self.btn_play_orig.clicked.connect(self.play_orig)
        btns_layout.addWidget(self.btn_play_orig)

        self.btn_play_denoised = QPushButton("✨ [B] 노이즈 제거 오디오 듣기")
        self.btn_play_denoised.setObjectName("btnPlayB")
        self.btn_play_denoised.setEnabled(False)
        self.btn_play_denoised.clicked.connect(self.play_denoised)
        btns_layout.addWidget(self.btn_play_denoised)
        player_layout.addLayout(btns_layout)

        self.btn_stop = QPushButton("⏹️ 정지")
        self.btn_stop.clicked.connect(self.stop_playback)
        player_layout.addWidget(self.btn_stop)

        layout.addWidget(player_group)

        # Close Button
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def generate_preview(self):
        start_sec = self.spin_start.value()
        self.status_lbl.setText("⏳ 선택 구간 오디오 추출 및 노이즈 제거 처리 중...")
        self.btn_generate.setEnabled(False)

        ffmpeg = get_ffmpeg_binary()
        # 1. Extract 30 seconds slice
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start_sec),
            "-t", "30",
            "-i", str(self.video_path),
            "-ar", "48000",
            "-c:a", "pcm_f32le",
            str(self.orig_preview_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **SUBPROCESS_FLAGS)
        if res.returncode != 0:
            QMessageBox.critical(self, "오류", "오디오 추출 실패")
            self.btn_generate.setEnabled(True)
            return

        # 2. Denoise with current settings
        data, sr = sf.read(str(self.orig_preview_wav), dtype="float32", always_2d=False)
        if self.engine_type == "spectral":
            noise_profile = extract_noise_sample_from_file(self.orig_preview_wav)
            denoiser = SpectralGatingDenoiser(
                strength=self.strength,
                voice_protection=self.voice_protection,
                noise_profile=noise_profile,
                enable_studio_cleanup=self.enable_studio_cleanup
            )
        elif self.engine_type == "rnnoise":
            denoiser = RNNoiseDenoiser(
                strength=self.strength,
                voice_protection=self.voice_protection,
                enable_studio_cleanup=self.enable_studio_cleanup
            )
        else:
            from core.denoise.deepfilter import DeepFilterDenoiser
            denoiser = DeepFilterDenoiser(
                strength=self.strength,
                voice_protection=self.voice_protection,
                enable_studio_cleanup=self.enable_studio_cleanup
            )

        clean = denoiser.process(data, sr)
        sf.write(str(self.denoised_preview_wav), clean, sr, subtype="FLOAT")

        self.status_lbl.setText("✅ 미리듣기 생성 완료! [A]와 [B]를 번갈아 누르며 음질을 비교해 보세요.")
        self.btn_play_orig.setEnabled(True)
        self.btn_play_denoised.setEnabled(True)
        self.btn_generate.setEnabled(True)

    def play_orig(self):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self.orig_preview_wav)))
        self.player.play()
        self.status_lbl.setText("▶️ 재생 중: [A] 원본 오디오 (배경 화이트 노이즈 있음)")

    def play_denoised(self):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(self.denoised_preview_wav)))
        self.player.play()
        self.status_lbl.setText("▶️ 재생 중: [B] 노이즈 제거 오디오 (화이트 노이즈 소거됨)")

    def stop_playback(self):
        self.player.stop()
        self.status_lbl.setText("⏹️ 재생 정지됨")

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)
