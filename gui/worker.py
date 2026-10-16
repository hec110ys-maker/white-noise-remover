"""
Background QThread worker for processing video audio without blocking the UI.
"""
import time
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QThread, Signal
from core.probe import probe_media, compare_media_metadata
from core.extract import extract_audio, remux_video_with_audio
from core.chunker import AudioChunker
from core.denoise.deepfilter import DeepFilterDenoiser
from core.denoise.spectral import SpectralGatingDenoiser
from core.denoise.rnnoise import RNNoiseDenoiser
from core.vad import extract_noise_sample_from_file
from core.qa import generate_spectrogram_image, analyze_audio_quality
from utils.paths import get_output_video_path, get_temp_work_dir, cleanup_temp_dir


class DenoiseWorker(QThread):
    # Signals to communicate with GUI
    progress_changed = Signal(dict)       # Progress stats (pct, eta, rtf, chunk)
    status_changed = Signal(str)         # Status text message
    finished_success = Signal(dict)      # Final report payload
    error_occurred = Signal(str)         # Error message

    def __init__(
        self,
        input_video_path: str | Path,
        output_video_path: Optional[str | Path] = None,
        engine_type: str = "deepfilter",
        strength: float = 1.0,
        voice_protection: float = 0.3,
        enable_studio_cleanup: bool = True,
        parent=None
    ):
        super().__init__(parent)
        # Increase QThread stack size to 8MB to prevent macOS OpenBLAS stack overflow
        self.setStackSize(8 * 1024 * 1024)
        self.input_video_path = Path(input_video_path).resolve()
        self.output_video_path = Path(output_video_path).resolve() if output_video_path else get_output_video_path(self.input_video_path)
        self.engine_type = engine_type
        self.strength = strength
        self.voice_protection = voice_protection
        self.enable_studio_cleanup = enable_studio_cleanup
        self._is_cancelled = False

    def cancel(self):
        """Signal thread to cancel operation."""
        self._is_cancelled = True

    def run(self):
        temp_dir = get_temp_work_dir(self.input_video_path)
        start_time = time.time()

        try:
            self.status_changed.emit("🔍 비디오 및 오디오 메타데이터 분석 중...")
            meta_orig = probe_media(self.input_video_path)
            if not meta_orig.has_audio:
                self.error_occurred.emit("입력 영상에 오디오 스트림이 없습니다.")
                return

            if self._is_cancelled:
                return

            # Step 1: Extract Audio
            self.status_changed.emit("🎙️ 무손실 48kHz 오디오 추출 중...")
            extracted_wav = temp_dir / "extracted_48k.wav"
            extract_audio(self.input_video_path, extracted_wav, sample_rate=48000)

            if self._is_cancelled:
                return

            # Step 2: Initialize Chosen Denoiser
            self.status_changed.emit(f"🧠 노이즈 제거 엔진 초기화 중 ({self.engine_type})...")
            if self.engine_type == "spectral":
                noise_profile = extract_noise_sample_from_file(extracted_wav, sample_duration_sec=1.5)
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
                denoiser = DeepFilterDenoiser(
                    strength=self.strength,
                    voice_protection=self.voice_protection,
                    enable_studio_cleanup=self.enable_studio_cleanup
                )

            if self._is_cancelled:
                return

            # Step 3: Stream Chunk Processing with Overlap-Add
            self.status_changed.emit("⚡ 청크별 딥러닝 노이즈 제거 및 크로스페이드 재합성 중...")
            denoised_wav = temp_dir / "denoised_48k.wav"
            checkpoint_file = temp_dir / "checkpoint.json"

            chunker = AudioChunker(chunk_duration_sec=30.0, overlap_duration_sec=1.0, sample_rate=48000)

            def progress_callback(info: dict):
                self.progress_changed.emit(info)

            chunker.process_file(
                input_wav_path=extracted_wav,
                output_wav_path=denoised_wav,
                process_fn=denoiser.process,
                checkpoint_path=checkpoint_file,
                progress_callback=progress_callback,
                cancel_check=lambda: self._is_cancelled
            )

            if self._is_cancelled:
                return

            # Step 4: Remux Video (-c:v copy)
            self.status_changed.emit("🎬 무손실 영상 재합성 중 (-c:v copy)...")
            remux_video_with_audio(
                input_video_path=self.input_video_path,
                processed_audio_path=denoised_wav,
                output_video_path=self.output_video_path,
                audio_codec="aac",
                audio_bitrate="192k"
            )

            if self._is_cancelled:
                return

            # Step 5: Spectrogram & QA Analysis
            self.status_changed.emit("📊 품질 분석 및 스펙트로그램 시각화 생성 중...")
            spec_img = temp_dir / "spectrogram_comparison.png"
            generate_spectrogram_image(extracted_wav, denoised_wav, spec_img, duration_limit_sec=30.0)

            meta_denoised = probe_media(self.output_video_path)
            meta_comp = compare_media_metadata(meta_orig, meta_denoised)
            qa_metrics = analyze_audio_quality(extracted_wav, denoised_wav)

            total_elapsed = time.time() - start_time
            result_payload = {
                "input_path": str(self.input_video_path),
                "output_path": str(self.output_video_path),
                "spectrogram_image": str(spec_img),
                "elapsed_sec": total_elapsed,
                "meta_comparison": meta_comp,
                "qa_metrics": qa_metrics
            }
            self.finished_success.emit(result_payload)

        except InterruptedError:
            self.status_changed.emit("작업이 사용자에 의해 취소되었습니다.")
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self._is_cancelled:
                cleanup_temp_dir(temp_dir)
