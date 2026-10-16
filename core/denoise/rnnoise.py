"""
RNNoise engine using FFmpeg's built-in arnndn recurrent neural network filter.
Extremely fast (150x+ real-time), lightweight RNN speech enhancement.
"""
from typing import Optional
from pathlib import Path
import subprocess
import numpy as np
from core.denoise.base import BaseDenoiser, apply_studio_cleanup
from core.probe import get_ffmpeg_binary
from utils.paths import app_base_dir, SUBPROCESS_FLAGS


class RNNoiseDenoiser(BaseDenoiser):
    """
    RNNoise (FFmpeg arnndn) neural network denoiser.
    Runs on CPU with ultra-low latency and minimal resource footprint.
    """
    def __init__(
        self,
        strength: float = 1.0,
        voice_protection: float = 0.3,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = "cpu",
        enable_studio_cleanup: bool = True
    ):
        super().__init__(strength=strength, voice_protection=voice_protection, device=device)
        self.enable_studio_cleanup = enable_studio_cleanup

        # Model path
        default_model = app_base_dir() / "models" / "bd.rnnn"
        self.model_path = Path(model_path).resolve() if model_path else default_model
        if not self.model_path.exists():
            raise FileNotFoundError(f"RNNoise model not found: {self.model_path}")

        self.ffmpeg_bin = get_ffmpeg_binary()

    @property
    def name(self) -> str:
        return "RNNoise (3순위 - 경량 순환신경망 RNN / 초경량 CPU)"

    def process(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        if self.strength <= 0.0:
            return chunk.copy()

        is_1d = (chunk.ndim == 1)
        data = chunk[:, np.newaxis] if is_1d else chunk
        num_samples, num_channels = data.shape
        data_f32 = np.ascontiguousarray(data, dtype=np.float32)

        # Pipe chunk into FFmpeg arnndn
        cmd = [
            self.ffmpeg_bin,
            "-v", "quiet",
            "-f", "f32le",
            "-ar", str(sr),
            "-ac", str(num_channels),
            "-i", "pipe:0",
            # Relative model name + cwd: a Windows drive colon (C:\) breaks ffmpeg filter-arg parsing
            "-af", f"arnndn=m='{self.model_path.name}'",
            "-f", "f32le",
            "pipe:1"
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.model_path.parent),
                **SUBPROCESS_FLAGS
            )
            raw_out, _ = proc.communicate(input=data_f32.tobytes())
            clean_data = np.frombuffer(raw_out, dtype=np.float32).reshape(-1, num_channels)

            # Conform length if slight delay/padding introduced
            if len(clean_data) != num_samples:
                if len(clean_data) > num_samples:
                    clean_data = clean_data[:num_samples]
                else:
                    pad = np.zeros((num_samples - len(clean_data), num_channels), dtype=np.float32)
                    clean_data = np.vstack([clean_data, pad])
        except Exception:
            clean_data = data_f32.copy()

        # Apply intelligent voice protection & strength
        result = self.apply_strength_and_protection(clean=clean_data, noisy=data, sr=sr)

        # Apply studio cleanup if enabled (eliminates residual hiss & rumble)
        if self.enable_studio_cleanup:
            result = apply_studio_cleanup(result, sr=sr)

        return result.squeeze(-1) if is_1d else result
