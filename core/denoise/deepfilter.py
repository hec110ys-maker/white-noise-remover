"""
DeepFilterNet (DeepFilterNet3) noise reduction engine.
State-of-the-art deep learning speech enhancement with real-time performance.
"""
from typing import Optional
import numpy as np
import torch
from core.denoise.base import BaseDenoiser
from utils.device import get_optimal_device


class DeepFilterDenoiser(BaseDenoiser):
    """
    DeepFilterNet3 denoiser implementation.
    Operates natively at 48kHz.
    """
    def __init__(
        self,
        strength: float = 1.0,
        voice_protection: float = 0.3,
        device: Optional[str] = None,
        atten_lim_db: Optional[float] = None,
        enable_studio_cleanup: bool = True
    ):
        super().__init__(strength=strength, voice_protection=voice_protection, device=device)
        self.atten_lim_db = atten_lim_db
        self.enable_studio_cleanup = enable_studio_cleanup
        
        # Determine compute device
        if self.device is None:
            dev_info = get_optimal_device()
            self.target_device = torch.device(dev_info.device_type if dev_info.device_type in ("cuda", "cpu") else "cpu")
        else:
            self.target_device = torch.device(self.device)

        # Lazy load model
        self.model = None
        self.df_state = None
        self._load_model()

    def _load_model(self) -> None:
        """Initialize and cache DeepFilterNet3 model."""
        from df.enhance import init_df
        from utils.paths import app_base_dir
        # DeepFilterNet3 is the recommended and standard 48kHz model.
        # Prefer the copy bundled in models/ so the packaged app works offline.
        bundled = app_base_dir() / "models" / "DeepFilterNet3"
        model, df_state, _ = init_df(
            model_base_dir=str(bundled) if (bundled / "config.ini").exists() else None,
            default_model="DeepFilterNet3",
            post_filter=True,
            log_level="ERROR",
            log_file=None
        )
        self.model = model.to(self.target_device)
        self.df_state = df_state

    @property
    def name(self) -> str:
        return "DeepFilterNet3 (AI 음성 향상)"

    def process(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio chunk with DeepFilterNet3.
        Supports mono (N,) or (N, 1) and stereo (N, 2).
        """
        if self.strength <= 0.0:
            return chunk.copy()

        from df.enhance import enhance

        is_1d = (chunk.ndim == 1)
        orig_shape = chunk.shape
        data = chunk[:, np.newaxis] if is_1d else chunk
        num_samples, num_channels = data.shape

        clean_channels = []
        for ch in range(num_channels):
            channel_data = data[:, ch].astype(np.float32)
            # Torch tensor in shape [1, samples]
            audio_tensor = torch.from_numpy(channel_data).unsqueeze(0).to(self.target_device)

            with torch.no_grad():
                enhanced_tensor = enhance(
                    self.model,
                    self.df_state,
                    audio_tensor,
                    pad=True,
                    atten_lim_db=self.atten_lim_db
                )

            enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            # Ensure exact sample length matching
            if len(enhanced_np) != num_samples:
                if len(enhanced_np) > num_samples:
                    enhanced_np = enhanced_np[:num_samples]
                else:
                    enhanced_np = np.pad(enhanced_np, (0, num_samples - len(enhanced_np)))

            clean_channels.append(enhanced_np)

        clean_data = np.column_stack(clean_channels)
        
        # Apply strength blending & speech formant protection
        result = self.apply_strength_and_protection(clean=clean_data, noisy=data, sr=sr)
        
        # Apply studio cleanup if enabled (eliminates residual 5% hiss & rumble)
        if self.enable_studio_cleanup:
            from core.denoise.base import apply_studio_cleanup
            result = apply_studio_cleanup(result, sr=sr)

        return result.squeeze(-1) if is_1d else result
