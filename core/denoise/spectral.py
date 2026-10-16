"""
Spectral Gating noise reduction engine using noisereduce.
Extremely fast on CPU, highly effective against stationary white noise/microphone hiss.
"""
from typing import Optional
import numpy as np
from core.denoise.base import BaseDenoiser


class SpectralGatingDenoiser(BaseDenoiser):
    """
    Spectral gating denoiser.
    Works directly with stationary background noise profiles.
    """
    def __init__(
        self,
        strength: float = 1.0,
        voice_protection: float = 0.3,
        noise_profile: Optional[np.ndarray] = None,
        device: Optional[str] = "cpu",
        enable_studio_cleanup: bool = True
    ):
        super().__init__(strength=strength, voice_protection=voice_protection, device=device)
        self.noise_profile = noise_profile
        self.enable_studio_cleanup = enable_studio_cleanup

    @property
    def name(self) -> str:
        return "noisereduce (스펙트럴 게이팅 - 초고속 CPU)"

    def set_noise_profile(self, noise_profile: np.ndarray) -> None:
        """Set stationary noise audio profile for subtraction."""
        self.noise_profile = noise_profile

    def process(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        if self.strength <= 0.0:
            return chunk.copy()

        import noisereduce as nr

        is_1d = (chunk.ndim == 1)
        data = chunk[:, np.newaxis] if is_1d else chunk
        num_samples, num_channels = data.shape

        clean_channels = []
        for ch in range(num_channels):
            y = data[:, ch].astype(np.float32)
            y_noise = None
            if self.noise_profile is not None:
                if self.noise_profile.ndim > 1:
                    y_noise = self.noise_profile[:, min(ch, self.noise_profile.shape[1] - 1)]
                else:
                    y_noise = self.noise_profile

            # Spectral gating noise reduction
            # prop_decrease controls reduction strength
            clean_ch = nr.reduce_noise(
                y=y,
                sr=sr,
                y_noise=y_noise,
                prop_decrease=float(self.strength),
                stationary=True,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                n_jobs=1
            )
            clean_channels.append(clean_ch.astype(np.float32))

        clean_data = np.column_stack(clean_channels)
        result = self.apply_strength_and_protection(clean=clean_data, noisy=data, sr=sr)

        # Apply studio cleanup if enabled (eliminates residual 5% hiss & rumble)
        if self.enable_studio_cleanup:
            from core.denoise.base import apply_studio_cleanup
            result = apply_studio_cleanup(result, sr=sr)

        return result.squeeze(-1) if is_1d else result
