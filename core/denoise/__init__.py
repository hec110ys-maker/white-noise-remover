"""
Denoiser implementations and factory.
"""
from core.denoise.base import BaseDenoiser, apply_studio_cleanup
from core.denoise.deepfilter import DeepFilterDenoiser
from core.denoise.spectral import SpectralGatingDenoiser
from core.denoise.rnnoise import RNNoiseDenoiser

__all__ = [
    "BaseDenoiser",
    "DeepFilterDenoiser",
    "SpectralGatingDenoiser",
    "RNNoiseDenoiser",
    "apply_studio_cleanup"
]


