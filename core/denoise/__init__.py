"""
Denoiser implementations and factory.
"""
import importlib.util

from core.denoise.base import BaseDenoiser, apply_studio_cleanup
from core.denoise.spectral import SpectralGatingDenoiser
from core.denoise.rnnoise import RNNoiseDenoiser


def deepfilter_available() -> bool:
    """True when PyTorch + DeepFilterNet are installed. The Windows lite build ships without them."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "df"))


def available_engines() -> list[str]:
    """Engine ids in recommended order; deepfilter only when its AI dependencies are present."""
    return (["deepfilter"] if deepfilter_available() else []) + ["spectral", "rnnoise"]


def __getattr__(name):
    # Lazy so importing this package never pulls in torch
    if name == "DeepFilterDenoiser":
        from core.denoise.deepfilter import DeepFilterDenoiser
        return DeepFilterDenoiser
    raise AttributeError(name)


__all__ = [
    "BaseDenoiser",
    "DeepFilterDenoiser",
    "SpectralGatingDenoiser",
    "RNNoiseDenoiser",
    "apply_studio_cleanup",
    "deepfilter_available",
    "available_engines",
]


