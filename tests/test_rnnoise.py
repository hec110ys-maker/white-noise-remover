import numpy as np
import pytest
from core.denoise.rnnoise import RNNoiseDenoiser


def test_rnnoise_denoiser_basic():
    """Test RNNoiseDenoiser processes mono and stereo chunks without error."""
    sr = 48000
    N = sr * 2  # 2 seconds
    denoiser = RNNoiseDenoiser(strength=1.0, voice_protection=0.3, enable_studio_cleanup=True)

    # 1. Stereo test
    stereo_data = (np.random.randn(N, 2) * 0.05).astype(np.float32)
    clean_stereo = denoiser.process(stereo_data, sr)
    assert clean_stereo.shape == stereo_data.shape
    assert clean_stereo.dtype == np.float32
    # Verify noise attenuation
    assert np.mean(clean_stereo ** 2) < np.mean(stereo_data ** 2)

    # 2. Mono test
    mono_data = (np.random.randn(N) * 0.05).astype(np.float32)
    clean_mono = denoiser.process(mono_data, sr)
    assert clean_mono.shape == mono_data.shape
    assert clean_mono.dtype == np.float32
    assert np.mean(clean_mono ** 2) < np.mean(mono_data ** 2)
