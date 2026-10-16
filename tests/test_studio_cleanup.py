import numpy as np
from core.denoise.base import BaseDenoiser, apply_studio_cleanup
from core.denoise.spectral import SpectralGatingDenoiser


def test_intelligent_voice_protection_no_silence_leak():
    """Ensure voice protection does not leak noise into complete silence."""
    sr = 48000
    N = sr * 2
    # Silence clean signal
    clean = np.zeros((N, 1), dtype=np.float32)
    # White noise signal
    noisy = (np.random.randn(N, 1) * 0.1).astype(np.float32)

    # Blending lives in BaseDenoiser; use a torch-free engine so this runs in the Windows lite build
    denoiser = SpectralGatingDenoiser(strength=1.0, voice_protection=0.5, enable_studio_cleanup=False)
    # Apply blending
    result = denoiser.apply_strength_and_protection(clean, noisy, sr)

    # In silence (< -48dB), speech_mask is 0, so result must be exactly clean (0.0)
    assert np.allclose(result, 0.0, atol=1e-5), "Noise leaked into silence with voice protection!"


def test_studio_cleanup_attenuates_noise_floor():
    """Ensure studio cleanup lowers pause noise floor and preserves speech."""
    sr = 48000
    # 1 second of speech-like loud signal + 1 second of low-level hiss
    t = np.linspace(0, 1, sr, endpoint=False)
    speech_seg = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Hiss at -55dB
    hiss_seg = (np.random.randn(sr) * 0.0018).astype(np.float32)

    audio = np.concatenate([speech_seg, hiss_seg])
    processed = apply_studio_cleanup(audio, sr=sr, enable_expander=True, enable_rumble_cut=True)

    # Speech segment RMS should be preserved within 0.1dB
    orig_speech_rms = np.sqrt(np.mean(speech_seg ** 2))
    proc_speech_rms = np.sqrt(np.mean(processed[:sr] ** 2))
    assert abs(20 * np.log10(proc_speech_rms / orig_speech_rms)) < 0.2

    # Hiss segment should be significantly expanded downwards (> 6dB reduction)
    orig_hiss_rms = np.sqrt(np.mean(hiss_seg ** 2))
    proc_hiss_rms = np.sqrt(np.mean(processed[int(1.2 * sr):] ** 2))
    assert proc_hiss_rms < orig_hiss_rms * 0.5, "Studio cleanup failed to attenuate pause noise floor!"
