"""
Abstract Base Class for all noise reduction engines.
Provides common interface, strength blending, and voice protection algorithms.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseDenoiser(ABC):
    """
    Common interface for all denoiser backends.
    """
    def __init__(
        self,
        strength: float = 1.0,
        voice_protection: float = 0.5,
        device: Optional[str] = None
    ):
        """
        :param strength: Noise reduction intensity from 0.0 (bypass) to 1.0 (full reduction).
        :param voice_protection: Speech preservation weight from 0.0 (no protection) to 1.0 (maximum voice warmth).
        :param device: 'cuda', 'mps', or 'cpu'. If None, auto-detected.
        """
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.voice_protection = float(np.clip(voice_protection, 0.0, 1.0))
        self.device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the denoiser engine."""
        pass

    @abstractmethod
    def process(self, chunk: np.ndarray, sr: int) -> np.ndarray:
        """
        Process an audio chunk.
        :param chunk: Audio data, shape (N, C) or (N,), float32 in [-1.0, 1.0].
        :param sr: Sample rate (typically 48000).
        :return: Denoised audio data with identical shape and dtype.
        """
        pass

    def apply_strength_and_protection(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Intelligently blend clean and noisy audio according to strength and voice protection settings.
        Protects speech vowel formants (150Hz - 2800Hz) during active speech while strictly
        preventing high-frequency white noise (hiss) and silence bleed.
        """
        if self.strength <= 0.0:
            return noisy.copy()

        # Base blending by strength
        if self.strength >= 1.0:
            blended = clean.copy()
        else:
            blended = self.strength * clean + (1.0 - self.strength) * noisy

        # Intelligent Voice Protection
        # Only restores speech formant resonance during active speech,
        # never leaks hiss into silence or high frequencies (>3kHz)
        if self.voice_protection > 0.0:
            from scipy.signal import butter, sosfilt

            is_1d = (clean.ndim == 1)
            c_data = clean[:, np.newaxis] if is_1d else clean
            n_data = noisy[:, np.newaxis] if is_1d else noisy

            # 1. Bandpass filter for human speech vowel formants (150Hz to 2800Hz)
            # High-frequency white noise hiss (>3kHz) is strictly excluded
            sos = butter(4, [150, min(2800, sr // 2 - 100)], btype="bandpass", fs=sr, output="sos")
            formant_noisy = sosfilt(sos, n_data, axis=0)
            formant_clean = sosfilt(sos, c_data, axis=0)

            # 2. Smooth speech envelope from clean audio (20ms window)
            frame_len = max(int(0.02 * sr), 16)
            power = np.mean(c_data ** 2, axis=1)
            win = np.ones(frame_len, dtype=np.float32) / frame_len
            env = np.sqrt(np.convolve(power, win, mode="same"))
            env_db = 20.0 * np.log10(np.maximum(env, 1e-9))

            # Speech activity mask: silence (< -48dB) = 0.0, speech (> -28dB) = 1.0
            speech_mask = np.clip((env_db - (-48.0)) / 20.0, 0.0, 1.0)
            speech_mask = speech_mask[:, np.newaxis]

            # 3. Additive formant body preservation during active speech only
            protection_weight = float(0.12 * self.voice_protection)
            formant_diff = formant_noisy - formant_clean
            body_addition = formant_diff * (speech_mask * protection_weight)

            blended_2d = blended[:, np.newaxis] if is_1d else blended
            blended_2d = blended_2d + body_addition
            blended = blended_2d.squeeze(-1) if is_1d else blended_2d

        return blended.astype(np.float32)


def apply_studio_cleanup(
    audio: np.ndarray,
    sr: int = 48000,
    enable_expander: bool = True,
    enable_rumble_cut: bool = True
) -> np.ndarray:
    """
    Studio-grade post-processing to eliminate remaining 5% micro-artifacts:
    1. 75Hz 2nd-order High-Pass filter: eliminates sub-bass mic rumble / AC hum.
    2. Dynamic Downward Expander: gently reduces residual noise floor in pause/breathing
       segments down to pristine studio black silence (-80dB+) without clipping speech tails.
    """
    if audio is None or len(audio) == 0:
        return audio

    from scipy.signal import butter, sosfilt, lfilter

    is_1d = (audio.ndim == 1)
    data = audio[:, np.newaxis] if is_1d else audio
    num_samples, num_channels = data.shape

    # 1. Rumble cut (< 75Hz)
    if enable_rumble_cut and sr > 150:
        sos_hp = butter(2, 75, btype="highpass", fs=sr, output="sos")
        data = sosfilt(sos_hp, data, axis=0)

    # 2. Dynamic Downward Expander
    if enable_expander:
        frame_len = max(int(0.015 * sr), 16)
        power = np.mean(data ** 2, axis=1)
        win = np.ones(frame_len, dtype=np.float32) / frame_len
        env = np.sqrt(np.convolve(power, win, mode="same"))
        env_db = 20.0 * np.log10(np.maximum(env, 1e-9))

        # Threshold: -45dB. Below this, expand downwards (ratio 2.0, max 24dB attenuation)
        thresh = -45.0
        ratio = 2.0
        gain_db = np.zeros_like(env_db)
        mask = env_db < thresh
        gain_db[mask] = (env_db[mask] - thresh) * (ratio - 1.0)
        gain_db = np.clip(gain_db, -24.0, 0.0)

        # Smooth gain envelope (approx. 40ms time constant) via 1-pole filter with exact analytical initial state
        # (avoiding scipy lfilter_zi -> OpenBLAS matrix solve stack overflow in worker threads)
        gain_linear = (10.0 ** (gain_db / 20.0)).astype(np.float32)
        alpha = float(np.exp(-1.0 / (0.040 * sr)))
        b = [1.0 - alpha]
        a = [1.0, -alpha]
        zi = np.array([gain_linear[0] * alpha], dtype=np.float32)
        smoothed_gain, _ = lfilter(b, a, gain_linear, zi=zi)
        smoothed_gain = np.clip(smoothed_gain.astype(np.float32), 0.05, 1.0)[:, np.newaxis]

        data = data * smoothed_gain

    result = data.squeeze(-1) if is_1d else data
    return result.astype(np.float32)

