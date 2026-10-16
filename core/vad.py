"""
Voice Activity Detection (VAD) and noise profile extraction.
Finds non-speech segments to capture stationary microphone background noise.
"""
from typing import Optional, Tuple
import numpy as np
import soundfile as sf
from pathlib import Path


def extract_noise_sample_from_audio(
    audio_data: np.ndarray,
    sr: int = 48000,
    sample_duration_sec: float = 1.0,
    manual_start_sec: Optional[float] = None,
    manual_end_sec: Optional[float] = None
) -> np.ndarray:
    """
    Extract a clean noise-only segment from the audio.
    If manual start/end are specified, use that exact slice.
    Otherwise, search for the lowest-energy non-speech segment using windowed RMS.
    """
    is_1d = (audio_data.ndim == 1)
    data = audio_data[:, np.newaxis] if is_1d else audio_data
    total_samples, channels = data.shape

    # Manual time slice
    if manual_start_sec is not None and manual_end_sec is not None:
        start_idx = int(manual_start_sec * sr)
        end_idx = int(manual_end_sec * sr)
        start_idx = max(0, min(start_idx, total_samples - 1))
        end_idx = max(start_idx + 100, min(end_idx, total_samples))
        sample = data[start_idx:end_idx]
        return sample.squeeze(-1) if is_1d else sample

    # Auto-detection: Search for the quietest 1-second segment (pure background noise)
    window_samples = int(sample_duration_sec * sr)
    if total_samples <= window_samples:
        return audio_data.copy()

    # Step in 0.2s increments
    step_samples = int(0.2 * sr)
    min_energy = float("inf")
    best_start = 0

    # Mono representation for energy computation
    mono = np.mean(data, axis=1)

    for i in range(0, total_samples - window_samples, step_samples):
        chunk = mono[i:i + window_samples]
        energy = np.mean(chunk ** 2)
        if energy < min_energy:
            min_energy = energy
            best_start = i

    best_end = best_start + window_samples
    noise_sample = data[best_start:best_end]
    return noise_sample.squeeze(-1) if is_1d else noise_sample


def extract_noise_sample_from_file(
    wav_path: str | Path,
    sample_duration_sec: float = 1.0,
    manual_start_sec: Optional[float] = None,
    manual_end_sec: Optional[float] = None
) -> np.ndarray:
    """Read WAV file and extract stationary noise sample."""
    path = Path(wav_path).resolve()
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return extract_noise_sample_from_audio(
        data,
        sr=sr,
        sample_duration_sec=sample_duration_sec,
        manual_start_sec=manual_start_sec,
        manual_end_sec=manual_end_sec
    )
