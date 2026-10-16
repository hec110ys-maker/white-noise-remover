"""
Unit tests for AudioChunker and Overlap-Add reconstruction.
Verifies exact sample count preservation and bit-exact passthrough.
"""
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from core.chunker import AudioChunker, create_fade_windows


def test_fade_windows_sum_to_one():
    """Verify Hann fade-in and fade-out sum to 1.0 everywhere."""
    overlap_len = 48000
    fade_in, fade_out = create_fade_windows(overlap_len)
    
    assert len(fade_in) == overlap_len
    assert len(fade_out) == overlap_len
    
    total = fade_in + fade_out
    np.testing.assert_allclose(total, np.ones(overlap_len), atol=1e-6)


@pytest.mark.parametrize("duration_sec", [5.0, 31.0, 75.0])
@pytest.mark.parametrize("channels", [1, 2])
def test_chunker_passthrough_reconstruction(tmp_path: Path, duration_sec: float, channels: int):
    """
    Generate synthetic test signal (sine wave + white noise) and pass through chunker
    with process_fn=None (lossless passthrough).
    Output must have exact same sample count and negligible reconstruction error.
    """
    sr = 48000
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    
    # Sine wave (440Hz) + low level noise
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.normal(0, 1, total_samples)
    sig = (sine + noise).astype(np.float32)
    
    if channels == 1:
        audio_data = sig[:, np.newaxis]
    else:
        # Stereo: left is sine+noise, right is inverted
        audio_data = np.column_stack([sig, -sig])

    in_wav = tmp_path / f"test_in_{duration_sec}s_ch{channels}.wav"
    out_wav = tmp_path / f"test_out_{duration_sec}s_ch{channels}.wav"
    
    sf.write(str(in_wav), audio_data, sr, subtype="FLOAT")
    
    # 30-sec chunk, 1-sec overlap
    chunker = AudioChunker(chunk_duration_sec=30.0, overlap_duration_sec=1.0, sample_rate=sr)
    
    progress_history = []
    result = chunker.process_file(
        input_wav_path=in_wav,
        output_wav_path=out_wav,
        process_fn=None,  # Passthrough
        progress_callback=lambda p: progress_history.append(p)
    )
    
    # 1. Exact sample count match
    assert result["exact_match"] is True
    assert result["total_samples"] == total_samples
    
    # Read back and compare waveform
    recon_data, recon_sr = sf.read(str(out_wav), dtype="float32", always_2d=True)
    assert recon_sr == sr
    assert recon_data.shape == audio_data.shape
    
    # Max absolute error must be negligible (< 1e-5 due to 32-bit float arithmetic)
    max_err = np.max(np.abs(recon_data - audio_data))
    mse = np.mean((recon_data - audio_data) ** 2)
    
    assert max_err < 1e-5, f"Reconstruction error too high: max_err={max_err}"
    assert mse < 1e-9, f"MSE too high: mse={mse}"
    
    # Progress callback check
    assert len(progress_history) > 0
    assert progress_history[-1]["progress_percent"] == 100.0
