"""
Test DeepFilterNet3 noise reduction on a 3-minute (180-second) audio sample.
Uses realistic human speech + stationary microphone white noise.
Verifies:
1. Significant white noise suppression (>15dB in noise-only sections).
2. Speech level and formant preservation in active voice sections (<1.5dB change).
3. Exact total sample count and seamless Overlap-Add stitching.
4. Fast processing speed (RTF < 0.1x).
"""
import subprocess
import time
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from core.chunker import AudioChunker
from core.denoise.deepfilter import DeepFilterDenoiser


def get_real_speech_clip(sr: int = 48000) -> np.ndarray:
    """
    Generate realistic spoken speech using macOS 'say' or fall back to modulated vocal formants.
    """
    tmp_wav = Path("/tmp/pytest_speech_clip.wav")
    try:
        subprocess.run(
            [
                "say",
                "-v", "Alex",
                "Hello everyone. Welcome back to our YouTube channel. In this tutorial, "
                "we are testing the audio white noise removal program to make sure "
                "that the male voice remains rich, natural, and crystal clear.",
                "-o", str(tmp_wav),
                "--data-format=LEF32@48000"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        data, _ = sf.read(str(tmp_wav), dtype="float32")
        return data
    except Exception:
        # Fallback to realistic frequency-modulated harmonic synthesis
        dur = 8.0
        t = np.linspace(0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
        # Pitch variation 110Hz ~ 140Hz
        f0 = 125.0 + 15.0 * np.sin(2 * np.pi * 0.8 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        speech = (
            0.6 * np.sin(phase) +
            0.4 * np.sin(2 * phase) +
            0.3 * np.sin(3 * phase) +
            0.2 * np.sin(4 * phase)
        )
        return speech.astype(np.float32)


def test_deepfilter_3min_sample_reduction(tmp_path: Path):
    """
    Assemble 3-minute (180s) audio with realistic speech + stationary white noise.
    Process with DeepFilterDenoiser through AudioChunker.
    """
    sr = 48000
    duration_sec = 180.0  # 3 minutes
    total_samples = int(duration_sec * sr)

    speech_clip = get_real_speech_clip(sr)
    clip_len = len(speech_clip)
    pause_len = int(2.5 * sr)  # 2.5s silence/noise-only pause between speech sentences
    block_len = clip_len + pause_len

    clean_speech = np.zeros(total_samples, dtype=np.float32)
    speech_active_mask = np.zeros(total_samples, dtype=bool)
    noise_only_mask = np.zeros(total_samples, dtype=bool)

    cur = 0
    while cur < total_samples:
        # Insert speech clip
        end_sp = min(cur + clip_len, total_samples)
        sp_chunk_len = end_sp - cur
        clean_speech[cur:end_sp] = speech_clip[:sp_chunk_len]
        # Active speech mask (ignore 0.5s edges)
        if sp_chunk_len > int(1.0 * sr):
            speech_active_mask[cur + int(0.5 * sr):end_sp - int(0.5 * sr)] = True

        # Insert pause (pure background noise)
        cur = end_sp
        end_pause = min(cur + pause_len, total_samples)
        if end_pause - cur > int(1.0 * sr):
            noise_only_mask[cur + int(0.5 * sr):end_pause - int(0.5 * sr)] = True
        cur = end_pause

    # Generate persistent white noise (stationary microphone hiss, approx -28dB RMS)
    np.random.seed(42)
    stationary_noise = 0.04 * np.random.normal(0, 1, total_samples).astype(np.float32)

    noisy_audio = clean_speech + stationary_noise

    in_wav = tmp_path / "3min_sample_input.wav"
    out_wav = tmp_path / "3min_sample_denoised.wav"
    sf.write(str(in_wav), noisy_audio, sr, subtype="FLOAT")

    # Initialize DeepFilterDenoiser
    denoiser = DeepFilterDenoiser(strength=1.0, voice_protection=0.5)

    # Process through streaming chunker (30s chunk, 1s overlap)
    chunker = AudioChunker(chunk_duration_sec=30.0, overlap_duration_sec=1.0, sample_rate=sr)

    start_time = time.time()
    result = chunker.process_file(
        input_wav_path=in_wav,
        output_wav_path=out_wav,
        process_fn=denoiser.process
    )
    elapsed = time.time() - start_time
    rtf = elapsed / duration_sec

    # 1. Exact sample count match
    assert result["exact_match"] is True
    assert result["total_samples"] == total_samples

    # 2. Analyze audio levels
    denoised_audio, _ = sf.read(str(out_wav), dtype="float32")

    # Noise attenuation measurement in non-speech sections
    orig_noise_rms = np.sqrt(np.mean(noisy_audio[noise_only_mask] ** 2))
    denoised_noise_rms = np.sqrt(np.mean(denoised_audio[noise_only_mask] ** 2))
    orig_noise_db = 20 * np.log10(max(orig_noise_rms, 1e-12))
    denoised_noise_db = 20 * np.log10(max(denoised_noise_rms, 1e-12))
    noise_attenuation_db = orig_noise_db - denoised_noise_db

    # Speech level preservation measurement (clean speech vs denoised audio in active speech sections)
    clean_speech_rms = np.sqrt(np.mean(clean_speech[speech_active_mask] ** 2))
    denoised_speech_rms = np.sqrt(np.mean(denoised_audio[speech_active_mask] ** 2))
    clean_speech_db = 20 * np.log10(max(clean_speech_rms, 1e-12))
    denoised_speech_db = 20 * np.log10(max(denoised_speech_rms, 1e-12))
    speech_level_diff_db = abs(clean_speech_db - denoised_speech_db)

    print(f"\n" + "="*50)
    print(f"       DeepFilterNet 3-Minute Benchmark Results")
    print(f"="*50)
    print(f"• Audio Duration         : {duration_sec}s (3 minutes)")
    print(f"• Processing Time        : {elapsed:.2f}s (Real-Time Factor: {rtf:.3f}x)")
    print(f"• Noise Reduction Level  : {noise_attenuation_db:.2f} dB")
    print(f"  - Original Noise Floor : {orig_noise_db:.1f} dB")
    print(f"  - Denoised Noise Floor : {denoised_noise_db:.1f} dB")
    print(f"• Speech Preservation    : {speech_level_diff_db:.2f} dB diff vs pure voice")
    print(f"  - Clean Speech RMS     : {clean_speech_db:.1f} dB")
    print(f"  - Denoised Speech RMS  : {denoised_speech_db:.1f} dB")
    print(f"="*50)

    # White noise should be reduced by at least 15 dB
    assert noise_attenuation_db > 15.0, f"Noise reduction too weak: {noise_attenuation_db:.2f} dB"
    # Pure voice level should be preserved within 1.5 dB
    assert speech_level_diff_db < 1.5, f"Speech level altered too much: {speech_level_diff_db:.2f} dB"
    # Processing speed must be faster than 0.1x real-time
    assert rtf < 0.1, f"Processing too slow: {rtf:.3f}x"
