"""
End-to-end integration test for Video Probe -> Audio Extract -> Chunker Passthrough -> Remux (-c:v copy).
Verifies:
1. Video stream is 100% preserved without re-encoding.
2. Frame rate, resolution, and total duration are identical.
3. Audio sync and exact length preservation.
4. Original file is completely untouched.
"""
import subprocess
import pytest
from pathlib import Path
from core.probe import probe_media, compare_media_metadata, get_ffmpeg_binary
from core.extract import extract_audio, remux_video_with_audio
from core.chunker import AudioChunker


def create_synthetic_test_video(output_mp4: Path, duration_sec: float = 6.0) -> Path:
    """Create a synthetic MP4 video (1920x1080 30fps with 440Hz sine audio) using ffmpeg."""
    ffmpeg = get_ffmpeg_binary()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        ffmpeg,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=1920x1080:rate=30",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_sec}:sample_rate=48000",
        "-c:v", "mpeg4",  # built-in encoder: the bundled LGPL FFmpeg has no libx264
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_mp4


def test_full_pipeline_passthrough_av_preservation(tmp_path: Path):
    """
    Run full step-2 pipeline:
    Create video -> probe -> extract wav -> chunker passthrough -> remux -> compare probe.
    """
    orig_video = tmp_path / "original_test_video.mp4"
    create_synthetic_test_video(orig_video, duration_sec=8.0)
    
    # Check original file size and hash/stat
    orig_stat_before = orig_video.stat()
    
    # 1. Probe original metadata
    meta_orig = probe_media(orig_video)
    assert meta_orig.has_video is True
    assert meta_orig.has_audio is True
    assert meta_orig.width == 1920
    assert meta_orig.height == 1080
    assert abs(meta_orig.fps - 30.0) < 0.01
    assert abs(meta_orig.duration_sec - 8.0) < 0.2
    
    # 2. Extract audio
    extracted_wav = tmp_path / "extracted_audio.wav"
    extract_audio(orig_video, extracted_wav, sample_rate=48000)
    assert extracted_wav.exists()
    
    # 3. Chunker passthrough (using small chunk/overlap to exercise multiple chunks on 8s audio)
    reconstructed_wav = tmp_path / "reconstructed_audio.wav"
    chunker = AudioChunker(chunk_duration_sec=3.0, overlap_duration_sec=0.5, sample_rate=48000)
    chunk_result = chunker.process_file(
        input_wav_path=extracted_wav,
        output_wav_path=reconstructed_wav,
        process_fn=None  # Passthrough
    )
    assert chunk_result["exact_match"] is True
    
    # 4. Remux with original video (-c:v copy)
    denoised_video = tmp_path / "original_test_video_denoised.mp4"
    remux_video_with_audio(
        input_video_path=orig_video,
        processed_audio_path=reconstructed_wav,
        output_video_path=denoised_video,
        audio_codec="aac",
        audio_bitrate="192k"
    )
    assert denoised_video.exists()
    
    # 5. Probe denoised output metadata
    meta_denoised = probe_media(denoised_video)
    
    # 6. Compare metadata
    comparison = compare_media_metadata(meta_orig, meta_denoised)
    
    assert comparison["valid"] is True, f"Metadata comparison failed: {comparison}"
    assert comparison["is_resolution_match"] is True
    assert comparison["is_fps_match"] is True
    assert comparison["is_video_codec_match"] is True
    assert comparison["duration_diff_sec"] < 0.05
    assert meta_denoised.audio_codec == "aac"
    
    # 7. Verify original file was strictly NOT modified
    orig_stat_after = orig_video.stat()
    assert orig_stat_before.st_mtime == orig_stat_after.st_mtime
    assert orig_stat_before.st_size == orig_stat_after.st_size
