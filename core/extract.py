"""
Audio extraction and video remuxing using FFmpeg.
Video stream is ALWAYS copied without re-encoding (-c:v copy).
"""
import subprocess
from pathlib import Path
from typing import Optional, Callable
from core.probe import get_ffmpeg_binary, probe_media
from utils.paths import SUBPROCESS_FLAGS


def extract_audio(
    input_video_path: str | Path,
    output_wav_path: str | Path,
    sample_rate: int = 48000,
    channels: Optional[int] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Path:
    """
    Extract audio track from video to 32-bit float PCM WAV (48kHz).
    32-bit float prevents digital clipping and preserves maximum dynamic range.
    """
    video_path = Path(input_video_path).resolve()
    wav_path = Path(output_wav_path).resolve()
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    ffmpeg = get_ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",  # Overwrite temp wav if exists
        "-i", str(video_path),
        "-vn",  # Disable video stream
        "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",  # 32-bit float PCM
    ]

    if channels:
        cmd.extend(["-ac", str(channels)])

    cmd.append(str(wav_path))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **SUBPROCESS_FLAGS
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"Extracted audio file is missing or empty: {wav_path}")

    return wav_path


def remux_video_with_audio(
    input_video_path: str | Path,
    processed_audio_path: str | Path,
    output_video_path: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k"
) -> Path:
    """
    Combine original video stream (copied losslessly via -c:v copy)
    with processed audio stream.
    Original video file is never modified.
    """
    video_in = Path(input_video_path).resolve()
    audio_in = Path(processed_audio_path).resolve()
    video_out = Path(output_video_path).resolve()
    video_out.parent.mkdir(parents=True, exist_ok=True)

    if not video_in.exists():
        raise FileNotFoundError(f"Input video not found: {video_in}")
    if not audio_in.exists():
        raise FileNotFoundError(f"Processed audio not found: {audio_in}")
    if video_in == video_out:
        raise ValueError("Output video path must be different from input video path.")

    ffmpeg = get_ffmpeg_binary()
    
    # -c:v copy: zero video quality loss, high speed
    # -c:a aac -b:a 192k: high quality audio encoding
    # -map 0:v:0: first video stream from input video
    # -map 1:a:0: first audio stream from processed audio
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",  # Web/YouTube streaming friendly
        str(video_out)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **SUBPROCESS_FLAGS
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg remuxing failed: {result.stderr}")

    if not video_out.exists() or video_out.stat().st_size == 0:
        raise RuntimeError(f"Remuxed video file is missing or empty: {video_out}")

    return video_out
