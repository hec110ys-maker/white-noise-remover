"""
Probe video and audio metadata using ffprobe.
"""
import json
import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from utils.paths import app_base_dir, SUBPROCESS_FLAGS


def ensure_ffmpeg_path():
    """Put the bundled bin/ folder first in PATH, then standard install locations (Homebrew, /usr/local/bin)."""
    search_dirs = [
        "/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        str(app_base_dir()),
        str(app_base_dir() / "bin"),
    ]
    current_path = os.environ.get("PATH", "")
    for d in search_dirs:
        if os.path.isdir(d) and d not in current_path.split(os.pathsep):
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"


# Run once on module import
ensure_ffmpeg_path()


def get_ffprobe_binary() -> str:
    """Find ffprobe in system PATH."""
    ensure_ffmpeg_path()
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        for candidate in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "ffprobe binary not found. Please ensure FFmpeg/ffprobe is installed and in your system PATH."
        )
    return ffprobe_path


def get_ffmpeg_binary() -> str:
    """Find ffmpeg in system PATH."""
    ensure_ffmpeg_path()
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        for candidate in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            "ffmpeg binary not found. Please ensure FFmpeg is installed and in your system PATH."
        )
    return ffmpeg_path


@dataclass
class MediaMetadata:
    filepath: Path
    duration_sec: float
    has_video: bool
    video_codec: Optional[str]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    video_frames: Optional[int]
    has_audio: bool
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]
    audio_duration_sec: Optional[float]
    num_video_streams: int
    num_audio_streams: int
    raw_info: Dict[str, Any]


def probe_media(filepath: str | Path) -> MediaMetadata:
    """
    Extract comprehensive video and audio metadata using ffprobe.
    Supports MP4, MOV, MKV and audio formats.
    """
    path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    ffprobe = get_ffprobe_binary()
    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            **SUBPROCESS_FLAGS
        )
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe execution failed for {path}: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe JSON output: {e}") from e

    format_info = data.get("format", {})
    duration_sec = float(format_info.get("duration", 0.0))

    has_video = False
    video_codec = None
    width = None
    height = None
    fps = None
    video_frames = None
    num_video_streams = 0

    has_audio = False
    audio_codec = None
    audio_sample_rate = None
    audio_channels = None
    audio_duration_sec = None
    num_audio_streams = 0

    streams = data.get("streams", [])
    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and not has_video:
            has_video = True
            num_video_streams += 1
            video_codec = s.get("codec_name")
            width = s.get("width")
            height = s.get("height")
            
            # Parse frame rate
            r_frame_rate = s.get("r_frame_rate", "0/0")
            if "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) != 0 else None
            else:
                fps = float(r_frame_rate) if r_frame_rate else None
                
            nb_frames = s.get("nb_frames")
            video_frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else None

        elif codec_type == "video":
            num_video_streams += 1

        elif codec_type == "audio" and not has_audio:
            has_audio = True
            num_audio_streams += 1
            audio_codec = s.get("codec_name")
            sr = s.get("sample_rate")
            audio_sample_rate = int(sr) if sr else None
            ch = s.get("channels")
            audio_channels = int(ch) if ch else None
            dur = s.get("duration")
            audio_duration_sec = float(dur) if dur else duration_sec

        elif codec_type == "audio":
            num_audio_streams += 1

    return MediaMetadata(
        filepath=path,
        duration_sec=duration_sec,
        has_video=has_video,
        video_codec=video_codec,
        width=width,
        height=height,
        fps=fps,
        video_frames=video_frames,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        audio_duration_sec=audio_duration_sec,
        num_video_streams=num_video_streams,
        num_audio_streams=num_audio_streams,
        raw_info=data
    )


def compare_media_metadata(orig: MediaMetadata, denoised: MediaMetadata) -> Dict[str, Any]:
    """
    Compare input vs output metadata to verify video preservation (-c:v copy)
    and audio sync integrity.
    """
    duration_diff = abs(orig.duration_sec - denoised.duration_sec)
    
    # Tolerances
    is_duration_match = duration_diff <= 0.1  # within 100ms
    is_resolution_match = (orig.width == denoised.width) and (orig.height == denoised.height)
    
    # FPS match within 0.05 fps
    orig_fps = orig.fps or 0.0
    denoised_fps = denoised.fps or 0.0
    is_fps_match = abs(orig_fps - denoised_fps) < 0.05
    
    is_video_codec_match = (orig.video_codec == denoised.video_codec)

    is_valid = (
        is_duration_match and
        is_resolution_match and
        is_fps_match and
        is_video_codec_match
    )

    return {
        "valid": is_valid,
        "duration_diff_sec": duration_diff,
        "is_duration_match": is_duration_match,
        "is_resolution_match": is_resolution_match,
        "is_fps_match": is_fps_match,
        "is_video_codec_match": is_video_codec_match,
        "orig_resolution": f"{orig.width}x{orig.height}" if orig.width else "N/A",
        "denoised_resolution": f"{denoised.width}x{denoised.height}" if denoised.width else "N/A",
        "orig_fps": round(orig_fps, 2),
        "denoised_fps": round(denoised_fps, 2),
        "orig_duration": round(orig.duration_sec, 2),
        "denoised_duration": round(denoised.duration_sec, 2),
        "orig_video_codec": orig.video_codec,
        "denoised_video_codec": denoised.video_codec,
        "output_audio_codec": denoised.audio_codec,
        "output_audio_sr": denoised.audio_sample_rate
    }
