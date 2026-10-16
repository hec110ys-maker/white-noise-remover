"""
Path utilities ensuring safe handling of Korean characters, spaces, and Windows/Mac paths.
"""
import os
import sys
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Optional


def app_base_dir() -> Path:
    """Root folder holding bundled resources (models/, bin/). Works in source and PyInstaller builds."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """Writable per-user folder for logs (the install folder may be read-only)."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path.home() / ".local" / "share"
    d = root / "WhiteNoiseRemover"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Hide the console window that each ffmpeg call would otherwise flash on Windows
SUBPROCESS_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def ensure_path(path: str | Path) -> Path:
    """Convert input to absolute resolved Path."""
    return Path(path).resolve()


def get_output_video_path(input_video_path: str | Path, output_dir: Optional[str | Path] = None) -> Path:
    """
    Generate output video path: {original_stem}_denoised.mp4
    Original file is NEVER modified or overwritten.
    """
    input_path = ensure_path(input_video_path)
    target_dir = Path(output_dir).resolve() if output_dir else input_path.parent
    output_filename = f"{input_path.stem}_denoised.mp4"
    return target_dir / output_filename


def get_temp_work_dir(input_video_path: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    """
    Create a unique working directory based on input file path hash to prevent collisions.
    """
    input_path = ensure_path(input_video_path)
    file_id = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()[:12]
    root_dir = Path(base_dir).resolve() if base_dir else input_path.parent
    temp_dir = root_dir / ".temp_denoise" / f"{input_path.stem}_{file_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_dir(temp_dir: str | Path) -> None:
    """Safely remove temporary directory and its contents."""
    p = Path(temp_dir)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
