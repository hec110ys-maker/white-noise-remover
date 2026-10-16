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


def estimate_work_bytes(duration_sec: float, channels: Optional[int]) -> int:
    """Disk needed for the temp work: extracted + denoised 48kHz float32 WAVs, plus 10% margin."""
    wav = duration_sec * 48000 * max(1, channels or 2) * 4
    return int(wav * 2 * 1.1)


def _has_room(directory: Path, needed_bytes: int) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return shutil.disk_usage(directory).free >= needed_bytes
    except OSError:
        return False


def get_temp_work_dir(
    input_video_path: str | Path,
    base_dir: Optional[str | Path] = None,
    needed_bytes: int = 0,
) -> Path:
    """
    Create a unique working directory based on input file path hash to prevent collisions.
    Uses the video's folder when it is writable and has room (e.g. a big D: drive); otherwise
    the system temp folder (read-only folders, USB sticks, full disks).
    """
    import tempfile
    input_path = ensure_path(input_video_path)
    file_id = hashlib.sha256(str(input_path).encode("utf-8")).hexdigest()[:12]
    roots = [Path(base_dir).resolve()] if base_dir else [input_path.parent, Path(tempfile.gettempdir())]
    for root in roots:
        temp_dir = root / ".temp_denoise" / f"{input_path.stem}_{file_id}"
        if _has_room(temp_dir, needed_bytes):
            return temp_dir
    need_gb = needed_bytes / 1024 ** 3
    raise RuntimeError(
        f"작업용 디스크 공간이 부족합니다. 약 {need_gb:.1f}GB가 필요합니다.\n"
        f"영상 폴더나 C: 드라이브의 공간을 확보한 뒤 다시 시도하세요."
    )


def cleanup_temp_dir(temp_dir: str | Path) -> None:
    """Safely remove temporary directory and its contents, plus the .temp_denoise parent once empty."""
    p = Path(temp_dir)
    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    parent = p.parent
    if parent.name == ".temp_denoise":
        try:
            parent.rmdir()  # only succeeds when no other job is using it
        except OSError:
            pass
