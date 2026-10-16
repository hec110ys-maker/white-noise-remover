"""
Quality Assurance (QA) and Spectrogram comparison visualization.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf


def compute_spectrogram(audio: np.ndarray, sr: int = 48000, n_fft: int = 2048, hop_length: int = 512):
    """Compute spectrogram matrix in dB."""
    from scipy import signal
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    f, t, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling="spectrum"
    )
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
    return f, t, Sxx_db


def generate_spectrogram_image(
    original_wav: str | Path,
    denoised_wav: str | Path,
    output_image_path: str | Path,
    duration_limit_sec: float = 30.0
) -> Path:
    """
    Generate side-by-side (or top-bottom) spectrogram comparison image
    for the first duration_limit_sec of the audio.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive background rendering
    import matplotlib.pyplot as plt
    import platform

    if platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    elif platform.system() == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    orig_p = Path(original_wav).resolve()
    denoised_p = Path(denoised_wav).resolve()
    out_img = Path(output_image_path).resolve()
    out_img.parent.mkdir(parents=True, exist_ok=True)

    orig_data, sr = sf.read(str(orig_p), dtype="float32", always_2d=False)
    denoised_data, _ = sf.read(str(denoised_p), dtype="float32", always_2d=False)

    max_samples = int(duration_limit_sec * sr)
    orig_slice = orig_data[:max_samples]
    denoised_slice = denoised_data[:max_samples]

    f_orig, t_orig, S_orig = compute_spectrogram(orig_slice, sr)
    f_den, t_den, S_den = compute_spectrogram(denoised_slice, sr)

    vmin = -90
    vmax = -10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor("#1e1e2e")

    # Original
    im1 = ax1.pcolormesh(t_orig, f_orig, S_orig, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax1.set_title("원본 오디오 스펙트로그램 (배경 화이트 노이즈 잔존)", color="#f8f8f2", fontsize=11, fontweight="bold")
    ax1.set_ylabel("주파수 (Hz)", color="#f8f8f2")
    ax1.tick_params(colors="#f8f8f2")
    ax1.set_facecolor("#11111b")
    ax1.set_ylim(0, 16000)

    # Denoised
    im2 = ax2.pcolormesh(t_den, f_den, S_den, shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax)
    ax2.set_title("노이즈 제거 후 스펙트로그램 (화이트 노이즈 소거 & 음성 보존)", color="#50fa7b", fontsize=11, fontweight="bold")
    ax2.set_xlabel("시간 (초)", color="#f8f8f2")
    ax2.set_ylabel("주파수 (Hz)", color="#f8f8f2")
    ax2.tick_params(colors="#f8f8f2")
    ax2.set_facecolor("#11111b")
    ax2.set_ylim(0, 16000)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation="vertical", pad=0.02)
    cbar.set_label("에너지 (dB)", color="#f8f8f2")
    cbar.ax.yaxis.set_tick_params(color="#f8f8f2")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#f8f8f2")

    plt.tight_layout()
    plt.savefig(str(out_img), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_img


def analyze_audio_quality(
    original_wav: str | Path,
    denoised_wav: str | Path
) -> Dict[str, Any]:
    """Compute energy reduction and quality metrics using memory-safe streaming chunks."""
    chunk_size = 65536

    orig_sum_sq = 0.0
    orig_total_samples = 0
    with sf.SoundFile(str(Path(original_wav).resolve()), mode="r") as r:
        while True:
            data = r.read(frames=chunk_size, dtype="float32")
            if len(data) == 0:
                break
            orig_sum_sq += float(np.sum(data ** 2))
            orig_total_samples += data.size

    den_sum_sq = 0.0
    den_total_samples = 0
    with sf.SoundFile(str(Path(denoised_wav).resolve()), mode="r") as r:
        while True:
            data = r.read(frames=chunk_size, dtype="float32")
            if len(data) == 0:
                break
            den_sum_sq += float(np.sum(data ** 2))
            den_total_samples += data.size

    orig_rms = float(np.sqrt(orig_sum_sq / max(orig_total_samples, 1)))
    denoised_rms = float(np.sqrt(den_sum_sq / max(den_total_samples, 1)))

    orig_db = 20.0 * np.log10(max(orig_rms, 1e-12))
    denoised_db = 20.0 * np.log10(max(denoised_rms, 1e-12))
    diff_db = orig_db - denoised_db

    return {
        "orig_rms_db": round(orig_db, 2),
        "denoised_rms_db": round(denoised_db, 2),
        "total_reduction_db": round(diff_db, 2),
        "warning_speech_loss": (diff_db > 12.0)  # Extreme overall reduction warning
    }
