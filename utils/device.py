"""
Hardware device detection utility (NVIDIA CUDA / Apple MPS / CPU).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceInfo:
    device_type: str  # 'cuda', 'mps', or 'cpu'
    device_name: str
    is_gpu: bool
    vram_gb: Optional[float] = None
    display_string: str = "CPU"


def get_optimal_device() -> DeviceInfo:
    """
    Detect the best available compute device for deep learning inference.
    Prioritizes CUDA -> MPS -> CPU fallback.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_vram = None
            try:
                total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            except Exception:
                pass

            vram_str = f" ({total_vram} GB VRAM)" if total_vram else ""
            return DeviceInfo(
                device_type="cuda",
                device_name=device_name,
                is_gpu=True,
                vram_gb=total_vram,
                display_string=f"⚡ GPU (CUDA): {device_name}{vram_str}"
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                device_type="mps",
                device_name="Apple Silicon (MPS)",
                is_gpu=True,
                display_string="⚡ GPU (Apple Silicon MPS)"
            )

    except ImportError:
        pass

    # CPU Fallback
    import os
    cpu_count = os.cpu_count() or 4
    return DeviceInfo(
        device_type="cpu",
        device_name=f"CPU ({cpu_count} threads)",
        is_gpu=False,
        display_string=f"💻 CPU 모드 ({cpu_count} 스레드)"
    )
