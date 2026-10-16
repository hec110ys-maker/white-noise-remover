"""
Global configuration settings for White Noise Remover.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    # Audio extraction and processing specs
    sample_rate: int = 48000
    audio_dtype: str = "float32"  # 32-bit float for maximum precision and headroom
    chunk_duration_sec: float = 30.0  # 30-second chunk
    overlap_duration_sec: float = 1.0  # 1-second overlap-add crossfade
    
    # Video & audio re-muxing settings
    output_audio_codec: str = "aac"
    output_audio_bitrate: str = "192k"
    video_codec_copy: bool = True  # Strict '-c:v copy' (no re-encoding)
    
    # Noise reduction tuning defaults
    default_engine: str = "deepfilter"
    noise_reduction_strength: float = 1.0  # 0.0 to 1.0 (0% to 100%)
    voice_protection_level: float = 0.3  # 0.0 to 1.0 (intelligent speech formant preservation)
    enable_studio_cleanup: bool = True  # Studio-grade rumble cut & dynamic downward expander
    
    # VAD settings
    vad_threshold: float = 0.5
    vad_min_silence_duration_sec: float = 0.5
    
    # Work & Temp directory
    temp_dir_name: str = ".temp_denoise"


# Singleton configuration instance
CONFIG = AppConfig()
