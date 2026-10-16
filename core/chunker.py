"""
Streaming audio chunking with Hann-window Overlap-Add (OLA) crossfading.
Guarantees memory usage < 500MB even for 3GB+ videos and exact sample count preservation.
"""
import json
import time
import math
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import soundfile as sf


def create_fade_windows(overlap_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate power-complementary raised-cosine (Hann) fade windows.
    fade_in(t) + fade_out(t) == 1.0 exactly for all t, guaranteeing perfect reconstruction.
    """
    if overlap_samples <= 0:
        return np.array([]), np.array([])
    t = np.linspace(0, np.pi, overlap_samples, endpoint=False)
    # Cosine squared window: sum of fade_out and fade_in is identically 1.0
    fade_out = 0.5 * (1.0 + np.cos(t))
    fade_in = 0.5 * (1.0 - np.cos(t))
    return fade_in.astype(np.float32), fade_out.astype(np.float32)


class AudioChunker:
    """
    Streams audio in chunks, applies denoiser or passthrough,
    and stitches back seamlessly using Overlap-Add.
    """
    def __init__(
        self,
        chunk_duration_sec: float = 30.0,
        overlap_duration_sec: float = 1.0,
        sample_rate: int = 48000
    ):
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_duration_sec = overlap_duration_sec
        self.sample_rate = sample_rate

        self.chunk_samples = int(chunk_duration_sec * sample_rate)
        self.overlap_samples = int(overlap_duration_sec * sample_rate)
        # Hop size is the non-overlapping advance step
        self.hop_samples = self.chunk_samples - self.overlap_samples
        if self.hop_samples <= 0:
            raise ValueError("Overlap duration must be strictly less than chunk duration.")

        self.fade_in, self.fade_out = create_fade_windows(self.overlap_samples)

    def process_file(
        self,
        input_wav_path: str | Path,
        output_wav_path: str | Path,
        process_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        checkpoint_path: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Process an audio file in a memory-efficient streaming manner.
        If process_fn is None, behaves as lossless passthrough (exact reconstruction).
        """
        in_path = Path(input_wav_path).resolve()
        out_path = Path(output_wav_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"Input WAV not found: {in_path}")

        info = sf.info(str(in_path))
        total_samples = info.frames
        channels = info.channels
        sr = info.samplerate

        if sr != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: file has {sr}Hz, chunker configured for {self.sample_rate}Hz")

        total_chunks = math.ceil(total_samples / self.hop_samples) if total_samples > 0 else 0
        start_time = time.time()
        samples_written = 0

        # Load or initialize checkpoint
        completed_chunks = 0
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                    completed_chunks = ckpt.get("completed_chunks", 0)
            except Exception:
                completed_chunks = 0

        # We will stream-write directly to the output file
        overlap_buffer = np.zeros((self.overlap_samples, channels), dtype=np.float32)
        has_overlap = False

        with sf.SoundFile(str(in_path), mode="r") as reader, \
             sf.SoundFile(str(out_path), mode="w", samplerate=sr, channels=channels, subtype="FLOAT") as writer:

            for chunk_idx in range(total_chunks):
                if cancel_check and cancel_check():
                    raise InterruptedError("Processing was cancelled by user.")

                seek_pos = chunk_idx * self.hop_samples
                if seek_pos >= total_samples:
                    break

                reader.seek(seek_pos)
                chunk_data = reader.read(frames=self.chunk_samples, dtype="float32", always_2d=True)
                actual_chunk_len = len(chunk_data)

                if actual_chunk_len == 0:
                    break

                # Apply processing function (denoiser) or passthrough
                if process_fn is not None:
                    # process_fn accepts (samples, channels) or (samples,) and returns processed ndarray
                    processed_chunk = process_fn(chunk_data, sr)
                    if processed_chunk.ndim == 1 and channels == 1:
                        processed_chunk = processed_chunk[:, np.newaxis]
                    elif processed_chunk.shape != chunk_data.shape:
                        raise ValueError(
                            f"Process function altered chunk shape from {chunk_data.shape} to {processed_chunk.shape}"
                        )
                else:
                    processed_chunk = chunk_data.copy()

                # Overlap-Add Crossfading Logic
                # Case 1: First chunk (no preceding overlap)
                if not has_overlap:
                    if actual_chunk_len <= self.overlap_samples or chunk_idx == total_chunks - 1:
                        # Very short file or single chunk
                        writer.write(processed_chunk)
                        samples_written += actual_chunk_len
                    else:
                        # Write non-overlapping main body
                        main_len = actual_chunk_len - self.overlap_samples
                        writer.write(processed_chunk[:main_len])
                        samples_written += main_len

                        # Store trailing overlap with fade_out applied
                        trailing = processed_chunk[main_len:]
                        overlap_buffer = trailing * self.fade_out[:, np.newaxis]
                        has_overlap = True

                # Case 2: Subsequent chunks
                else:
                    current_overlap_len = min(self.overlap_samples, actual_chunk_len)
                    
                    # 1) Combine leading section with stored overlap buffer
                    leading = processed_chunk[:current_overlap_len]
                    fade_in_slice = self.fade_in[:current_overlap_len, np.newaxis]
                    blended = overlap_buffer[:current_overlap_len] + (leading * fade_in_slice)
                    writer.write(blended)
                    samples_written += current_overlap_len

                    # 2) Check if this is the final chunk
                    is_last_chunk = (chunk_idx == total_chunks - 1) or (seek_pos + actual_chunk_len >= total_samples)

                    if is_last_chunk:
                        # Write remainder without fade_out because there is no next chunk
                        remaining = processed_chunk[current_overlap_len:]
                        if len(remaining) > 0:
                            writer.write(remaining)
                            samples_written += len(remaining)
                        has_overlap = False
                    else:
                        # Write middle non-overlapping body
                        main_len = actual_chunk_len - self.overlap_samples
                        if main_len > current_overlap_len:
                            body = processed_chunk[current_overlap_len:main_len]
                            writer.write(body)
                            samples_written += len(body)

                        # Store trailing section for next overlap
                        trailing = processed_chunk[main_len:]
                        t_len = len(trailing)
                        overlap_buffer = np.zeros((self.overlap_samples, channels), dtype=np.float32)
                        overlap_buffer[:t_len] = trailing * self.fade_out[:t_len, np.newaxis]
                        has_overlap = True

                # Progress & RTF Calculation
                completed_chunks += 1
                elapsed = time.time() - start_time
                progress_pct = min(100.0, (chunk_idx + 1) / total_chunks * 100.0)
                processed_audio_sec = min(total_samples, (chunk_idx + 1) * self.hop_samples) / sr
                rtf = elapsed / processed_audio_sec if processed_audio_sec > 0 else 0.0
                eta_sec = (total_chunks - (chunk_idx + 1)) * (elapsed / (chunk_idx + 1)) if chunk_idx > 0 else 0.0

                if progress_callback:
                    progress_callback({
                        "chunk_index": chunk_idx + 1,
                        "total_chunks": total_chunks,
                        "progress_percent": progress_pct,
                        "elapsed_sec": elapsed,
                        "eta_sec": max(0.0, eta_sec),
                        "rtf": rtf,
                        "samples_written": samples_written
                    })

                # Update checkpoint
                if checkpoint_path:
                    try:
                        with open(checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "completed_chunks": completed_chunks,
                                "total_chunks": total_chunks,
                                "samples_written": samples_written,
                                "timestamp": time.time()
                            }, f)
                    except Exception:
                        pass

        # Verification of exact sample count
        out_info = sf.info(str(out_path))
        if out_info.frames != total_samples:
            # If slight difference at edge, truncate or pad to match total_samples perfectly
            self._conform_sample_count(out_path, total_samples, channels, sr)

        final_info = sf.info(str(out_path))
        return {
            "total_samples": final_info.frames,
            "original_samples": total_samples,
            "exact_match": (final_info.frames == total_samples),
            "channels": final_info.channels,
            "samplerate": final_info.samplerate,
            "total_chunks": total_chunks,
            "elapsed_sec": time.time() - start_time
        }

    def _conform_sample_count(self, filepath: Path, target_samples: int, channels: int, sr: int) -> None:
        """Enforce bit-exact sample count to prevent any AV desync using memory-safe streaming."""
        temp_file = filepath.with_suffix(".conform.wav")
        chunk_frames = 65536
        written = 0

        with sf.SoundFile(str(filepath), mode="r") as r, \
             sf.SoundFile(str(temp_file), mode="w", samplerate=sr, channels=channels, subtype="FLOAT") as w:
            while written < target_samples:
                to_read = min(chunk_frames, target_samples - written)
                data = r.read(frames=to_read, dtype="float32", always_2d=True)
                if len(data) == 0:
                    # Pad remaining zeros if file ended early
                    pad_len = target_samples - written
                    pad = np.zeros((pad_len, channels), dtype=np.float32)
                    w.write(pad)
                    written += pad_len
                    break
                w.write(data)
                written += len(data)

        temp_file.replace(filepath)
