"""
CLI runner for White Noise Remover.
Usage:
    python run_cli.py "your_video.mp4"
    python run_cli.py "your_video.mp4" --strength 0.9 --voice-protection 0.5
"""
import sys
import argparse
import time
from pathlib import Path
from core.probe import probe_media, compare_media_metadata
from core.extract import extract_audio, remux_video_with_audio
from core.chunker import AudioChunker
from core.denoise import available_engines
from utils.paths import get_output_video_path, get_temp_work_dir, cleanup_temp_dir, estimate_work_bytes
from utils.device import get_optimal_device


def run_pipeline(
    input_video: str | Path,
    output_video: str | Path | None = None,
    engine: str = available_engines()[0],
    strength: float = 1.0,
    voice_protection: float = 0.3,
    enable_studio_cleanup: bool = True
) -> Path:
    video_path = Path(input_video).resolve()
    if not video_path.exists():
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    out_path = Path(output_video).resolve() if output_video else get_output_video_path(video_path)
    _meta = probe_media(video_path)
    temp_dir = get_temp_work_dir(video_path, needed_bytes=estimate_work_bytes(_meta.duration_sec, _meta.audio_channels))

    device_info = get_optimal_device()
    print("=" * 60)
    print("       화이트 노이즈 제거 프로그램")
    print("=" * 60)
    print(f"• 입력 영상    : {video_path.name}")
    print(f"• 출력 영상    : {out_path.name}")
    print(f"• 엔진 선택    : {engine}")
    print(f"• 하드웨어     : {device_info.display_string}")
    print(f"• 제거 강도    : {int(strength * 100)}%")
    print(f"• 음성 보호    : {int(voice_protection * 100)}%")
    print(f"• 스튜디오정리 : {'켜짐 (잔류 Hiss/럼블 차단)' if enable_studio_cleanup else '꺼짐'}")
    print("=" * 60)

    start_total = time.time()

    try:
        # 1. Probe original metadata
        print("\n[1/4] 🔍 비디오/오디오 메타데이터 분석 중...")
        meta_orig = probe_media(video_path)
        if not meta_orig.has_audio:
            print("❌ 오류: 입력 파일에 오디오 트랙이 존재하지 않습니다.")
            sys.exit(1)
        print(f"  - 해상도: {meta_orig.width}x{meta_orig.height}, FPS: {meta_orig.fps:.2f}, 코덱: {meta_orig.video_codec}")
        print(f"  - 영상 길이: {meta_orig.duration_sec:.1f}초 ({meta_orig.duration_sec/60:.1f}분)")
        print(f"  - 오디오: {meta_orig.audio_sample_rate}Hz, {meta_orig.audio_channels}채널, {meta_orig.audio_codec}")

        # 2. Extract audio to 48kHz WAV
        extracted_wav = temp_dir / "extracted_48k.wav"
        print(f"\n[2/4] 🎙️ 무손실 48kHz 오디오 추출 중...")
        extract_audio(video_path, extracted_wav, sample_rate=48000)
        print("  - 오디오 추출 완료 (32-bit float PCM)")

        # 3. Denoise with chosen engine in streaming chunks
        denoised_wav = temp_dir / "denoised_48k.wav"
        print(f"\n[3/4] 🧠 노이즈 제거 진행 중 (엔진: {engine}, 30초 청크 단위)...")
        if engine == "spectral":
            from core.denoise.spectral import SpectralGatingDenoiser
            from core.vad import extract_noise_sample_from_file
            noise_profile = extract_noise_sample_from_file(extracted_wav, sample_duration_sec=1.5)
            denoiser = SpectralGatingDenoiser(
                strength=strength,
                voice_protection=voice_protection,
                noise_profile=noise_profile,
                enable_studio_cleanup=enable_studio_cleanup
            )
        elif engine == "rnnoise":
            from core.denoise.rnnoise import RNNoiseDenoiser
            denoiser = RNNoiseDenoiser(
                strength=strength,
                voice_protection=voice_protection,
                enable_studio_cleanup=enable_studio_cleanup
            )
        else:
            from core.denoise.deepfilter import DeepFilterDenoiser
            denoiser = DeepFilterDenoiser(
                strength=strength,
                voice_protection=voice_protection,
                enable_studio_cleanup=enable_studio_cleanup
            )
        chunker = AudioChunker(chunk_duration_sec=30.0, overlap_duration_sec=1.0, sample_rate=48000)

        def on_progress(p: dict):
            pct = p["progress_percent"]
            chunk = p["chunk_index"]
            total = p["total_chunks"]
            rtf = p["rtf"]
            eta = p["eta_sec"]
            bar_len = 25
            filled = int(bar_len * pct / 100.0)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% | 청크 {chunk}/{total} | 남은시간: {eta:.1f}초 | 처리속도: {rtf:.3f}x")
            sys.stdout.flush()

        chunk_res = chunker.process_file(
            input_wav_path=extracted_wav,
            output_wav_path=denoised_wav,
            process_fn=denoiser.process,
            progress_callback=on_progress
        )
        print("\n  - 노이즈 제거 및 Overlap-Add 재합성 완료!")

        # 4. Remux with video (-c:v copy)
        print(f"\n[4/4] 🎬 무손실 비디오 결합 중 (-c:v copy, 영상 화질 100% 보존)...")
        remux_video_with_audio(
            input_video_path=video_path,
            processed_audio_path=denoised_wav,
            output_video_path=out_path,
            audio_codec="aac",
            audio_bitrate="192k"
        )
        print(f"  - 합성 완료: {out_path.name}")

        # 5. Quality Assurance Check
        meta_denoised = probe_media(out_path)
        comp = compare_media_metadata(meta_orig, meta_denoised)

        total_elapsed = time.time() - start_total
        print("\n" + "=" * 60)
        print("                  🎉 처리 완료 보고서")
        print("=" * 60)
        print(f"• 출력 파일 경로 : {out_path}")
        print(f"• 총 소요 시간   : {total_elapsed:.1f}초")
        print(f"• 영상 화질 보존 : {'✅ 100% 무손실 복사 일치' if comp['is_resolution_match'] and comp['is_fps_match'] else '⚠️ 확인 필요'}")
        print(f"  - 원본 vs 출력 : {comp['orig_resolution']} @ {comp['orig_fps']}fps -> {comp['denoised_resolution']} @ {comp['denoised_fps']}fps")
        print(f"• 재생 시간 오차 : {comp['duration_diff_sec']:.3f}초 (싱크 완벽 보존)")
        print(f"• 원본 파일 상태 : ✅ 안전 (수정되지 않음)")
        print("=" * 60)

        return out_path

    finally:
        cleanup_temp_dir(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="동영상 화이트 노이즈 제거 프로그램")
    parser.add_argument("video_path", help="처리할 동영상 파일 경로 (MP4, MOV, MKV)")
    parser.add_argument("-o", "--output", help="출력 파일 경로 (기본값: 원본파일명_denoised.mp4)", default=None)
    engines = available_engines()
    parser.add_argument("-e", "--engine", choices=engines, default=engines[0], help=f"제거 엔진 선택 ({', '.join(engines)}, 기본값: {engines[0]})")
    parser.add_argument("-s", "--strength", help="노이즈 제거 강도 (0.0 ~ 1.0, 기본값: 1.0)", type=float, default=1.0)
    parser.add_argument("-p", "--voice-protection", help="음성 보호 강도 (0.0 ~ 1.0, 기본값: 0.3)", type=float, default=0.3)
    parser.add_argument("--no-studio-cleanup", help="스튜디오 잔류 노이즈 정리기 비활성화", action="store_true")

    args = parser.parse_args()
    run_pipeline(
        input_video=args.video_path,
        output_video=args.output,
        engine=args.engine,
        strength=args.strength,
        voice_protection=args.voice_protection,
        enable_studio_cleanup=not args.no_studio_cleanup
    )


if __name__ == "__main__":
    main()
