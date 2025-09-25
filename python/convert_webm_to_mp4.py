# Use Python 3.11
# py -3.11 -m venv ./venv-3.11
# ./venv-3.11/Scripts/activate

import ffmpeg
import os
import sys
from pathlib import Path


def convert_webm_to_mp4(input_file, output_file=None, video_codec='libx264', audio_codec='aac', overwrite=True):
    """
    WebM 파일을 MP4로 변환하는 함수

    Args:
        input_file (str): 입력 WebM 파일 경로
        output_file (str, optional): 출력 MP4 파일 경로. None이면 자동 생성
        video_codec (str): 비디오 코덱 (기본값: 'libx264')
        audio_codec (str): 오디오 코덱 (기본값: 'aac')
        overwrite (bool): 기존 파일 덮어쓰기 여부

    Returns:
        bool: 변환 성공 여부
        str: 출력 파일 경로 또는 에러 메시지
    """
    try:
        # 입력 파일 존재 확인
        if not os.path.exists(input_file):
            return False, f"Input file not found: {input_file}"

        # 출력 파일 경로 자동 생성
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix('.mp4'))

        # 출력 디렉토리 생성
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        print(f"Converting: {input_file} -> {output_file}")

        # FFmpeg 변환 실행
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(
            stream,
            output_file,
            vcodec=video_codec,
            acodec=audio_codec,
            **{
                'strict': 'experimental',
                'avoid_negative_ts': 'make_zero'
            }
        )

        if overwrite:
            stream = ffmpeg.overwrite_output(stream)

        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

        # 변환 결과 확인
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"Successfully converted! Output size: {file_size:.2f} MB")
            return True, output_file
        else:
            return False, "Output file was not created"

    except ffmpeg.Error as e:
        error_msg = f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown FFmpeg error'}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        print(error_msg)
        return False, error_msg


def convert_multiple_webm_to_mp4(input_dir, output_dir=None, file_pattern="*.webm"):
    """
    여러 WebM 파일을 일괄 변환하는 함수

    Args:
        input_dir (str): 입력 디렉토리 경로
        output_dir (str, optional): 출력 디렉토리 경로. None이면 입력 디렉토리 사용
        file_pattern (str): 파일 패턴 (기본값: "*.webm")

    Returns:
        list: [(success, input_file, output_file, message), ...]
    """
    from glob import glob

    if output_dir is None:
        output_dir = input_dir

    webm_files = glob(os.path.join(input_dir, file_pattern))
    results = []

    print(f"Found {len(webm_files)} WebM files to convert")

    for i, input_file in enumerate(webm_files, 1):
        print(f"\n[{i}/{len(webm_files)}] Processing: {os.path.basename(input_file)}")

        # 출력 파일명 생성
        input_name = Path(input_file).stem
        output_file = os.path.join(output_dir, f"{input_name}.mp4")

        success, message = convert_webm_to_mp4(input_file, output_file)
        results.append((success, input_file, output_file, message))

    return results


def main():
    """메인 함수"""
    # 단일 파일 변환 예제
    input_file = './webm/unknown_전체평가녹화_2025-09-16T01-39-17.webm'
    output_file = './mp4/unknown_전체평가녹화_2025-09-16T01-39-17.mp4'

    success, result = convert_webm_to_mp4(input_file, output_file)

    if success:
        print(f"\n✅ Conversion completed: {result}")
    else:
        print(f"\n❌ Conversion failed: {result}")
        sys.exit(1)

    # 일괄 변환 예제 (주석 처리)
    # print("\n" + "="*50)
    # print("Batch conversion example:")
    # results = convert_multiple_webm_to_mp4('./webm', './mp4')

    # successful = sum(1 for success, _, _, _ in results if success)
    # total = len(results)
    # print(f"\nBatch conversion completed: {successful}/{total} files converted successfully")


if __name__ == "__main__":
    main()