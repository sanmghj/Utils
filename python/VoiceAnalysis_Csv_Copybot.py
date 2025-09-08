import pandas as pd
import sys
import os
import shutil
from pathlib import Path

# 전역 변수
OUTPUT_FOLDER = "parsed_wav_repo"
REQUIRED_SCORE_COLUMNS = ['문자점수', '단어점수']
FILE_COLUMN_NAME = '파일명'
SCORE_THRESHOLD = 70

def extract_low_score_files(excel_path):
    """
    엑셀 파일에서 문자점수 또는 단어점수가 70 미만인 행의 파일명을 추출

    Args:
        excel_path (str): 엑셀 파일 경로

    Returns:
        list: 조건에 해당하는 파일명 리스트
    """
    try:
        # 파일 확장자에 따라 적절한 읽기 함수 사용
        file_extension = os.path.splitext(excel_path)[1].lower()

        if file_extension == '.csv':
            # CSV 파일 읽기 (인코딩 자동 감지 시도)
            try:
                df = pd.read_csv(excel_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(excel_path, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(excel_path, encoding='euc-kr')
        elif file_extension in ['.xlsx', '.xls']:
            # 엑셀 파일 읽기
            df = pd.read_excel(excel_path)
        else:
            print(f"지원하지 않는 파일 형식입니다: {file_extension}")
            print("지원 형식: .csv, .xlsx, .xls")
            return []

        # 필요한 컬럼이 있는지 확인
        missing_score_columns = [col for col in REQUIRED_SCORE_COLUMNS if col not in df.columns]

        if missing_score_columns:
            print(f"오류: 다음 컬럼을 찾을 수 없습니다: {missing_score_columns}")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return []

        # 파일명 컬럼 찾기
        if FILE_COLUMN_NAME not in df.columns:
            print(f"오류: 파일명 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return []

        # 전체 데이터프레임을 순회하며 조건 확인
        file_list = []
        total_rows = len(df)

        print(f"전체 {total_rows}개 행을 확인 중...")

        for index, row in df.iterrows():
            try:
                # 문자점수와 단어점수 값 확인
                char_score = row['문자점수']
                word_score = row['단어점수']
                filename = row[FILE_COLUMN_NAME]

                # NaN 값 처리
                if pd.isna(char_score) or pd.isna(word_score) or pd.isna(filename):
                    continue

                # int형으로 변환 시도
                char_score = int(char_score)
                word_score = int(word_score)

                # 조건 확인: 문자점수 또는 단어점수가 70 미만
                if char_score < SCORE_THRESHOLD or word_score < SCORE_THRESHOLD:
                    file_list.append(filename)
                    print(f"조건 충족: {filename} (문자점수: {char_score}, 단어점수: {word_score})")

            except (ValueError, TypeError) as e:
                print(f"행 {index + 1} 처리 중 오류: {e}")
                continue

        return file_list

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {excel_path}")
        return []
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return []

def copy_wav_files(filename_list):
    """
    파일명 리스트의 WAV 파일들을 OUTPUT_FOLDER로 복사

    Args:
        filename_list (list): WAV 파일 경로들이 담긴 리스트

    Returns:
        int: 복사된 파일 수
    """
    if not filename_list:
        print("복사할 파일이 없습니다.")
        return 0

    # 출력 폴더 생성 (없을 때만)
    output_path = Path(OUTPUT_FOLDER)
    if not output_path.exists():
        output_path.mkdir(exist_ok=True)
        print(f"출력 폴더 생성: {output_path.absolute()}")
    else:
        print(f"기존 출력 폴더 사용: {output_path.absolute()}")

    copied_count = 0

    for wav_path in filename_list:
        try:
            source_file = Path(wav_path)

            # 파일이 존재하는지 확인
            if not source_file.exists():
                print(f"파일이 존재하지 않음: {wav_path}")
                continue

            # 대상 파일 경로
            dest_file = output_path / source_file.name

            # 파일 복사
            shutil.copy2(source_file, dest_file)
            print(f"복사 완료: {source_file.name}")
            copied_count += 1

        except Exception as e:
            print(f"복사 실패 {wav_path}: {e}")

    print(f"\n총 {copied_count}개 파일이 '{OUTPUT_FOLDER}' 폴더로 복사되었습니다.")
    return copied_count

def main():
    # 명령줄 인수 확인
    if len(sys.argv) != 2:
        print("사용법: python VoiceAnalysis_Csv_Copybot.py <파일경로>")
        print("예시: python VoiceAnalysis_Csv_Copybot.py data.csv")
        print("예시: python VoiceAnalysis_Csv_Copybot.py comparison_results_250905/data.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다: {file_path}")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        sys.exit(1)

    # 점수가 70 미만인 파일명 추출
    low_score_files = extract_low_score_files(file_path)

    if low_score_files:
        print(f"문자점수 또는 단어점수가 70 미만인 파일 ({len(low_score_files)}개):")
        for i, filename in enumerate(low_score_files, 1):
            print(f"{i}. {filename}")

        # WAV 파일 복사
        copy_wav_files(low_score_files)
    else:
        print("조건에 해당하는 파일이 없습니다.")

if __name__ == "__main__":
    main()