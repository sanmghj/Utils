import mne
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # Windows: 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')

plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

def load_sleep_edfx(psg_file, hypno_file):
    """Sleep-EDFx 데이터 로드 및 에폭 라벨 추출"""
    # PSG 데이터 로드
    raw = mne.io.read_raw_edf(psg_file, preload=True)
    print(f"PSG 파일 로드 완료: {len(raw.ch_names)}개 채널, {raw.n_times}개 샘플")
    print(f"채널 이름: {raw.ch_names}")
    print(f"샘플링 주파수: {raw.info['sfreq']} Hz")
    print(f"총 기록 시간: {raw.times[-1]:.1f}초 ({raw.times[-1]/60:.1f}분)")

    raw.filter(0.5, 45)  # 필터링 (선택적)

    # Hypnogram 로드
    annotations = mne.read_annotations(hypno_file)
    print(f"\n어노테이션 로드 완료: {len(annotations)}개 어노테이션")
    print(f"어노테이션 설명 종류:")
    for desc in sorted(set(annotations.description)):
        count = sum(1 for d in annotations.description if d == desc)
        print(f"  - {desc}: {count}개")

    # Hypnogram의 어노테이션을 PSG raw 데이터에 추가
    raw.set_annotations(annotations)

    # Sleep stage 이벤트만 추출 (대소문자 구분)
    # "Sleep stage"로 시작하는 어노테이션만 가져오기
    events, event_id = mne.events_from_annotations(raw, event_id='auto',
                                                     regexp='^Sleep stage')

    print(f"\n추출된 이벤트 ID 맵핑:")
    for label, code in sorted(event_id.items(), key=lambda x: x[1]):
        print(f"  {label} -> {code}")
    print(f"총 이벤트 수: {len(events)}")

    if len(events) == 0:
        raise ValueError("어노테이션에서 Sleep stage 이벤트를 찾을 수 없습니다. Hypnogram 파일을 확인하세요.")

    # 이벤트를 30초 에폭으로 변환
    # tmax는 각 어노테이션의 duration을 사용 (대부분 30초)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=29.99,
                        baseline=None, preload=True, reject_by_annotation=False)

    return epochs, event_id, annotations

def plot_hypnogram(epochs, event_id):
    """Hypnogram 플롯"""
    if len(epochs) == 0:
        print("에폭이 비어있어 플롯을 생성할 수 없습니다.")
        return

    # event_id를 역으로 매핑 (코드 -> 라벨)
    reverse_event_id = {v: k for k, v in event_id.items()}

    # Sleep stage 라벨을 숫자로 매핑 (시각화용)
    stage_mapping = {
        'Sleep stage W': 5,
        'Sleep stage R': 4,
        'Sleep stage 1': 3,
        'Sleep stage 2': 2,
        'Sleep stage 3': 1,
        'Sleep stage 4': 0,
        'Sleep stage ?': -1
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

    # 1. Hypnogram (시간에 따른 수면 단계)
    stages = [stage_mapping.get(reverse_event_id.get(code, ''), -2)
              for code in epochs.events[:, -1]]
    times = epochs.events[:, 0] / epochs.info['sfreq'] / 60  # 분 단위로 변환

    ax1.plot(times, stages, drawstyle='steps-post', linewidth=2)
    ax1.set_xlabel('시간 (분)')
    ax1.set_ylabel('수면 단계')
    ax1.set_yticks([0, 1, 2, 3, 4, 5, -1])
    ax1.set_yticklabels(['Stage 4', 'Stage 3', 'Stage 2', 'Stage 1', 'REM', 'Wake', '?'])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Hypnogram')

    # 2. 라벨 분포 바 차트
    label_counts = Counter(epochs.events[:, -1])
    if len(label_counts) == 0:
        print("라벨 분포가 비어있습니다.")
        return

    # 라벨 이름과 카운트 정렬
    sorted_items = sorted([(reverse_event_id.get(code, f'Unknown_{code}'), count)
                           for code, count in label_counts.items()],
                          key=lambda x: stage_mapping.get(x[0], -2), reverse=True)
    labels, counts = zip(*sorted_items)

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#b2bec3']
    ax2.barh(labels, counts, color=colors[:len(labels)])
    ax2.set_xlabel('에폭 수 (30초 단위)')
    ax2.set_title('Sleep Stage Distribution')
    ax2.grid(True, axis='x', alpha=0.3)

    # 각 바에 숫자 표시
    for i, (label, count) in enumerate(zip(labels, counts)):
        percentage = count / len(epochs) * 100
        ax2.text(count + max(counts)*0.01, i, f'{count} ({percentage:.1f}%)',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# 사용 예시 (파일 경로 수정하세요)
psg_file = './edf/SC4001E0-PSG.edf'  # 실제 파일 경로
hypno_file = './edf/SC4001EC-Hypnogram.edf'

try:
    epochs, event_id, annotations = load_sleep_edfx(psg_file, hypno_file)

    # 1. 라벨 정보 출력
    print("\n=== 수면 라벨 정보 ===")
    print(f"총 에폭 수: {len(epochs)} (30초 단위)")
    print(f"라벨 종류: {list(event_id.keys())}")

    # 라벨 분포
    if len(epochs) > 0:
        label_counts = Counter(epochs.events[:, -1])

        # event_id를 역으로 매핑 (코드 -> 라벨)
        reverse_event_id = {v: k for k, v in event_id.items()}

        print("\n=== 수면 단계별 분포 ===")
        for code, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            label = reverse_event_id.get(code, f'Unknown_{code}')
            duration_min = count * 0.5  # 30초 = 0.5분
            print(f"{label:20s}: {count:3d} 에폭 ({count/len(epochs)*100:5.1f}%) = {duration_min:6.1f}분")

        # 수면 효율성 계산
        total_sleep_stages = sum(count for code, count in label_counts.items()
                                 if 'Sleep stage W' not in reverse_event_id.get(code, '')
                                 and 'Sleep stage ?' not in reverse_event_id.get(code, ''))
        sleep_efficiency = (total_sleep_stages / len(epochs)) * 100
        print(f"\n수면 효율성: {sleep_efficiency:.1f}% (각성 및 미분류 제외)")

        # 2. Hypnogram 플롯
        plot_hypnogram(epochs, event_id)

        # 3. 첫 100 에폭 라벨 시퀀스 출력
        first_100_count = min(100, len(epochs))
        first_100_labels = [reverse_event_id.get(code, f'Unknown_{code}').replace('Sleep stage ', '')
                            for code in epochs.events[:first_100_count, -1]]
        print(f"\n=== 첫 {first_100_count} 에폭 라벨 시퀀스 (30초 단위) ===")
        # 10개씩 끊어서 출력
        for i in range(0, len(first_100_labels), 10):
            epoch_range = f"{i}-{min(i+9, len(first_100_labels)-1)}"
            labels_line = ' '.join(first_100_labels[i:i+10])
            print(f"에폭 {epoch_range:6s}: {labels_line}")
    else:
        print("에폭이 생성되지 않았습니다.")

except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    print("Sleep-EDFx 데이터셋의 PSG.edf와 Hypnogram.edf 파일을 다운로드하세요.")
except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()
