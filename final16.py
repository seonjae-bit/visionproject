import time
import numpy as np
import sounddevice as sd
import cv2

# ==========================================
# 1. 시스템 기본 파라미터 설정
# ==========================================
SAMPLE_RATE = 44100
WIDTH = 16
HEIGHT = 16
F0 = 60.0  # Base frequency (Hz) -> 1주기 = 1/60초 (약 16.67ms)

# 각 행(Row)별 주파수 배수 설정 (고음 위쪽 -> 저음 아래쪽)
HARMONIC_MULTIPLIERS = np.linspace(35, 4, HEIGHT, dtype=int)
FREQS = HARMONIC_MULTIPLIERS * F0

# 열당 3주기 배정 (0.05초 = 50ms)
CYCLES_PER_COL = 3
DUR_PER_COL = CYCLES_PER_COL / F0  # 3/60s = 0.05초
SAMPLES_PER_COL = int(SAMPLE_RATE * DUR_PER_COL)  # 2205 샘플

# [Desmos 핵심] 보간 구간: 정확히 1주기(1/f0 = 약 16.67ms)만 보간
TRANSITION_SAMPLES = int(SAMPLE_RATE / F0)  # 735 샘플

# 장면 끝난 후 휴식 시간 (0.2초)
PAUSE_DURATION = 0.2
PAUSE_SAMPLES = int(SAMPLE_RATE * PAUSE_DURATION)
silence_buffer = np.zeros(PAUSE_SAMPLES, dtype=np.float32)

# ==========================================
# 2. Desmos 기반 오디오 생성 함수
# ==========================================
def generate_audio_frame(grid_data):
    """
    grid_data: 현재 프레임 (HEIGHT, WIDTH) - 0.0 ~ 1.0 밝기 값
    """
    total_samples = SAMPLES_PER_COL * WIDTH
    combined_signal = np.zeros(total_samples, dtype=np.float32)

    # 열 단위 시간 축 배열
    t_col = np.linspace(0, DUR_PER_COL, SAMPLES_PER_COL, endpoint=False)

    for c in range(WIDTH):
        col_signal = np.zeros(SAMPLES_PER_COL, dtype=np.float32)
        
        curr_col = grid_data[:, c]
        next_col = grid_data[:, c + 1] if c < WIDTH - 1 else grid_data[:, 0]

        start_idx = c * SAMPLES_PER_COL
        
        for r in range(HEIGHT):
            freq = FREQS[r]
            L_curr = curr_col[r]
            L_next = next_col[r]

            # 1) 기본 고정 진폭 구간 (A_m) : 열 전체를 기본 밝기로 설정
            amp_envelope = np.full(SAMPLES_PER_COL, L_curr, dtype=np.float32)

            # 2) 전환 보간 구간 (B_m) : 열의 맨 뒤 '정확히 1주기' 동안 L[m] -> L[m+1] 집중 보간
            if TRANSITION_SAMPLES > 0:
                trans_t = np.linspace(0, 1, TRANSITION_SAMPLES, endpoint=False)
                interpolated_amp = L_curr + (L_next - L_curr) * trans_t
                amp_envelope[-TRANSITION_SAMPLES:] = interpolated_amp

            # 3) C_m = B_m(x) * sin(2*pi*F*x) 위상 연속성 보장 연산
            wave = amp_envelope * np.sin(2 * np.pi * freq * t_col)
            col_signal += wave

        combined_signal[start_idx : start_idx + SAMPLES_PER_COL] = col_signal

    # 클리핑 방지 정규화
    max_val = np.max(np.abs(combined_signal))
    if max_val > 0:
        combined_signal = (combined_signal / max_val) * 0.8

    return combined_signal

# ==========================================
# 3. 메인 파이프라인
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("1주기 보간 + 0.2초 휴식 알고리즘 적용 완료. 실행 중...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 전처리 (흑백, 16x16 축소, 상하반전)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            flipped = cv2.flip(resized, 0)  # 고음이 위쪽으로 가도록 반전

            # 노이즈 컷오프 및 정규화
            flipped[flipped < 25] = 0
            grid = flipped.astype(np.float32) / 255.0

            # 1) 16열 스캔 오디오 재생 (0.8초)
            audio_data = generate_audio_frame(grid)
            sd.play(audio_data, SAMPLE_RATE)
            sd.wait()

            # 2) 🔇 전체 스캔 완결 후 0.2초 휴식
            sd.play(silence_buffer, SAMPLE_RATE)
            sd.wait()

    except KeyboardInterrupt:
        print("\n프로그램 종료")
    finally:
        cap.release()

if __name__ == "__main__":
    main()