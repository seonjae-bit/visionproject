import cv2
import numpy as np
import sounddevice as sd
import time

# --- 환경 설정 ---
SAMPLE_RATE = 44100  # 오디오 샘플 레이트 (Hz)
RESOLUTION = 8       # 8x8 해상도
SCAN_DURATION = 0.5  # 한 장면 전체 재생 시간 (초)
COL_DURATION = SCAN_DURATION / RESOLUTION  # 1개 열 재생 시간 (0.0625초)
SILENCE_DURATION = SCAN_DURATION * 0.1     # 장면 종료 후 무음 시간 (0.05초)

# 8개 행에 부여할 주파수 배열 (위에서 아래 순서)
# 픽셀이 위에 있으면 높은 주파수(2093Hz), 아래에 있으면 낮은 주파수(131Hz)
FREQUENCIES = np.linspace(2093, 131, RESOLUTION)

def generate_tone(frequency, duration, volume):
    """특정 주파수와 볼륨을 가진 사인파 오디오 배열을 생성합니다."""
    if volume == 0:
        return np.zeros(int(SAMPLE_RATE * duration))
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # 볼륨은 픽셀 밝기(0~255) 기반이므로, 소리가 찢어지지 않게 최대치 제한(0.1 정도)을 둡니다.
    wave = (volume / 255.0) * 0.1 * np.sin(2 * np.pi * frequency * t)
    return wave

def process_frame(frame):
    """이미지를 8x8 흑백으로 변환하고, 열 단위로 사운드를 합성하여 재생합니다."""
    # 1. 8x8 해상도로 축소
    resized = cv2.resize(frame, (RESOLUTION, RESOLUTION))
    
    # 2. 흑백(밝기 정보만) 변환
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 전체 장면 소리를 담을 빈 리스트
    full_sound = []
    
    # 4. 왼쪽 열(0열)부터 오른쪽 열(7열)까지 순서대로 처리
    for col in range(RESOLUTION):
        col_sound = np.zeros(int(SAMPLE_RATE * COL_DURATION))
        
        for row in range(RESOLUTION):
            brightness = gray[row, col] # 각 픽셀의 밝기 (0 ~ 255)
            freq = FREQUENCIES[row]
            
            # 주파수별 소리를 생성해 해당 열의 사운드에 중첩(다성음)
            col_sound += generate_tone(freq, COL_DURATION, brightness)
            
        full_sound.append(col_sound)
        
    # 모든 열의 소리를 하나로 이어 붙임
    full_sound = np.concatenate(full_sound)
    
    # 5. 총 재생 시간의 10%만큼 무음(Silence) 추가
    silence = np.zeros(int(SAMPLE_RATE * SILENCE_DURATION))
    final_output = np.concatenate([full_sound, silence])
    
    # 오디오 재생 (재생되는 동안 대기)
    sd.play(final_output, SAMPLE_RATE)
    sd.wait()

def main():
    # USB 카메라 연결 (기본 카메라 0번)
    cap = cv2.VideoCapture(0)
    
    # 라즈베리파이 내부 카메라 버퍼 크기를 최소한(1개)으로 설정하여 자체 지연 방지
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다. 연결을 확인하세요.")
        return

    print("실시간 지연 방지 모드 가동 중... 종료하려면 'q'를 누르세요.")
    
    try:
        while True:
            # 1. 소리가 재생되는 동안 버퍼에 쌓인 "과거 프레임"들을 전부 폐기 처리합니다.
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    break
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == 0: 
                    break

            # 2. 버퍼가 비워졌으므로 "지금 이 순간"의 가장 최신 프레임을 가져옵니다.
            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 가져올 수 없습니다.")
                    break
                
            # 디버깅용 화면 출력 (라즈베리파이 GUI 환경에서 확인 가능)
            debug_frame = cv2.resize(frame, (400, 400))
            cv2.imshow("Realtime Sound Camera", debug_frame)
            
            # 3. 0.55초 동안 소리를 변환하고 재생합니다.
            process_frame(frame)
            
            # 키 입력 대기 ('q' 누르면 종료)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
