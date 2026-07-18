import cv2
import numpy as np
import sounddevice as sd
import time

# --- 환경 설정 ---
SAMPLE_RATE = 44100
RESOLUTION = 8
SCAN_DURATION = 0.5
COL_DURATION = SCAN_DURATION / RESOLUTION
SILENCE_DURATION = SCAN_DURATION * 0.1
FREQUENCIES = np.linspace(2093, 131, RESOLUTION)

# ⚠️ 중요: 오디오 장치 번호 강제 지정 
# python3 -m sounddevice 명령어로 확인한 유선 이어폰(AV Jack) 번호를 넣으세요.
# 예: sd.default.device = 1  (기본값으로 안 되면 주석을 풀고 번호를 적으세요)
# sd.default.device = 1 

def generate_tone(frequency, duration, volume):
    if volume == 0:
        return np.zeros(int(SAMPLE_RATE * duration))
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # 소리가 너무 작게 들린다면 0.1을 0.3~0.5 정도로 키워보세요.
    wave = (volume / 255.0) * 0.2 * np.sin(2 * np.pi * frequency * t)
    return wave

def process_frame(frame):
    resized = cv2.resize(frame, (RESOLUTION, RESOLUTION))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    full_sound = []
    for col in range(RESOLUTION):
        col_sound = np.zeros(int(SAMPLE_RATE * COL_DURATION))
        for row in range(RESOLUTION):
            brightness = gray[row, col]
            freq = FREQUENCIES[row]
            col_sound += generate_tone(freq, COL_DURATION, brightness)
        full_sound.append(col_sound)
        
    full_sound = np.concatenate(full_sound)
    silence = np.zeros(int(SAMPLE_RATE * SILENCE_DURATION))
    final_output = np.concatenate([full_sound, silence])
    
    # 사운드 재생
    sd.play(final_output, SAMPLE_RATE)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다.")
        return

    print("프로그램 시작됨... 종료하려면 반드시 '카메라 화면 창'을 클릭하고 'q'를 누르세요.")
    
    try:
        while True:
            # 버퍼 비우기
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    break
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == 0: 
                    break

            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
                if not ret:
                    break
                
            debug_frame = cv2.resize(frame, (400, 400))
            cv2.imshow("Realtime Sound Camera", debug_frame)
            
            # 소리 생성 및 재생 지시 (재생이 시작되면 코드는 멈추지 않고 바로 다음으로 넘어감)
            process_frame(frame)
            
            # 오디오가 완전히 재생될 때까지 0.55초 동안 키 입력을 체크하면서 대기 (먹통 방지)
            start_wait = time.time()
            quit_program = False
            while time.time() - start_wait < (SCAN_DURATION + SILENCE_DURATION):
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    quit_program = True
                    break
            
            if quit_program:
                break
                
    finally:
        sd.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()