import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6  # 한 장면(프레임) 재생 시간을 0.6초로 설정
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

# 0.6초에 맞춘 총 오디오 샘플 수 및 타임라인 계산
total_samples = int(FS * FRAME_TIME)
t = np.arange(total_samples) / FS

# 주파수 배열 생성 (위쪽 픽셀=고음, 아래쪽 픽셀=저음)
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)

# 페이드(Fade) 없이 부드럽게 이어질 기저 사인파 매트릭스 생성
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# --- 라즈베리파이 카메라 버퍼 지연 방지 스레드 ---
class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
        self.ok, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok: 
                self.ok = ok
                self.frame = frame
            else: 
                time.sleep(0.01)

    def read(self): 
        return self.ok, self.frame
        
    def release(self): 
        self.running = False
        self.cap.release()

# 카메라 스레드 기동
cam = CameraStream()
print("Press q to quit.")

while True:
    ok, frame = cam.read()
    if not ok: break
    
    # 1. 전처리 및 픽셀 리사이즈
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 암부 자글거림 1차 방지 필터
    small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 2. 암부 노이즈 문턱값(Threshold) 처리
    THRESHOLD_LEVEL = 15
    small[small < THRESHOLD_LEVEL] = 0
    
    # 3. 0~255 값을 0.0~1.0의 절대 진폭 값으로 직접 매핑 (표준화 제거)
    amp_matrix = (small.astype(np.float32) / 255.0) ** 2
    
    # 4. 가로축(시간축) 방향으로 진폭을 오디오 샘플 수만큼 부드럽게 선형 보간(Interpolation)
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    # 5. 오디오 신호 합성
    audio_matrix = base_waves * smooth_amps
    audio = np.sum(audio_matrix, axis=0)
    
    # 6. 전체 화면이 암흑일 때의 완벽한 음소거 처리
    avg_brightness = np.mean(small)
    if avg_brightness < 1.0:
        audio = np.zeros_like(audio)
    else:
        # 모든 채널이 흰색일 때 볼륨이 찢어지는 현상을 방지하기 위한 안전 장치(SAFETY_MARGIN) 적용
        SAFETY_MARGIN = 2.5
        audio = (audio / HEIGHT) * SAFETY_MARGIN
    
    # 최종 물리적 클리핑 안전선 구축
    audio = np.clip(audio, -0.9, 0.9)
    
    # 7. 화면 출력 및 사운드 재생
    cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
    
    sd.default.device = (None, 1)
    sd.play(audio, FS)
    
    # 키보드 q 입력 시 안전하게 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sd.stop()
        break
    
    sd.wait()  # 오디오가 0.6초 동안 완벽히 출력될 때까지 대기

cam.release()
cv2.destroyAllWindows()