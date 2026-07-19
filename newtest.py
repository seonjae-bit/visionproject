import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 시스템 설정 값 ---
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6  # 한 장면(프레임) 재생 시간
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
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 자글거리는 하드웨어 노이즈 완화 필터
    small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 2. 0~255 값을 0.0~1.0의 절대 진폭 값으로 직접 매핑 및 제곱 처리
    amp_matrix = (small.astype(np.float32) / 255.0) ** 2
    
    # 3. 가로축(시간축) 방향으로 진폭을 오디오 샘플 수만큼 부드럽게 선형 보간(Interpolation)
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    # 4. 오디오 신호 합성 (각 주파수는 이제 온전한 자기 밝기 값을 가집니다)
    audio_matrix = base_waves * smooth_amps
    audio = np.sum(audio_matrix, axis=0)
    
    # 5. [수정] 오디오 찢어짐(Clipping) 방지를 위한 소프트 리미터 (하이퍼볼릭 탄젠트)
    # 신호가 작을 때는(예: 0.25) 탄젠트 함수 특성상 변형 없이 거의 그대로 0.25로 나옵니다.
    # 하지만 화면 전체가 흰색이라 소리가 다 합쳐져서 1.0을 훌쩍 넘어가더라도, 
    # 오디오 파형이 찢어지지 않고 최대 0.95 선에서 부드럽게 감쇄 곡선을 그리며 압축됩니다.
    audio = np.tanh(audio) * 0.95
    
    # 6. 화면 출력 및 사운드 재생
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