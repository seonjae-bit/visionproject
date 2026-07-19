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

# [개선] 최저/최고 주파수 설정
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

# 0.6초에 맞춘 총 오디오 샘플 수 및 타임라인 계산
total_samples = int(FS * FRAME_TIME)
t = np.arange(total_samples) / FS

# -----------------------------------------------------------------
# [수정] 일차함수(linspace)를 지수함수(geomspace)로 변경!
# 피아노 음계처럼 고음으로 갈수록 주파수가 기하급수적으로 증가하는 등비수열을 만듭니다.
# 위쪽 픽셀 = 고음, 아래쪽 픽셀 = 저음 매핑 유지
# -----------------------------------------------------------------
freqs = np.geomspace(MAX_FREQ, MIN_FREQ, HEIGHT)

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
    
    # 2. 0~255 값을 0.0~1.0의 절대 진폭 값으로 변환
    raw_amp = small.astype(np.float32) / 255.0
    
    # -----------------------------------------------------------------
    # [수정] 행별 독립 노이즈 게이트 적용 (드르륵 잔진동 소리 원천 차단)
    # 카메라가 검은 화면에서 혼자 감도를 높여 만든 미세한 노이즈(0.12 이하)는 0으로 밀어버립니다.
    # 만약 흰색 선을 그렸는데 소리가 뚝뚝 끊기면 이 값을 0.08~0.1 사이로 낮춰보세요.
    # -----------------------------------------------------------------
    raw_amp[raw_amp < 0.12] = 0.0  
    
    # 노이즈가 깔끔하게 잘려 나간 청정 상태에서 제곱(대비)을 먹입니다.
    amp_matrix = raw_amp ** 2
    # -----------------------------------------------------------------
    
    # 3. 가로축(시간축) 방향으로 진폭을 오디오 샘플 수만큼 부드럽게 선형 보간(Interpolation)
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    # 4. 오디오 신호 합성
    audio_matrix = base_waves * smooth_amps
    audio = np.sum(audio_matrix, axis=0)
    
    # 5. 오디오 찢어짐(Clipping) 방지를 위한 소프트 리미터 (하이퍼볼릭 탄젠트)
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