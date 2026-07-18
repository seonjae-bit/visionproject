import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# --- 이 값들만 바꾸면 16, 32, 64 모두 부드럽게 작동합니다 ---
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.5  # 해상도가 높아지므로 스캔 시간을 조금 늘려주면 듣기 좋습니다.
FS = 44100
MIN_FREQ = 131.0
MAX_FREQ = 2093.0

# 총 샘플 수 및 기본 오디오 타임라인
total_samples = int(FS * FRAME_TIME)
t = np.arange(total_samples) / FS

# 주파수 배열 (위쪽이 고음, 아래쪽이 저음)
freqs = np.linspace(MAX_FREQ, MIN_FREQ, HEIGHT)

# 페이드 앰프(env) 대신, 처음부터 끝까지 이어지는 순수 거대한 사인파 기저 생성
# 모양: (HEIGHT, total_samples)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 카메라 지연 방지 스레드
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
            if ok: self.ok, self.frame = ok, frame
            else: time.sleep(0.01)

    def read(self): return self.ok, self.frame
    def release(self): self.running = False; self.cap.release()

cam = CameraStream()
print("Press q to quit.")

while True:
    ok, frame = cam.read()
    if not ok: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    # 1. 각 픽셀의 밝기(진폭) 매트릭스 계산 (HEIGHT, WIDTH)
    amp_matrix = (small.astype(np.float32) / 255.0) ** 2
    
    # 2. [핵심] 가로축(WIDTH) 방향의 진폭을 전체 오디오 샘플 수(total_samples)로 부드럽게 확대(보간)
    # 각 행(주파수)별로 뚝뚝 끊기는 밝기를 부드러운 곡선으로 만듭니다.
    x_old = np.linspace(0, 1, WIDTH)
    x_new = np.linspace(0, 1, total_samples)
    
    smooth_amps = np.zeros((HEIGHT, total_samples), dtype=np.float32)
    for r in range(HEIGHT):
        smooth_amps[r] = np.interp(x_new, x_old, amp_matrix[r, :])
    
    # 3. 부드럽게 변하는 진폭 곡선을 주파수 신호에 그대로 곱해줌
    audio_matrix = base_waves * smooth_amps
    
    # 4. 모든 주파수 성분을 하나로 합성
    audio = np.sum(audio_matrix, axis=0)
    
    # 볼륨 정규화 (찢어짐 방지)
    m = np.max(np.abs(audio))
    if m > 1e-6: 
        audio *= (0.9 / m)
        
    # 화면 출력 및 재생
    cv2.imshow("Sonification", cv2.resize(small, (320, 320), interpolation=cv2.INTER_NEAREST))
    sd.default.device = (None, 1)
    sd.play(audio, FS)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sd.stop()
        break
    sd.wait()

cam.release()
cv2.destroyAllWindows()