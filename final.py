import cv2
import numpy as np
import sounddevice as sd
import threading
import time

# =====================================================================
# ⚙️ 기본 설정
# =====================================================================
WIDTH = 32
HEIGHT = 32
FRAME_TIME = 0.6       # 재생 시간 0.6초
FS = 44100
MIN_FREQ = 220.0       # 220Hz (1옥타브 라 / A3)
MAX_FREQ = 880.0       # 880Hz (3옥타브 라 / A5)
TOTAL_SAMPLES = int(FS * FRAME_TIME)

# 휴식 시간 (0.06초)
DELAY_TIME = FRAME_TIME * 0.10

# 1. 0.6초 전체 통째로 흐르는 시간(t) 축 생성
t = np.arange(TOTAL_SAMPLES) / FS

# 2. 220Hz ~ 880Hz 지수(Logarithmic) 스케일 배치
freqs = np.geomspace(MAX_FREQ, MIN_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 저음 20% 감쇄 가중치
freq_weights = np.linspace(1.0, 0.8, HEIGHT, dtype=np.float32)

# 3. 32개 열 보간 가중치 행렬 미리 계산 (CPU 최적화)
col_centers = (np.arange(WIDTH) + 0.5) * (TOTAL_SAMPLES / WIDTH)
sample_indices = np.arange(TOTAL_SAMPLES)

col_weights = np.array([
    np.interp(sample_indices, col_centers, (np.arange(WIDTH) == c).astype(float))
    for c in range(WIDTH)
], dtype=np.float32)


class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
            
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)

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


cam = CameraStream()
print("Vision.py Running (Linear Brightness, Log Scale 220~880Hz)... Press Ctrl+C to quit.")

sd.default.device = None

try:
    while True:
        ok, frame = cam.read()
        if not ok: 
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0  # 노이즈 제거
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 0 ~ 9 단계 양자화
        small_step = (small / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        
        # 🚀 [수정] 제곱(**2)을 제거하고 선형(Linear) 비율로 전달
        amps_per_col = small_step / 9.0
        amps_per_col *= freq_weights[:, None]

        # C언어 기반 NumPy 행렬곱 진폭 보간
        smooth_amps = np.dot(amps_per_col, col_weights)

        # 파형 합성 및 정규화
        combined_wave = np.sum(base_waves * smooth_amps, axis=0)
        
        m = np.max(np.abs(combined_wave))
        if m > 1e-6:
            combined_wave *= 0.9 / m

        # 오디오 재생
        sd.play(combined_wave, FS)
        sd.wait()
        
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")