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
FRAME_TIME = 0.6
FS = 44100
MIN_FREQ = 220.0       # 220Hz (1옥타브 라)
MAX_FREQ = 880.0       # 880Hz (3옥타브 라)
TOTAL_SAMPLES = int(FS * FRAME_TIME)
DELAY_TIME = 0.06

# 1. 고정 주파수 사인파 생성 (sin(2*pi*f*t))
t = np.arange(TOTAL_SAMPLES) / FS
freqs = np.geomspace(MIN_FREQ, MAX_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 각 열의 중심 샘플 위치 (0 ~ 31)
col_centers = (np.arange(WIDTH) + 0.5) * (TOTAL_SAMPLES / WIDTH)
sample_indices = np.arange(TOTAL_SAMPLES)

# 저음 보정 가중치
freq_weights = np.linspace(0.8, 1.0, HEIGHT, dtype=np.float32)

# 스피커 팝 노이즈 방지용 3ms 극소 페이드
FADE_SAMPLES = int(FS * 0.003)
fade_envelope = np.ones(TOTAL_SAMPLES, dtype=np.float32)
fade_envelope[:FADE_SAMPLES] = np.linspace(0, 1, FADE_SAMPLES)
fade_envelope[-FADE_SAMPLES:] = np.linspace(1, 0, FADE_SAMPLES)


class CameraStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): raise RuntimeError("Camera open failed")
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
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
sd.default.device = None
print("Vision.py Running (Math Envelope Mode)... Press Ctrl+C to quit.")

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 화면 위 = 고음, 화면 아래 = 저음
        small_flipped = np.flipud(small)
        
        # 0 ~ 9 단계 양자화
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        
        # 밝기 진폭 단계 (0.0 ~ 1.0)
        amps_per_col = (small_step / 9.0) * freq_weights[:, None]  # Shape: (32, 32)

        # 🚀 [수식 구현] 각 시간(sample_indices) 위치에서 밝기값(amps_per_col)을
        # 연속된 선형 함수(Linear Piecewise Envelope)로 보간 연결함
        # f(x) = (1/2)*(x-9) + 1 과 완벽히 동일한 연속 직선 연결
        smooth_envelope = np.zeros((HEIGHT, TOTAL_SAMPLES), dtype=np.float32)
        for h in range(HEIGHT):
            smooth_envelope[h] = np.interp(sample_indices, col_centers, amps_per_col[h])

        # 연속된 진폭 f(x) 와 고정 주파수 sin(pi * x) 를 곱함: y = f(x) * sin(...)
        combined_wave = np.sum(base_waves * smooth_envelope, axis=0)
        
        # 3ms 극소 페이드 적용 (양 끝 팝 노이즈만 제거)
        combined_wave *= fade_envelope
        combined_wave /= (HEIGHT * 0.4)

        # 재생
        sd.play(combined_wave, FS)
        sd.wait()
        
        time.sleep(DELAY_TIME)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    sd.stop()
    cam.release()
    print("Program terminated successfully.")