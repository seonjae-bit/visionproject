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

t = np.arange(TOTAL_SAMPLES) / FS

# 1. 주파수 스케일 (220Hz ~ 880Hz 지수 배치)
freqs = np.geomspace(MIN_FREQ, MAX_FREQ, HEIGHT)
base_waves = np.array([np.sin(2 * np.pi * f * t) for f in freqs], dtype=np.float32)

# 2. 🚀 [핵심] 주파수별 맞춤형 S-Curve 보간 행렬 생성
# (HEIGHT, WIDTH, TOTAL_SAMPLES) -> 각 주파수(HEIGHT)마다 보간 곡선을 다르게 계산
col_centers = (np.arange(WIDTH) + 0.5) * (TOTAL_SAMPLES / WIDTH)
sample_indices = np.arange(TOTAL_SAMPLES)

col_weights_by_freq = np.zeros((HEIGHT, WIDTH, TOTAL_SAMPLES), dtype=np.float32)

for h, f in enumerate(freqs):
    # 주파수(f)가 낮을수록(저음) 잔상이 남지 않게 감쇄 곡선을 가파르게(exponent 높임) 설정
    # 고음(880Hz): p ~ 1.0 (완만) / 저음(220Hz): p ~ 2.2 (가파름)
    p = 1.0 + 1.2 * ((MAX_FREQ - f) / (MAX_FREQ - MIN_FREQ))
    
    for c in range(WIDTH):
        # 기본 삼각 가중치 (0.0 ~ 1.0)
        tri = np.interp(sample_indices, col_centers, (np.arange(WIDTH) == c).astype(float))
        # 주파수별 지수(S-Curve) 변환 적용하여 저음 잔상 억제
        col_weights_by_freq[h, c] = np.power(tri, p)


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
print("Vision.py Running (Frequency-Adaptive S-Curve applied)... Press Ctrl+C to quit.")

try:
    while True:
        ok, frame = cam.read()
        if not ok: continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[gray < 25] = 0
        
        small = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 상하 반전 (화면 위 = 고음, 화면 아래 = 저음)
        small_flipped = np.flipud(small)
        
        small_step = (small_flipped / 28.44).astype(np.float32)
        small_step = np.clip(small_step, 0.0, 9.0)
        amps = small_step / 9.0  # (32, 32)

        # 3. 주파수별 개별 보간 적용 (Einsum 연산)
        # amps: (H, W), col_weights_by_freq: (H, W, T) -> smooth_amps: (H, T)
        smooth_amps = np.einsum('hw,hwt->ht', amps, col_weights_by_freq)

        # 전체 파형 합성
        combined_wave = np.sum(base_waves * smooth_amps, axis=0)
        
        # 3ms 극소 페이드 및 음량 스케일링
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